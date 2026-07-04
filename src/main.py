from collections import defaultdict
import random
import time

import ray
import numpy as np
import os

from cluster import build_cluster, teardown_cluster
from config import (
    LEARNING_RATE,
    NUM_ITERATIONS,
    NUM_REPLICAS,
    NUM_SERVERS,
    NUM_WEIGHTS,
    NUM_WORKERS,
    SYNC_MODE,
    SyncMode,
    CHECKPOINT_DIR,
    SEED,
)
from load_mnist import load_mnist_data
from sidecar_evaluator import SidecarEvaluator

def gather_full_weights(servers, hash_ring, num_weights):
    full_weights = np.zeros(num_weights)
    keys_by_server = defaultdict(list)
    for i in range(num_weights):
        server_id = hash_ring.get_server(i)
        keys_by_server[server_id].append(i)

    for server_id, indices in keys_by_server.items():
        weights_ref = servers[server_id].pull_weights.remote(indices)
        weights_dict: dict[int, float] = ray.get(weights_ref)
        for idx, val in weights_dict.items():
            full_weights[idx] = val

    return full_weights


def evaluate_global_model(weights, X_test, y_test):
    logits = X_test @ weights
    preds = 1 / (1 + np.exp(-logits))
    preds = (preds >= 0.5).astype(np.float32)
    return np.mean(preds == y_test)

def clear_checkpoints():
       if not os.path.isdir(CHECKPOINT_DIR):
           os.makedirs(CHECKPOINT_DIR, exist_ok=True)
           return
       for f in os.listdir(CHECKPOINT_DIR):
           path = os.path.join(CHECKPOINT_DIR, f)
           if os.path.isfile(path):
               os.remove(path)


def run_training(
    num_workers,
    num_weights,
    learning_rate,
    sync_mode=SYNC_MODE,
    num_iterations: int | None = None,
    eval_every: int = 10,
    random_seed: int | None = None,
    return_wall_time: bool = False,
):
    if num_iterations is None:
        num_iterations = NUM_ITERATIONS
    rng_seed = SEED if random_seed is None else random_seed

    np.random.seed(rng_seed)
    random.seed(rng_seed)
    X_train, y_train, X_test, y_test = load_mnist_data()
    clear_checkpoints()
    # Reseed before sharding: shard_data() uses np.random; cluster uses random.Random(rng_seed)
    # for weight init, so re-seed here to match a fresh run's shard split.
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    ring, servers, workers = build_cluster(
        num_workers=num_workers,
        num_servers=NUM_SERVERS,
        num_weights=num_weights,
        num_replicas=NUM_REPLICAS,
        learning_rate=learning_rate,
        X_train=X_train,
        y_train=y_train,
        sync_mode=sync_mode,
        weight_init_seed=rng_seed,
    )

    training_history = []
    loop_start = time.perf_counter()
    try:
        w0 = gather_full_weights(servers, ring, num_weights)
        acc0 = evaluate_global_model(w0, X_test, y_test)
        training_history.append(
            {
                "iteration": -1,
                "steps_per_worker": 0,
                "before_training": True,
                "accuracy": float(acc0),
            }
        )
        if sync_mode == SyncMode.ASYNCHRONOUS:
            print(f"Before training (async): Accuracy = {acc0:.4f}")
        else:
            print(f"Before training (sequential BSP): Accuracy = {acc0:.4f}")

        if sync_mode == SyncMode.ASYNCHRONOUS:
            remaining = num_iterations
            total_steps = 0
            while remaining > 0:
                chunk = min(eval_every, remaining)
                ray.get([w.train_loop_async.remote(chunk) for w in workers])
                total_steps += chunk
                remaining -= chunk
                weights = gather_full_weights(servers, ring, num_weights)
                acc = evaluate_global_model(weights, X_test, y_test)
                print(f"After {total_steps} steps per worker (async): Accuracy = {acc:.4f}")
                training_history.append(
                    {
                        "iteration": int(total_steps),
                        "steps_per_worker": int(total_steps),
                        "before_training": False,
                        "accuracy": float(acc),
                    }
                )

        else:
            for iteration in range(num_iterations):
                ray.get([worker.run_iteration.remote(iteration) for worker in workers])
                if iteration % eval_every == 0:
                    weights = gather_full_weights(servers, ring, num_weights)
                    acc = evaluate_global_model(weights, X_test, y_test)
                    print(
                        f"Iteration {iteration} (sequential BSP): Accuracy = {acc:.4f}"
                    )
                    training_history.append(
                        {
                            "iteration": int(iteration),
                            "before_training": False,
                            "accuracy": float(acc),
                        }
                    )

        loop_wall_time = time.perf_counter() - loop_start
    finally:
        teardown_cluster(servers, workers)

    if return_wall_time:
        return training_history, loop_wall_time
    return training_history


def run_training_with_sidecar(
    num_workers,
    num_weights,
    learning_rate,
    sync_mode=SYNC_MODE,
    num_iterations: int | None = None,
    checkpoint_every: int = 10,
    random_seed: int | None = None,
):
    """
    Training with sidecar evaluation: the driver never pauses workers to
    evaluate. Servers checkpoint every `checkpoint_every` iterations and a
    SidecarEvaluator actor evaluates each checkpoint as it appears, so the
    accuracy curve is produced at checkpoint cadence with zero training stalls.

    Returns (history, train_wall_time). History entries carry iteration_min /
    iteration_max because a snapshot assembled from independent per-server
    checkpoints can span iterations (torn snapshot).
    """
    if num_iterations is None:
        num_iterations = NUM_ITERATIONS
    rng_seed = SEED if random_seed is None else random_seed

    np.random.seed(rng_seed)
    random.seed(rng_seed)
    X_train, y_train, X_test, y_test = load_mnist_data()
    clear_checkpoints()
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    ring, servers, workers = build_cluster(
        num_workers=num_workers,
        num_servers=NUM_SERVERS,
        num_weights=num_weights,
        num_replicas=NUM_REPLICAS,
        learning_rate=learning_rate,
        X_train=X_train,
        y_train=y_train,
        sync_mode=sync_mode,
        weight_init_seed=rng_seed,
        checkpoint_every=checkpoint_every,
    )
    evaluator = SidecarEvaluator.remote(X_test, y_test, num_weights)

    try:
        ray.get(evaluator.start.remote())
        start = time.perf_counter()

        if sync_mode == SyncMode.ASYNCHRONOUS:
            ray.get([w.train_loop_async.remote(num_iterations) for w in workers])
        else:
            for iteration in range(num_iterations):
                ray.get([w.run_iteration.remote(iteration) for w in workers])

        train_wall_time = time.perf_counter() - start
        # make sure in-flight background checkpoint writes are on disk before
        # the evaluator takes its final poll
        ray.get([s.flush_checkpoints.remote() for s in servers.values()])
        ray.get(evaluator.stop.remote())
        history = ray.get(evaluator.get_history.remote())
    finally:
        teardown_cluster(servers, workers)
        ray.kill(evaluator)

    return history, train_wall_time


if __name__ == "__main__":
    ray.shutdown()
    ray.init()
    run_training(NUM_WORKERS, NUM_WEIGHTS, LEARNING_RATE, sync_mode=SYNC_MODE)
