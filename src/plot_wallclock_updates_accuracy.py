"""
End-to-end Ray benchmark: wall-clock weight updates/sec (fixed steps per
worker) and test accuracy vs wall-clock time, BSP and async sync modes.

Writes PNGs to plots/plot_wallclock_updates_accuracy/:
  throughput_workers.png, throughput_servers.png, throughput_replicas.png,
  accuracy_vs_time.png

  python src/plot_wallclock_updates_accuracy.py
"""
from __future__ import annotations

import os
import random
import sys
import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import ray

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from cluster import build_cluster, teardown_cluster
from config import (
    LEARNING_RATE,
    NUM_WEIGHTS,
    SEED,
    SyncMode,
)
from load_mnist import load_mnist_data
from main import evaluate_global_model, gather_full_weights

WORKER_SWEEP = [1, 2, 4, 6, 8]
SERVER_SWEEP = [1, 2, 4, 8]
REPLICA_SWEEP = [1, 2, 10, 50, 200]

FIXED_WORKERS = 6
FIXED_SERVERS = 2
FIXED_REPLICA_WORKERS = 6
FIXED_REPLICA_SERVERS = 4
NUM_RING_REPLICAS = 2

STEPS_PER_WORKER = 100
ACCURACY_TRAIN_STEPS = 100
ACCURACY_EVAL_EVERY = 10

SYNC_LABELS = {
    SyncMode.SEQUENTIAL_BSP: "BSP",
    SyncMode.ASYNCHRONOUS: "async",
}


def _reset_seeds() -> None:
    """Shard placement uses np.random; weight init uses random.Random(SEED) in build_cluster."""
    np.random.seed(SEED)
    random.seed(SEED)


def _run_training_chunk(workers, n_steps: int, mode: SyncMode) -> None:
    if mode == SyncMode.ASYNCHRONOUS:
        ray.get([w.train_loop_async.remote(n_steps) for w in workers])
    else:
        raise ValueError(f"Not a polling mode: {mode}")


def run_throughput_trial(
    num_workers: int,
    num_servers: int,
    num_replicas: int,
    X,
    y,
    mode: SyncMode,
) -> float:
    _reset_seeds()
    ring, servers, workers = build_cluster(
        num_workers=num_workers,
        num_servers=num_servers,
        num_weights=NUM_WEIGHTS,
        num_replicas=num_replicas,
        learning_rate=LEARNING_RATE,
        X_train=X,
        y_train=y,
        sync_mode=mode,
    )

    start = time.perf_counter()

    if mode == SyncMode.ASYNCHRONOUS:
        ray.get([w.train_loop_async.remote(STEPS_PER_WORKER) for w in workers])
    else:
        for it in range(STEPS_PER_WORKER):
            ray.get([w.run_iteration.remote(it) for w in workers])

    elapsed = time.perf_counter() - start
    teardown_cluster(servers, workers)

    total_updates = num_workers * STEPS_PER_WORKER
    return total_updates / elapsed


def run_accuracy_curve(
    num_workers: int,
    num_servers: int,
    num_replicas: int,
    X,
    y,
    X_test,
    y_test,
    mode: SyncMode,
) -> tuple[list[float], list[float]]:
    _reset_seeds()
    ring, servers, workers = build_cluster(
        num_workers=num_workers,
        num_servers=num_servers,
        num_weights=NUM_WEIGHTS,
        num_replicas=num_replicas,
        learning_rate=LEARNING_RATE,
        X_train=X,
        y_train=y,
        sync_mode=mode,
    )

    times: list[float] = []
    accs: list[float] = []
    start = time.perf_counter()
    total_steps = 0

    try:
        if mode == SyncMode.ASYNCHRONOUS:
            weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
            acc = evaluate_global_model(weights, X_test, y_test)
            times.append(0.0)
            accs.append(acc)
            while total_steps < ACCURACY_TRAIN_STEPS:
                _run_training_chunk(workers, ACCURACY_EVAL_EVERY, mode)
                total_steps += ACCURACY_EVAL_EVERY
                weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
                acc = evaluate_global_model(weights, X_test, y_test)
                times.append(time.perf_counter() - start)
                accs.append(acc)
        else:
            weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
            acc = evaluate_global_model(weights, X_test, y_test)
            times.append(0.0)
            accs.append(acc)
            for it in range(ACCURACY_TRAIN_STEPS):
                ray.get([w.run_iteration.remote(it) for w in workers])
                if it % ACCURACY_EVAL_EVERY == 0:
                    weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
                    acc = evaluate_global_model(weights, X_test, y_test)
                    times.append(time.perf_counter() - start)
                    accs.append(acc)
    finally:
        teardown_cluster(servers, workers)

    return times, accs


def _sweep_throughput_1d(
    x_values: list[int],
    X,
    y,
    triplet: Callable[[int], tuple[int, int, int]],
) -> dict[SyncMode, list[float]]:
    out: dict[SyncMode, list[float]] = {mode: [] for mode in SyncMode}
    for x in x_values:
        nw, ns, nr = triplet(x)
        for mode in SyncMode:
            t = run_throughput_trial(nw, ns, nr, X, y, mode)
            out[mode].append(t)
    return out


def _plot_throughput_lines(
    x_values: list[int],
    results: dict[SyncMode, list[float]],
    xlabel: str,
    title: str,
    out_path: str,
    logx: bool = False,
) -> None:
    plt.figure()
    for mode in SyncMode:
        plt.plot(
            x_values,
            results[mode],
            marker="o",
            label=SYNC_LABELS[mode],
        )
    plt.xlabel(xlabel)
    plt.ylabel("Updates / sec")
    plt.title(title)
    if logx:
        plt.xscale("log")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    out_dir = os.path.join(
        os.path.dirname(_SRC), "plots", "plot_wallclock_updates_accuracy"
    )
    os.makedirs(out_dir, exist_ok=True)

    _reset_seeds()
    X, y, X_test, y_test = load_mnist_data()

    w = _sweep_throughput_1d(
        WORKER_SWEEP,
        X,
        y,
        lambda nw: (nw, FIXED_SERVERS, NUM_RING_REPLICAS),
    )
    _plot_throughput_lines(
        WORKER_SWEEP,
        w,
        "Workers",
        "Throughput vs Workers",
        os.path.join(out_dir, "throughput_workers.png"),
    )

    s = _sweep_throughput_1d(
        SERVER_SWEEP,
        X,
        y,
        lambda ns: (FIXED_WORKERS, ns, NUM_RING_REPLICAS),
    )
    _plot_throughput_lines(
        SERVER_SWEEP,
        s,
        "Servers",
        "Throughput vs Servers",
        os.path.join(out_dir, "throughput_servers.png"),
    )

    r = _sweep_throughput_1d(
        REPLICA_SWEEP,
        X,
        y,
        lambda nr: (FIXED_REPLICA_WORKERS, FIXED_REPLICA_SERVERS, nr),
    )
    _plot_throughput_lines(
        REPLICA_SWEEP,
        r,
        "Replicas",
        "Throughput vs Replicas",
        os.path.join(out_dir, "throughput_replicas.png"),
        logx=True,
    )

    out_accuracy = os.path.join(out_dir, "accuracy_vs_time.png")
    plt.figure()
    for mode in SyncMode:
        times, accs = run_accuracy_curve(
            FIXED_WORKERS,
            FIXED_SERVERS,
            NUM_RING_REPLICAS,
            X,
            y,
            X_test,
            y_test,
            mode,
        )
        plt.plot(times, accs, marker="o", label=SYNC_LABELS[mode])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy vs Wall-Clock Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_accuracy, dpi=150)
    plt.close()
    print(f"Wrote {out_accuracy}")

    ray.shutdown()


if __name__ == "__main__":
    main()
