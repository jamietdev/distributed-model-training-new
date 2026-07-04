"""Fault tolerance tests for remove-and-reshard recovery.

  test_static_reshard — kill a server without worrying abt training step, 
  reshard, verify every weight is still accounted for and the absorbed values match the checkpoint.

  test_live_reshard   — kill a server MID-TRAINING, reshard, resume, verify accuracy continues to improve.
"""
import os
import sys
import time
import ray
import matplotlib.pyplot as plt
from typing import Dict
import shutil
from cluster import build_cluster, teardown_cluster
from config import (
    CHECKPOINT_EVERY,
    LEARNING_RATE,
    NUM_WEIGHTS,
    SyncMode,
    NUM_WORKERS,
    NUM_SERVERS,
    NUM_REPLICAS,
)
import config
from load_mnist import load_mnist_data
from main import clear_checkpoints, evaluate_global_model, gather_full_weights
from recovery import reshard_after_failure
RUN_MODE = "experiment"

def _run_iterations(workers, start, end):
    for it in range(start, end):
        ray.get([w.run_iteration.remote(it) for w in workers])


def _kill_iteration_multiple_of_checkpoint(n):
    if n % CHECKPOINT_EVERY == 0:
        return n
    return ((n // CHECKPOINT_EVERY) + 1) * CHECKPOINT_EVERY


def test_static_reshard():
    print("\n=== Static reshard ===")
    clear_checkpoints()
    X_train, y_train, _, _ = load_mnist_data()

    ring, servers, workers = build_cluster(
        num_workers=NUM_WORKERS,
        num_servers=NUM_SERVERS,
        num_weights=NUM_WEIGHTS,
        num_replicas=NUM_REPLICAS,
        learning_rate=LEARNING_RATE,
        X_train=X_train,
        y_train=y_train,
        sync_mode=SyncMode.SEQUENTIAL_BSP,
    )

    try:
        kill_after = _kill_iteration_multiple_of_checkpoint(10)
        _run_iterations(workers, 0, kill_after)

        victim_id = "server_0"
        victim_indices = ring.indices_for_server(victim_id)
        pre_kill_orphans: Dict[int, float] = ray.get(
            servers[victim_id].pull_weights.remote(victim_indices)
        )
        print(f"  killing {victim_id} with {len(victim_indices)} weights "
              f"at iter {kill_after}")

        ray.kill(servers[victim_id])

        info = reshard_after_failure(
            dead_server_id=victim_id,
            ring=ring,
            servers=servers,
            workers=workers,
        )
        print(f"  reshard info: {info}")


        # verify thatevery orphaned weight ended up on exactly one survivor, and its value matches the checkpoint
        for orphan_idx, expected_val in pre_kill_orphans.items():
            new_owner = ring.get_server(orphan_idx)
            assert new_owner != victim_id, (
                f"weight {orphan_idx} still routes to dead {victim_id}"
            )
            assert new_owner in servers, (
                f"weight {orphan_idx} routes to unknown {new_owner}"
            )
            actual = ray.get(
                servers[new_owner].pull_weights.remote([orphan_idx])
            )
            assert abs(actual[orphan_idx] - expected_val) < 1e-9, (
                f"weight {orphan_idx} wrong after absorb: "
                f"expected {expected_val}, got {actual[orphan_idx]}"
            )

        # verify that the union of all survivor shards covers every weight index exactly once
        total_owned = 0
        for sid, handle in servers.items():
            idxs = ray.get(handle.get_weight_indices.remote())
            total_owned += len(idxs)
        assert total_owned == NUM_WEIGHTS, (
            f"weight accounting broken: {total_owned} owned total, "
            f"expected {NUM_WEIGHTS}"
        )

        print(f"  all {len(pre_kill_orphans)} orphans correctly absorbed")
        print(f"  all {NUM_WEIGHTS} weights accounted for across survivors")
        print("  PASS")
    finally:
        teardown_cluster(servers, workers)


def run_live_reshard_trial(kill_after):
    if os.path.exists(config.CHECKPOINT_DIR):
        shutil.rmtree(config.CHECKPOINT_DIR)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    clear_checkpoints()
    X_train, y_train, X_test, y_test = load_mnist_data()

    ring, servers, workers = build_cluster(
        num_workers=NUM_WORKERS,
        num_servers=NUM_SERVERS,
        num_weights=NUM_WEIGHTS,
        num_replicas=NUM_REPLICAS,
        learning_rate=LEARNING_RATE,
        X_train=X_train,
        y_train=y_train,
        sync_mode=SyncMode.SEQUENTIAL_BSP,
    )

    try:
        print(f"\n=== Trial: kill_after={kill_after} ===")    
        _run_iterations(workers, 0, kill_after)

        pre_weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
        pre_kill_acc = evaluate_global_model(pre_weights, X_test, y_test)

        victim_id = "server_1"

        # start timing
        start = time.time()
        ray.kill(servers[victim_id])
        reshard_after_failure(
            dead_server_id=victim_id,
            ring=ring,
            servers=servers,
            workers=workers,
        )
        ray.get([w.run_iteration.remote(kill_after) for w in workers])
        recovery_time = time.time() - start

        # fewer servers now, but workers have the new ring.
        final_weights = gather_full_weights(servers, ring, NUM_WEIGHTS)
        final_acc = evaluate_global_model(final_weights, X_test, y_test)

        return {
            "kill_after": kill_after,
            "pre_acc": pre_kill_acc,
            "final_acc": final_acc,
            "recovery_time": recovery_time,
        } 
    finally:
        teardown_cluster(servers, workers)

def run_experiments():
    print(f"\n[MODE] {config.RECOVERY_MODE}")

    kill_points = [40, 60, 80, 100]
    results = []
    for k in kill_points:
        try:
            res = run_live_reshard_trial(k)
            results.append(res)
        except Exception as e:
            print(f"Failed at k={k}: {e}")
            results.append({
                "kill_after": k,
                "recovery_time": None
            })
    return results

def plot_results(all_results):
    plt.figure()

    for mode, results in all_results.items():
        kill = []
        recovery = []
        for r in results:
            if r["recovery_time"] is not None:
                kill.append(r["kill_after"])
                recovery.append(r["recovery_time"])
            else:
                plt.scatter(r["kill_after"], 0, marker="x", color="red")

        plt.scatter(kill, recovery, marker="o", label=mode)

    plt.title("Recovery Time Comparison")
    plt.xlabel("Failure Iteration")
    plt.ylabel("Recovery Time (seconds)")
    plt.legend()

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "plots",
        "fault_tolerance",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "recovery_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    try:
        if RUN_MODE == "test":
            test_static_reshard()
            print("\nStatic reshard tests passed.")

        elif RUN_MODE == "experiment":
            all_results = {}

            for mode in ["chain", "checkpoint"]:
                config.RECOVERY_MODE = mode
                results = run_experiments()
                all_results[mode] = results

            plot_results(all_results)
        

    finally:
        ray.shutdown()
