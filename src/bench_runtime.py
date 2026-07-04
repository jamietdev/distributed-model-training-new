import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
import ray

from cluster import build_cluster, teardown_cluster
from hash_ring import HashRing
from load_mnist import load_mnist_data
from config import (
    LEARNING_RATE,
    NUM_WEIGHTS,
    WARMUP_ITERS,
    TIMED_ITERS,
    ITER_TIMEOUT_S,
    NUM_WORKERS,
    NUM_SERVERS,
    NUM_REPLICAS,
    SyncMode,
)

def summarize(samples, label):
    s = sorted(samples)
    n = len(s)
    mean = statistics.mean(s)
    std = statistics.stdev(s) if n > 1 else 0.0
    p50 = s[n // 2]
    p95 = s[min(int(n * 0.95), n - 1)]
    throughput = 1.0 / mean if mean > 0 else float("inf")
    print(
        f"  {label:30s} n={n:3d}  "
        f"mean={mean*1000:7.2f}ms  std={std*1000:6.2f}ms  "
        f"p50={p50*1000:7.2f}ms  p95={p95*1000:7.2f}ms  "
        f"throughput={throughput:6.1f} it/s"
    )
    return {"mean": mean, "std": std, "p50": p50, "p95": p95, "throughput": throughput}


def run_one_trial(
    num_workers,
    num_servers,
    num_weights,
    num_iterations,
    num_replicas,
    learning_rate,
    X_train,
    y_train,
    warmup_iters=WARMUP_ITERS,
    sync_mode: SyncMode = SyncMode.SEQUENTIAL_BSP,
):
    _, servers, workers = build_cluster(
        num_workers=num_workers,
        num_servers=num_servers,
        num_weights=num_weights,
        num_replicas=num_replicas,
        learning_rate=learning_rate,
        X_train=X_train,
        y_train=y_train,
        sync_mode=sync_mode,
    )

    # timed loop: one sample = wall time for all workers to finish the same global iteration
    iter_times = []
    try:
        for it in range(num_iterations):
            t0 = time.perf_counter()
            ray.get(
                [w.run_iteration.remote(it) for w in workers],
                timeout=ITER_TIMEOUT_S,
            )
            iter_times.append(time.perf_counter() - t0)
    except ray.exceptions.GetTimeoutError:
        print(
            f"  !! iteration {len(iter_times)} timed out after "
            f"{ITER_TIMEOUT_S}s — likely the pull_weights busy-wait deadlock"
        )
    finally:
        teardown_cluster(servers, workers)

    return iter_times[warmup_iters:]


def run_async_trial(
    num_workers,
    num_servers,
    num_weights,
    num_iterations,
    num_replicas,
    learning_rate,
    X_train,
    y_train,
    warmup_iters=WARMUP_ITERS,
):
    """
    Time asynchronous training: each sample is wall-clock for one "round" where
    every worker runs train_loop_async(1) (one local step per worker, concurrent).
    Throughput in rounds/s is comparable in spirit to run_one_trial (one global
    sync iteration = all workers finish one step).
    """
    _, servers, workers = build_cluster(
        num_workers=num_workers,
        num_servers=num_servers,
        num_weights=num_weights,
        num_replicas=num_replicas,
        learning_rate=learning_rate,
        X_train=X_train,
        y_train=y_train,
        sync_mode=SyncMode.ASYNCHRONOUS,
    )
    step_times: list[float] = []
    try:
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            ray.get(
                [w.train_loop_async.remote(1) for w in workers],
                timeout=ITER_TIMEOUT_S,
            )
            step_times.append(time.perf_counter() - t0)
    except ray.exceptions.GetTimeoutError:
        print(
            f"  !! async step {len(step_times)} timed out after "
            f"{ITER_TIMEOUT_S}s"
        )
    finally:
        teardown_cluster(servers, workers)

    return step_times[warmup_iters:]


def bench_baseline(X_train, y_train):
    print("\n=== Baseline (workers=6, servers=2, replicas=2) ===")
    times = run_one_trial(
        num_workers=6, num_servers=2,
        num_weights=NUM_WEIGHTS, num_iterations=TIMED_ITERS + WARMUP_ITERS,
        num_replicas=2, learning_rate=LEARNING_RATE,
        X_train=X_train, y_train=y_train,
    )
    summarize(times, "per-iter")
    print(f"  total wall-clock (timed): {sum(times):.2f}s")


def bench_scaling_workers(X_train, y_train):
    """Hold servers fixed, vary workers. Measures synchronous iteration latency
    and global iteration throughput."""
    print("\n=== Scaling: num_workers (servers=2, replicas=2) ===")

    results = []
    for nw in [1, 2, 4, 6, 8]:
        times = run_one_trial(
            num_workers=nw,
            num_servers=2,
            num_weights=NUM_WEIGHTS,
            num_iterations=TIMED_ITERS + WARMUP_ITERS,
            num_replicas=2,
            learning_rate=LEARNING_RATE,
            X_train=X_train,
            y_train=y_train,
        )

        # skip runs that timed out or are incomplete
        if len(times) < TIMED_ITERS:
            print(f"Skipping workers={nw} due to timeout")
            continue

        stats = summarize(times, f"workers={nw}")
        results.append((nw, stats["mean"], stats["throughput"]))

    x = [r[0] for r in results]
    latency_ms = [r[1] * 1000 for r in results]  
    throughput = [r[2] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(x, latency_ms, marker='o')
    ax1.set_title("Iteration Latency")
    ax1.set_xlabel("Number of Workers")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid()
    ax2.plot(x, throughput, marker='x')
    ax2.set_title("Iteration Throughput")
    ax2.set_xlabel("Number of Workers")
    ax2.set_ylabel("Throughput (iterations/sec)")
    ax2.grid()

    plt.suptitle("Scaling Workers (Synchronous Training)")
    plt.tight_layout()
    plt.show()

def bench_scaling_servers(X_train, y_train):
    """Hold workers fixed, vary servers. Measures synchronous iteration latency
    and global iteration throughput."""
    print("\n=== Scaling: num_servers (workers=6, replicas=2) ===")

    results = []
    for ns in [1, 2, 4, 8]:
        times = run_one_trial(
            num_workers=NUM_WORKERS,
            num_servers=ns,
            num_weights=NUM_WEIGHTS,
            num_iterations=TIMED_ITERS + WARMUP_ITERS,
            num_replicas=NUM_REPLICAS,
            learning_rate=LEARNING_RATE,
            X_train=X_train,
            y_train=y_train,
        )

        # Skip incomplete runs (timeouts)
        if len(times) < TIMED_ITERS:
            print(f"Skipping servers={ns} due to timeout")
            continue

        stats = summarize(times, f"servers={ns}")
        results.append((ns, stats["mean"], stats["throughput"]))

    x = [r[0] for r in results]
    latency_ms = [r[1] * 1000 for r in results]
    throughput = [r[2] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(x, latency_ms, marker='o')
    ax1.set_title("Iteration Latency")
    ax1.set_xlabel("Number of Servers")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid()
    ax2.plot(x, throughput, marker='x')
    ax2.set_title("Iteration Throughput")
    ax2.set_xlabel("Number of Servers")
    ax2.set_ylabel("Throughput (iterations/sec)")
    ax2.grid()

    plt.suptitle("Scaling Servers (Synchronous)")
    plt.tight_layout()
    plt.show()

def bench_replicas(X_train, y_train):
    """Vary virtual nodes per server. Measures effect on load balance and
    steady-state iteration performance."""
    print("\n=== Virtual-node count (workers=6, servers=4) ===")

    results = []
    for nr in [1, 2, 10, 50, 200]:
        times = run_one_trial(
            num_workers=NUM_WORKERS,
            num_servers=4,
            num_weights=NUM_WEIGHTS,
            num_iterations=TIMED_ITERS + WARMUP_ITERS,
            num_replicas=nr,
            learning_rate=LEARNING_RATE,
            X_train=X_train,
            y_train=y_train,
        )

        # skip incomplete runs
        if len(times) < TIMED_ITERS:
            print(f"Skipping replicas={nr} due to timeout")
            continue

        stats = summarize(times, f"replicas={nr}")
        results.append((nr, stats["mean"], stats["throughput"]))

    x = [r[0] for r in results]
    latency_ms = [r[1] * 1000 for r in results]
    throughput = [r[2] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(x, latency_ms, marker='o')
    ax1.set_title("Iteration Latency")
    ax1.set_xlabel("Number of Virtual Nodes (Replicas)")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid()
    ax2.plot(x, throughput, marker='x')
    ax2.set_title("Iteration Throughput")
    ax2.set_xlabel("Number of Virtual Nodes (Replicas)")
    ax2.set_ylabel("Throughput (iterations/sec)")
    ax2.grid()

    plt.suptitle("Effect of Consistent Hashing Replication")
    plt.tight_layout()
    plt.show()

def bench_load_balance():
    """Not a runtime metric per se, but worth printing: how lopsided
    is the weight-to-server assignment at low replica counts?"""
    print("\n=== Load distribution by replica count (servers=4) ===")
    for nr in [1, 2, 10, 50, 200]:
        ring = HashRing(NUM_WEIGHTS, nr)
        for i in range(4):
            ring.add_server(f"server_{i}")
        by_server = ring.all_server_indices()
        counts = sorted(len(v) for v in by_server.values())
        ideal = NUM_WEIGHTS / 4
        max_deviation = max(abs(c - ideal) for c in counts) / ideal
        print(
            f"  replicas={nr:3d}  shard sizes={counts}  "
            f"max_dev={max_deviation*100:5.1f}% from ideal ({ideal:.1f})"
        )


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    X_train, y_train, _, _ = load_mnist_data()

    bench_baseline(X_train, y_train)
    bench_load_balance()
    bench_scaling_workers(X_train, y_train)
    bench_scaling_servers(X_train, y_train)
    bench_replicas(X_train, y_train)

    ray.shutdown()