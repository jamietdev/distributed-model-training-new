"""
Sidecar evaluation benchmark: inline eval (driver pauses all workers at every
eval point) vs sidecar eval (separate actor evaluates checkpoints; training
never stalls).

Outputs to plots/sidecar_eval/:
  accuracy_vs_time.png   — accuracy curves, inline vs sidecar, BSP and async
  training_wall_time.png — wall-clock to finish the same number of steps

  python src/bench_sidecar_eval.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import ray

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import LEARNING_RATE, NUM_WEIGHTS, NUM_WORKERS, SyncMode
from main import run_training, run_training_with_sidecar

NUM_ITERATIONS = 100
EVAL_EVERY = 10  # inline eval cadence and sidecar checkpoint cadence

MODES = [
    (SyncMode.SEQUENTIAL_BSP, "BSP"),
    (SyncMode.ASYNCHRONOUS, "async"),
]


def run_inline(mode: SyncMode) -> tuple[list[dict], float]:
    history, wall_time = run_training(
        NUM_WORKERS,
        NUM_WEIGHTS,
        LEARNING_RATE,
        sync_mode=mode,
        num_iterations=NUM_ITERATIONS,
        eval_every=EVAL_EVERY,
        return_wall_time=True,
    )
    return history, wall_time


def main() -> None:
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    out_dir = os.path.join(os.path.dirname(_SRC), "plots", "sidecar_eval")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for mode, label in MODES:
        print(f"\n=== {label}: inline eval ===")
        _, inline_total = run_inline(mode)
        print(f"  total wall time (train + eval stalls): {inline_total:.2f}s")

        print(f"=== {label}: sidecar eval ===")
        history, train_time = run_training_with_sidecar(
            NUM_WORKERS,
            NUM_WEIGHTS,
            LEARNING_RATE,
            sync_mode=mode,
            num_iterations=NUM_ITERATIONS,
            checkpoint_every=EVAL_EVERY,
        )
        print(f"  training wall time (no stalls): {train_time:.2f}s")
        print(f"  sidecar evaluated {len(history)} checkpoint versions")
        for h in history:
            torn = "" if h["iteration_min"] == h["iteration_max"] else (
                f"  [torn: iters {h['iteration_min']}-{h['iteration_max']}]"
            )
            print(
                f"    t={h['time']:.2f}s  iter={h['iteration_max']}  "
                f"acc={h['accuracy']:.4f}{torn}"
            )
        results[label] = {
            "inline_total": inline_total,
            "sidecar_train": train_time,
            "sidecar_history": history,
        }

    # accuracy vs iteration: sidecar curve per mode
    plt.figure()
    for label, r in results.items():
        h = r["sidecar_history"]
        plt.plot(
            [p["iteration_max"] for p in h],
            [p["accuracy"] for p in h],
            marker="o",
            label=f"{label} (sidecar)",
        )
    plt.xlabel("Iteration (checkpoint version)")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy from Sidecar Evaluation (training never stalled)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, "accuracy_vs_iteration.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nWrote {out}")

    # wall-clock comparison: same steps, inline vs sidecar
    plt.figure()
    labels = list(results.keys())
    x = range(len(labels))
    width = 0.35
    inline_times = [results[l]["inline_total"] for l in labels]
    sidecar_times = [results[l]["sidecar_train"] for l in labels]
    plt.bar([i - width / 2 for i in x], inline_times, width, label="inline eval")
    plt.bar([i + width / 2 for i in x], sidecar_times, width, label="sidecar eval")
    plt.xticks(list(x), labels)
    plt.ylabel("Wall-clock time (s)")
    plt.title(f"Time to run {NUM_ITERATIONS} steps/worker (eval every {EVAL_EVERY})")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(out_dir, "training_wall_time.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")

    ray.shutdown()


if __name__ == "__main__":
    main()
