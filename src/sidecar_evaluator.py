import json
import os
import threading
import time

import numpy as np
import ray

from config import CHECKPOINT_DIR


def _read_checkpoint_files(checkpoint_dir):
    """Read every per-server checkpoint. Returns None if the directory is
    missing/empty or any file is unreadable (retry on the next poll)."""
    if not os.path.isdir(checkpoint_dir):
        return None
    snapshots = {}
    for fname in os.listdir(checkpoint_dir):
        if not (fname.startswith("checkpoint_") and fname.endswith(".json")):
            continue
        try:
            with open(os.path.join(checkpoint_dir, fname), "r") as f:
                snapshots[fname] = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    return snapshots or None


@ray.remote
class SidecarEvaluator:
    def __init__(
        self,
        X_test,
        y_test,
        num_weights,
        checkpoint_dir: str = CHECKPOINT_DIR,
        poll_interval_s: float = 0.02,
    ):
        self.X_test = X_test
        self.y_test = y_test
        self.num_weights = num_weights
        self.checkpoint_dir = checkpoint_dir
        self.poll_interval_s = poll_interval_s
        self._history = []
        self._last_version = None
        self._start_time = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        self._start_time = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._poll_once()

    def get_history(self):
        with self._lock:
            return list(self._history)

    def _loop(self):
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.poll_interval_s)

    def _poll_once(self):
        snapshots = _read_checkpoint_files(self.checkpoint_dir)
        if snapshots is None:
            return

        version = tuple(
            sorted((fname, data["iteration"]) for fname, data in snapshots.items())
        )
        if version == self._last_version:
            return

        weights = np.zeros(self.num_weights)
        covered: set[int] = set()
        for data in snapshots.values():
            for idx, val in data["weights"].items():
                weights[int(idx)] = val
                covered.add(int(idx))
        if len(covered) < self.num_weights:
            # not every server has checkpointed yet; wait for full coverage
            return

        accuracy = self._evaluate(weights)
        iterations = [data["iteration"] for data in snapshots.values()]
        with self._lock:
            self._history.append(
                {
                    "time": time.perf_counter() - self._start_time,
                    "iteration_min": int(min(iterations)),
                    "iteration_max": int(max(iterations)),
                    "accuracy": float(accuracy),
                }
            )
        self._last_version = version

    def _evaluate(self, weights):
        logits = self.X_test @ weights
        preds = 1 / (1 + np.exp(-logits))
        preds = (preds >= 0.5).astype(np.float32)
        return np.mean(preds == self.y_test)
