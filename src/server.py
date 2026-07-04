# owns a weight shard
# receives gradient updates, applies them, serve current values on request

import threading
import time
import json
import ray
import os
import config
from config import SyncMode, CHECKPOINT_DIR

# dead server's actor is gone; need this to read its last checkpoint from disk to know updated values
# need this here b/c of ray flag
def read_checkpoint_file(server_id):

    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{server_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "iteration": data["iteration"],
        "weights": {int(k): v for k, v in data["weights"].items()},
    }

@ray.remote
class ParameterServer:
    def __init__(
        self,
        server_id,
        weight_indices,
        num_weights,
        learning_rate,
        weight_vals,
        current_iteration,
        num_expected_workers,
        sync_mode: SyncMode = SyncMode.SEQUENTIAL_BSP,
        checkpoint_every: int = config.CHECKPOINT_EVERY,
    ):
        self.server_id = server_id
        self.weight_indices = list(weight_indices)
        self.num_weights = num_weights
        self.learning_rate = learning_rate
        self.weight_vals = dict(weight_vals)
        self.current_iteration = current_iteration
        self.gradient_store = {k: [] for k in self.weight_indices}
        self.num_expected_workers = num_expected_workers
        self.workers_seen = set()
        self.sync_mode = sync_mode
        self.async_updates = sync_mode == SyncMode.ASYNCHRONOUS
        self.checkpoint_every = checkpoint_every
        self.push_count = 0  # async: pushes applied (no global rounds to count)
        self.replicas = []  # list of (server_id, handle)
        self.replicated_shards = {}  # master server id -> {weight_idx: val}
        self._checkpoint_thread = None
        # initial checkpoint: the sidecar evaluator gets a pre-training
        # baseline point and full shard coverage from t=0
        if self.checkpoint_every > 0:
            self.add_checkpoint()

    def pull_weights(self, indices, expected_iteration=None) -> dict[int, float]:
        # Stale / latest: async does not block until a global BSP clock.
        wait_bsp = not self.async_updates and expected_iteration is not None
        if wait_bsp:
            while self.current_iteration < expected_iteration:
                time.sleep(0.001)
        return {i: self.weight_vals[i] for i in indices}

    def push_gradients(
        self, gradient_dict: dict[int, float], worker_id, iteration
    ) -> None:
        if self.async_updates:
            self._apply_gradients_immediate(gradient_dict)
            return

        if iteration != self.current_iteration:
            return
        for idx, grad in gradient_dict.items():
            assert idx in self.gradient_store, f"Unexpected weight index {idx}"
            self.gradient_store[idx].append(grad)

        self.workers_seen.add(worker_id)
        if len(self.workers_seen) == self.num_expected_workers:
            self.update_weights()

    def _apply_gradients_immediate(self, gradient_dict: dict[int, float]) -> None:
        for idx, grad in gradient_dict.items():
            assert idx in self.gradient_store, f"Unexpected weight index {idx}"
            self.weight_vals[idx] -= self.learning_rate * grad
        # No global rounds in async; N pushes ~ one BSP round of update volume,
        # so checkpoint cadence matches BSP's "every checkpoint_every rounds".
        self.push_count += 1
        self.current_iteration = self.push_count // self.num_expected_workers
        if (
            self.checkpoint_every > 0
            and self.push_count % (self.checkpoint_every * self.num_expected_workers) == 0
        ):
            self.replicate_to_replicas()
            self.add_checkpoint()

    def update_weights(self):
        self.workers_seen = set()
        for weight_index in self.weight_indices:
            grads = self.gradient_store[weight_index]
            if len(grads) == 0:
                continue
            average_gradient = sum(grads) / len(grads)
            self.weight_vals[weight_index] -= self.learning_rate * average_gradient

        self.gradient_store = {weight_index: [] for weight_index in self.weight_indices}
        self.current_iteration += 1

        # replicate to followers (replicas) for fault tolerance
        self.replicate_to_replicas()

        if self.checkpoint_every > 0 and self.current_iteration % self.checkpoint_every == 0:
            self.add_checkpoint()

    
    # absorb the additional weights inherited from a failed server
    def absorb_weights(self, new_weight_dict: dict[int, float]) -> int:
        for idx, val in new_weight_dict.items():
            assert idx not in self.gradient_store, (
                f"{self.server_id} already owns weight {idx}; cannot absorb"
            )
            self.weight_indices.append(idx)
            self.weight_vals[idx] = val
            self.gradient_store[idx] = []
        return len(self.weight_indices)

    def get_iteration(self) -> int:
        return self.current_iteration

    def set_iteration(self, iteration: int) -> None:
        
        self.current_iteration = iteration

    def get_weight_indices(self) -> list:
        return list(self.weight_indices)

    def _checkpoint_path(self) -> str:
        return os.path.join(CHECKPOINT_DIR, f"checkpoint_{self.server_id}.json")


    def add_checkpoint(self):
        print(f"[CHECKPOINT] {self.server_id} at iteration {self.current_iteration}")
        # snapshot synchronously (consistency), write in the background so the
        # update path never blocks on disk
        checkpoint_data = {
            "iteration": int(self.current_iteration),
            "weights": {
                str(i): float(self.weight_vals[i]) for i in self.weight_indices
            },
        }
        # serialize writes: an older checkpoint must never replace a newer one
        self.flush_checkpoints()
        self._checkpoint_thread = threading.Thread(
            target=self._write_checkpoint, args=(checkpoint_data,), daemon=True
        )
        self._checkpoint_thread.start()

    def _write_checkpoint(self, checkpoint_data):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        # write-then-rename so concurrent readers (sidecar evaluator, recovery)
        # never see a partially written file
        tmp_path = self._checkpoint_path() + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(checkpoint_data, f)
        os.replace(tmp_path, self._checkpoint_path())

    def flush_checkpoints(self):
        """Block until any in-flight checkpoint write has hit disk."""
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            self._checkpoint_thread.join()

    def load_checkpoint(self):
        path = self._checkpoint_path()
        if not os.path.isfile(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
 
        loaded_weights = {int(k): v for k, v in data["weights"].items()}
 
        # Fail if ring assignment changed under us
        assert set(loaded_weights.keys()) == set(self.weight_indices), (
            f"Checkpoint indices don't match current shard assignment for "
            f"{self.server_id}. Checkpoint has "
            f"{len(loaded_weights)} indices, server owns "
            f"{len(self.weight_indices)}."
        )
        self.weight_vals = loaded_weights
        self.current_iteration = data["iteration"]
        return self.current_iteration

    # functions for chain replication
    def set_replicas(self, replicas):
        self.replicas = replicas 
    
    def replicate(self, leader_id, weight_dict, iteration):
        self.replicated_shards[leader_id] = dict(weight_dict)
    
    def get_replicated_shards(self, leader_id):
        return self.replicated_shards.get(leader_id, None)

    def replicate_to_replicas(self):
        if not self.replicas:
            return 
        snapshot = dict(self.weight_vals)
       # push updated weights to replica servers (chain replication) using replicate() function. eventual replication
        for _, handle in self.replicas:
            handle.replicate.remote(
                self.server_id,
                snapshot,
                self.current_iteration
            )

        # we can't use this because it introduces deadlocks / circular waits where servers might be waiting for each other
        # ray.get([handle.replicate.remote(
        #     self.server_id,
        #     snapshot,
        #     self.current_iteration
        # ) for _, handle in self.replicas])