import os
from enum import Enum


class SyncMode(str, Enum):
    """Distributed training synchronization strategy."""

    SEQUENTIAL_BSP = "sequential_bsp"
    ASYNCHRONOUS = "asynchronous"


NUM_SERVERS = 3
NUM_WORKERS = 10
NUM_WEIGHTS = 785  # 784 input features + 1 bias
NUM_ITERATIONS = 50
LEARNING_RATE = 0.2
NUM_REPLICAS = 2

SYNC_MODE = SyncMode.SEQUENTIAL_BSP

# for runtime testing
WARMUP_ITERS = 5  # discarded; first few iters are dominated by JIT/RPC setup
TIMED_ITERS = 30  # samples actually used for stats
ITER_TIMEOUT_S = 10.0

# for fault tolerance
RECOVERY_MODE = "chain"  # or "checkpoint"
CHAIN_REPLICAS = 2
# anchored to the repo root so checkpoints land in one place regardless of cwd
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"
)
CHECKPOINT_EVERY = 50

SEED = 67
