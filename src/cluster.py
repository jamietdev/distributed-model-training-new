import ray
import random
from config import CHECKPOINT_EVERY, RECOVERY_MODE, SyncMode, SEED, CHAIN_REPLICAS
from hash_ring import HashRing
from load_mnist import shard_data
from server import ParameterServer
from worker import Worker

def get_ring_ordered_servers(ring, servers):
    """Returns a deterministic ordering of server IDs based on their positions
    on the consistent hash ring."""

    return sorted(
        servers.keys(),
        key=lambda sid: min(
            ring.hash(f"{sid}#v{i}")
            for i in range(ring.num_virtual_servers)
        )
    )

def assign_chain_replicas(ring, servers, k): # we replicate the weights on server_i to the next k servers in ring order
    ordered_servers = get_ring_ordered_servers(ring, servers)
    n = len(ordered_servers)
    for i, server_id in enumerate(ordered_servers):
        replica_ids = [ordered_servers[(i + j + 1) % n] for j in range(min(k, n - 1))]
        replicas = [(s, servers[s]) for s in replica_ids]

        ray.get(servers[server_id].set_replicas.remote(replicas))

def build_cluster(
    num_workers,
    num_servers,
    num_weights,
    num_replicas,
    learning_rate,
    X_train,
    y_train,
    sync_mode: SyncMode = SyncMode.SEQUENTIAL_BSP,
    weight_init_seed: int | None = None,
    checkpoint_every: int = CHECKPOINT_EVERY,
):
    shards = shard_data(X_train, y_train, num_workers)

    ring = HashRing(num_weights, num_replicas)
    server_ids = [f"server_{i}" for i in range(num_servers)]
    for sid in server_ids:
        ring.add_server(sid)

    wseed = SEED if weight_init_seed is None else weight_init_seed
    rng = random.Random(wseed)
    servers = {}
    for sid in server_ids:
        owned = ring.indices_for_server(sid)
        wvals = {k: float(rng.uniform(-0.1, 0.1)) for k in owned}
        servers[sid] = ParameterServer.remote(
            server_id=sid,
            weight_indices=owned,
            num_weights=num_weights,
            learning_rate=learning_rate,
            weight_vals=wvals,
            current_iteration=0,
            num_expected_workers=num_workers,
            sync_mode=sync_mode,
            checkpoint_every=checkpoint_every,
        )

    if RECOVERY_MODE == "chain":
        assign_chain_replicas(ring, servers, CHAIN_REPLICAS)

    workers = []
    for i in range(num_workers):
        w = Worker.remote(
            worker_id=f"worker_{i}",
            hash_ring=ring,
            num_weights=num_weights,
            learning_rate=learning_rate,
            current_iteration=0,
            X_train_batch=shards[i][0],
            y_train_batch=shards[i][1],
            servers=servers,
            sync_mode=sync_mode,
        )
        workers.append(w)

    return ring, servers, workers


def teardown_cluster(servers, workers):
    for w in workers:
        ray.kill(w)
    for s in servers.values():
        ray.kill(s)
