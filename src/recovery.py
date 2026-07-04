import os

import ray
from typing import Dict, Tuple, Optional, cast
from cluster import assign_chain_replicas
from server import read_checkpoint_file
import config
def get_recovery_weights(dead_server_id, ring, servers) -> Tuple[Dict[int, float], Optional[int]]:
    if config.RECOVERY_MODE == "chain":
        # get from replicas 
        for sid, handle in servers.items():
            try:
                shard = cast(Dict[int, float], ray.get(
                    handle.get_replicated_shards.remote(dead_server_id),
                    timeout=1.0
                ))
                if shard is not None:
                    return shard, None
            except Exception:
                continue
        # fail if no replica found
        raise RuntimeError(f"Chain recovery failed: no replica found for {dead_server_id}")

    elif config.RECOVERY_MODE == "checkpoint":
        checkpoint = read_checkpoint_file(dead_server_id)
        if checkpoint is None:
            raise RuntimeError(
                f"No checkpoint found for dead server {dead_server_id}. "
                f"Cannot reshard. Check that CHECKPOINT_EVERY is set and at "
                f"least one checkpoint fired before the failure."
            )
        return checkpoint["weights"], checkpoint["iteration"]
    raise RuntimeError(f"Unknown recovery mode: {config.RECOVERY_MODE}")
    
def reshard_after_failure(
    dead_server_id,
    ring,
    servers,
    workers,
):
    # read dead server's last checkpoint
    orphaned_weights, checkpoint_iter = get_recovery_weights(dead_server_id, ring, servers)

    ring.remove_server(dead_server_id)
    # fix the weight map, remove the handle for the ray actor to the dead server
    del servers[dead_server_id]

    # remove the dead server's checkpoint file: survivors re-checkpoint the
    # absorbed indices, and a leftover file would overlap them with stale values
    # (the sidecar evaluator assembles snapshots from all files on disk)
    dead_ckpt = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_{dead_server_id}.json")
    if os.path.isfile(dead_ckpt):
        os.remove(dead_ckpt)

    # determining new owners of the dropped weights
    owner_map: dict[str, dict[int, float]] = {}
    for idx, val in orphaned_weights.items():
        new_owner = ring.get_server(idx)
        owner_map.setdefault(new_owner, {})[idx] = val

    # get each survivor to absorb its new weights
    absorb_refs = [] # references to ray actors
    absorb_targets = []
    for owner_id, weight_dict in owner_map.items():
        absorb_refs.append(servers[owner_id].absorb_weights.remote(weight_dict))
        absorb_targets.append(owner_id)

    new_sizes = ray.get(absorb_refs)

    # after failure, everyone agrees to continue from the latest completed iteration
    iters = ray.get([s.get_iteration.remote() for s in servers.values()])
    max_iter = max(iters)
    ray.get([s.set_iteration.remote(max_iter) for s in servers.values()])
    ray.get([w.reconfigure.remote(ring, servers) for w in workers])
    ray.get([w.set_iteration.remote(max_iter) for w in workers])

    # reassign chain replicas among survivors; otherwise every server keeps
    # replicating to the dead actor (errors silently swallowed) and the
    # replication factor degrades with each failure
    if config.RECOVERY_MODE == "chain" and len(servers) > 1:
        assign_chain_replicas(ring, servers, config.CHAIN_REPLICAS)

    return {
        "dead_server": dead_server_id,
        "checkpoint_iteration": checkpoint_iter,
        "orphaned_weight_count": len(orphaned_weights),
        "absorbed_by": {
            target: size for target, size in zip(absorb_targets, new_sizes)
        },
    }