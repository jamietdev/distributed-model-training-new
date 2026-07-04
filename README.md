# distributed-model-training
```
Setup:
source venv/bin/activate
pip install -r requirements.txt
cd src

To run training:
python3 main.py
```

# WRITEUP
### Project Overview
Implementing https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf.

We implemented the distributed parameter server architecture from Li et al. (OSDI 2014) to train a logistic regression model on MNIST for binary even/odd classification. The model uses 785 weights (784 pixels + 1 bias). Parameter servers store weight shards partitioned via consistent hashing, while worker nodes compute gradients over disjoint data shards. Servers push updated weights back to workers after each round.

Features from the paper we implemented:
1. **Parameter server / worker architecture** on Ray. Servers own weight shards, workers own data shards, push/pull RPC interface. **Consistent hash ring** with virtual nodes (MD5-based) for load-balanced weight partitioning across servers
2. **Synchronization modes:**
  - BSP synchronization = all workers are on the same iteration step
  - Asynchronous synchronization = immediate gradient application, no coordination
  - Experiments:
     - Accuracy vs. wall-clock time benchmarks — comparing convergence speed across sync modes
     - Throughput across sync modes
3. **Fault tolerance strategies:**
  - Disk checkpointing = periodic JSON snapshots of each server's weight shard to disk
  - Chain replication = fire-and-forget weight replication to k clockwise ring neighbors after each update
  - Fault tolerance tests:
     - Static reshard — kill a server outside training, redistribute orphaned weights to survivors, verify correctness
     - Live reshard — kill a server mid-training, recover weights, reconfigure ring and workers, resume training
  - Experiments: compared checkpoint vs chain replication recovery at different failure points
4. **Sidecar evaluation**

See writeup.pdf for more details
