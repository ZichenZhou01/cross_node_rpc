import torch

import time
import rpc_rdma

client = rpc_rdma.Client(host="192.168.233.18", port="7471")
res = client.add(1,3)

for _ in range(10):
    start_time = time.time()
    res = client.add(1,3)
    end_time = time.time()
    print("Time in microseconds:", (end_time - start_time) * 1e6)
    

