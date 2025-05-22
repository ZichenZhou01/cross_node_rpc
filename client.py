import torch

import time
import rpc_rdma

print("\n start connection")
client = rpc_rdma.Client(host="192.168.233.18", port="7471")
print("\n finish connection")
res = client.add(1,3)

for _ in range(10):
    start_time = time.time()
    res = client.add(1,3)
    res2 = client.mul(2, 3, 4)
    end_time = time.time()
    print("\n add result is {res}, mul result is {res2}")
    print("\nTime in microseconds:", (end_time - start_time) * 1e6)
    

