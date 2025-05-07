import torch

import rpc_rdma

client = rpc_rdma.Client(host="192.168.233.18", port="7471")
res = client.add(1,3)

