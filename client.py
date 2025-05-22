import torch

import time
import rpc_rdma

print("\n start connection")
client = rpc_rdma.Client(host="192.168.233.18", port="7471")
print("\n finish connection")
res = client.add(1,3)

batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64
send_shape = (batch_size, num_heads, seq_len, head_dim)
recv_shape = (batch_size, num_heads, seq_len, head_dim)

client_send_tensor = torch.randn(send_shape, dtype=torch.float16, device='cuda')
send_size = client_send_tensor.numel()
recv_size = recv_shape[0] * recv_shape[1] * recv_shape[2] * recv_shape[3]
client_recv_tensor = torch.empty(recv_shape, dtype=torch.float16, device='cuda')

for _ in range(10):
    # start_time = time.time()
    # res = client.add(1,3)
    # res2 = client.mul(2, 3, 4)
    # end_time = time.time()
    # print(f"\n add result is {res}, mul result is {res2}")
    # print("\nTime in microseconds:", (end_time - start_time) * 1e6)
    start_time = time.time()
    received_addr2 = client.exchange_gpu_data(
        client_send_tensor.data_ptr(),
        send_size,
        recv_size,
        client_recv_tensor.data_ptr() 
    )
    end_time = time.time()

    exchange_time = (end_time - start_time) * 1000

    if received_addr2 != 0:
        print(f"\nExchange completed in {exchange_time:.1f} ms")
        print(f"\nUsed provided buffer: 0x{received_addr2:x}")
        print(f"\nBuffer matches: {received_addr2 == client_recv_tensor.data_ptr()}")
        
    else:
        print("\n Pre-allocated exchange failed")


    

