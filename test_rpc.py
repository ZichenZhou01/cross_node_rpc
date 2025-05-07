import torch
import rpc_rdma
import time
rpc_rdma.start_server(port="7471")

while True:
    try:
        # Simulate server activity
        time.sleep(5)  # Sleep for a while to simulate work
    except KeyboardInterrupt:
        print("Server stopped.")
        break
