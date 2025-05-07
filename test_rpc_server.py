import signal
import rpc_rdma
import time

def handle_sigint(sig, frame):
    print("Stopping server...")
    rpc_rdma.stop_server()
    exit(0)

signal.signal(signal.SIGINT, handle_sigint)

rpc_rdma.start_server("7471")
while True:
    time.sleep(5)
