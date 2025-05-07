import signal
import torch
import rpc_rdma
import time
import sys
import traceback

def handle_sigint(sig, frame):
    print("[PYTHON] Ctrl+C caught (SIGINT). Invoking stop_server()...")
    try:
        rpc_rdma.stop_server()
        print("[PYTHON] stop_server() completed")
    except Exception as e:
        print("[PYTHON] Exception in stop_server():", e)
        traceback.print_exc()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

print("[PYTHON] Starting server...")
rpc_rdma.start_server("7471")
print("[PYTHON] Server started on port 7471")

while True:
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("[PYTHON] Unexpected KeyboardInterrupt inside loop (shouldn't happen if signal works)")
        break
