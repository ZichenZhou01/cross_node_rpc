# zmq_client.py
import zmq
import struct
import time

class Client:
    def __init__(self, host="127.0.0.1", port="7471"):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

    def add(self, a: int, b: int) -> int:
        payload = struct.pack('ii', a, b)
        self.socket.send(b'add' + payload)
        reply = self.socket.recv()
        if len(reply) != 4:
            return None
        return struct.unpack('i', reply)[0]

if __name__ == "__main__":
    client = Client(host="192.168.1.18", port="7471")
    print("Result:", client.add(1, 3))

    for _ in range(10):
        start_time = time.time()
        result = client.add(1, 3)
        end_time = time.time()
        print("Time in microseconds:", (end_time - start_time) * 1e6)
