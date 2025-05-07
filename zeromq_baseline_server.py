# zmq_server.py
import zmq
import struct

def handle_add(data: bytes) -> bytes:
    if len(data) < 8:
        return b''
    a, b = struct.unpack('ii', data[:8])
    result = a + b
    return struct.pack('i', result)

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:7471")

    print("Server listening on port 7471...")
    while True:
        message = socket.recv()
        if message.startswith(b'add'):
            payload = message[3:]  # strip 'add' prefix
            response = handle_add(payload)
            socket.send(response)
        else:
            socket.send(b'')

if __name__ == "__main__":
    main()
