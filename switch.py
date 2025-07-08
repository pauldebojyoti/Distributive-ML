import socket
import threading
import struct
import numpy as np

# Configuration
SWITCH_PORT = 4000
WORKER_NODES = [("worker_1", 5000), ("worker_2", 5001)]

def forward_to_worker(client_data, worker_address):
    """
    Forwards data (weights and batch) to a worker node and receives the updated weights.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as worker_socket:
        worker_socket.connect(worker_address)
        worker_socket.sendall(client_data)  # Send data to the worker
        response = b''
        while True:
            data = worker_socket.recv(1024)
            if not data:
                break
            response += data
    return response  # Return the updated weights from the worker

def handle_connection(client_socket):
    """
    Handles the connection from the central node, forwards data to all worker nodes,
    aggregates the weights, and sends the aggregated weights back to the central node.
    """
    # Receive data (weights + batch) from the central node
    client_data = b''
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        client_data += data

    # Forward data to all worker nodes and collect their responses
    responses = []
    for worker_address in WORKER_NODES:
        response = forward_to_worker(client_data, worker_address)
        responses.append(np.frombuffer(response, dtype=np.float32))

    # Aggregate weights (e.g., simple average)
    aggregated_weights = np.mean(responses, axis=0)

    # Send the aggregated weights back to the central node
    client_socket.sendall(aggregated_weights.tobytes())

    # Forward the aggregated weights to all worker nodes
    for worker_address in WORKER_NODES:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as worker_socket:
            worker_socket.connect(worker_address)
            worker_socket.sendall(aggregated_weights.tobytes())

def start_virtual_switch():
    """
    Starts the virtual switch to listen for connections from the central node
    and forward data to the worker nodes.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', SWITCH_PORT))
    server.listen(len(WORKER_NODES))
    print(f"[Switch] Listening on port {SWITCH_PORT}...")

    while True:
        client_socket, addr = server.accept()
        print(f"[Switch] Connection from {addr}")
        threading.Thread(target=handle_connection, args=(client_socket,)).start()

if __name__ == "__main__":
    start_virtual_switch()