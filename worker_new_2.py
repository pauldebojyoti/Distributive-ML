import socket
import struct
import numpy as np

PORT = 5001  # Port for the second worker
NUM_WEIGHTS = 10
LEARNING_RATE = 0.01

def recv_exact(sock, n):
    """
    Receives exactly `n` bytes from the socket.
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Connection closed unexpectedly while receiving data")
        data += packet
    return data

def train_sgd(weights, dataset):
    """
    Performs Stochastic Gradient Descent (SGD) on the received dataset.
    """
    for features, label in dataset:
        pred = np.dot(weights, features)
        error = pred - label
        for j in range(NUM_WEIGHTS):
            weights[j] -= LEARNING_RATE * error * features[j]  # Include error in the update
    return weights

def receive_data(sock):
    """
    Receives weights and dataset from the virtual switch.
    """
    # Receive number of samples (4 bytes)
    size_data = recv_exact(sock, 4)
    data_size = struct.unpack('I', size_data)[0]

    # Receive weights
    weight_data = recv_exact(sock, NUM_WEIGHTS * 4)
    weights = np.frombuffer(weight_data, dtype=np.float32).copy()

    # Receive dataset
    dataset = []
    for _ in range(data_size):
        sample_data = recv_exact(sock, (NUM_WEIGHTS + 1) * 4)
        data = struct.unpack('f' * (NUM_WEIGHTS + 1), sample_data)
        features = list(data[:NUM_WEIGHTS])
        label = data[-1]
        dataset.append((features, label))

    return weights, dataset

def main():
    """
    Main function to handle worker operations.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', PORT))
    server.listen(1)
    print(f"[Worker] Listening on port {PORT}...")

    while True:
        client, addr = server.accept()
        print(f"[Worker] Connection from {addr}")
        try:
            # Receive weights and dataset from the virtual switch
            weights, dataset = receive_data(client)

            # Train using SGD
            weights = train_sgd(weights, dataset)

            # Send updated weights back to the virtual switch
            client.sendall(weights.tobytes())
        except Exception as e:
            print(f"[Worker] Error: {e}")
        finally:
            client.close()

if __name__ == "__main__":
    main()
