import socket
import threading
import sys
import time

class Trustline:
    def __init__(self, balance=0):
        self.balance = balance

    def pay(self, amount):
        self.balance -= amount
        return self.balance

    def receive(self, amount):
        self.balance += amount
        return self.balance

    def get_balance(self):
        return self.balance


class TrustlineServer:
    def __init__(self, host, port, trustline):
        self.host = host
        self.port = port
        self.trustline = trustline
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print("Waiting for connection on port", self.port)
        self.client_socket, self.address = self.server_socket.accept()
        print(f"Connected to {self.address}")

    def listen_for_transactions(self):
        while True:
            try:
                data = self.client_socket.recv(1024)
                if not data:
                    break
                amount = int(data.decode('utf-8'))
                self.trustline.receive(amount)
                print(f"\nYou were paid {amount}!\n> ", end="")
            except ConnectionResetError:
                break

    def notify_payment(self, amount):
        self.client_socket.sendall(str(amount).encode('utf-8'))

    def close_connection(self):
        self.client_socket.close()
        self.server_socket.close()


class TrustlineClient:
    def __init__(self, host, port, trustline):
        self.host = host
        self.port = port
        self.trustline = trustline
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.socket.connect((self.host, self.port))
                break
            except ConnectionRefusedError:
                print("Connection refused, retrying...")
                time.sleep(1)

    def send_payment(self, amount):
        self.socket.sendall(str(amount).encode('utf-8'))
        self.trustline.pay(amount)

    def close_connection(self):
        self.socket.close()


def start_trustline_interface(role, host, port):
    trustline = Trustline()
    connection = None

    if role.lower() == 'server':
        connection = TrustlineServer(host, port, trustline)
        listener_thread = threading.Thread(target=connection.listen_for_transactions)
        listener_thread.start()
    elif role.lower() == 'client':
        connection = TrustlineClient(host, port, trustline)

    print("Welcome to your trustline!")

    try:
        while True:
            command = input("> ").strip().lower()
            if command == 'balance':
                print(trustline.get_balance())
            elif command.startswith('pay '):
                amount = int(command.split()[1])
                if role.lower() == 'client':
                    connection.send_payment(amount)
                else:
                    connection.notify_payment(amount)
                print("Sent")
            elif command == 'exit':
                print("Goodbye.")
                break
            else:
                print("Unknown command.")
    except KeyboardInterrupt:
        print("\nGoodbye.")
    finally:
        if connection:
            connection.close_connection()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python start_trustline.py [server|client] [HOST] [PORT]")
        sys.exit(1)

    role = sys.argv[1]
    host = sys.argv[2]
    port = int(sys.argv[3])

    start_trustline_interface(role, host, port)
