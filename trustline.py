import socket
import threading
import sys
import time


#make sure cant pay -150


#one client and server
#both computers - CLI - should have their own clients

#one computer = client, the other computer = server


#both computers = client
#another computer = server that validates and verifies both bank accounts in a Sql / kv store


#one server exists somewhere else -> make sure that the security 



#client server + server 

#multiple 


#simpler way - flask + http routes 

#make a route for each client


#3rd computer r-pi -> cloud instance -> flask server


#clients = command line program -> send request(/pay/) 
# requestBody = { userId, amount }
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
                if data.startswith('message:'):
                    print(data.replace('message:', ''))
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



#client allows client to pay server -> but not the other way around
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

    def send_message(self, message):
        self.socket.sendall(str(message).encode('utf-8'))

    def send_payment(self, amount):
        self.socket.sendall(str(amount).encode('utf-8'))
        self.trustline.pay(amount)

    def close_connection(self):
        self.socket.close()


#send message from client -> server prints out 
#



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

            elif command.startswith('say '):
                connection.send_message('message:' + command.replace('say ', ''))

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
