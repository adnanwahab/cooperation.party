class Trustline:
    def __init__(self):
        self.balance = 0

    def pay(self, amount):
        self.balance -= amount
        print("Sent")

    def receive(self, amount):
        self.balance += amount
        print(f"You were paid {amount}!")

    def get_balance(self):
        return self.balance

    def display_balance(self):
        print(self.balance)

    def start(self):
        print("Welcome to your trustline!")
        while True:
            command = input("> ").strip().lower()
            if command == 'balance':
                self.display_balance()
            elif command.startswith('pay'):
                _, amount_str = command.split()
                try:
                    amount = int(amount_str)
                    self.pay(amount)
                except ValueError:
                    print("Invalid amount. Please enter a number.")
            elif command == 'exit':
                print("Goodbye.")
                break
            else:
                print("Unknown command. Please try again.")

if __name__ == "__main__":
    user_name = input("Please enter your name: ")
    print(f"Starting trustline for {user_name}...")
    tl = Trustline()
    tl.start()
