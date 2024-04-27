import sys
from web3 import Web3

assert len(sys.argv) == 2, "Usage: create_account.py <private_keys_path>"

path = sys.argv[1]

w3 = Web3()
account = w3.eth.account.create()

with open(path, "a") as file:
    file.write(account.key.hex()+'\n')
