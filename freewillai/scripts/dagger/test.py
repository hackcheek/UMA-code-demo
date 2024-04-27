import time
import sys
from web3 import Web3, HTTPProvider

url = sys.argv[1]

w3 = Web3(HTTPProvider(url))
while not w3.is_connected():
    time.sleep(1)


print(w3.eth.accounts)
