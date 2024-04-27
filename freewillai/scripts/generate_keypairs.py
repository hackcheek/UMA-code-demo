import sys
import json
from web3 import Web3, HTTPProvider

assert 3 >= len(sys.argv) > 1
rpc_url = sys.argv[1]
dest_path = sys.argv[2] if len(sys.argv) == 3 else None


dest_path = dest_path or "/tmp/eth_account.json"
w3 = Web3(HTTPProvider(rpc_url))

account = w3.eth.account.create()

data = dict(
    public_key = account.address,
    private_key = account.key.hex(),
)

with open(dest_path, "w") as jsonfile:
    jsonfile.write(json.dumps(data))

print(f"[*] Generated key pairs at {dest_path}")
