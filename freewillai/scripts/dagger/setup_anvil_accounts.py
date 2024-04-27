"""
Script to generate the necesaries accounts, 
deploy contracts and mint fwai and eth to run demo on anvil
"""
import json
import argparse
import requests
from web3 import Web3, HTTPProvider
from scripts.token_owner import main as token_owner, TokenArgs


def set_balance_to_100ETH(account, anvil_endpoint):
    data = json.dumps({
        "method": "anvil_setBalance",
        "params": [account.address, "0x56BC75E2D63100000"],
        "id": 1,
        "jsonrpc": "2.0"
    })

    resp = requests.post(
        anvil_endpoint,
        headers={"Content-Type": "application/json"},
        data=data
    )
    print(f"Set 100ETH on {account.address}:", resp.text)


def mint_fwai(account, anvil_endpoint, owner_private_key):
    args = TokenArgs(
        mint=100_000_000_000_000_000_000_000,
        to=account.address,
        network="devnet/anvil",
        rpc_url=anvil_endpoint,
        private_key=owner_private_key,
    )
    print(f"[*] Minting 100FWAI to {account.address}")
    token_owner(args)


def dump_private_keys(path, accounts):
    with open(path, "w") as file:
        file.write("\n".join(a.key.hex() for a in accounts))


def main(rpc_url, num_of_nodes, out_path, token_address):
    import time
    time.sleep(20)
    w3 = Web3(HTTPProvider(rpc_url))
    print(f"[DEBUG] {token_address=}")

    if w3.eth.get_code(token_address).hex() == '0x':
        raise RuntimeError(
            f"Not found token contract with this address={token_address}"
        )

    nodes = [w3.eth.account.create() for _ in range(num_of_nodes+1)]
    owner_pk = nodes[0].key.hex()
    
    for account in nodes:
        set_balance_to_100ETH(account, rpc_url)
        mint_fwai(account, rpc_url, owner_pk)

    return nodes
    # dump_private_keys(out_path, accounts=nodes)



def cli():
    print("[DEBUG]--HERE-1--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-url", type=str)
    parser.add_argument("--num-of-nodes", type=int)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--token-address", type=str)

    args = parser.parse_args()

    return main(
        args.rpc_url,
        args.num_of_nodes,
        args.out_path,
        args.token_address
    )


if __name__ == "__main__":
    cli()
