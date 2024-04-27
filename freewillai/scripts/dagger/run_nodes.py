import os
import json
import argparse
from typing import List
import requests
import asyncio
from freewillai.globals import Global
from freewillai.node import Node
from freewillai.utils import get_account
from freewillai.common import Network
from web3 import Web3, HTTPProvider
from scripts.token_owner import main as token_owner, TokenArgs

# DEMO_NETWORKS = ["devnet/anvil", "testnet/optimism", "testnet/goerli"]
DEMO_NETWORKS = ["devnet/anvil"]

def get_balance(account, anvil_rpc):
    data = json.dumps({
        "method": "eth_getBalance",
        "params": [account.address, "latest"],
        "id": 1,
        "jsonrpc": "2.0"
    })

    resp = requests.post(
        anvil_rpc,
        headers={"Content-Type": "application/json"},
        data=data
    )
    balance = int(resp.json()["result"], 16)
    print(f"Balance of {account.address}:", balance)
    return balance


def set_balance_to_100ETH(account, anvil_rpc):
    data = json.dumps({
        "method": "anvil_setBalance",
        "params": [account.address, "0x56BC75E2D63100000"],
        "id": 1,
        "jsonrpc": "2.0"
    })

    resp = requests.post(
        anvil_rpc,
        headers={"Content-Type": "application/json"},
        data=data
    )
    print(f"Set 100ETH on {account.address}:", resp.text)


def mint_fwai(account, anvil_rpc, owner_private_key):
    args = TokenArgs(
        mint=100_000_000_000_000_000_000_000,
        to=account.address,
        network="devnet/anvil",
        rpc_url=anvil_rpc,
        private_key=owner_private_key,
    )
    print(f"[*] Minting 100FWAI to {account.address}")
    token_owner(args)


async def run_workers(workers, stake, cooling_time, anvil_rpc):
    nodes: List[Node] = []
    log_dir = "/logs/fwai-workers-log-{}/"
    num = 1

    def _check_and_get_log_dir(num):
        if os.path.exists(log_dir.format(num)):
            return _check_and_get_log_dir(num+1)
        if not os.path.exists(log_dir.format(num)):
            os.mkdir(log_dir.format(num))
        return log_dir.format(num)

    log_dir = _check_and_get_log_dir(num)
    
    for id, worker in enumerate(workers):
        for network in DEMO_NETWORKS:
            # Intance nodes
            net = Network(network)
            if network == "devnet/anvil" and net.rpc_url != anvil_rpc:
                net = net.with_custom_rpc(anvil_rpc)

            node = Node(
                private_key=worker.key.hex(),
                network=net,
                cooling_time=cooling_time,
                log_file=os.path.join(log_dir, f"worker{id+1}")
            )
            node.stake(stake)
            nodes.append(node)

    await asyncio.gather(*map(lambda n: n.spin_up(), nodes))


async def main(
    anvil_rpc:str,
    token_address:str,
    stake:int,
    cooling_time:int,
    owner_pk:str,
    num_of_workers:int
):
    w3 = Web3(HTTPProvider(anvil_rpc))
    Global.ipfs_host = 'ipfs'
    Global.ipfs_port = 5001

    workers = [w3.eth.account.create() for _ in range(num_of_workers)]
    owner = get_account(owner_pk)

    if w3.eth.get_code(Web3.to_checksum_address(token_address)).hex() == '0x':
        raise RuntimeError(
            f"Not found token contract with this address={token_address}"
        )

    # If minting to owner
    if get_balance(owner, anvil_rpc) < 10**20:
        set_balance_to_100ETH(owner, anvil_rpc)
    mint_fwai(owner, anvil_rpc, owner_pk)

    for account in workers:
        set_balance_to_100ETH(account, anvil_rpc)
        mint_fwai(account, anvil_rpc, owner_pk)

    await run_workers(workers, stake, cooling_time, anvil_rpc)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anvil-rpc", type=str)
    parser.add_argument("--token-address", type=str)
    parser.add_argument("--stake", type=int)
    parser.add_argument("--owner-pk", type=str)
    parser.add_argument("--num-of-workers", type=int)
    parser.add_argument("--cooling-time", type=int)
    parser.add_argument("--ipfs-port", type=int)

    args = parser.parse_args()

    asyncio.run(main(
        args.anvil_rpc,
        args.token_address,
        args.stake,
        args.cooling_time,
        args.owner_pk,
        args.num_of_workers
    ))

if __name__ == "__main__":
    cli()
