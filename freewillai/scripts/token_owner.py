import os
import argparse

from freewillai.contract import TokenContract
from freewillai.common import Network
from freewillai.exceptions import UserRequirement
from freewillai.utils import get_account, load_global_env
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenArgs(argparse.Namespace):
    to: str
    mint: Optional[int] = None
    burn: Optional[int] = None
    wait: bool = False
    network: str = "devnet/anvil"
    rpc_url: Optional[str] = None
    private_key: Optional[str] = None
    env_file: Optional[str] = None
    

def main(args: TokenArgs):
    assert args.network

    if not args.private_key:
        # Load environ variables 
        env_filename = args.env_file or ".env"
        load_global_env(env_filename)
        if not os.environ.get("PRIVATE_KEY"):
            raise UserRequirement(
                    f"Please declare the following environment variables in {env_filename}\n"
                    "  > PRIVATE_KEY\n",
                    )

    # Get user and Make sure that the user is the owner
    account = get_account()

    # Just one option
    assert args.mint or args.burn and not (args.mint and args.burn)

    network = Network(args.network)
    if args.rpc_url:
        network = network.with_custom_rpc(args.rpc_url)
        print(network)

    print(f"[DEBUG] {network.rpc_url=}")

    token = TokenContract(account, network=network)
    print(f"{token.owner=}")
    token.approve(args.to, args.mint)
    try: 
        tx_hash = token.initialize()
        token.wait_for_transaction(tx_hash)
    except: ...

    if args.mint and not args.to:
        raise UserRequirement("Please especify which is the address to mint") 

    if args.burn and not args.to:
        raise UserRequirement("Please especify which is the address to burn") 

    if args.mint and args.to:
        print(f"[*] Minting {args.mint} FWAI to {args.to}")
        tx_hash = token.mint(args.to, args.mint)

    if args.burn and args.to:
        print(f"[*] Burning {args.burn} FWAI to {args.to}")
        tx_hash = token.burn(args.to, args.burn)

    if args.wait:
        token.wait_for_transaction(tx_hash)
        balance = token.get_balance_of(args.to)
        print(f"[*] Address: {args.to} has {balance} fwai")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mint', type=int)
    parser.add_argument('-b', '--burn', type=int)
    parser.add_argument('-w', '--wait', action=argparse.BooleanOptionalAction)
    parser.add_argument('-o', '--to', type=str)
    parser.add_argument('-n', '--network', type=str)
    parser.add_argument('-r', '--rpc-url', type=str)
    parser.add_argument('-p', '--private-key', type=str)
    parser.add_argument('-e', '--env-file', type=str)

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli()
