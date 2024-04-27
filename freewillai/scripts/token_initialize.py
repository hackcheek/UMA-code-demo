import argparse
from freewillai.common import Network
from freewillai.contract import TokenContract
from freewillai.utils import get_account


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--address', type=str)
    parser.add_argument('-n', '--network', type=str)
    parser.add_argument('-p', '--private-key', type=str)

    args = parser.parse_args()

    assert args.address and args.private_key and args.network

    network = Network.by_network_id(args.network).build()
    token = TokenContract(get_account(args.private_key), network=network)
    tx_hash = token.initialize()
    print(
        f"[*] Initialized"
        f"  > {tx_hash=}"
    )
