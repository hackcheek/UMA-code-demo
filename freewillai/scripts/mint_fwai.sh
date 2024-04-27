#!/bin/bash

# Addresses of eth accounts in this order:
# Main, worker1, worker2
envs="
    .env
    worker1.env
    worker2.env
    worker3.env
"

network=$1

# Mint 1k FWAI to these accounts
for env in ${envs}; do
    source ${env}
    address=`cast wallet address ${PRIVATE_KEY}`
    python3 -m scripts.token_owner --mint 100000000000000000000000000000000000 -n ${network} --to ${address}
done
