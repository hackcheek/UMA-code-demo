#!/bin/bash

set -x

rpc=$1

if [ ! $rpc ];then
    rpc="http://127.0.0.1:8545"
fi

dot_envs="
.env
worker1.env
worker2.env
worker3.env
"

# Set eth to nodes and client
for env in $dot_envs; do

    # Get private key from env
    source ${env}

    # Get address from private key
    address=`cast wallet address ${PRIVATE_KEY}`

    # Set account balance
    curl $rpc \
        -X POST \
        -H "Content-Type: application/json" \
        --data "{
            \"method\":\"anvil_setBalance\",
            \"params\":[\"${address}\", \"10000000000000000000000000000000000000\"],
            \"id\":1,
            \"jsonrpc\":\"2.0\"
        }"

    # Deploy contracts if is the client
    if [[ "${env}" == ".env" ]]; then
        bash scripts/deploy_contracts_v2.sh ${PRIVATE_KEY} ${rpc}
    fi
done
