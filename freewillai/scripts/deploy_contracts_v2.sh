#!/bin/sh

PRIVATE_KEY=$1
rpc_url=$2
LOG_FILE=deploy_contracts.log

if [ ! $PRIVATE_KEY ]; then
    source .env
fi

if [[ "$PRIVATE_KEY" != "0x*" ]]; then
    PRIVATE_KEY=0x$PRIVATE_KEY
fi

if [ ! $rpc_url ]; then
    rpc_url=http://0.0.0.0:8545
fi

touch $LOG_FILE
PRIVATE_KEY=$PRIVATE_KEY forge script scripts/deploy_contracts.s.sol --rpc-url $rpc_url --broadcast -vvvv | tee $LOG_FILE

echo "FreeWillAI Token address: `cat $LOG_FILE | grep FreeWillAI@ | sed -E 's/^.*FreeWillAI@(.*)$/\1/' | head -n 1`"
echo "FreeWillAI Task Runner address: `cat $LOG_FILE | grep TaskRunner@ | sed -E 's/^.*TaskRunner@(.*)$/\1/' | head -n 1`"
