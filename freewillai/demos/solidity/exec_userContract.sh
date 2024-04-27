#!/bin/bash


source demos/userContract.env

task_runner_cmd=`forge create contracts/TaskRunner.sol:TaskRunner \
    --rpc-url $RPC_URL \
    --private-key $PRIVATE_KEY \
    --legacy`

task_runner_address=`echo "$task_runner_cmd" | grep 'Deployed to:' | cut -d ' ' -f 3`
task_runner_tx_hash=`echo "$task_runner_cmd" | grep 'Transaction hash:' | cut -d ' ' -f 3`

echo -e "\nDeployed TaskRunner Contract"
echo -e "\tContract address: $task_runner_address"
echo -e "\tTransaction hash: $task_runner_tx_hash"
