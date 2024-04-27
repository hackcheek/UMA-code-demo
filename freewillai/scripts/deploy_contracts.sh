#!/bin/sh

private_key=$1
rpc_url=$2

if [ ! $private_key ]; then
    echo "[!] No private key provided"
    echo "  > Usage: deploy_contracts.sh \$PRIVATE_KEY"
fi

# Unexpectaly if ! [ -z ... ] is not working
if [ ! $rpc_url ]; then
    rpc_url=http://0.0.0.0:8545
fi


if [ -z "${FREEWILLAI_DIRECTORY}" ]; then
    app_dir=.
else
    app_dir=${FREEWILLAI_DIRECTORY}
fi


cd $app_dir
token_std=`forge create contracts/FreeWillAIToken.sol:FreeWillAI \
    --rpc-url $rpc_url \
    --private-key $private_key \
    --legacy`

echo ">>> $token_std"


deployer=`echo "$token_std" | grep 'Deployer:' | cut -d ' ' -f 2`
token_address=`echo "$token_std" | grep 'Deployed to:' | cut -d ' ' -f 3`
token_tx_hash=`echo "$token_std" | grep 'Transaction hash:' | cut -d ' ' -f 3`

echo -e "\nDeploy by: $deployer"
echo -e "\nDeployed Token Contract (FWAI)"
echo -e "\tContract address: $token_address"
echo -e "\tTransaction hash: $token_tx_hash"


task_runner_cmd=`forge create contracts/TaskRunner.sol:TaskRunner \
    --rpc-url $rpc_url \
    --private-key $private_key \
    --constructor-args $token_address \
    --legacy`

task_runner_address=`echo "$task_runner_cmd" | grep 'Deployed to:' | cut -d ' ' -f 3`
task_runner_tx_hash=`echo "$task_runner_cmd" | grep 'Transaction hash:' | cut -d ' ' -f 3`

echo -e "\nDeployed TaskRunner Contract"
echo -e "\tContract address: $task_runner_address"
echo -e "\tTransaction hash: $task_runner_tx_hash"

# Build envfile to save contracts' addresses:
if [ -z "${3}" ]; then
    out_envfile=.contract.env
else
    out_envfile=${3}
fi
    cat << EOF > ${out_envfile}
export PRIVATE_KEY=${private_key}
export FREEWILLAI_TOKEN_ADDRESS=${token_address}
export FREEWILLAI_TASK_RUNNER_ADDRESS=${task_runner_address}
export FREEWILLAI_TOKEN_ABI_PATH=./contracts/FreeWillAITokenABI.json
export FREEWILLAI_TASK_RUNNER_ABI_PATH=./contracts/TaskRunnerABI.json
EOF
