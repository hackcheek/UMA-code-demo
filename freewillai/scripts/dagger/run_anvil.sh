#!/bin/sh

host=localhost
port=$1
rpc_url=http://$host:$port

if [ $host == localhost ];then
    anvil_host=0.0.0.0
else
    anvil_host=$host
fi
anvil --host $anvil_host --port $port &
anvil_pid=$!

echo $anvil_pid > /anvil/pid

cd /app

# Get PRIVATE_KEY
source .env

# Waiting enough to ensure that anvil starts
sleep 1

# Set account balance
address=`cast wallet address ${PRIVATE_KEY}`
curl $rpc_url \
    -X POST \
    -H "Content-Type: application/json" \
    --data "{
    \"method\":\"anvil_setBalance\",
    \"params\":[\"${address}\", \"0x56BC75E2D63100000\"],
    \"id\":1,
    \"jsonrpc\":\"2.0\"
}"

token_std=`forge create contracts/FreeWillAIToken.sol:FreeWillAI \
    --rpc-url $rpc_url \
    --gas-limit 30000000 \
    --private-key $PRIVATE_KEY \
    --legacy`

deployer=`echo "$token_std" | grep 'Deployer:' | cut -d ' ' -f 2`
token_address=`echo "$token_std" | grep 'Deployed to:' | cut -d ' ' -f 3`
token_tx_hash=`echo "$token_std" | grep 'Transaction hash:' | cut -d ' ' -f 3`

echo -e "TokenAddress:$token_address"

task_runner_cmd=`forge create contracts/TaskRunner.sol:TaskRunner \
    --rpc-url $rpc_url \
    --private-key $PRIVATE_KEY \
    --constructor-args $token_address \
    --legacy`

task_runner_address=`echo "$task_runner_cmd" | grep 'Deployed to:' | cut -d ' ' -f 3`
task_runner_tx_hash=`echo "$task_runner_cmd" | grep 'Transaction hash:' | cut -d ' ' -f 3`

echo -e "TaskRunnerAddress:$task_runner_address"


cat << EOF > /anvil/contracts.env
export OWNER_PRIVATE_KEY=${private_key}
export FREEWILLAI_TOKEN_ADDRESS=${token_address}
export FREEWILLAI_TASK_RUNNER_ADDRESS=${task_runner_address}
export FREEWILLAI_TOKEN_ABI_PATH=/app/contracts/FreeWillAITokenABI.json
export FREEWILLAI_TASK_RUNNER_ABI_PATH=/app/contracts/TaskRunnerABI.json
EOF

# To Attach anvil
wait $anvil_pid
