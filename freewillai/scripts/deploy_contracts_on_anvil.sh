#!/bin/bash

set -x

source .env
bash ./scripts/deploy_contracts.sh ${PRIVATE_KEY} http://127.0.0.1:8545 anvil.contracts.env
