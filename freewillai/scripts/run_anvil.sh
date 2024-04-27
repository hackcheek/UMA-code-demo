#!/bin/bash
set -x  # Turn on trace mode to echo every command before executing them
anvil --config-out anvil_configs.json --host 0.0.0.0 --port 8545
