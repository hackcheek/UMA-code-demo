#!/bin/bash
set -x  # Turn on trace mode to echo every command before executing them
python3 -m venv venv
source venv/bin/activate
export ANVIL_CONFIG_PATH=anvil_configs.json
DEMO=1 python3 demo.py
