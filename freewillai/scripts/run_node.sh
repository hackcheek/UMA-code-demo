#!/bin/bash
set -x  # Turn on trace mode to echo every command before executing them
if [ -z "$1" ]; then
	echo "Usage: ./run_node.sh <private_key>"
	echo "Example: ./run_node.sh 0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97"
else
	python3 -m venv venv
	source venv/bin/activate
	DEMO=1 python3 -m freewillai.node -s 100 -n devnet/anvil -p $1 
fi
