#!/bin/bash
set -x  # Turn on trace mode to echo every command before executing them
if [ -z "$1" ]; then
	echo "Usage: ./run_node1.sh <network>"
	echo "Example: ./run_node1.sh devnet/anvil"
else
	python3 -m venv venv
	source venv/bin/activate
	DEMO=1 python3 -m freewillai.node -s 500 -n $1 -e worker1.env 
fi
