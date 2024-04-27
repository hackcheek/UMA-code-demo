#!/bin/sh

# Check for the presence of the .ipfs directory in the user's home directory
set -x  # Turn on trace mode to echo every command

api_port=$1
gateway_port=$2

# Init ipfs
ipfs init

# Configs
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'
ipfs config Addresses.API /ip4/0.0.0.0/tcp/${api_port}
ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/${gateway_port}

# Run IPFS daemon
ipfs daemon
