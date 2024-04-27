#!/bin/bash
set -x  # Turn on trace mode to echo every command before executing them
python3 -m venv venv
source venv/bin/activate

if [ -z "$1" ]; then
    echo "Usage: ./mint.sh <public_key>"
    echo "Example: ./mint.sh 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
else
    python3 -m scripts.token_owner -n demo --mint 200000000000000000000 --to $1
fi



