#!/bin/sh

# Remove venv if exists
rm -rf venv 2>/dev/null

# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/hackcheek/tensorflow-onnx
