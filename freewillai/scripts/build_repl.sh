#!/bin/bash

set -x

IMAGE_NAME=py_executor
DOCKERFILE=containers/repl2.dockerfile

docker build -f $DOCKERFILE -t $IMAGE_NAME .
