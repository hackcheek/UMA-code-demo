#!/bin/bash
app="demo"
docker build -t ${app} -f repl.dockerfile .
docker run -d -p 80:80 \
  --name=${app} \
  -v $PWD:/app ${app}
