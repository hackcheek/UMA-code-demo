#!/bin/sh

echo $1 > saracatunga.txt
python exec_code.py
exit $?
