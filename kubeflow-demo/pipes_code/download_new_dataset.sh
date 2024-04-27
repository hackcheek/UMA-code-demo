#!/bin/bash

dataset_link='https://raw.githubusercontent.com/Pratik94229/Diamond-Price-Prediction-End-to-End-Project/main/artifacts/raw.csv'
new_dataset_path='datasets/diamonds/new_records.csv'

echo "[*] Downloading new records from a random selection of the full dataset"
echo "  > Data rows: 2k"
echo "  > Full Dataset: $dataset_link"

content=`curl -s $dataset_link | sed 's/[^,]*,//'`
header=`echo "$content" | head -1`
rows=`echo "$content" | tail +2 | shuf -n 2000`

new_dataset=`echo -e "$header\n$rows"`
echo "$new_dataset" > $new_dataset_path
