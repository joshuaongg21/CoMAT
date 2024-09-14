#!/bin/bash

# Check if the dataset argument is provided
if [ -z "$1" ]; then
    echo "Error: No dataset provided. Please provide a dataset as the first argument."
    echo "Usage: $0 <dataset> [CUDA_VISIBLE_DEVICES]"
    exit 1
fi

# Check if the CUDA_VISIBLE_DEVICES argument is provided, if not default to "0"
if [ -z "$2" ]; then
    export CUDA_VISIBLE_DEVICES="0,1"
else
    export CUDA_VISIBLE_DEVICES="$2"
fi

dataset=$1
methods=("non-cot" "cot" "symbolicot")
model="llama3.1_70b"
dataconfig="normal"

for method in "${methods[@]}"; do
    echo "Running: python main.py --dataset $dataset --method $method --model $model --dataconfig $dataconfig"
    python main.py --dataset "$dataset" --method "$method" --model "$model" --dataconfig "$dataconfig"
done
