#!/bin/bash

# Check if the dataset argument is provided
if [ -z "$1" ]; then
    echo "Error: No dataset provided. Please provide a dataset as the first argument."
    echo "Usage: $0 <dataset> [CUDA_VISIBLE_DEVICES]"
    exit 1
fi

# Check if the CUDA_VISIBLE_DEVICES argument is provided, if not default to "0"
if [ -z "$3" ]; then
    export CUDA_VISIBLE_DEVICES="0"
else
    export CUDA_VISIBLE_DEVICES="$3"
fi

dataset=$1
method=$2
model="qwen2-72b"
dataconfig="normal"

echo "Running: python main.py --dataset $dataset --method $method --model $model --dataconfig $dataconfig"
python main.py --dataset "$dataset" --method "$method" --model "$model" --dataconfig "$dataconfig"
