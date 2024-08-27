#!/bin/bash

datasets=("mmlu" "aqua" "mmlu-pro" "GPQA" "gaokao")
methods=("non-cot" "cot" "symbolicot")
model="llama3.1_8b"
dataconfig=("normal" "shuffle" "swapping")

# Set the visible CUDA devices to GPU 0 and GPU 1
export CUDA_VISIBLE_DEVICES=0,1

for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        echo "Running: python main.py --dataset $dataset --method $method --model $model --dataconfig $dataconfig"
        python main.py --dataset "$dataset" --method "$method" --model "$model"
    done
done
