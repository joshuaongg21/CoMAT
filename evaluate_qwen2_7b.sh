#!/bin/bash

datasets=("gsm8k" "olympiadbench-en" "olympiadbench-cn" "mgsm")
methods=("non-cot" "cot" "symbolicot")
model="qwen2-7b"
dataconfigs=("normal" "shuffle" "swapping")

# Set the visible CUDA devices 
export CUDA_VISIBLE_DEVICES=0

for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
          for dataconfig in "${dataconfigs[@]}"; do
              echo "Running: python main.py --dataset $dataset --method $method --model $model --dataconfig $dataconfig"
              python main.py --dataset "$dataset" --method "$method" --model "$model" --dataconfig "$dataconfig"
          done
    done
done
