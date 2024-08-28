#!/bin/bash

datasets=("mmlu" "aqua" "GPQA" "gaokao" "mmlu-pro" "gsm8k" "mgsm")
methods=("non-cot" "cot" "symbolicot")
model="codestral"
dataconfigs=("normal" "shuffle" "swapping")

# Set the visible CUDA devices 
export CUDA_VISIBLE_DEVICES=0,1

for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
          for dataconfig in "${dataconfigs[@]}"; do
              echo "Running: python main.py --dataset $dataset --method $method --model $model --dataconfig $dataconfig"
              python main.py --dataset "$dataset" --method "$method" --model "$model" --dataconfig "$dataconfig"
          done
    done
done
