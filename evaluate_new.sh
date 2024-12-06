datasets=("svamp" "multiarath")
methods=("non-cot" "cot" "comat")
models=("qwen2-7b" "qwen2-72b" "gpt" "gemini" )
dataconfigs=("normal")

export CUDA_VISIBLE_DEVICES=0,1

for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        for model in "${models[@]}"; do
            for dataconfig in "${dataconfigs[@]}"; do
                echo "Running: python main.py --dataset $dataset --method $method --model $model --dataconfig $dataconfig"
                python main.py --dataset "$dataset" --method "$method" --model "$model" --dataconfig "$dataconfig"
            done
        done
    done
done