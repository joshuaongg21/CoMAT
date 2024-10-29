datasets=("mmlu" "aqua" "gaokao" "gsm8k" "olympiadbench-en" "olympiadbench-cn" "mmlu-redux-abstract_algebra" "mmlu-redux-elementary_mathematics" "mmlu-redux-high_school_mathematics" "mmlu-redux-college_mathematics")
methods=("non-cot" "cot" "symbolicot")
models=("gpt" "gemini" "qwen2-7b" "qwen2-72b")
dataconfigs=("normal" "swapping")

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