import json
import os
from tqdm import tqdm
from datasets import load_dataset
from utils import predict_gpt, predict_llama
import re
import argparse
from data_preprocess.aqua import load_aqua_questions, process_aqua_questions, process_aqua_questions_swapping_simple, process_aqua_questions_swapping_complex
from data_preprocess.gaokao import load_gaokao_questions, process_gaokao_questions, process_gaokao_questions_swap_complex
from data_preprocess.mmlu import process_mmlu_questions, process_mmlu_questions_swap_complex, process_mmlu_questions_shuffled
from data_preprocess.mmlupro import process_mmlu_pro_questions_shuffled, process_mmlu_pro_questions, process_mmlu_pro_questions_swap_complex
from data_preprocess.gsm8k import load_gsm8k_questions, process_gsm8k_questions
from data_preprocess.mgsm import load_mgsm_questions, process_mgsm_questions
from data_preprocess.gpqa import process_gpqa_questions
from data_preprocess.math import process_competition_math_questions

from dotenv import load_dotenv
import openai
import anthropic
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import random
from torch.nn import DataParallel

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
login(token=os.getenv('HUGGING_FACE_HUB_TOKEN'))
anthropic_client = anthropic.Client(api_key=os.getenv('CLAUDE_API_KEY'))

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(description="Process MMLU, MMLU-Pro, AQUA, GaoKao, TruthfulQA, Math, GPQA, MGSM, or GSM8K questions")
    parser.add_argument("--dataset", choices=["mmlu", "mmlu-pro", "aqua", "gaokao", "truthfulqa", "math", "gpqa", "mgsm", "gsm8k"], required=True, help="Choose the dataset")
    parser.add_argument("--method", choices=["cot", "non-cot", "symbolicot"], required=True, help="Choose the method")
    parser.add_argument("--model", choices=["gpt", "llama3.1_8b", "phi-3", "codestral", "llama3.1_70b", "qwen2"], required=True, help="Choose the model")
    parser.add_argument("--dataconfig", choices=["normal", "shuffle", "swapping"], default="normal", help="Choose the data configuration")
    args = parser.parse_args()

    output_dir = f"results/{args.dataset}/{args.method}/{args.model}"
    output_file_path = f"{output_dir}/{args.method}_{args.model}_{args.dataconfig}.json"
    log_file_path = f"{output_dir}/{args.method}_{args.model}_{args.dataconfig}_log.txt"

    # Ensure the directory exists
    ensure_dir(output_file_path)

    # Create the output file
    with open(output_file_path, 'w') as f:
        json.dump([], f)
    print(f"Created output file: {output_file_path}")

    # Create and write to the log file
    with open(log_file_path, 'w') as f:
        f.write(f"Start evaluating the {args.dataset} dataset with {args.method} method using {args.model} model and {args.dataconfig} configuration\n")
    print(f"Created log file: {log_file_path}")

    if args.dataset == "mmlu":
        prompt_dir = 'prompts/MMLU-Mathematics'
    elif args.dataset == "mmlu-pro":
        prompt_dir = 'prompts/MMLU-Pro-Mathematics'
    elif args.dataset == "aqua":
        prompt_dir = 'prompts/AQUA-Mathematics'
    elif args.dataset == "truthfulqa":
        prompt_dir = 'prompts/TruthfulQA'
    elif args.dataset == "math":
        prompt_dir = 'prompts/MATH'
    elif args.dataset == "mgsm":
        prompt_dir = 'prompts/MGSM'
    elif args.dataset == "gsm8k":
        prompt_dir = 'prompts/gsm8k'
    elif args.dataset == "gpqa":
        prompt_dir = 'prompts/gpqa'
    elif args.dataset == 'gaokao':  
        prompt_dir = 'prompts/GaoKao-Math'
    elif args.dataset == 'mgsm':
        prompt_dir = 'prompts/mgsm'
    else:
        raise ValueError ("prompts not inside the folder, please select a suitable dataset")

    formulation_prompt_path = f"{prompt_dir}/{args.method}.txt"

    # Ensure the directory exists
    ensure_dir(output_file_path)

    # Create the output file
    with open(output_file_path, 'w') as f:
        json.dump([], f)
    print(f"Created output file: {output_file_path}")

    # Initialize device, model, and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    tokenizer = None

    if args.model == "gpt":
        model = openai
    elif args.model == "llama3.1_8b":
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
    elif args.model == "llama3.1_70b":
        model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
    elif args.model == "phi-3":
        model_id = "microsoft/Phi-3.5-mini-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
    elif args.model == "codestral":
        model_id = "mistralai/Codestral-22B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
    elif args.model == "qwen2":
        model_id = "Qwen/Qwen2-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
    else: 
        raise ValueError("Model does not exist")

    if args.dataset == "mmlu":
        dataset = load_dataset("cais/mmlu", "college_mathematics", split="test")
        if args.dataconfig == "normal":
            results, accuracy = process_mmlu_questions(dataset, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        elif args.dataconfig == "shuffle":
            results, accuracy = process_mmlu_questions_shuffled(dataset, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        elif args.dataconfig == "swapping":
            results, accuracy = process_mmlu_questions_swap_complex(dataset, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
    elif args.dataset == "mmlu-pro":
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        math_dataset = dataset.filter(lambda example: example['category'] == 'math')
        if args.dataconfig == "normal":
            results, accuracy = process_mmlu_pro_questions(dataset, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        elif args.dataconfig == "shuffle":
            results, accuracy = process_mmlu_pro_questions_shuffled(dataset, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        elif args.dataconfig == "swapping":
            results, accuracy = process_mmlu_pro_questions_swap_complex(dataset, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
    elif args.dataset == "aqua":
        questions = load_aqua_questions('prompts/AQUA-Mathematics/test.json')
        if args.dataconfig == "normal":
            results, accuracy = process_aqua_questions(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        elif args.dataconfig == "shuffle":
            results, accuracy = process_aqua_questions_swapping_simple(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        elif args.dataconfig == "swapping":
            results, accuracy = process_aqua_questions_swapping_complex(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
    # elif args.dataset == "truthfulqa": ###REMOVING TRUTHFULQA
    #     # Load TruthfulQA dataset
    #     dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    #     results, accuracy = process_truthfulqa_questions(dataset, output_file_path, formulation_prompt_path, openai)
    elif args.dataset == "math":
        dataset = load_dataset("hendrycks/competition_math", split="test")
        results, accuracy = process_competition_math_questions(dataset, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
    elif args.dataset == "gpqa":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
        results, accuracy = process_gpqa_questions(dataset, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
    elif args.dataset == "mgsm":
        questions = load_mgsm_questions(load_dataset)
        if args.dataconfig == "normal":
            results, accuracy = process_mgsm_questions(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        else:
            raise ValueError("Please select --dataconfig normal")
    elif args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
        questions = load_gsm8k_questions(dataset)
        if args.dataconfig == "normal":
            results, accuracy = process_gsm8k_questions(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        else:
            raise ValueError("Please select --dataconfig normal")
    elif args.dataset == "gaokao":  
        questions = load_gaokao_questions('prompts/GaoKao-Math/2023_Math_MCQs.json')
        if args.dataconfig == "normal":
            results, accuracy = process_gaokao_questions(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        elif args.dataconfig == "swapping":
            results, accuracy = process_gaokao_questions_swap_complex(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        elif args.dataconfig == "swapping":
            results, accuracy = process_gaokao_questions_swap_complex(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        else:
            print("Shuffle configuration not implemented for GaoKao. Using normal processing.")
            results, accuracy = process_gaokao_questions(questions, output_file_path, formulation_prompt_path, openai)
    else:
        print("LOAD YOUR DATASET")

    print(results)
    print(f"Final results saved to {output_file_path}")
    print(f"Final Accuracy: {accuracy:.2%}")

    with open(log_file_path, 'a') as f:
        f.write(f"Final Accuracy: {accuracy:.2%}\n")

    print(f"Log file updated: {log_file_path}")

if __name__ == "__main__":
    main()
