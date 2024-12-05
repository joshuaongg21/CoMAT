import json
import os
import re
import argparse
from tqdm import tqdm
from datasets import load_dataset
import torch
from dotenv import load_dotenv
import openai
import anthropic
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

from utils import predict_gpt
from data_preprocess.aqua import load_aqua_questions, process_aqua_questions, process_aqua_questions_swapping_complex
from data_preprocess.gaokao import load_gaokao_questions, process_gaokao_questions, process_gaokao_questions_swap_complex
from data_preprocess.mmlu_redux import process_mmlu_redux_questions, process_mmlu_redux_questions_swap_complex
from data_preprocess.gsm8k import load_gsm8k_questions, process_gsm8k_questions
from data_preprocess.mgsm import load_mgsm_questions, process_mgsm_questions
from data_preprocess.olympiadbench import process_olympiadbench_questions
from data_preprocess.AIME import load_aime_questions, process_aime_questions  # Added import for AIME

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
anthropic_client = anthropic.Client(api_key=os.getenv('CLAUDE_API_KEY'))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    DATASET_CHOICES = [
        "mmlu-redux-abstract_algebra",
        "mmlu-redux-elementary_mathematics",
        "mmlu-redux-high_school_mathematics",
        "mmlu-redux-college_mathematics",
        "mmlu",
        "aqua",
        "gaokao",
        "mgsm",
        "gsm8k",
        "olympiadbench-en",
        "olympiadbench-cn",
        "aime"  # Added AIME to the dataset choices
    ]
    parser = argparse.ArgumentParser(description="Process datasets")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True, help="Choose the dataset")
    parser.add_argument("--method", choices=["cot", "non-cot", "comat"], required=True, help="Choose the method")
    parser.add_argument("--model", choices=["gpt", "qwen2-7b", "qwen2-72b", "gemini"], required=True, help="Choose the model")
    parser.add_argument("--dataconfig", choices=["normal", "swapping"], default="normal", help="Choose the data configuration")
    args = parser.parse_args()

    output_dir = f"final_results/{args.dataset}/{args.method}/{args.model}"
    output_file_path = f"{output_dir}/{args.method}_{args.model}_{args.dataconfig}.json"
    log_file_path = f"{output_dir}/{args.method}_{args.model}_{args.dataconfig}_log.txt"

    ensure_dir(output_file_path)

    with open(output_file_path, 'w') as f:
        json.dump([], f)
    print(f"Created output file: {output_file_path}")

    with open(log_file_path, 'w') as f:
        f.write(f"Start evaluating the {args.dataset} dataset with {args.method} method using {args.model} model and {args.dataconfig} configuration\n")
    print(f"Created log file: {log_file_path}")

    start_time = time.time()  # Start timing the evaluation

    if args.dataset == "mmlu-redux-abstract_algebra":
        prompt_dir = 'prompts/MMLU-Redux-abstract_algebra'
    elif args.dataset == "mmlu-redux-elementary_mathematics":
        prompt_dir = 'prompts/MMLU-Redux-elementary_mathematics'
    elif args.dataset == "mmlu-redux-high_school_mathematics":
        prompt_dir = 'prompts/MMLU-Redux-high_school_mathematics'
    elif args.dataset == "mmlu-redux-college_mathematics":
        prompt_dir = 'prompts/MMLU-Redux-college_mathematics'
    elif args.dataset == "aqua":
        prompt_dir = 'prompts/AQUA-Mathematics'
    elif args.dataset == "mgsm":
        prompt_dir = 'prompts/mgsm'
    elif args.dataset == "gsm8k":
        prompt_dir = 'prompts/gsm8k'
    elif args.dataset == 'gaokao':  
        prompt_dir = 'prompts/GaoKao-Math'
    elif args.dataset == "olympiadbench-en" or args.dataset == "olympiadbench-cn":
        prompt_dir = 'prompts/OlympiadBench'
        if not os.path.isdir(prompt_dir):
            prompt_dir = 'prompts/olympiadbench'
    elif args.dataset == "aime":
        prompt_dir = 'prompts/AIME'  # Added prompt directory for AIME
    else:
        raise ValueError ("Prompts not inside the folder, please select a suitable dataset")

    formulation_prompt_path = f"{prompt_dir}/{args.method}.txt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    tokenizer = None

    if args.model == "gpt":
        model = openai
    elif args.model == "gemini":
        model = genai
    elif args.model == "qwen2-7b":
        model_id = "Qwen/Qwen2-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
    elif args.model == "qwen2-72b":
        model_id = "Qwen/Qwen2-72B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
    else: 
        raise ValueError("Model does not exist")

    if args.dataset.startswith("mmlu-redux"):
        subject = args.dataset.split("-")[-1]
        dataset = load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject, split="test")
        if args.dataconfig == "normal":
            results, accuracy = process_mmlu_redux_questions(dataset, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        elif args.dataconfig == "swapping":
            results, accuracy = process_mmlu_redux_questions_swap_complex(dataset, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
    elif args.dataset == "aqua":
        questions = load_aqua_questions('prompts/AQUA-Mathematics/test.json')
        if args.dataconfig == "normal":
            results, accuracy = process_aqua_questions(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
        elif args.dataconfig == "swapping":
            results, accuracy = process_aqua_questions_swapping_complex(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
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
        else:
            print("Shuffle configuration not implemented for GaoKao. Using normal processing.")
            results, accuracy = process_gaokao_questions(questions, output_file_path, formulation_prompt_path, openai)
    elif args.dataset == "olympiadbench-en":
        dataset = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
        questions = [
            {
                "question": item['question'],
                "final_answer": item['final_answer'][0] 
            }
            for item in dataset
        ]
        results, accuracy = process_olympiadbench_questions(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
    elif args.dataset == "olympiadbench-cn":
        dataset = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_zh_COMP", split="train")
        questions = [
            {
                "question": item['question'],
                "final_answer": item['final_answer'][0] 
            }
            for item in dataset
        ]
        results, accuracy = process_olympiadbench_questions(questions, output_file_path, formulation_prompt_path, args.model, model, tokenizer, device)
    elif args.dataset == "aime":
        questions = load_aime_questions('prompts/AIME/AIME_Dataset_1983_2024.csv')  # Update the path as needed
        if args.dataconfig == "normal":
            results, accuracy = process_aime_questions(
                questions,
                output_file_path,
                formulation_prompt_path,
                args.model,
                model,
                tokenizer,
                device
            )
        else:
            raise ValueError("Please select --dataconfig normal")
    else:
        raise ValueError ("Dataset not found")


    end_time = time.time()
    duration = end_time - start_time
    print(results)
    print(f"Final results saved to {output_file_path}")
    print(f"Final Accuracy: {accuracy:.2%}")
    print(f"Evaluation Duration: {duration:.2f} seconds")


    with open(log_file_path, 'a') as f:
        f.write(f"Final Accuracy: {accuracy:.2%}\n")
        f.write(f"Evaluation Duration: {duration:.2f} seconds\n")

    print(f"Log file updated: {log_file_path}")

if __name__ == "__main__":
    main()
