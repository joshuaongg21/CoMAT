import json
import os
from tqdm import tqdm
from datasets import load_dataset
from utils import predict_gpt
import re
import argparse
from data_preprocessing import process_mmlu_questions, process_mmlu_pro_questions, load_aqua_questions, process_aqua_questions, load_gaokao_questions, process_gaokao_questions
from dotenv import load_dotenv
import openai
import anthropic

# Load environment variables
load_dotenv()

# Set up your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up your Anthropic API key
anthropic_client = anthropic.Client(api_key=os.getenv('CLAUDE_API_KEY'))

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(description="Process MMLU, MMLU-Pro, AQUA, or GaoKao questions")
    parser.add_argument("--dataset", choices=["mmlu", "mmlu-pro", "aqua", "gaokao"], required=True, help="Choose the dataset: 'mmlu', 'mmlu-pro', 'aqua', or 'gaokao'")
    args = parser.parse_args()

    # Set up the output file path based on the dataset
    output_dir = f"results/{args.dataset}"
    output_file_path = f"{output_dir}/testing.json"

    # Set up the formulation prompt path based on the dataset
    if args.dataset == "mmlu":
        formulation_prompt_path = 'prompts/MMLU-Mathematics/formulation.txt'
    elif args.dataset == "mmlu-pro":
        formulation_prompt_path = 'prompts/MMLU-Pro-Mathematics/formulation.txt'
    elif args.dataset == "aqua":
        formulation_prompt_path = 'prompts/AQUA-Mathematics/CoT.txt'
    else:  # gaokao
        formulation_prompt_path = 'prompts/GaoKao-Math/formulation.txt'

    # Ensure the directory exists
    ensure_dir(output_file_path)

    # Create the output file
    with open(output_file_path, 'w') as f:
        json.dump([], f)
    print(f"Created output file: {output_file_path}")

    if args.dataset == "mmlu":
        # Load MMLU dataset
        dataset = load_dataset("cais/mmlu", "college_mathematics", split="test")
        results, accuracy = process_mmlu_questions(dataset, output_file_path, formulation_prompt_path, openai)
    elif args.dataset == "mmlu-pro":
        # Load MMLU-Pro dataset
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        math_dataset = dataset.filter(lambda example: example['category'] == 'math')
        results, accuracy = process_mmlu_pro_questions(math_dataset, output_file_path, formulation_prompt_path, openai)
    elif args.dataset == "aqua":
        # Load AQUA dataset
        questions = load_aqua_questions('prompts/AQUA-Mathematics/test.json')
        results, accuracy = process_aqua_questions(questions, output_file_path, formulation_prompt_path, openai)
    else:  # gaokao
        # Load GaoKao dataset
        questions = load_gaokao_questions('prompts/GaoKao-Math/2023_Math_MCQs.json')
        results, accuracy = process_gaokao_questions(questions, output_file_path, formulation_prompt_path, openai)

    print(results)
    print(f"Final results saved to {output_file_path}")
    print(f"Final Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()