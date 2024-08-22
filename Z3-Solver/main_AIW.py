import json
from tqdm import tqdm
from datasets import load_dataset
from utils import predict_gpt, gpt4o_mini_decoder
from config import output_file_path, openai, formulation_prompt_path
import re
import openai

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def extract_final_answer(gpt_result):
    # Try to find an explicit "Answer: X" pattern first
    explicit_answer = re.search(r"Answer:\s*(\d+)", gpt_result, re.IGNORECASE)
    if explicit_answer:
        return explicit_answer.group(1)
    
    # If no explicit answer, look for the last number in the text
    numbers = re.findall(r'\d+', gpt_result)
    if numbers:
        return numbers[-1]
    
    # If still no number found, return "null"
    return "null"

def process_aiw_questions(dataset):
    results = []
    correct_count = 0
    total_count = 0

    for example in tqdm(dataset, desc="Processing questions"):
        prompt = example['prompt']
        correct_answer = example['right_answer']
        description = example['description']
        print(f"Processing question: {prompt}")  # Debug print

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        gpt_result = predict_gpt(openai, messages)
        print(f"GPT result: {gpt_result}")  # Debug print

        # Parse the final answer
        final_answer = extract_final_answer(gpt_result)

        is_correct = (final_answer == str(correct_answer))
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "prompt": prompt,
            "gpt_result": gpt_result,
            "final_answer": final_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "description": description
        })

        # Save results after each successful question
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for question {total_count}")  # Debug print

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy

def main():
    # Create the output file
    with open(output_file_path, 'w') as f:
        json.dump([], f)
    print(f"Created output file: {output_file_path}")

    # Load dataset
    dataset = load_dataset("AIW_dataset/AIW-Dataset.json")

    # Process all AIW questions
    results, accuracy = process_aiw_questions(dataset)
    print(results)
    print(f"Final results saved to {output_file_path}")
    print(f"Final Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()