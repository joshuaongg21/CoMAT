import json
from tqdm import tqdm
from datasets import load_dataset
from utils import predict_gpt, evaluate_gpt4o_mini
from config import output_file_path, openai, formulation_prompt_path
import re

def process_gpqa_questions(dataset, limit=10):
    results = []
    correct_count = 0
    total_count = 0

    for i, example in tqdm(enumerate(dataset), desc="Processing questions", total=min(limit, len(dataset))):
        if i >= limit:
            break

        question = example['Pre-Revision Question']
        correct_answer = example['Pre-Revision Correct Answer']

        print(f"Processing question: {question}")  # Debug print

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        gpt_result = predict_gpt(openai, messages)
        print(f"GPT result: {gpt_result}")  # Debug print

        # Extract final answer
        final_answer_match = re.search(r"Final Answer:(.+)$", gpt_result, re.MULTILINE | re.DOTALL)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else "No final answer found"

        # Evaluate the answer using GPT-4o-mini
        evaluation_result = evaluate_gpt4o_mini(question, final_answer, correct_answer)
        
        is_correct = evaluation_result != 'gpt-output is wrong, try again'
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question,
            "gpt_result": gpt_result,
            "final_answer": final_answer,
            "correct_answer": correct_answer,
            "is_correct": int(is_correct),  # Convert boolean to 0 or 1
            "evaluation_result": evaluation_result
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
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split='train')
    
    # Process GPQA questions (limited to 5)
    results, accuracy = process_gpqa_questions(dataset, limit=5)
    print(json.dumps(results, indent=2))
    print(f"Final results saved to {output_file_path}")
    print(f"Final Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()