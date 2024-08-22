import json
from tqdm import tqdm
from datasets import load_dataset
from utils import predict_gpt, gpt4o_mini_decoder
from config import output_file_path, openai, formulation_prompt_path
import re
import openai

def process_mmlu_pro_questions(dataset):
    results = []
    correct_count = 0
    total_count = 0

    # Convert the dataset to a dictionary
    data_dict = dataset.to_dict()
    
    # Get the number of questions (length of any column)
    num_questions = len(data_dict['question'])

    for i in tqdm(range(num_questions), desc="Processing questions"):
        question = data_dict['question'][i]
        options = data_dict['options'][i]
        correct_answer = data_dict['answer'][i]
        correct_answer_index = data_dict['answer_index'][i]

        print(f"Processing question: {question}")  # Debug print

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        # Format options as A, B, C, D, etc.
        formatted_options = "\n".join([f"{chr(65+j)}. {option}" for j, option in enumerate(options)])

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        
        gpt_result = predict_gpt(openai, messages)
        print(f"GPT result: {gpt_result}")  # Debug print

        # Parse the final answer
        final_answer_match = re.search(r"Final Answer: ([A-Z])", gpt_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
            final_answer_numeric = ord(final_answer_letter) - ord('A')
        else:
            final_answer_numeric = -1  # Invalid answer

        is_correct = (final_answer_numeric == correct_answer_index)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question,
            "options": options,
            "gpt_result": gpt_result,
            "final_answer": final_answer_numeric,
            "correct_answer": correct_answer,
            "correct_answer_index": correct_answer_index,
            "is_correct": is_correct
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
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    
    # Filter for math category
    math_dataset = dataset.filter(lambda example: example['category'] == 'math')

    # Process MMLU-Pro questions (math category from test set)
    results, accuracy = process_mmlu_pro_questions(math_dataset)
    print(results)
    print(f"Final results saved to {output_file_path}")
    print(f"Final Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()