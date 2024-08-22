import json
from tqdm import tqdm
from utils import predict_gpt
from config import output_file_path, openai, formulation_prompt_path
import re

def load_aqua_questions(file_path):
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                question = json.loads(line.strip())
                questions.append(question)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}")
                print(f"Error message: {str(e)}")
    return questions

def process_aqua_questions(questions):
    results = []
    correct_count = 0
    total_count = 0

    for example in tqdm(questions, desc="Processing questions"):
        question = example['question']
        options = example['options']
        correct_answer = example['correct']
        
        print(f"Processing question: {question}")  # Debug print

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        # Format options as A, B, C, D, E
        formatted_options = "\n".join([f"{option}" for option in options])

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        
        gpt_result = predict_gpt(openai, messages)
        print(f"GPT result: {gpt_result}")  # Debug print

        # Parse the final answer
        final_answer_match = re.search(r"Final Answer: ([ABCDE])", gpt_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
        else:
            final_answer_letter = "Invalid"  # Invalid answer

        is_correct = (final_answer_letter == correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question,
            "options": options,
            "gpt_result": gpt_result,
            "final_answer": final_answer_letter,
            "correct_answer": correct_answer,
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
    questions = load_aqua_questions('prompts/AQUA-Mathematics/test.json')

    # Process all AQuA questions
    results, accuracy = process_aqua_questions(questions)
    print(results)
    print(f"Final results saved to {output_file_path}")
    print(f"Final Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()