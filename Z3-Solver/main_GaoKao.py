import json
from tqdm import tqdm
from utils import predict_gpt
from config import output_file_path, openai, formulation_prompt_path
import re

def load_gaokao_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['example']

def process_gaokao_questions(questions):
    results = []
    correct_count = 0
    total_count = 0

    for example in tqdm(questions, desc="Processing questions"):
        question = example['question']
        options = re.findall(r'([A-D]\..*?)(?=[A-D]\.|\Z)', question, re.DOTALL)
        options = [opt.strip() for opt in options]
        correct_answer = example['answer'][0]  # Assuming the first answer is the correct one
        
        print(f"Processing question: {question}")  # Debug print

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        # Format options as A, B, C, D
        formatted_options = "\n".join(options)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        
        gpt_result = predict_gpt(openai, messages)
        print(f"GPT result: {gpt_result}")  # Debug print

        # Parse the final answer
        final_answer_match = re.search(r"Final Answer: ([ABCD])", gpt_result)
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
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for question {total_count}")  # Debug print

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy

def main():
    # Create the output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump([], f)
    print(f"Created output file: {output_file_path}")

    # Load dataset
    questions = load_gaokao_questions('prompts/GaoKao-Math/2023_Math_MCQs.json')

    # Process all GaoKao Math questions
    results, accuracy = process_gaokao_questions(questions)
    print(results)
    print(f"Final results saved to {output_file_path}")
    print(f"Final Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()