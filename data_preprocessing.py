import json
from tqdm import tqdm
import re
from utils import predict_gpt

def process_mmlu_questions(dataset, output_file_path, formulation_prompt_path, openai):
    results = []
    correct_count = 0
    total_count = 0

    for example in tqdm(dataset, desc="Processing questions"):
        question = example['question']
        options = example['choices']
        correct_answer = example['answer']
        
        print(f"Processing question: {question}")  # Debug print

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        formatted_options = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        
        gpt_result = predict_gpt(openai, messages)
        print(f"GPT result: {gpt_result}")  # Debug print

        final_answer_match = re.search(r"Final Answer: ([ABCD])", gpt_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
            final_answer_numeric = ord(final_answer_letter) - ord('A')
        else:
            final_answer_numeric = -1  # Invalid answer

        is_correct = (final_answer_numeric == correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question,
            "options": options,
            "gpt_result": gpt_result,
            "final_answer": final_answer_numeric,
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

def process_mmlu_pro_questions(dataset, output_file_path, formulation_prompt_path, openai):
    results = []
    correct_count = 0
    total_count = 0

    data_dict = dataset.to_dict()
    num_questions = len(data_dict['question'])

    for i in tqdm(range(num_questions), desc="Processing questions"):
        question = data_dict['question'][i]
        options = data_dict['options'][i]
        correct_answer = data_dict['answer'][i]
        correct_answer_index = data_dict['answer_index'][i]

        print(f"Processing question: {question}")  # Debug print

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        formatted_options = "\n".join([f"{chr(65+j)}. {option}" for j, option in enumerate(options)])

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        
        gpt_result = predict_gpt(openai, messages)
        print(f"GPT result: {gpt_result}")  # Debug print

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

def process_aqua_questions(questions, output_file_path, formulation_prompt_path, openai):
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

def load_gaokao_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['example']

def process_gaokao_questions(questions, output_file_path, formulation_prompt_path, openai):
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