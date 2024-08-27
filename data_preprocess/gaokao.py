import re
from datasets import load_dataset
import json
from tqdm import tqdm
from utils import predict_gpt, evaluate_gpt4o_mini, predict_llama
import random

def load_gaokao_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['example']

def process_gaokao_questions(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
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

        if model_type == "gpt":
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
            ]
            model_result = predict_gpt(model, messages)
        else:  
            prompt = f"{system_content}\n\nQuestion: {question}\n\nOptions:\n{formatted_options}"
            model_result = predict_llama(model, tokenizer, prompt, max_new_tokens=1024)

        # Parse the final answer
        final_answer_match = re.search(r"Final Answer: ([ABCD])", model_result)
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
            "model_result": model_result,
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

def process_gaokao_questions_swap_complex(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    correct_count = 0
    total_count = 0

    additional_options = [
        "Blank, ignore this option",
        "Real Madrid is the Best Team",
        "Karma is my Boyfriend",
        "I was enhanced to meet you",
        "May the force be with you"
    ]

    for example in tqdm(questions, desc="Processing questions"):
        question = example['question']
        options = re.findall(r'([A-D]\..*?)(?=[A-D]\.|\Z)', question, re.DOTALL)
        options = [opt.strip() for opt in options]
        correct_answer = example['answer'][0]  # Assuming the first answer is the correct one
        
        print(f"Processing question: {question}")  # Debug print

        # Add a random additional option
        additional_option = random.choice(additional_options)
        options.append(f"E. {additional_option}")

        # Safely extract option contents
        option_contents = []
        for opt in options:
            parts = opt.split('.', 1)
            if len(parts) > 1:
                option_contents.append(parts[1].strip())
            else:
                option_contents.append(opt.strip())

        # Randomly shuffle the contents of the options
        random.shuffle(option_contents)

        # Create new options with shuffled content
        shuffled_options = [f"{chr(65+i)}. {content}" for i, content in enumerate(option_contents)]

        # Find the new position of the correct answer
        correct_content = options[ord(correct_answer) - 65].split('.', 1)[1].strip()
        new_correct_answer = chr(65 + option_contents.index(correct_content))

        with open(formulation_prompt_path, 'r', encoding='utf-8') as f:
            system_content = f.read()

        # Format shuffled options
        formatted_options = "\n".join(shuffled_options)

        # Remove options from the original question
        question_without_options = re.sub(r'[A-D]\..*?(?=[A-D]\.|\Z)', '', question, flags=re.DOTALL)
        question_without_options = question_without_options.strip()

        if model_type == "gpt":
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
            ]
            model_result = predict_gpt(model, messages)
        else:  
            prompt = f"{system_content}\n\nQuestion: {question}\n\nOptions:\n{formatted_options}"
            model_result = predict_llama(model, tokenizer, prompt, max_new_tokens=1024)

        # Parse the final answer
        final_answer_match = re.search(r"Final Answer: ([A-E])", model_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
        else:
            final_answer_letter = "Invalid"  # Invalid answer

        is_correct = (final_answer_letter == new_correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question_without_options,
            "original_options": options[:4],  # Original A, B, C, D options
            "shuffled_options_with_additional": shuffled_options,
            "model_result": model_result,
            "final_answer": final_answer_letter,
            "original_correct_answer": correct_answer,
            "new_correct_answer": new_correct_answer,
            "is_correct": is_correct,
            "additional_option": additional_option
        })

        # Save results after each successful question
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for question {total_count}")  # Debug print

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy

def process_gaokao_questions_shuffled(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    correct_count = 0
    total_count = 0

    for example in tqdm(questions, desc="Processing questions"):
        question = example['question']
        options = re.findall(r'([A-D]\..*?)(?=[A-D]\.|\Z)', question, re.DOTALL)
        options = [opt.strip() for opt in options]
        correct_answer = example['answer'][0]  # Assuming the first answer is the correct one

        print(f"Processing question: {question}")

        # Randomly swap options
        shuffled_options = options.copy()
        random.shuffle(shuffled_options)

        # Create a mapping of new positions to old positions
        option_mapping = {new: old for new, old in enumerate(shuffled_options)}

        # Update the correct answer based on the new positions
        new_correct_answer = chr(65 + shuffled_options.index(options[ord(correct_answer) - 65]))

        with open(formulation_prompt_path, 'r', encoding='utf-8') as f:
            system_content = f.read()

        # Format shuffled options as A, B, C, D
        formatted_options = "\n".join([f"{chr(65 + i)}. {option.split('.', 1)[1].strip()}" for i, option in enumerate(shuffled_options)])

        # Remove options from the original question
        question_without_options = re.sub(r'[A-D]\..*?(?=[A-D]\.|\Z)', '', question, flags=re.DOTALL)
        question_without_options = question_without_options.strip()

        if model_type == "gpt":
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
            ]
            model_result = predict_gpt(model, messages)
        else:  
            prompt = f"{system_content}\n\nQuestion: {question}\n\nOptions:\n{formatted_options}"
            model_result = predict_llama(model, tokenizer, prompt, max_new_tokens=1024)

        # Parse the final answer
        final_answer_match = re.search(r"Final Answer: ([ABCD])", model_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
        else:
            final_answer_letter = "Invalid"  # Invalid answer

        is_correct = (final_answer_letter == new_correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question_without_options,
            "original_options": options,
            "shuffled_options": shuffled_options,
            "model_result": model_result,
            "final_answer": final_answer_letter,
            "original_correct_answer": correct_answer,
            "new_correct_answer": new_correct_answer,
            "is_correct": is_correct
        })

        # Save results after each successful question
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for question {total_count}")

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy