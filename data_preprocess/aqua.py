import re
from datasets import load_dataset
import json
from tqdm import tqdm
from utils import predict_gpt, predict_llama, model_evaluation
import random
import torch
import openai
from transformers import pipeline

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

def process_aqua_questions(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    correct_count = 0
    total_count = 0

    for example in tqdm(questions, desc="Processing questions"):
        question = example['question']
        options = example['options']
        correct_answer = example['correct']

        print(f"Processing question: {question}")

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        formatted_options = "\n".join([f"{option}" for option in options])

        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device)

        print(f"Model result: {model_result}")

        final_answer_match = re.search(r"Final Answer: ([ABCDE])", model_result)
        final_answer_letter = final_answer_match.group(1) if final_answer_match else "Invalid"

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

        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for question {total_count}")

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy
    
def process_aqua_questions_swapping_simple(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    correct_count = 0
    total_count = 0

    for example in tqdm(questions, desc="Processing questions"):
        question = example['question']
        options = example['options']
        correct_answer = example['correct']
        
        print(f"Processing question: {question}")

        shuffled_options = options.copy()
        random.shuffle(shuffled_options)
        
        option_mapping = {old: new for new, old in enumerate(shuffled_options)}
        
        new_correct_answer = chr(65 + option_mapping[options[ord(correct_answer) - 65]])

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        formatted_options = "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(shuffled_options)])

        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device)

        print(f"Model result: {model_result}")

        final_answer_match = re.search(r"Final Answer: ([ABCDE])", model_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
        else:
            final_answer_letter = "Invalid"  

        is_correct = (final_answer_letter == new_correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question,
            "original_options": options,
            "shuffled_options": shuffled_options,
            "model_result": model_result,
            "final_answer": final_answer_letter,
            "original_correct_answer": correct_answer,
            "new_correct_answer": new_correct_answer,
            "is_correct": is_correct
        })


        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for question {total_count}")  

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy

def process_aqua_questions_swapping_complex(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    correct_count = 0
    total_count = 0

    additional_options = [
        "Blank, ignore this option",
        "Real Madrid is the Best Team",
        "Karma is my Boyfriend",
        "I was enchanted to meet you",
        "May the force be with you"
    ]

    for example in tqdm(questions, desc="Processing questions"):
        question = example['question']
        options = example['options']
        correct_answer = example['correct']
        
        print(f"Processing question: {question}")  

        additional_option = random.choice(additional_options)
        options.append(f"{chr(65+len(options))}){additional_option}")

        option_contents = []
        for opt in options:
            parts = opt.split(')', 1)
            if len(parts) > 1:
                option_contents.append(parts[1].strip())
            else:
                option_contents.append(opt.strip())

        random.shuffle(option_contents)

        shuffled_options = [f"{chr(65+i)}){content}" for i, content in enumerate(option_contents)]

        correct_content = options[ord(correct_answer) - 65].split(')', 1)[1].strip()
        new_correct_answer = chr(65 + option_contents.index(correct_content))

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        formatted_options = "\n".join(shuffled_options)
        
        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device)

        print(f"Model result: {model_result}")

        final_answer_match = re.search(r"Final Answer: ([A-" + chr(65+len(shuffled_options)-1) + "])", model_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
        else:
            final_answer_letter = "Invalid"  

        is_correct = (final_answer_letter == new_correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question,
            "original_options": example['options'],
            "shuffled_options_with_additional": shuffled_options,
            "gpt_result": model_result,
            "final_answer": final_answer_letter,
            "original_correct_answer": correct_answer,
            "new_correct_answer": new_correct_answer,
            "is_correct": is_correct,
            "additional_option": additional_option
        })

        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for question {total_count}")  

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy