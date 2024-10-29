import re
from datasets import load_dataset
import json
from tqdm import tqdm
from utils import predict_gpt, evaluate_gpt4o_mini, predict_llama, model_evaluation
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
        correct_answer = example['answer'][0]  
        
        print(f"Processing question: {question}")  

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        formatted_options = "\n".join(options)

        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device)

        print(f"Model result: {model_result}")

        final_answer_match = re.search(r"Final Answer: ([ABCD])", model_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
        else:
            final_answer_letter = "Invalid" 

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

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for question {total_count}") 

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
        correct_answer = example['answer'][0]  
        
        print(f"Processing question: {question}") 

        additional_option = random.choice(additional_options)
        options.append(f"E. {additional_option}")

        option_contents = []
        for opt in options:
            parts = opt.split('.', 1)
            if len(parts) > 1:
                option_contents.append(parts[1].strip())
            else:
                option_contents.append(opt.strip())

        random.shuffle(option_contents)

        shuffled_options = [f"{chr(65+i)}. {content}" for i, content in enumerate(option_contents)]

        correct_content = options[ord(correct_answer) - 65].split('.', 1)[1].strip()
        new_correct_answer = chr(65 + option_contents.index(correct_content))

        with open(formulation_prompt_path, 'r', encoding='utf-8') as f:
            system_content = f.read()

        formatted_options = "\n".join(shuffled_options)

        question_without_options = re.sub(r'[A-D]\..*?(?=[A-D]\.|\Z)', '', question, flags=re.DOTALL)
        question_without_options = question_without_options.strip()

        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device)

        print(f"Model result: {model_result}")

        final_answer_match = re.search(r"Final Answer: ([A-E])", model_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
        else:
            final_answer_letter = "Invalid"  

        is_correct = (final_answer_letter == new_correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question_without_options,
            "original_options": options[:4], 
            "shuffled_options_with_additional": shuffled_options,
            "model_result": model_result,
            "final_answer": final_answer_letter,
            "original_correct_answer": correct_answer,
            "new_correct_answer": new_correct_answer,
            "is_correct": is_correct,
            "additional_option": additional_option
        })

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for question {total_count}") 

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
        correct_answer = example['answer'][0]  

        print(f"Processing question: {question}")

        shuffled_options = options.copy()
        random.shuffle(shuffled_options)

        option_mapping = {new: old for new, old in enumerate(shuffled_options)}

        new_correct_answer = chr(65 + shuffled_options.index(options[ord(correct_answer) - 65]))

        with open(formulation_prompt_path, 'r', encoding='utf-8') as f:
            system_content = f.read()

        formatted_options = "\n".join([f"{chr(65 + i)}. {option.split('.', 1)[1].strip()}" for i, option in enumerate(shuffled_options)])

        question_without_options = re.sub(r'[A-D]\..*?(?=[A-D]\.|\Z)', '', question, flags=re.DOTALL)
        question_without_options = question_without_options.strip()

        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device)

        print(f"Model result: {model_result}")

        final_answer_match = re.search(r"Final Answer: ([ABCD])", model_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
        else:
            final_answer_letter = "Invalid"  

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

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for question {total_count}")

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy