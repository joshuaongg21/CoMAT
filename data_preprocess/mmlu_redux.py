import random
import re
import json
from tqdm import tqdm
from utils import predict_gpt, predict_llama, model_evaluation

def process_mmlu_redux_questions(dataset, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    correct_count = 0
    total_count = 0

    for example in tqdm(dataset, desc="Processing questions"):
        if example["error_type"] != "ok":
            continue
        question = example['question']
        options = example['choices']
        correct_answer = example['answer']

        print(f"Processing question: {question}")

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        formatted_options = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])

        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device)

        print(f"Model result: {model_result}")

        final_answer_match = re.search(r"Final Answer: ([ABCD])", model_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
            final_answer_numeric = ord(final_answer_letter) - ord('A')
        else:
            final_answer_numeric = -1  

        is_correct = (final_answer_numeric == correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question,
            "options": options,
            "model_result": model_result,
            "final_answer": final_answer_numeric,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        })

        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for question {total_count}")

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy

def process_mmlu_redux_questions_swap_complex(dataset, output_file_path, formulation_prompt_path, model_type, model, model_evaluation, tokenizer=None, device=None):
    results = []
    correct_count = 0
    total_count = 0

    additional_options = [
        "Blank, ignore this option",
        "Real Madrid is the Best Team",
        "Karma is my Boyfriend",
        "It was enchanted to meet you",
        "May the force be with you"
    ]

    for example in tqdm(dataset, desc="Processing questions"):
        if example["error_type"] != "ok":
            continue
        question = example['question']
        options = example['choices']
        correct_answer = example['answer']

        print(f"Processing question: {question}")

        additional_option = random.choice(additional_options)
        options.append(additional_option)

        option_contents = options.copy()

        random.shuffle(option_contents)

        shuffled_options = [f"{chr(65+i)}. {content}" for i, content in enumerate(option_contents)]

        correct_content = example['choices'][correct_answer]
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
            "original_options": example['choices'],
            "shuffled_options_with_additional": shuffled_options,
            "model_result": model_result,
            "final_answer": final_answer_letter,
            "original_correct_answer": chr(65 + correct_answer),
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