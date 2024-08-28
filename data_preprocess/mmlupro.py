
import random
import re
import json
from tqdm import tqdm
from utils import predict_gpt, predict_llama, model_evaluation

def process_mmlu_pro_questions(dataset, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
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
    
        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device)

        print(f"Model result: {model_result}")

        final_answer_match = re.search(r"Final Answer: ([A-Z])", model_result)
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
            "model_result": model_result,
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


def process_mmlu_pro_questions_shuffled(dataset, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
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

        print(f"Processing question: {question}")

        # Randomly swap options
        shuffled_options = options.copy()
        random.shuffle(shuffled_options)

        # Create a mapping of new positions to old positions
        option_mapping = {new: old for new, old in enumerate(shuffled_options)}

        # Update the correct answer based on the new positions
        new_correct_answer = shuffled_options.index(options[correct_answer_index])

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        # Format shuffled options as A, B, C, D
        formatted_options = "\n".join([f"{chr(65+j)}. {option}" for j, option in enumerate(options)])

        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device)

        print(f"Model result: {model_result}")

        # Parse the final answer
        final_answer_match = re.search(r"Final Answer: ([A-Z])", model_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
            final_answer_numeric = ord(final_answer_letter) - ord('A')
        else:
            final_answer_numeric = -1  # Invalid answer

        is_correct = (final_answer_numeric == new_correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question,
            "original_options": options,
            "shuffled_options": shuffled_options,
            "model_result": model_result,
            "final_answer": final_answer_numeric,
            "original_correct_answer": correct_answer,
            "original_correct_answer_index": correct_answer_index,
            "new_correct_answer": new_correct_answer,
            "is_correct": is_correct
        })

        # Save results after each successful question
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for question {total_count}")

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy

def process_mmlu_pro_questions_swap_complex(dataset, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
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

    data_dict = dataset.to_dict()
    num_questions = len(data_dict['question'])

    for i in tqdm(range(num_questions), desc="Processing questions"):
        question = data_dict['question'][i]
        options = data_dict['options'][i]
        correct_answer = data_dict['answer'][i]
        correct_answer_index = data_dict['answer_index'][i]

        print(f"Processing question: {question}")  # Debug print

        # Add a random additional option
        additional_option = random.choice(additional_options)
        options.append(additional_option)

        # Safely extract option contents
        option_contents = options.copy()

        # Randomly shuffle the contents of the options
        random.shuffle(option_contents)

        # Create new options with shuffled content
        shuffled_options = [f"{chr(65+i)}. {content}" for i, content in enumerate(option_contents)]

        # Find the new position of the correct answer
        correct_content = data_dict['options'][i][correct_answer_index]
        new_correct_answer = chr(65 + option_contents.index(correct_content))

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        # Format shuffled options
        formatted_options = "\n".join(shuffled_options)

        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device)

        print(f"Model result: {model_result}")

        # Parse the final answer
        final_answer_match = re.search(r"Final Answer: ([A-" + chr(65+len(shuffled_options)-1) + "])", model_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
        else:
            final_answer_letter = "Invalid"  # Invalid answer

        is_correct = (final_answer_letter == new_correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question,
            "original_options": data_dict['options'][i],
            "shuffled_options_with_additional": shuffled_options,
            "model_result": model_result,
            "final_answer": final_answer_letter,
            "original_correct_answer": correct_answer,
            "original_correct_answer_index": correct_answer_index,
            "new_correct_answer": new_correct_answer,
            "is_correct": is_correct,
            "additional_option": additional_option
        })

        # Save results after each successful question
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for question {total_count}")  # Debug print

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy