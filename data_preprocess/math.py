import re
from datasets import load_dataset
import json
from tqdm import tqdm
from utils import predict_gpt, predict_llama, evaluate_gpt4o_mini, model_evaluation
import random

def process_competition_math_questions(dataset, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    correct_count = 0
    total_count = 0

    print(f"Dataset loaded. Number of items: {len(dataset)}")

    with open(formulation_prompt_path, 'r') as f:
        system_content = f.read()

    for example in tqdm(dataset, desc="Processing questions"):
        question = example['problem']
        correct_answer = example['solution']

        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, None, device)

        print(f"Model result: {model_result}")

        # Evaluate the result
        evaluation = evaluate_gpt4o_mini(question, model_result, correct_answer)
        is_correct = (evaluation == '1')

        if is_correct:
            correct_count += 1
        total_count += 1

        result = {
            "question": question,
            "model_result": model_result,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        }
        results.append(result)
        print(f"Evaluation result: {'Correct' if is_correct else 'Incorrect'}")

        # Save results after each successful question
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for question {total_count}")

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy