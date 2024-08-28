import re
from datasets import load_dataset
import json
from tqdm import tqdm
from utils import predict_gpt, predict_llama, evaluate_gpt4o_mini, model_evaluation
import random

def process_gpqa_questions(dataset, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    correct_count = 0
    total_count = 0

    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset keys: {dataset.keys()}")

    # Assuming 'train' is the correct split, adjust if necessary
    train_dataset = dataset['train']
    print(f"Train dataset type: {type(train_dataset)}")
    print(f"Train dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        print(f"First item keys: {train_dataset[0].keys()}")
        print(f"First item: {json.dumps(train_dataset[0], indent=2)}")

    with open(formulation_prompt_path, 'r') as f:
        system_content = f.read()

    for example in tqdm(train_dataset, desc="Processing questions"):
        try:
            question = example['Pre-Revision Question']
            correct_answer = example['Pre-Revision Correct Answer']

            print(f"\nProcessing question: {question[:100]}...")  # Print first 100 chars of question

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

        except KeyError as e:
            print(f"KeyError: {e}. Skipping this example.")
            print(f"Example content: {json.dumps(example, indent=2)}")
        except Exception as e:
            print(f"Unexpected error: {e}. Skipping this example.")
            print(f"Example content: {json.dumps(example, indent=2)}")

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy