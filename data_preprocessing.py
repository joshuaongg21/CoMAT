import re
from datasets import load_dataset
import json
from tqdm import tqdm
from utils import predict_gpt, predict_llama, evaluate_gpt4o_mini
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

            if model_type == "gpt":
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"Question: {question}"}
                ]
                model_result = predict_gpt(model, messages)
            else:  # llama or llama3.1_8b
                prompt = f"{system_content}\n\nQuestion: {question}"
                model_result = predict_llama(model, tokenizer, prompt, max_new_tokens=1024)

            print(f"Model result: {model_result[:100]}...")  # Print first 100 chars of result

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

        print(f"\nProcessing question: {question[:100]}...")  # Print first 100 chars of question

        if model_type == "gpt":
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Question: {question}"}
            ]
            model_result = predict_gpt(model, messages)
        else:  # llama or llama3.1_8b
            prompt = f"{system_content}\n\nQuestion: {question}"
            model_result = predict_llama(model, tokenizer, prompt, max_new_tokens=1024)

        print(f"Model result: {model_result[:100]}...")  # Print first 100 chars of result

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

def load_mgsm_questions(dataset):
    questions = {}
    configs = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']
    for config in configs:
        try:
            # Load the dataset for each configuration
            config_dataset = load_dataset("juletxara/mgsm", config, split="test")
            questions[config] = []
            for item in config_dataset:
                questions[config].append({
                    'question': item['question'],
                    'answer': str(item['answer_number'])  # Convert to string for consistency
                })
            print(f"Loaded {len(questions[config])} questions for {config} configuration")
        except Exception as e:
            print(f"Error loading {config} configuration: {str(e)}")
    return questions

def process_mgsm_questions(questions, output_file_path, formulation_prompt_path, openai):
    results = {}
    total_correct = 0
    total_questions = 0

    with open(formulation_prompt_path, 'r') as f:
        system_content = f.read()

    for config, config_questions in questions.items():
        results[config] = []
        correct_count = 0
        
        for example in tqdm(config_questions, desc=f"Processing MGSM questions ({config})"):
            question = example['question']
            correct_answer = example['answer']
            
            print(f"Processing question ({config}): {question}")  # Debug print

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Question: {question}"}
            ]
            
            gpt_result = predict_gpt(openai, messages)
            print(f"GPT result: {gpt_result}")  # Debug print

            # Extract the numeric answer from the model's response
            final_answer_match = re.search(r"Final Answer: (\d+)", gpt_result)
            if final_answer_match:
                final_answer = final_answer_match.group(1)
            else:
                final_answer = "Invalid"  # Invalid answer

            is_correct = (final_answer == correct_answer)
            if is_correct:
                correct_count += 1
            
            result = {
                "question": question,
                "gpt_result": gpt_result,
                "final_answer": final_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct
            }
            results[config].append(result)

            # Save results after each question
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved results for question {len(results[config])} in {config}")  # Debug print

        config_accuracy = correct_count / len(config_questions) if len(config_questions) > 0 else 0
        print(f"Accuracy for {config}: {config_accuracy:.2%}")
        
        total_correct += correct_count
        total_questions += len(config_questions)

    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    return results, overall_accuracy


# def load_gsm8k_questions(dataset):
#     questions = []
#     for item in dataset:
#         question = item['question']
#         # Extract the final answer after the ####
#         answer_match = re.search(r'####\s*(\d+)', item['answer'])
#         if answer_match:
#             answer = answer_match.group(1)
#             questions.append({
#                 'question': question,
#                 'answer': answer
#             })
#     return questions

# def process_gsm8k_questions(questions, output_file_path, formulation_prompt_path, openai):
#     results = []
#     total_correct = 0
#     total_questions = 0

#     with open(formulation_prompt_path, 'r') as f:
#         system_content = f.read()

#     for example in tqdm(questions, desc="Processing GSM8K questions"):
#         question = example['question']
#         correct_answer = example['answer']
        
#         print(f"Processing question: {question}")  # Debug print

#         messages = [
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": f"Question: {question}"}
#         ]
        
#         gpt_result = predict_gpt(openai, messages)
#         print(f"GPT result: {gpt_result}")  # Debug print

#         # Extract the numeric answer from the model's response
#         final_answer_match = re.search(r"Final Answer: (\d+)", gpt_result)
#         if final_answer_match:
#             final_answer = final_answer_match.group(1)
#         else:
#             final_answer = "Invalid"  # Invalid answer

#         is_correct = (final_answer == correct_answer)
#         if is_correct:
#             total_correct += 1
#         total_questions += 1
        
#         result = {
#             "question": question,
#             "gpt_result": gpt_result,
#             "final_answer": final_answer,
#             "correct_answer": correct_answer,
#             "is_correct": is_correct
#         }
#         results.append(result)

#         # Save results after each question
#         with open(output_file_path, 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
#         print(f"Saved results for question {len(results)}")  # Debug print

#     accuracy = total_correct / total_questions if total_questions > 0 else 0
#     print(f"Overall Accuracy: {accuracy:.2%}")
#     return results, accuracy