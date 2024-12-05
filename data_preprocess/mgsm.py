import random
import re
import json
from tqdm import tqdm
from utils import predict_gpt, model_evaluation
from datasets import load_dataset

def load_mgsm_questions(dataset):
    questions = {}
    configs = ['bn', 'sw', 'te', 'th', 'zh']
    for config in configs:
        try:
            config_dataset = load_dataset("juletxara/mgsm", config, split="test")
            questions[config] = []
            for item in config_dataset:
                questions[config].append({
                    'question': item['question'],
                    'answer': str(item['answer_number'])
                })
            print(f"Loaded {len(questions[config])} questions for {config} configuration")
        except Exception as e:
            print(f"Error loading {config} configuration: {str(e)}")
    return questions

def process_mgsm_questions(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = {}
    total_correct = 0
    total_questions = 0

    with open(formulation_prompt_path, 'r') as f:
        system_content = f.read()

    for config, config_questions in questions.items():
        results[config] = []
        
        for example in tqdm(config_questions, desc=f"Processing MGSM questions ({config})"):
            question = example['question']
            correct_answer = example['answer']
            
            print(f"Processing question ({config}): {question}")

            model_result = model_evaluation(model_type, model, tokenizer, system_content, question, None, device)

            print(f"Model result: {model_result}")

            final_answer_match = re.search(r"Final Answer: (\d+)", model_result)
            if final_answer_match:
                final_answer = final_answer_match.group(1)
            else:
                final_answer = "Invalid"

            is_correct = (final_answer == correct_answer)
            if is_correct:
                total_correct += 1
            
            result = {
                "question": question,
                "model_result": model_result,
                "final_answer": final_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct
            }
            results[config].append(result)
            total_questions += 1

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for {config}")

    final_accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"Final Accuracy: {final_accuracy:.2%}") #print final accuracy

    return results, final_accuracy