import random
import re
import json
from tqdm import tqdm
from utils import predict_gpt, model_evaluation

def load_gsm8k_questions(dataset):
    questions = []
    for item in dataset:
        question = item['question']
        answer_match = re.search(r'####\s*(\d+)', item['answer'])
        if answer_match:
            answer = answer_match.group(1)
            questions.append({
                'question': question,
                'answer': answer
            })
    return questions

def process_gsm8k_questions(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    total_correct = 0
    total_questions = 0

    with open(formulation_prompt_path, 'r') as f:
        system_content = f.read()

    for example in tqdm(questions, desc="Processing GSM8K questions"):
        question = example['question']
        correct_answer = example['answer']

        print(f"Processing question: {question}")  

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
        total_questions += 1

        result = {
            "question": question,
            "model_result": model_result,
            "final_answer": final_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        }
        results.append(result)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for question {len(results)}") 

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"Overall Accuracy: {accuracy:.2%}")
    return results, accuracy