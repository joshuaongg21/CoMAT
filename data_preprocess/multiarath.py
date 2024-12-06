# data_preprocess/multiarath.py

import json
import re
from tqdm import tqdm
from utils import model_evaluation, evaluate_gpt4o_mini

def load_multiarath_questions(dataset):
    """
    Load SVAMP questions from a dataset object.
    The dataset is expected to be an iterable of items, each having 'question' and 'answer'.
    Adjust as needed if the actual structure is different.
    """
    questions = []
    for item in dataset:
        question = item['question'].strip()
        answer = str(item['final_ans']).strip() 
        questions.append({
            'question': question,
            'answer': answer
        })
    return questions

def process_multiarath_questions(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    total_correct = 0
    total_questions = 0

    with open(formulation_prompt_path, 'r', encoding='utf-8') as f:
        system_content = f.read()

    for example in tqdm(questions, desc="Processing MultiArath questions"):
        question = example['question']
        correct_answer = example['answer']

        print(f"Processing question: {question}")

        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, None, device)
        print(f"Model result: {model_result}")

        last_three_sentences = ' '.join(model_result.split('.')[-3:]).strip()
        print(f"Last three sentences extracted: {last_three_sentences}")

        evaluation_result = evaluate_gpt4o_mini(question, last_three_sentences, correct_answer)
        is_correct = (evaluation_result == '1') 

        if is_correct:
            total_correct += 1
        total_questions += 1

        result = {
            "question": question,
            "model_result": model_result,
            "final_answer": last_three_sentences,
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
