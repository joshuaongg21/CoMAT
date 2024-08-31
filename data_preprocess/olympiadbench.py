import json
from tqdm import tqdm
from utils import predict_gpt, predict_llama, evaluate_gpt4o_mini, model_evaluation
import os

def process_olympiadbench_questions(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    print(f"Starting to process {len(questions)} questions")
    
    # Load existing results if file exists
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")
    else:
        results = []
        print("No existing results found, starting fresh")
    
    correct_count = 0
    total_count = 0
    
    with open(formulation_prompt_path, 'r') as f:
        system_content = f.read()
    
    for i, question in enumerate(tqdm(questions, desc="Processing questions")):
        try:
            question_text = question['question']
            correct_answer = question['final_answer']
            print(f"\nProcessing question {i+1}/{len(questions)}: {question_text[:100]}...")
            
            model_result = model_evaluation(model_type, model, tokenizer, system_content, question_text, None, device)
            print(f"Model result: {model_result}")
            
            evaluation = evaluate_gpt4o_mini(question_text, model_result, correct_answer)
            is_correct = (evaluation == '1')
            if is_correct:
                correct_count += 1
            total_count += 1
            
            result = {
                "question": question_text,
                "model_result": model_result,
                "correct_answer": correct_answer,
                "is_correct": is_correct
            }
            results.append(result)
            print(f"Evaluation result: {'Correct' if is_correct else 'Incorrect'}")
            
            # Append new result to file
            try:
                with open(output_file_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved results for question {len(results)}")
            except Exception as write_error:
                print(f"Error writing to file: {write_error}")
            
        except Exception as e:
            print(f"Unexpected error processing question {i+1}: {e}. Skipping this example.")
            print(f"Example content: {json.dumps(question, indent=2)}")
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Processed {total_count} questions. Accuracy: {accuracy:.2%}")
    return results, accuracy