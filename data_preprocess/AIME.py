# data_preprocess/AIME.py

import csv
import re
import json
from tqdm import tqdm
from utils import model_evaluation, evaluate_gpt4o_mini

def load_aime_questions(csv_file_path):
    questions = []
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = row['Question']
            answer = row['Answer'].strip()
            questions.append({
                'question': question,
                'answer': answer
            })
    return questions

def process_aime_questions(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    total_correct = 0
    total_questions = 0

    # Load the prompt system content
    with open(formulation_prompt_path, 'r', encoding='utf-8') as f:
        system_content = f.read()

    for example in tqdm(questions, desc="Processing AIME questions"):
        question = example['question']
        correct_answer = example['answer']

        print(f"Processing question: {question}")

        # Get the model result for the question
        model_result = model_evaluation(model_type, model, tokenizer, system_content, question, None, device)
        print(f"Model result: {model_result}")

        # Extract the last 3 sentences from the model result
        last_three_sentences = ' '.join(model_result.split('.')[-3:]).strip()
        print(f"Last three sentences extracted: {last_three_sentences}")

        # Use GPT-4o-mini for evaluation
        evaluation_result = evaluate_gpt4o_mini(question, last_three_sentences, correct_answer)
        is_correct = (evaluation_result == '1')  # '1' means correct, '0' means incorrect

        if is_correct:
            total_correct += 1
        total_questions += 1

        # Store the results for this question
        result = {
            "question": question,
            "model_result": model_result,
            "final_answer": last_three_sentences,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        }
        results.append(result)

        # Save results after each question to the output file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for question {len(results)}")

    # Calculate overall accuracy
    accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"Overall Accuracy: {accuracy:.2%}")
    return results, accuracy


# def process_aime_questions(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
#     results = []
#     total_correct = 0
#     total_questions = 0

#     with open(formulation_prompt_path, 'r', encoding='utf-8') as f:
#         system_content = f.read()

#     for example in tqdm(questions, desc="Processing AIME questions"):
#         question = example['question']
#         correct_answer = example['answer']

#         print(f"Processing question: {question}")

#         model_result = model_evaluation(model_type, model, tokenizer, system_content, question, None, device)

#         print(f"Model result: {model_result}")

#         # Extract the final answer from the model's response
#         final_answer_match = re.search(r"Final Answer:\s*(.+)", model_result)
#         if final_answer_match:
#             final_answer = final_answer_match.group(1).strip()
#         else:
#             final_answer = "Invalid"  # Handle invalid or unexpected responses

#         # Use exact match for evaluation
#         is_correct = (final_answer == correct_answer)
#         if is_correct:
#             total_correct += 1
#         total_questions += 1

#         result = {
#             "question": question,
#             "model_result": model_result,
#             "final_answer": final_answer,
#             "correct_answer": correct_answer,
#             "is_correct": is_correct
#         }
#         results.append(result)

#         # Save results after each question
#         with open(output_file_path, 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
#         print(f"Saved results for question {len(results)}")

#     accuracy = total_correct / total_questions if total_questions > 0 else 0
#     print(f"Overall Accuracy: {accuracy:.2%}")
#     return results, accuracy
