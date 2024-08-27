import random
import re
import json
from tqdm import tqdm
from utils import predict_gpt, predict_llama

def load_gsm8k_questions(dataset):
    questions = []
    for item in dataset:
        question = item['question']
        # Extract the final answer after the ####
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
        
        print(f"Processing question: {question}")  # Debug print
        formatted_options = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])

        if model_type == "gpt":
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
            ]
            model_result = predict_gpt(model, messages)
        else:  
            prompt = f"{system_content}\n\nQuestion: {question}\n\nOptions:\n{formatted_options}"
            model_result = predict_llama(model, tokenizer, prompt, max_new_tokens=1024)

        # Extract the numeric answer from the model's response
        final_answer_match = re.search(r"Final Answer: (\d+)", model_result)
        if final_answer_match:
            final_answer = final_answer_match.group(1)
        else:
            final_answer = "Invalid"  # Invalid answer

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

        # Save results after each question
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for question {len(results)}")  # Debug print

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"Overall Accuracy: {accuracy:.2%}")
    return results, accuracy

def process_gsm8k_questions_shuffled(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    total_correct = 0
    total_questions = 0

    with open(formulation_prompt_path, 'r') as f:
        system_content = f.read()

    for example in tqdm(questions, desc="Processing GSM8K questions (shuffled)"):
        question = example['question']
        correct_answer = example['answer']
        
        print(f"Processing question: {question}")

        # Shuffle the words in the question
        words = question.split()
        random.shuffle(words)
        shuffled_question = ' '.join(words)

        if model_type == "gpt":
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
            ]
            model_result = predict_gpt(model, messages)
        else:  
            prompt = f"{system_content}\n\nQuestion: {question}\n\nOptions:\n{formatted_options}"
            model_result = predict_llama(model, tokenizer, prompt, max_new_tokens=1024)

        # Extract the numeric answer from the model's response
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
            "original_question": question,
            "shuffled_question": shuffled_question,
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

def process_gsm8k_questions_swap_complex(questions, output_file_path, formulation_prompt_path, model_type, model, tokenizer=None, device=None):
    results = []
    total_correct = 0
    total_questions = 0

    additional_sentences = [
        "The sky is blue.",
        "Cats are furry animals.",
        "Apples grow on trees.",
        "Water boils at 100 degrees Celsius.",
        "The Earth orbits around the Sun."
    ]

    with open(formulation_prompt_path, 'r') as f:
        system_content = f.read()

    for example in tqdm(questions, desc="Processing GSM8K questions (complex swap)"):
        question = example['question']
        correct_answer = example['answer']
        
        print(f"Processing question: {question}")

        # Add a random additional sentence
        additional_sentence = random.choice(additional_sentences)
        complex_question = question + " " + additional_sentence

        # Shuffle the words in the complex question
        words = complex_question.split()
        random.shuffle(words)
        shuffled_complex_question = ' '.join(words)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {shuffled_complex_question}"}
        ]
        
        model_result = predict_gpt(openai, messages)
        print(f"GPT result: {model_result}")

        # Extract the numeric answer from the model's response
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
            "original_question": question,
            "complex_question": complex_question,
            "shuffled_complex_question": shuffled_complex_question,
            "model_result": model_result,
            "final_answer": final_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "additional_sentence": additional_sentence
        }
        results.append(result)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for question {len(results)}")

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"Overall Accuracy: {accuracy:.2%}")
    return results, accuracy