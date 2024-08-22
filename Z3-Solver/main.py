# import json
# from tqdm import tqdm
# from datasets import load_dataset
# from utils import z3_evaluate_mmlu, gpt4o_mini_decoder
# from config import output_file_path

# def process_mmlu_questions(dataset, limit=20):
#     results = []
#     correct_count = 0
#     total_count = 0

#     for example in tqdm(dataset.select(range(limit)), total=limit, desc="Processing questions"):
#         question = example['question']
#         options = example['choices']
#         correct_answer = example['answer']
#         print(f"Processing question: {question}")  # Debug print

#         symbolic_form, z3_code, z3_result = z3_evaluate_mmlu(question)
#         print(f"Z3 result: {z3_result}")  # Debug print
#         gpt4o_mini_answer = gpt4o_mini_decoder(question, options, z3_result)
#         print(f"GPT-4 mini answer: {gpt4o_mini_answer}")  # Debug print

#         is_correct = (gpt4o_mini_answer == correct_answer) and (gpt4o_mini_answer != "z3-code is wrong, try again")
#         if is_correct:
#             correct_count += 1
#         total_count += 1

#         results.append({
#             "question": question,
#             "options": options,
#             "symbolic_form": symbolic_form,
#             "z3_code": z3_code,
#             "z3_result": z3_result,
#             "gpt4o_mini_answer": gpt4o_mini_answer,
#             "correct_answer": correct_answer,
#             "is_correct": is_correct
#         })

#         # Save results after each successful question
#         with open(output_file_path, 'w') as f:
#             json.dump(results, f, indent=2)
#         print(f"Saved results for question {total_count}")  # Debug print

#     accuracy = correct_count / total_count if total_count > 0 else 0
#     print(f"Accuracy: {accuracy:.2%}")
#     return results, accuracy

# # def process_mmlu_questions(dataset, limit=20):
# #     results = []
# #     correct_count = 0
# #     total_count = 0

# #     for example in tqdm(dataset.select(range(limit)), total=limit, desc="Processing questions"):
# #         question = example['question']
# #         options = example['choices']
# #         correct_answer = example['answer']
# #         print(f"Processing question: {question}")  # Debug print
# #         try:
# #             symbolic_form, z3_code, z3_result = z3_evaluate_mmlu(question)
# #             print(f"Z3 result: {z3_result}")  # Debug print
# #             gpt4o_mini_answer = gpt4o_mini_decoder(question, options, z3_result)
# #             print(f"GPT-4 mini answer: {gpt4o_mini_answer}")  # Debug print

# #             is_correct = (gpt4o_mini_answer == correct_answer) and (gpt4o_mini_answer != "z3-code is wrong, try again")
# #             if is_correct:
# #                 correct_count += 1
# #             total_count += 1

# #             results.append({
# #                 "question": question,
# #                 "options": options,
# #                 "symbolic_form": symbolic_form,
# #                 "z3_code": z3_code,
# #                 "z3_result": z3_result,
# #                 "gpt4o_mini_answer": gpt4o_mini_answer,
# #                 "correct_answer": correct_answer,
# #                 "is_correct": is_correct
# #             })

# #             # Save results after each successful question
# #             with open(output_file_path, 'w') as f:
# #                 json.dump(results, f, indent=2)
# #             print(f"Saved results for question {total_count}")  # Debug print
# #         except Exception as e:
# #             print(f"\nError processing question: {question}")
# #             print(f"Error message: {str(e)}")
# #             continue

# #     accuracy = correct_count / total_count if total_count > 0 else 0
# #     print(f"Accuracy: {accuracy:.2%}")
# #     return results, accuracy

# def main():
#     # Create the output file
#     with open(output_file_path, 'w') as f:
#         json.dump([], f)
#     print(f"Created output file: {output_file_path}")

#     # Load dataset (you need to specify how to load your dataset here)
#     dataset = load_dataset("cais/mmlu", "college_mathematics", split="test")

#     # Process MMLU questions (limited to 20)
#     results, accuracy = process_mmlu_questions(dataset, limit=1)
#     print(results)
#     print(f"Final results saved to {output_file_path}")
#     print(f"Final Accuracy: {accuracy:.2%}")

# if __name__ == "__main__":
#     main()



# import json
# from tqdm import tqdm
# from datasets import load_dataset
# from utils import predict_claude, gpt4o_mini_decoder, predict_gpt
# from config import output_file_path, anthropic_client, formulation_prompt_path

# def process_mmlu_questions(dataset, limit=20):
#     results = []
#     correct_count = 0
#     total_count = 0

#     for example in tqdm(dataset.select(range(limit)), total=limit, desc="Processing questions"):
#         question = example['question']
#         options = example['choices']
#         correct_answer = example['answer']
#         print(f"Processing question: {question}")  # Debug print

#         with open(formulation_prompt_path, 'r') as f:
#             system_content = f.read()

#         messages = [
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": f"Question: {question}"}
#         ]
        
#         claude_result = predict_claude(anthropic_client, messages)
#         print(f"Claude result: {claude_result}")  # Debug print
#         gpt4o_mini_answer = gpt4o_mini_decoder(question, options, claude_result)
#         print(f"GPT-4 mini answer: {gpt4o_mini_answer}")  # Debug print

#         is_correct = (gpt4o_mini_answer == correct_answer) and (gpt4o_mini_answer != "z3-code is wrong, try again")
#         if is_correct:
#             correct_count += 1
#         total_count += 1

#         results.append({
#             "question": question,
#             "options": options,
#             "claude_result": claude_result,
#             "gpt4o_mini_answer": gpt4o_mini_answer,
#             "correct_answer": correct_answer,
#             "is_correct": is_correct
#         })

#         # Save results after each successful question
#         with open(output_file_path, 'w') as f:
#             json.dump(results, f, indent=2)
#         print(f"Saved results for question {total_count}")  # Debug print

#     accuracy = correct_count / total_count if total_count > 0 else 0
#     print(f"Accuracy: {accuracy:.2%}")
#     return results, accuracy

# def main():
#     # Create the output file
#     with open(output_file_path, 'w') as f:
#         json.dump([], f)
#     print(f"Created output file: {output_file_path}")

#     # Load dataset
#     dataset = load_dataset("cais/mmlu", "college_mathematics", split="test")

#     # Process MMLU questions (limited to 1 for testing)
#     results, accuracy = process_mmlu_questions(dataset, limit=20)
#     print(results)
#     print(f"Final results saved to {output_file_path}")
#     print(f"Final Accuracy: {accuracy:.2%}")

# if __name__ == "__main__":
#     main()

import json
from tqdm import tqdm
from datasets import load_dataset
from utils import predict_gpt, gpt4o_mini_decoder
from config import output_file_path, openai, formulation_prompt_path

import re

# def process_mmlu_questions(dataset, limit=101):
#     results = []
#     correct_count = 0
#     total_count = 0

#     for example in tqdm(dataset.select(range(limit)), total=limit, desc="Processing questions"):
#         question = example['question']
#         options = example['choices']
#         correct_answer = example['answer']
#         print(f"Processing question: {question}")  # Debug print

#         with open(formulation_prompt_path, 'r') as f:
#             system_content = f.read()

#         # Format options as A, B, C, D
#         formatted_options = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])

#         messages = [
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
#         ]
        
#         gpt_result = predict_gpt(openai, messages)
#         print(f"GPT result: {gpt_result}")  # Debug print

#         # Parse the final answer
#         final_answer_match = re.search(r"Final Answer: ([ABCD])", gpt_result)
#         if final_answer_match:
#             final_answer_letter = final_answer_match.group(1)
#             final_answer_numeric = ord(final_answer_letter) - ord('A')
#         else:
#             final_answer_numeric = -1  # Invalid answer

#         is_correct = (final_answer_numeric == correct_answer)
#         if is_correct:
#             correct_count += 1
#         total_count += 1

#         results.append({
#             "question": question,
#             "options": options,
#             "gpt_result": gpt_result,
#             "final_answer": final_answer_numeric,
#             "correct_answer": correct_answer,
#             "is_correct": is_correct
#         })

#         # Save results after each successful question
#         with open(output_file_path, 'w') as f:
#             json.dump(results, f, indent=2)
#         print(f"Saved results for question {total_count}")  # Debug print

#     accuracy = correct_count / total_count if total_count > 0 else 0
#     print(f"Accuracy: {accuracy:.2%}")
#     return results, accuracy

#INCLUDE WITH GPT-MINI
# def process_mmlu_questions(dataset, limit=20):
#     results = []
#     correct_count = 0
#     total_count = 0

#     for example in tqdm(dataset.select(range(limit)), total=limit, desc="Processing questions"):
#         question = example['question']
#         options = example['choices']
#         correct_answer = example['answer']
#         print(f"Processing question: {question}")  # Debug print

#         with open(formulation_prompt_path, 'r') as f:
#             system_content = f.read()

#         # Format options as A, B, C, D
#         formatted_options = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])

#         messages = [
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
#         ]
        
#         gpt_result = predict_gpt(openai, messages)
#         print(f"GPT result: {gpt_result}")  # Debug print
#         gpt4o_mini_answer = gpt4o_mini_decoder(question, options, gpt_result)
#         print(f"GPT-4 mini answer: {gpt4o_mini_answer}")  # Debug print

#         is_correct = (gpt4o_mini_answer == correct_answer) and (gpt4o_mini_answer != "z3-code is wrong, try again")
#         if is_correct:
#             correct_count += 1
#         total_count += 1

#         results.append({
#             "question": question,
#             "options": options,
#             "gpt_result": gpt_result,
#             "gpt4o_mini_answer": gpt4o_mini_answer,
#             "correct_answer": correct_answer,
#             "is_correct": is_correct
#         })

#         # Save results after each successful question
#         with open(output_file_path, 'w') as f:
#             json.dump(results, f, indent=2)
#         print(f"Saved results for question {total_count}")  # Debug print

#     accuracy = correct_count / total_count if total_count > 0 else 0
#     print(f"Accuracy: {accuracy:.2%}")
#     return results, accuracy

# def main():
#     # Create the output file
#     with open(output_file_path, 'w') as f:
#         json.dump([], f)
#     print(f"Created output file: {output_file_path}")

#     # Load dataset
#     dataset = load_dataset("cais/mmlu", "college_mathematics", split="test")

#     # Process MMLU questions (limited to 20)
#     results, accuracy = process_mmlu_questions(dataset, limit=101)
#     print(results)
#     print(f"Final results saved to {output_file_path}")
#     print(f"Final Accuracy: {accuracy:.2%}")

# if __name__ == "__main__":
#     main()



def process_mmlu_questions(dataset):
    results = []
    correct_count = 0
    total_count = 0

    for example in tqdm(dataset, desc="Processing questions"):
        question = example['question']
        options = example['choices']
        correct_answer = example['answer']
        print(f"Processing question: {question}")  # Debug print

        with open(formulation_prompt_path, 'r') as f:
            system_content = f.read()

        # Format options as A, B, C, D
        formatted_options = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        
        gpt_result = predict_gpt(openai, messages)
        print(f"GPT result: {gpt_result}")  # Debug print

        # Parse the final answer
        final_answer_match = re.search(r"Final Answer: ([ABCD])", gpt_result)
        if final_answer_match:
            final_answer_letter = final_answer_match.group(1)
            final_answer_numeric = ord(final_answer_letter) - ord('A')
        else:
            final_answer_numeric = -1  # Invalid answer

        is_correct = (final_answer_numeric == correct_answer)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            "question": question,
            "options": options,
            "gpt_result": gpt_result,
            "final_answer": final_answer_numeric,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        })

        # Save results after each successful question
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for question {total_count}")  # Debug print

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    return results, accuracy

def main():
    # Create the output file
    with open(output_file_path, 'w') as f:
        json.dump([], f)
    print(f"Created output file: {output_file_path}")

    # Load dataset
    dataset = load_dataset("cais/mmlu", "college_mathematics", split="test")

    # Process MMLU questions (full dataset)
    results, accuracy = process_mmlu_questions(dataset)
    print(results)
    print(f"Final results saved to {output_file_path}")
    print(f"Final Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()