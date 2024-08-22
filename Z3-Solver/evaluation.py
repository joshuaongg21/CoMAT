import openai
from z3 import *
import json
import io
import sys
from datasets import load_dataset
from tqdm import tqdm

# Set up your OpenAI API key
openai.api_key = ''

# Create the output file
output_file_path = 'mmlu_college_mathematics_results.json'
with open(output_file_path, 'w') as f:
    json.dump([], f)
print(f"Created output file: {output_file_path}")

def z3_converter(question):
    messages = [
        {"role": "system", "content": "Please convert the question to Z3 code based on the question. DO NOT PROVIDE ANYTHING ELSE (INCLUDING NO EXPLANATION) EXCEPT THE Z3 CODE."},
        {"role": "user", "content": f"Question: {question}"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        temperature=0.0
    )
    
    code = response.choices[0].message['content'].strip()
    code = code.replace('```python', '').replace('```', '').strip()
    return code

def execute_z3_code(z3_code: str):
    global_namespace = {'Solver': Solver, 'Int': Int, 'Real': Real, 'sat': sat, 'print': print}
    
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        exec(z3_code, global_namespace)
        output = new_stdout.getvalue().strip()
        
        lines = output.split('\n')
        if len(lines) >= 2:
            sat_result = lines[-2]
            model_result = lines[-1]
            if sat_result == 'sat':
                return model_result
            else:
                return "unsat"
        else:
            return "Error: Unexpected output format"
    except Exception as e:
        raise Exception(f"Error executing Z3 code: {str(e)}")
    finally:
        sys.stdout = old_stdout

def z3_evaluate_mmlu(question):
    z3_code = z3_converter(question)
    result = execute_z3_code(z3_code)
    return z3_code, result

def process_mmlu_questions(dataset, limit=20):
    results = []
    for example in tqdm(dataset.select(range(limit)), total=limit, desc="Processing questions"):
        question = example['question']
        try:
            z3_code, final_answer = z3_evaluate_mmlu(question)
            results.append({
                "question": question,
                "z3_code": z3_code,
                "final_answer": final_answer
            })
            
            # Save results after each successful question
            with open(output_file_path, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"\nError processing question: {question}")
            print(f"Error message: {str(e)}")
            break
    return results

# Load the dataset
dataset = load_dataset("edinburgh-dawg/mmlu-redux", "college-mathematics", split="test")

# Process MMLU questions (limited to 20)
results = process_mmlu_questions(dataset, limit=20)

print(f"Final results saved to {output_file_path}")
