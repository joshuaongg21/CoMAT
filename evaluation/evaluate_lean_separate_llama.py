import subprocess
import os
import openai
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

INSTRUCTION_LEAN = (
    "# Task: \n"
    "You are a logician with background in mathematics that translates natural language reasoning text to Lean4 code so that these natural language reasoning problems can be solved. \n"
    "During the translation, please keep close attention to the predicates and entities and MAKE SURE IT'S TRANSLATED CORRECTLY TO LEAN4. \n"
    "Please provide only the Lean4 code in your response, without any additional explanations or JSON formatting."
    "# Example Answer:\n"
    '-- Define a type for Person\n'
    'inductive Person where\n'
    '| alice : Person\n'
    '| brother : Person\n\n'
    '-- Define a function that returns the number of sisters a person has\n'
    'def has_sisters : Person → Nat\n'
    '| Person.alice => 2\n'
    '| Person.brother => 2\n\n'
    '-- Define a function that returns the number of brothers a person has\n'
    'def has_brothers : Person → Nat\n'
    '| Person.alice => 5\n'
    '| Person.brother => 5\n\n'
    '-- Axioms stating the number of sisters and brothers for Alice\n'
    'axiom A1 : has_sisters Person.alice = 2\n'
    'axiom A2 : has_brothers Person.alice = 5\n\n'
    '-- Verify the number of sisters Alice\'s brother has\n'
    '-- example : has_sisters Person.brother = 2 := A1\n'
    '#eval has_sisters Person.brother"'
)

INSTRUCTION_SOLVE = (
    "# Task: \n"
    "You are a mathematician with background in mathematics that solves natural language reasoning problems. \n"
    "You are specialised in solving reasoning questions, while reviewing it step by step\n"
    "Please provide a clear and concise answer to the given question."
)

def predict_llama(model, tokenizer, system_prompt, prompt, max_new_tokens, device):
    input_text = f"{system_prompt}\n\n{prompt}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    pad_token_id = tokenizer.pad_token_id
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=False,
        temperature=0.0
    )
    prediction = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    return prediction

def generate_lean(prompt):
    lean_code = predict_llama(model, tokenizer, INSTRUCTION_LEAN, prompt, 1000, device)
    return lean_code.replace("```lean", "").replace("```", "").strip()

def generate_answer(prompt):
    return predict_llama(model, tokenizer, INSTRUCTION_SOLVE, prompt, 1000, device)

def run_lean(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    result = subprocess.run(['lean', file_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"LEAN execution failed with error:\n{result.stderr}")
    
    return result.stdout.strip()

if __name__ == "__main__":
    question = "Alice has 4 brothers and she also has 1 sister. How many sisters does Alice's brother have?"
    
    lean_prompt = f"Question: {question}\nLean Code:"
    lean_code = generate_lean(prompt=lean_prompt)
    print(f"Generated Lean Code:\n{lean_code}")
    
    os.makedirs("output_lean", exist_ok=True)
    
    lean_file_name = "verification.lean"
    lean_file_path = os.path.join("output_lean", lean_file_name)
    with open(lean_file_path, 'w') as file:
        file.write(lean_code)
    
    try:
        lean_output = run_lean(lean_file_path)
        print("\nLean Output:\n", lean_output)
    except Exception as e:
        print("Error:", e)
        lean_output = f"Lean execution failed: {str(e)}"
    
    solve_prompt = f"Question: {question}\nAnswer:"
    llama_answer = generate_answer(prompt=solve_prompt)
    print(f"\nGenerated Answer:\n{llama_answer}")
    
    json_answer = {
        "LEAN Form": lean_code,
        "CodeLlama Answer": llama_answer,
        "LEAN Answer": lean_output
    }
    
    os.makedirs("output_json", exist_ok=True)
    
    json_file_name = "answer.json"
    json_file_path = os.path.join("output_json", json_file_name)
    with open(json_file_path, 'w') as file:
        json.dump(json_answer, file, indent=2)
    print(f"\nJSON file saved as: {json_file_path}")