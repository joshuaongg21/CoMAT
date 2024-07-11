import sys
import os

# Get the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (which should contain the 'src' folder)
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

import json
from tqdm import tqdm
from src.lean_generator import generate_lean
from src.answer_generator import generate_answer
from src.lean_executor import run_lean
from src.textgrad_improver import apply_textgrad

def main():
    question = "Alice has 4 brothers and she also has 1 sister. How many sisters does Alice's brother have?"
    
    lean_prompt = f"Question: {question}\nLean Code:"
    initial_lean_code = generate_lean(prompt=lean_prompt, model="gpt-4o")
    print(f"Initial Generated Lean Code:\n{initial_lean_code}")
    
    textgrad_system_prompt = """You will evaluate the Lean code solution to the given question. 
    Identify any errors or improvements that can be made to the code. Be concise and focus on correctness and clarity."""
    
    improved_lean_code = apply_textgrad(initial_lean_code, textgrad_system_prompt, question)
    print(f"\nFinal Improved Lean Code:\n{improved_lean_code}")
    
    os.makedirs("output_lean", exist_ok=True)
    
    lean_file_name = "verification.lean"
    lean_file_path = os.path.join("output_lean", lean_file_name)
    with open(lean_file_path, 'w') as file:
        file.write(improved_lean_code)
    
    lean_output = run_lean(lean_file_path)
    print("\nLean Output:\n", lean_output)
    
    solve_prompt = f"Question: {question}\nAnswer:"
    gpt_answer = generate_answer(prompt=solve_prompt, model="gpt-4o")
    print(f"\nGenerated Answer:\n{gpt_answer}")
    
    json_answer = {
        "Question": question,
        "Initial LEAN Form": initial_lean_code,
        "Improved LEAN Form": improved_lean_code,
        "GPT Answer": gpt_answer,
        "LEAN Answer": lean_output
    }
    
    os.makedirs("output_json", exist_ok=True)
    
    json_file_name = "answer.json"
    json_file_path = os.path.join("output_json", json_file_name)
    with open(json_file_path, 'w') as file:
        json.dump(json_answer, file, indent=2)
    print(f"\nJSON file saved as: {json_file_path}")

if __name__ == "__main__":
    main()