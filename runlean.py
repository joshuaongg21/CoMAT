import subprocess
import os
import openai
import json
import re
import json_repair

INSTRUCTION = (
    "# Task: \n"
    "You are a logician with background in mathematics that translates natural language reasoning text to Lean4 code so that these natural language reasoning problems can be solved. \n"
    "During the translation, please keep close attention to the predicates and entities and MAKE SURE IT'S TRANSLATED CORRECTLY TO LEAN4. \n"
    "After you have translated the text to Lean4, please write the Lean4 code in a file and run it to verify the answer. \n"
    "Provide your assessment in JSON format with keys 'LEAN Form', 'LEAN File', 'GPT Answer', and 'LEAN Answer'.\n"
    "# Example Answer:\n"
    '{\n'
    '  "LEAN Form": "\n'
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
    '\n'
    '  "LEAN File": "path/to/lean/file/verification.lean"'
    '\n'
    '  "GPT Answer": "Alice\'s brother has 6 sisters."'
    '\n'
    '  "LEAN Answer": "2"'
    '}\n'
    "Please ensure that the JSON is well-formed and does not include any additional text or explanations outside the JSON object."
)

def generate_answer(prompt, model):
    openai.api_key = ''
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0.0
    )
    return response.choices[0].message['content'].strip()

def run_lean(file_path):
    """
    Run the given LEAN file and return its output.
    
    Parameters:
    file_path (str): Path to the LEAN file to be executed.
    
    Returns:
    str: Output from the LEAN execution.
    """
    # Ensure the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Run the LEAN file
    result = subprocess.run(['lean', file_path], capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(f"LEAN execution failed with error:\n{result.stderr}")
    
    return result.stdout.strip()

if __name__ == "__main__":
    question = "Alice has 4 brothers and she also has 1 sister. How many sisters does Alice's brother have?"
    answer_prompt = f"Question: {question}\nAnswer:"

    generated_answer = generate_answer(prompt=answer_prompt, model="gpt-4")
    print(f"Generated Answer:\n{generated_answer}")

    try:
        # Repair the JSON string using json-repair
        repaired_answer = json_repair.repair_json(generated_answer)
        json_answer = json.loads(repaired_answer)
        
        lean_form = json_answer['LEAN Form']
        gpt_answer = json_answer['GPT Answer']
        lean_answer = json_answer['LEAN Answer']

        # Create the "lean" folder if it doesn't exist
        os.makedirs("lean", exist_ok=True)

        # Write the Lean form to a file in the "lean" folder
        lean_file_name = "verification.lean"
        lean_file_path = os.path.join("lean", lean_file_name)
        with open(lean_file_path, 'w') as file:
            file.write(lean_form)

        # Run the Lean file and capture the output
        try:
            output = run_lean(lean_file_path)
            print("\nLean Output:\n", output)
        except Exception as e:
            print("Error:", e)

        # Create the "json" folder if it doesn't exist
        os.makedirs("json", exist_ok=True)

        # Save the JSON file in the "json" folder
        json_file_name = "answer.json"
        json_file_path = os.path.join("json", json_file_name)
        with open(json_file_path, 'w') as file:
            json.dump(json_answer, file, indent=2)
        print(f"\nJSON file saved as: {json_file_path}")

        print("\nGPT Answer:", gpt_answer)
        print("Lean Answer:", lean_answer)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in the generated answer. Details: {str(e)}")
        print("Raw Generated Answer:\n", generated_answer)