import subprocess
import os
import openai
import json
from dotenv import load_dotenv
import textgrad as tg


load_dotenv()

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

anthropic_api_key = os.getenv("CLAUDE_API_KEY")
if anthropic_api_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
else:
    raise ValueError("CLAUDE_API_KEY is not set in the .env file.")

def generate_lean(prompt, model):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": INSTRUCTION_LEAN},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0.0
    )
    lean_code = response.choices[0].message['content'].strip()
    
    # Parsing - Remove ```lean and ``` from the generated Lean code
    lean_code = lean_code.replace("```lean", "").replace("```", "").strip()
    
    return lean_code

def generate_answer(prompt, model):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": INSTRUCTION_SOLVE},
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

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    result = subprocess.run(['lean', file_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"LEAN execution failed with error:\n{result.stderr}")
    
    return result.stdout.strip()

def apply_textgrad(initial_solution, system_prompt, model="gpt-4o", feedback_model="claude-3-5-sonnet-20240620"):
    tg.set_backward_engine(tg.get_engine(feedback_model))
    
    solution = tg.Variable(initial_solution,
                           requires_grad=True,
                           role_description="Lean code solution")
    
    loss_system_prompt = tg.Variable(system_prompt,
                                     requires_grad=False,
                                     role_description="system prompt")
    
    loss_fn = tg.TextLoss(loss_system_prompt)
    optimizer = tg.TGD([solution])
    loss = loss_fn(solution)
    loss.backward()
    optimizer.step()
    
    return solution.value

if __name__ == "__main__":
    question = "Alice has 4 brothers and she also has 1 sister. How many sisters does Alice's brother have?"
    
    lean_prompt = f"Question: {question}\nLean Code:"
    initial_lean_code = generate_lean(prompt=lean_prompt, model="gpt-4o")
    print(f"Initial Generated Lean Code:\n{initial_lean_code}")
    
    textgrad_system_prompt = """You will evaluate the Lean code solution to the given question. 
    Identify any errors or improvements that can be made to the code. Be concise and focus on correctness and clarity."""
    
    improved_lean_code = apply_textgrad(initial_lean_code, textgrad_system_prompt)
    print(f"\nImproved Lean Code:\n{improved_lean_code}")
    
    os.makedirs("output_lean", exist_ok=True)
    
    lean_file_name = "verification.lean"
    lean_file_path = os.path.join("output_lean", lean_file_name)
    with open(lean_file_path, 'w') as file:
        file.write(improved_lean_code)
    
    try:
        lean_output = run_lean(lean_file_path)
        print("\nLean Output:\n", lean_output)
    except Exception as e:
        print("Error:", e)
    
    solve_prompt = f"Question: {question}\nAnswer:"
    gpt_answer = generate_answer(prompt=solve_prompt, model="gpt-4o")
    print(f"\nGenerated Answer:\n{gpt_answer}")
    
    json_answer = {
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