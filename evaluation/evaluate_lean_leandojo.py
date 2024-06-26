import subprocess
import os
import openai
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

INSTRUCTION_LEAN = (
    "# Task: \n"
    "You are a logician with a background in mathematics that translates natural language reasoning text to Lean4 code so that these natural language reasoning problems can be solved. \n"
    "During the translation, please pay close attention to the predicates and entities. \n"
    "Remember to consider all perspectives in family relationships. For example, if A is B's sibling, B is also A's sibling.\n"

    "Review and translate the code step by step: \n"
    "1. Start with the type definitions.\n"
    "- **Types**: Include all the types and entities correctly, make sure that all the entities are included.\n\n"

    "2. Proceed with the function definitions.\n"
    "- **Functions**: Include all the functions and entities correctly. Do make sure that all the entities in the questions are clearly analysed and included. DO NOT MISS OUT ANY ENTITIES OR CHARACTERS inside.\n\n"

    "3. Then state the axioms.\n"
    "- **Axioms**: Ensure the axioms are correctly stated and the numbers are correctly assigned.\n\n"

    "4. Finally, after making sure the functions and axioms are correct, include the evaluation statement.\n"
    "- **Evaluation**: Make sure the evaluation statement reflects the correct number based on the function definitions.\n\n"

    "Please provide only the Lean4 code in your response, without any additional explanations or JSON formatting."
)

FEW_SHOT_EXAMPLES = [
    {
        "question": "Rapunzel has 3 sisters and 2 brothers. How many cousins does Emily's brother have?",
        "answer": "4 (including Emily)",
        "code": '''
-- Define a type for Person
inductive Person where
| rapunzel : Person
| sister1 : Person
| sister2 : Person
| sister3 : Person
| brother1 : Person
| brother2 : Person
deriving DecidableEq

-- Define a function that returns the number of sisters a person has
def has_sisters : Person → Nat
| Person.rapunzel => 3
| Person.sister1 => 3
| Person.sister2 => 3
| Person.sister3 => 3
| Person.brother1 => 4  -- Including Rapunzel
| Person.brother2 => 4  -- Including Rapunzel

-- Define a function that returns the number of brothers a person has
def has_brothers : Person → Nat
| Person.rapunzel => 2
| Person.sister1 => 2
| Person.sister2 => 2
| Person.sister3 => 2
| Person.brother1 => 1
| Person.brother2 => 1

-- Axioms stating the number of sisters and brothers for Rapunzel
axiom A1 : has_sisters Person.rapunzel = 3
axiom A2 : has_brothers Person.rapunzel = 2

-- Verify the number of sisters Rapunzel's brother has (including Rapunzel)
#eval has_sisters Person.brother1
'''
    }
]

INSTRUCTION_SOLVE = (
    "# Task: \n"
    "You are a mathematician with background in mathematics that solves natural language reasoning problems. \n"
    "You are specialized in solving reasoning questions, while reviewing it step by step\n"
    "Please provide a clear and concise answer to the given question."
)

def generate_lean(prompt, model):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    messages = [
        {"role": "system", "content": INSTRUCTION_LEAN},
    ]
    
    # Add FEW_SHOT_EXAMPLES
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["question"]})
        messages.append({"role": "assistant", "content": example["code"]})
    
    # Add the current user prompt
    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=3000,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    lean_code = response.choices[0].message['content'].strip()
    
    # Remove ```lean and ``` from the generated Lean code
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

def generate_tactics(lean_code):
    """
    Use the Transformers-based model to generate tactics for the given Lean code.

    Parameters:
    lean_code (str): Lean code to be processed.

    Returns:
    str: Generated tactics or error message.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small")

        input_text = f"Generate tactics for:\n{lean_code}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        outputs = model.generate(input_ids, max_length=200, num_return_sequences=1, temperature=0.0)
        tactics = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return f"Generated tactics:\n{tactics}"
    except Exception as e:
        return f"Error generating tactics: {str(e)}"

def run_lean(file_path, theorem_statement, tactics=None):
    """
    Run the given LEAN file with the provided theorem statement and tactics (if any) and return its output.
    
    Parameters:
    file_path (str): Path to the LEAN file to be executed.
    theorem_statement (str): Theorem statement to be included in the LEAN file.
    tactics (str, optional): Tactics to be included in the LEAN file.
    
    Returns:
    str: Output from the LEAN execution.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        lean_code = file.read()
    
    if tactics:
         lean_code_with_tactics = f"{lean_code}\n\ntheorem proof_attempt : {theorem_statement} := rfl"
    else:
        lean_code_with_tactics = f"{lean_code}\n\ntheorem proof_attempt : {theorem_statement} := rfl"
    
    with open(file_path, 'w') as file:
        file.write(lean_code_with_tactics)
    
    # Run the LEAN file
    result = subprocess.run(['lean', file_path], capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        print("Lean file content for debugging:")
        with open(file_path, 'r') as file:
            print(file.read())
        print("Lean execution stderr for debugging:")
        print(result.stderr)
        raise RuntimeError(f"LEAN execution failed with error:\n{result.stderr}")
    
    return result.stdout.strip()

def generate_theorem_statement(question):
    try:
        tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small")

        input_text = f"Generate theorem statement for: {question}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        outputs = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.0)
        theorem_statement = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove redundant "theorem proof_attempt :" and clean up the statement
        theorem_statement = theorem_statement.replace("theorem proof_attempt :", "").strip()
        theorem_statement = theorem_statement.split(":=")[0].strip()  # Remove everything after :=
        theorem_statement = ' '.join(theorem_statement.split())  # Remove extra whitespace

        # Construct the final theorem statement
        theorem_statement = f"theorem proof_attempt : {theorem_statement} := by"

        return theorem_statement
    except Exception as e:
        print(f"Error generating theorem statement: {str(e)}")
        return None
    

if __name__ == "__main__":
    question = "Alice has 3 sisters. Her mother has 1 sister who does not have children - she has 7 nephews and nieces and also 2 brothers. Alice's father has a brother who has 5 nephews and nieces in total, and who has also 1 son. How many cousins does Alice's sister have?"
    
    # Generate Lean code
    lean_prompt = f"Question: {question}\nLean Code:"
    lean_code = generate_lean(prompt=lean_prompt, model="gpt-4o")
    print(f"Generated Lean Code:\n{lean_code}")
    
    tactics = None
    try:
        tactics_output = generate_tactics(lean_code)
        print("\nGenerated Tactics:\n", tactics_output)
        
        # Extract actual tactics from the output
        tactics = tactics_output.split("Generated tactics:\n")[1].strip()
    except Exception as e:
        print("Error generating tactics:", e)

    os.makedirs("output_lean", exist_ok=True)
    lean_file_name = "verification.lean"
    lean_file_path = os.path.join("output_lean", lean_file_name)
    with open(lean_file_path, 'w') as file:
        file.write(lean_code)
    
    theorem_statement = generate_theorem_statement(question)
    
    try:
        lean_output = run_lean(lean_file_path, theorem_statement, tactics)
        print("\nLean Output:\n", lean_output)
    except Exception as e:
        lean_output = f"Error: {str(e)}"
        print("Error:", e)
    
    solve_prompt = f"Question: {question}\nAnswer:"
    gpt_answer = generate_answer(prompt=solve_prompt, model="gpt-4o")
    print(f"\nGenerated Answer:\n{gpt_answer}")
    
    json_answer = {
        "LEAN Form": lean_code,
        "GPT Answer": gpt_answer,
        "LEAN Answer": lean_output,
        "Tactics Output": tactics_output
    }
    
    os.makedirs("output_json", exist_ok=True)
    json_file_name = "answer.json"
    json_file_path = os.path.join("output_json", json_file_name)
    with open(json_file_path, 'w') as file:
        json.dump(json_answer, file, indent=2)
    print(f"\nJSON file saved as: {json_file_path}")


#Alternative code
# import subprocess
# import os
# import openai
# import json
# from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load_dotenv()

# INSTRUCTION_LEAN = (
#     "# Task: \n"
#     "You are a logician with background in mathematics that translates natural language reasoning text to Lean4 code so that these natural language reasoning problems can be solved. \n"
#     "During the translation, please keep close attention to the predicates and entities and MAKE SURE IT'S TRANSLATED CORRECTLY TO LEAN4. \n"
#     "Please provide only the Lean4 code in your response, without any additional explanations or JSON formatting."
#     "# Example Answer:\n"
#     '-- Define a type for Person\n'
#     'inductive Person where\n'
#     '| alice : Person\n'
#     '| brother : Person\n\n'
#     '-- Define a function that returns the number of sisters a person has\n'
#     'def has_sisters : Person → Nat\n'
#     '| Person.alice => 2\n'
#     '| Person.brother => 2\n\n'
#     '-- Define a function that returns the number of brothers a person has\n'
#     'def has_brothers : Person → Nat\n'
#     '| Person.alice => 5\n'
#     '| Person.brother => 5\n\n'
#     '-- Axioms stating the number of sisters and brothers for Alice\n'
#     'axiom A1 : has_sisters Person.alice = 2\n'
#     'axiom A2 : has_brothers Person.alice = 5\n\n'
#     '-- Verify the number of sisters Alice\'s brother has\n'
#     '#eval has_sisters Person.brother"'
# )

# INSTRUCTION_SOLVE = (
#     "# Task: \n"
#     "You are a mathematician with background in mathematics that solves natural language reasoning problems. \n"
#     "You are specialized in solving reasoning questions, while reviewing it step by step\n"
#     "Please provide a clear and concise answer to the given question."
# )

# def generate_lean(prompt, model):
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": INSTRUCTION_LEAN},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=3000,
#         temperature=0.0
#     )
#     lean_code = response.choices[0].message['content'].strip()
    
#     # Remove ```lean and ``` from the generated Lean code
#     lean_code = lean_code.replace("```lean", "").replace("```", "").strip()
    
#     return lean_code

# def generate_answer(prompt, model):
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": INSTRUCTION_SOLVE},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=3000,
#         temperature=0.0
#     )
#     return response.choices[0].message['content'].strip()

# def generate_tactics(lean_code):
#     """
#     Use the Transformers-based model to generate tactics for the given Lean code.

#     Parameters:
#     lean_code (str): Lean code to be processed.

#     Returns:
#     str: Generated tactics or error message.
#     """
#     try:
#         tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small")
#         model = AutoModelForSeq2SeqLM.from_pretrained("kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small")

#         input_text = f"Generate tactics for:\n{lean_code}"
#         input_ids = tokenizer(input_text, return_tensors="pt").input_ids

#         outputs = model.generate(input_ids, max_length=200, num_return_sequences=1, temperature=0.7)
#         tactics = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return f"Generated tactics:\n{tactics}"
#     except Exception as e:
#         return f"Error generating tactics: {str(e)}"

# def run_lean(file_path, tactics):
#     """
#     Run the given LEAN file with the provided tactics and return its output.
    
#     Parameters:
#     file_path (str): Path to the LEAN file to be executed.
#     tactics (str): Tactics to be included in the LEAN file.
    
#     Returns:
#     str: Output from the LEAN execution.
#     """
#     if not os.path.isfile(file_path):
#         raise FileNotFoundError(f"The file {file_path} does not exist.")
    
#     with open(file_path, 'r') as file:
#         lean_code = file.read()
    
#     lean_code_with_tactics = f"{lean_code}\n\ntheorem proof_attempt : true :=\nbegin\n{tactics}\nend"
    
#     with open(file_path, 'w') as file:
#         file.write(lean_code_with_tactics)
    
#     # Run the LEAN file
#     result = subprocess.run(['lean', file_path], capture_output=True, text=True)
    
#     # Check for errors
#     if result.returncode != 0:
#         raise RuntimeError(f"LEAN execution failed with error:\n{result.stderr}")
    
#     return result.stdout.strip()

# if __name__ == "__main__":
#     question = "Alice has 4 brothers and she also has 1 sister. How many sisters does Alice's brother have?"
    
#     # Generate Lean code
#     lean_prompt = f"Question: {question}\nLean Code:"
#     lean_code = generate_lean(prompt=lean_prompt, model="gpt-3.5-turbo")
#     print(f"Generated Lean Code:\n{lean_code}")
    
#     try:
#         tactics_output = generate_tactics(lean_code)
#         print("\nGenerated Tactics:\n", tactics_output)
        
#         # Extract actual tactics from the output
#         tactics = tactics_output.split("Generated tactics:\n")[1].strip()
#     except Exception as e:
#         tactics = "sorry"  # Default tactic if generation fails
#         print("Error generating tactics:", e)

#     os.makedirs("output_lean", exist_ok=True)
#     lean_file_name = "verification.lean"
#     lean_file_path = os.path.join("output_lean", lean_file_name)
#     with open(lean_file_path, 'w') as file:
#         file.write(lean_code)
    
#     try:
#         lean_output = run_lean(lean_file_path, tactics)
#         print("\nLean Output:\n", lean_output)
#     except Exception as e:
#         lean_output = f"Error: {str(e)}"
#         print("Error:", e)
    
#     solve_prompt = f"Question: {question}\nAnswer:"
#     gpt_answer = generate_answer(prompt=solve_prompt, model="gpt-3.5-turbo")
#     print(f"\nGenerated Answer:\n{gpt_answer}")
    
#     json_answer = {
#         "LEAN Form": lean_code,
#         "GPT Answer": gpt_answer,
#         "LEAN Answer": lean_output,
#         "Tactics Output": tactics_output
#     }
    
#     os.makedirs("output_json", exist_ok=True)
#     json_file_name = "answer.json"
#     json_file_path = os.path.join("output_json", json_file_name)
#     with open(json_file_path, 'w') as file:
#         json.dump(json_answer, file, indent=2)
#     print(f"\nJSON file saved as: {json_file_path}")
