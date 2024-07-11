import os
from dotenv import load_dotenv

load_dotenv()

INSTRUCTION_LEAN = """
# Task: 
You are a logician with background in mathematics that translates natural language reasoning text to Lean4 code so that these natural language reasoning problems can be solved. 
During the translation, please keep close attention to the predicates and entities and MAKE SURE IT'S TRANSLATED CORRECTLY TO LEAN4. 
Please provide only the Lean4 code in your response, without any additional explanations or JSON formatting.
# Example Answer:
-- Define a type for Person
inductive Person where
| alice : Person
| brother : Person

-- Define a function that returns the number of sisters a person has
def has_sisters : Person → Nat
| Person.alice => 2
| Person.brother => 2

-- Define a function that returns the number of brothers a person has
def has_brothers : Person → Nat
| Person.alice => 5
| Person.brother => 5

-- Axioms stating the number of sisters and brothers for Alice
axiom A1 : has_sisters Person.alice = 2
axiom A2 : has_brothers Person.alice = 5

-- Verify the number of sisters Alice's brother has
-- example : has_sisters Person.brother = 2 := A1
#eval has_sisters Person.brother
"""

INSTRUCTION_SOLVE = """
# Task: 
You are a mathematician with background in mathematics that solves natural language reasoning problems. 
You are specialised in solving reasoning questions, while reviewing it step by step
Please provide a clear and concise answer to the given question.
"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("CLAUDE_API_KEY")

if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
else:
    raise ValueError("CLAUDE_API_KEY is not set in the .env file.")