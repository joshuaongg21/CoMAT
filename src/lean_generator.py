import openai
from src.config import INSTRUCTION_LEAN, OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def generate_lean(prompt, model):
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
    return lean_code.replace("```lean", "").replace("```", "").strip()

def refine_lean_code(lean_code, inconsistencies, question):
    refinement_prompt = (
        f"Question: {question}\n\n"
        f"The following Lean code has issues:\n\n{lean_code}\n\n"
        f"Identified inconsistencies:\n"
        f"{chr(10).join(inconsistencies)}\n\n"
        f"Please refine the code to resolve these issues. "
        f"Ensure all identifiers are properly defined, the code is logically consistent, "
        f"and it correctly answers the question."
    )
    return generate_lean(refinement_prompt, model="gpt-4o")