import openai
from src.config import INSTRUCTION_SOLVE, OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def generate_answer(prompt, model):
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