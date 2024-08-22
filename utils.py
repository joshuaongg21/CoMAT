import io
import sys
from z3 import *
import openai

def predict_claude(anthropic, messages):
    system_message = None
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
        else:
            formatted_messages.append({
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}]
            })
    
    response = anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=700,
        temperature=0,
        system=system_message,
        messages=formatted_messages
    )
    prediction = response.content[0].text
    return prediction

def predict_gpt(openai, messages):
    system_message = None
    formatted_messages = []
    
    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
        else:
            formatted_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
    
    if system_message:
        formatted_messages.insert(0, {"role": "system", "content": system_message})
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",  # You can change this to the desired GPT model
        messages=formatted_messages,
        max_tokens=1000,
        temperature=0
    )
    
    prediction = response.choices[0].message['content'].strip()
    return prediction

def gpt4o_mini_decoder(question, options, z3_result):
    messages = [
        {"role": "system", "content": "You are a decoder that selects the correct MMLU option based on the Z3 code output. If the Z3 output doesn't align with any option, respond with 'z3-code is wrong, try again'. DO NOT PROVIDE YOUR OWN ANSWER OR REASONING"},
        {"role": "user", "content": f"Question: {question}\nOptions: {options}\nZ3 Result: {z3_result}\n\nBased on the Z3 result, which option is correct? If none match, say 'z3-code is wrong, try again'."}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0
    )

    return response.choices[0].message['content'].strip()

def evaluate_gpt4o_mini(question, gpt_result, correct_answer):
    messages = [
        {"role": "system", "content": "You are a decider that decides whether the answer is the same as the correct answer. If the the output doesn't align with the correct answer, respond with '0', whereas if it's correct, then respond with '1'. DO NOT PROVIDE YOUR OWN ANSWER OR REASONING, JUST SELECT '0' OR '1'."},
        {"role": "user", "content": f"Question: {question}\nGPT-4o Result: {gpt_result}\nCorrect Answer: {correct_answer}. Answer with 0 (Wrong) or 1 (Correct)."}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0
    )

    return response.choices[0].message['content'].strip()
   
# def symbolic_representation(question):
#     with open(formulation_prompt_path, 'r') as f:
#         system_content = f.read()

#     messages = [
#         {"role": "system", "content": system_content},
#         {"role": "user", "content": f"Question: {question}"}
#     ]

#     symbolic_form = predict_claude(anthropic_client, messages)
#     return symbolic_form

# def z3_converter(question, symbolic_form):
#     with open(z3_solver_prompt_path, 'r') as f:
#         system_content = f.read()

#     messages = [
#         {"role": "system", "content": system_content},
#         {"role": "user", "content": f"Question: {question}\nSymbolic form: {symbolic_form}"}
#     ]

#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=messages,
#         temperature=0.0
#     )

#     code = response.choices[0].message['content'].strip()
#     code = code.replace('```python', '').replace('```', '').strip()
#     return code

# def execute_z3_code(z3_code: str):
#     global_namespace = {
#         'Solver': Solver,
#         'Int': Int,
#         'Real': Real,
#         'sat': sat,
#         'print': print,
#         'exp': lambda x: 2.718281828459045**x,
#         'Exp': lambda x: 2.718281828459045**x,
#         'And': And,
#         'Or': Or,
#         'Not': Not,
#         'If': If,
#         'pi': 3.141592653589793,
#         'Sum': Sum,
#         'sqrt': lambda x: x**(0.5),
#         'ForAll': ForAll,
#         'Exists': Exists,
#         'Function': Function,
#         'BitVec': BitVec,
#         'BitVecVal': BitVecVal,
#         'Extract': Extract,
#         'Concat': Concat,
#         'BV2Int': BV2Int,
#         'Int2BV': Int2BV,
#         'sin': lambda x: Sin(x),
#         'cos': lambda x: Cos(x),
#         'tan': lambda x: Tan(x),
#         'log': lambda x: Log(x),
#     }

#     old_stdout = sys.stdout
#     new_stdout = io.StringIO()
#     sys.stdout = new_stdout

#     try:
#         exec(z3_code, global_namespace)
#         output = new_stdout.getvalue().strip()
#         return output
#     except Exception as e:
#         return f"Error executing Z3 code: {str(e)}"
#     finally:
#         sys.stdout = old_stdout

# def z3_evaluate_mmlu(question):
#     symbolic_form = symbolic_representation(question)
#     print("Generated symbolic representation:")
#     print(symbolic_form)

#     z3_code = z3_converter(question, symbolic_form)
#     print("Generated Z3 code:")
#     print(z3_code)

#     result = execute_z3_code(z3_code)
#     print("Execution result:")
#     print(result)
#     return symbolic_form, z3_code, result

# def gpt4o_mini_decoder(question, options, z3_result):
#     messages = [
#         {"role": "system", "content": "You are a decoder that selects the correct MMLU option based on the Z3 code output. If the Z3 output doesn't align with any option, respond with 'z3-code is wrong, try again'. DO NOT PROVIDE YOUR OWN ANSWER OR REASONING"},
#         {"role": "user", "content": f"Question: {question}\nOptions: {options}\nZ3 Result: {z3_result}\n\nBased on the Z3 result, which option is correct? If none match, say 'z3-code is wrong, try again'."}
#     ]

#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         temperature=0.0
#     )

#     return response.choices[0].message['content'].strip()