import io
import sys
from z3 import *
import openai
import torch
from torch.nn import DataParallel
from transformers import pipeline


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
        model="gpt-4o-2024-08-06",  
        messages=formatted_messages,
        max_tokens=4000,
        temperature=0
    )
    
    prediction = response.choices[0].message['content'].strip()
    return prediction



def predict_phi3(pipe, prompt, max_new_tokens, model, tokenizer):
    messages = [
        {"role": "user", "content": prompt}
    ]
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

# def predict_llama(model, tokenizer, prompt, max_new_tokens, device):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             num_return_sequences=1,
#             do_sample=False,
#             temperature=0.0
#         )
    
#     prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return prediction

def predict_llama(model, tokenizer, prompt, max_new_tokens, device):
    model_to_use = model.module if isinstance(model, DataParallel) else model
    device = next(model_to_use.parameters()).device

    # Tokenize and move tensors to the same device as the model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    pad_token_id = tokenizer.pad_token_id
    
    # Generate output
    output = model_to_use.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=False,
        temperature=0.0
    )
    
    # Decode output
    prediction = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
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
        {"role": "system", "content": "You are a decider that decides whether the answer is the same as the correct answer. If the output doesn't align with the correct answer, respond with '0', whereas if it's correct, then respond with '1'. DO NOT PROVIDE YOUR OWN ANSWER OR REASONING, JUST SELECT '0' OR '1'."},
        {"role": "user", "content": f"GPT-4o Result: {gpt_result}\nCorrect Answer: {correct_answer}. Answer with 0 (Wrong) or 1 (Correct)."}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0
    )

    return response.choices[0].message['content'].strip()

def model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device=None):
    if model_type == "gpt":
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        model_result = predict_gpt(model, messages)
    elif model_type == "phi-3":
        prompt = f"{system_content}\n\nQuestion: {question}\n\nOptions:\n{formatted_options}"
        model_result = predict_phi3(model, prompt, max_new_tokens=3500)
    elif model_type == "llama3.1_8b":
        prompt = f"{system_content}\n\nQuestion: {question}\n\nOptions:\n{formatted_options}"
        model_result = predict_llama(model, tokenizer, prompt, max_new_tokens=3500, device=device)
    else: 
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Model result: {model_result}")
    return model_result
