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

def predict_phi3(model, tokenizer, prompt, max_new_tokens=3500):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# def predict_llama(model, tokenizer, prompt, max_new_tokens=3500):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             temperature=0.0
#         )

#     return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def predict_llama(model, tokenizer, prompt, max_new_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)
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

def predict_codestral(model, tokenizer, prompt, max_new_tokens=3000):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def predict_qwen2(model, tokenizer, system_content, question, formatted_options, max_new_tokens=3500):
    prompt = f"Question: {question}\n\nOptions:\n{formatted_options}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )
    
    response = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
    return response

def model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, device=None):
    if model_type == "gpt":
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        model_result = predict_gpt(model, messages)
    elif model_type == "phi-3":
        prompt = f"{system_content}\n\nQuestion: {question}\n\nOptions:\n{formatted_options}"
        model_result = predict_phi3(model, tokenizer, prompt, max_new_tokens=3500)
    elif model_type == "llama3.1_8b" or model_type == "llama3.1_70b":
        prompt = f"{system_content}\n\nQuestion: {question}\n\nOptions:\n{formatted_options}"
        model_result = predict_llama(model, tokenizer, prompt, max_new_tokens=3500)
    elif model_type == "codestral":
        prompt = f"{system_content}\n\nQuestion: {question}\n\nOptions:\n{formatted_options}"
        model_result = predict_codestral(model, tokenizer, prompt, max_new_tokens=3500)
    elif model_type == "qwen2":
        prompt = f"{system_content}\n\nQuestion: {question}\n\nOptions:\n{formatted_options}"
        model_result = predict_qwen2(model, tokenizer, system_content, question, formatted_options)
    else: 
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Model result: {model_result}")
    return model_result