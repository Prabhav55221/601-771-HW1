"""Sampling comparison analysis with DistilGPT2.

Author: Prabhav Singh
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def load_model():
    """Load DistilGPT2 model and tokenizer."""
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_text(prompt, model, tokenizer, strategy="greedy", temperature=1.0, max_new_tokens=500):
    """Generate text using different sampling strategies."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        if strategy == "greedy" or temperature == 0:
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_sampling_comparison():
    """Run sampling comparison with different temperature values."""
    model, tokenizer = load_model()
    prompt = "Once upon a time"
    
    temperatures = [0, 0.3, 0.6, 0.9, 1.2, 1.5]
    
    os.makedirs('results', exist_ok=True)
    with open('results/sampling_results.txt', 'w+') as f:
        f.write("SAMPLING COMPARISON\n")
        f.write("===================\n\n")
        f.write(f"Prompt: '{prompt}'\n")
        f.write(f"Generation length: 500 new tokens\n\n")
        
        # Greedy decoding
        f.write("GREEDY DECODING:\n")
        f.write("-" * 50 + "\n")
        greedy_output = generate_text(prompt, model, tokenizer, strategy="greedy")
        f.write(f"{greedy_output}\n\n")
        
        # Temperature sampling
        for temp in temperatures:
            if temp == 0:
                f.write(f"TEMPERATURE SAMPLING (T={temp}) - Equivalent to Greedy:\n")
            else:
                f.write(f"TEMPERATURE SAMPLING (T={temp}):\n")
            f.write("-" * 50 + "\n")
            temp_output = generate_text(prompt, model, tokenizer, strategy="temperature", temperature=temp, max_new_tokens=500)
            f.write(f"{temp_output}\n\n")

if __name__ == "__main__":
    run_sampling_comparison()