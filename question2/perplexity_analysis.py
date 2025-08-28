import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random

def load_model():
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def compute_perplexity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()

def shuffle_paragraph(text):
    sentences = text.strip().split('. ')
    sentences = [s + '.' if not s.endswith('.') else s for s in sentences if s.strip()]
    if sentences[-1].endswith('..'):
        sentences[-1] = sentences[-1][:-1]
    
    random.shuffle(sentences)
    return ' '.join(sentences)

def run_perplexity_analysis():
    model, tokenizer = load_model()
    
    with open('paragraph.txt', 'r') as f:
        original_text = f.read().strip()
    
    random.seed(42)
    shuffled_text = shuffle_paragraph(original_text)
    
    original_perplexity = compute_perplexity(original_text, model, tokenizer)
    shuffled_perplexity = compute_perplexity(shuffled_text, model, tokenizer)
    
    with open('results/perplexity_results.txt', 'w') as f:
        f.write("PERPLEXITY ANALYSIS\n")
        f.write("==================\n\n")
        f.write("Original paragraph:\n")
        f.write(f"{original_text}\n\n")
        f.write(f"Original perplexity: {original_perplexity:.2f}\n\n")
        f.write("Shuffled paragraph:\n")
        f.write(f"{shuffled_text}\n\n")
        f.write(f"Shuffled perplexity: {shuffled_perplexity:.2f}\n\n")
        f.write(f"Difference: {shuffled_perplexity - original_perplexity:.2f}\n")
        f.write(f"Ratio (shuffled/original): {shuffled_perplexity/original_perplexity:.2f}\n")

if __name__ == "__main__":
    run_perplexity_analysis()