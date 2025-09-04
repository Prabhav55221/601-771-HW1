# CS 601.771 - Advanced Self-Supervised Statistical NLP

Homework Assignment 1 - Prabhav Singh

## Project Structure

- **`profiling_self_attention/`** - Self-Attention Profiling (FLOPS, memory, timing analysis)
- **`perpexity/`** - DistilGPT2 Analysis (perplexity and sampling experiments)  
- **`sgd_behaviour/`** - SGD Optimization (momentum and weight decay effects)
- **`fineturning_moderbert/`** - ModernBERT Fine-tuning (head-only vs LoRA comparison)
- **`retrieval/`** - Information Retrieval (FAISS-based SciFact search)

## Setup

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate nlp-hw1
```

2. Navigate to specific question directory to run experiments