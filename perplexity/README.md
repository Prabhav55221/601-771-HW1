# Question 2: DistilGPT2 Analysis

Empirical analysis of DistilGPT2 including perplexity and sampling experiments.

## Overview

Two main experiments:
- **Perplexity Analysis**: Compare original vs word-shuffled text perplexity
- **Sampling Comparison**: Generate text with different sampling strategies

## Usage

### Run Both Experiments
```bash
python main.py
```

### Individual Experiments
```bash
python perplexity_analysis.py    # Perplexity only
python sampling_comparison.py    # Sampling only
```

## Experiments

### Perplexity Analysis
- Compares coherent vs shuffled text
- Uses `paragraph.txt` as input  
- Word-level shuffling for better evaluation

### Sampling Comparison
- Prompt: "Once upon a time"
- Generates 500 tokens
- Methods: Greedy + Temperature sampling (T=0, 0.3, 0.6, 0.9, 1.2, 1.5)

## Results

- `results/perplexity_results.txt` - Perplexity comparison with analysis
- `results/sampling_results.txt` - Text samples with diversity/quality commentary