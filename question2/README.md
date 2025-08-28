# Question 2: Empirical Analysis with DistilGPT2

## Setup

1. Install transformers:
```bash
pip install transformers torch
```

2. Add your paragraph to `paragraph.txt`

## Usage

Run both analyses:
```bash
python main.py
```

Or run individually:
```bash
python perplexity_analysis.py
python sampling_comparison.py
```

## Output

- `results/perplexity_results.txt` - Perplexity comparison
- `results/sampling_results.txt` - Generated text samples