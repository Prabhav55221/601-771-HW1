# Question 1: Self-Attention Profiling

Profile self-attention mechanisms and analyze computational complexity, memory usage, and wall clock time.

## Files

- `config.py` - Configuration parameters
- `attention_layers.py` - Single-head and multi-head attention implementations
- `profiling_utils.py` - FLOPS counting, timing utilities
- `plot_results.py` - Plotting functions
- `self_attention_profiler.py` - Main profiling script

## Usage

1. Create conda environment:
```bash
conda env create -f ../environment.yml
conda activate nlp-hw1
```

2. Run profiling:
```bash
python self_attention_profiler.py
```

## Output

- Plots showing scaling trends with error bars
- Summary of FLOPS, memory usage, and timing results
- CPU vs GPU performance comparisons