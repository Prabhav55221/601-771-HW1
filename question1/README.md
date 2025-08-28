# Question 1: Self-Attention Profiling

This directory contains code to profile self-attention mechanisms and analyze their computational complexity, memory usage, and wall clock time.

## Files

- `config.py` - Configuration parameters (embedding dimensions, sequence lengths, etc.)
- `attention_layers.py` - Single-head and multi-head attention implementations
- `profiling_utils.py` - FLOPS counting, memory profiling, and timing utilities
- `plot_results.py` - Seaborn-based plotting with error bars
- `self_attention_profiler.py` - Main profiling script
- `results/` - Output directory for plots and data

## Usage

1. Make sure conda environment is activated:
```bash
conda activate nlp-hw1
```

2. Run the profiling experiment:
```bash
python self_attention_profiler.py
```

## What it measures

1. **FLOPS** (Floating Point Operations) - Using torchprofile/fvcore or manual counting
2. **Memory Usage** - Peak GPU/CPU memory consumption during forward pass
3. **Wall Clock Time** - Actual execution time with multiple runs and error bars

## Output

- CSV files with detailed numerical results
- PNG/PDF plots showing scaling trends with error bars
- Summary tables comparing single-head vs multi-head attention
- Both CPU and GPU performance comparisons (if GPU available)

## Configuration

Edit `config.py` to modify:
- Sequence lengths to test (default: [10, 100, 1000, 10000])
- Embedding dimension (default: 768)
- Number of attention heads (default: 8)
- Number of runs for averaging (default: 20)
- Which tests to run (single-head, multi-head, CPU, GPU)