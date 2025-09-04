# Question 1: Self-Attention Profiling

Profiles self-attention mechanisms for computational complexity analysis.

## Overview

Implements and profiles single-head and multi-head attention layers, measuring:
- FLOPS (computational complexity) 
- Memory usage
- Wall clock time

Experiments run across sequence lengths: 10, 100, 1K, 10K tokens.

## Usage

### SLURM Execution
```bash
sbatch run_cpu.sh    # CPU profiling
sbatch run_gpu.sh    # GPU profiling  
```

### Local Testing
```bash
python self_attention_profiler.py
```

## Results

- Plots saved to `results/` directory
- Shows complexity scaling trends with error bars
- Compares single-head vs multi-head attention
- CPU vs GPU performance analysis