# Question 4: ModernBERT Fine-tuning

Fine-tuning ModernBERT on StrategyQA with head-only vs LoRA comparison.

## Overview

Compares two fine-tuning approaches:
- **Head-Only**: Freeze base model, train classification head (1538 parameters)
- **LoRA**: Low-rank adaptation targeting last attention layer (1536 parameters)

## Usage

### SLURM Execution
```bash
sbatch run_experiments.sh
```

### Local Testing
```bash
python strategy_qa_trainer.py
```

## Experiments

- **4.1** Classification head fine-tuning only
- **4.2** LoRA with matched parameter count (rank=1, W_o layers)

## Key Features

- Parameter count matching between approaches
- Training/validation curve tracking
- Best model selection on dev set
- Test set evaluation with final model

## Results

- Training plots showing accuracy over epochs
- Classification results table
- JSON output with detailed metrics