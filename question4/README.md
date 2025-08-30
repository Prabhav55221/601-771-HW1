# Question 4: ModernBERT Fine-tuning on StrategyQA

Fine-tune ModernBERT on StrategyQA dataset comparing head-only vs LoRA approaches.

## Files

- `config.py` - Training configuration parameters
- `strategy_qa_trainer.py` - Complete training pipeline for both approaches

## Usage

1. Create conda environment:
```bash
conda env create -f ../environment.yml
conda activate nlp-hw1
pip install peft datasets
```

2. Run experiments:
```bash
python strategy_qa_trainer.py
```

## Experiments

**(4.1) Head-Only Fine-tuning**: Freeze ModernBERT base, train only classification head
**(4.2) LoRA Fine-tuning**: Use Low-Rank Adaptation with matching parameter count

## Key Features

- **Parameter Matching**: LoRA rank automatically calculated to match head-only parameter count
- **Proper Evaluation**: Best model selected based on validation accuracy, evaluated on test set
- **Training Visualization**: Plots training/validation accuracy curves
- **Results Table**: Generates formatted results matching assignment requirements

## Output

- `results/training_curves.png` - Training and validation accuracy plots
- `results/results_table.txt` - Formatted results table 
- `results/results_summary.json` - Detailed results in JSON format
- `results/head_only/` - Head-only fine-tuning checkpoints
- `results/lora/` - LoRA fine-tuning checkpoints