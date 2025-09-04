"""Configuration for ModernBERT StrategyQA fine-tuning experiments.

Author: Prabhav Singh
"""


class Config:
    """Configuration parameters for StrategyQA experiments."""
    model_name = "answerdotai/ModernBERT-base"
    dataset_name = "wics/strategy-qa"
    
    num_epochs = 5
    batch_size = 16
    learning_rate = 2e-5
    max_length = 512
    
    lora_r = 1  # Rank=1 to match head parameter count
    lora_alpha = 16
    lora_dropout = 0.1
    
    random_seed = 42
    save_strategy = "epoch"
    evaluation_strategy = "epoch"
    
    output_dir_head = "results/head_only"
    output_dir_lora = "results/lora"