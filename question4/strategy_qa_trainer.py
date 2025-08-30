import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    set_seed
)
from peft import get_peft_model, LoraConfig, TaskType
import os
from typing import Dict, List, Tuple
import json

from config import Config


class StrategyQATrainer:
    """Fine-tune ModernBERT on StrategyQA with head-only and LoRA approaches."""
    
    def __init__(self, config: Config):
        self.config = config
        set_seed(config.random_seed)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.dataset = self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        """Load and preprocess StrategyQA dataset."""
        dataset = load_dataset("wics/strategy-qa")
        full_dataset = dataset["test"]  # only has test split
        
        train_rest_split = full_dataset.train_test_split(test_size=0.2, seed=self.config.random_seed)
        val_test_split = train_rest_split["test"].train_test_split(test_size=0.5, seed=self.config.random_seed)
        
        dataset_splits = {
            "train": train_rest_split["train"],
            "validation": val_test_split["train"], 
            "test": val_test_split["test"]
        }
        
        def preprocess_function(examples):
            return self.tokenizer(
                examples["question"],
                truncation=True,
                padding=False,
                max_length=self.config.max_length
            )
        
        def extract_labels(examples):
            examples["labels"] = [1 if answer else 0 for answer in examples["answer"]]
            return examples
        
        for split_name in dataset_splits:
            dataset_splits[split_name] = dataset_splits[split_name].map(extract_labels, batched=True)
            dataset_splits[split_name] = dataset_splits[split_name].map(preprocess_function, batched=True)
        
        return dataset_splits
    
    def create_base_model(self):
        """Create base ModernBERT model for classification."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
            id2label={0: "False", 1: "True"},
            label2id={"False": 0, "True": 1}
        )
        return model
    
    def count_trainable_parameters(self, model):
        """Count number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def freeze_base_model(self, model):
        """Freeze all parameters except classification head."""
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        
        head_params = self.count_trainable_parameters(model)
        print(f"Head-only trainable parameters: {head_params}")
        return head_params
    
    def calculate_lora_rank(self, target_params: int, model):
        """Calculate LoRA rank to match target parameter count."""
        
        def estimate_lora_params(r, alpha=self.config.lora_alpha):
            lora_params = 0
            for name, module in model.named_modules():
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    if any(target in name for target in ['query', 'key', 'value', 'dense']):
                        lora_params += (module.in_features * r) + (r * module.out_features)
            return lora_params
        
        for r in range(1, 64):
            estimated_params = estimate_lora_params(r)
            if estimated_params >= target_params:
                print(f"Selected LoRA rank: {r}, estimated parameters: {estimated_params}")
                return r
        
        print(f"Using default rank: {self.config.lora_r}")
        return self.config.lora_r
    
    def create_lora_model(self, target_params: int):
        """Create LoRA model with matching parameter count."""
        base_model = self.create_base_model()
        
        calculated_rank = self.calculate_lora_rank(target_params, base_model)
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=calculated_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["query", "key", "value", "dense"]
        )
        
        model = get_peft_model(base_model, peft_config)
        lora_params = self.count_trainable_parameters(model)
        print(f"LoRA trainable parameters: {lora_params}")
        
        return model, lora_params
    
    def compute_metrics(self, eval_pred):
        """Compute accuracy metric."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
    
    def train_model(self, model, output_dir: str, run_name: str):
        """Train model and return training history."""
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            run_name=run_name,
            seed=self.config.random_seed,
            eval_on_start=True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        class TrainingCallback:
            def __init__(self):
                self.train_accuracies = []
                
            def on_epoch_end(self, args, state, control, model=None, **kwargs):
                train_results = trainer.evaluate(eval_dataset=self.dataset["train"])
                self.train_accuracies.append(train_results["eval_accuracy"])
        
        callback = TrainingCallback()
        trainer.add_callback(callback)
        trainer.train()
        
        test_results = trainer.evaluate(eval_dataset=self.dataset["test"])
        
        eval_logs = [log for log in trainer.state.log_history if "eval_accuracy" in log]
        
        training_history = {
            "train_accuracy": callback.train_accuracies,
            "eval_accuracy": [log["eval_accuracy"] for log in eval_logs],
            "test_accuracy": test_results["eval_accuracy"],
            "best_eval_accuracy": max([log["eval_accuracy"] for log in eval_logs]) if eval_logs else 0
        }
        
        return training_history
    
    def run_head_only_experiment(self):
        """Run head-only fine-tuning experiment."""
        print("Starting Head-Only Fine-tuning Experiment...")
        
        model = self.create_base_model()
        head_params = self.freeze_base_model(model)
        
        os.makedirs(self.config.output_dir_head, exist_ok=True)
        history = self.train_model(model, self.config.output_dir_head, "head_only")
        
        return history, head_params
    
    def run_lora_experiment(self, target_params: int):
        """Run LoRA fine-tuning experiment."""
        print("Starting LoRA Fine-tuning Experiment...")
        
        model, lora_params = self.create_lora_model(target_params)
        
        os.makedirs(self.config.output_dir_lora, exist_ok=True)
        history = self.train_model(model, self.config.output_dir_lora, "lora")
        
        return history, lora_params
    
    def plot_training_curves(self, head_history: Dict, lora_history: Dict):
        """Plot training and validation accuracy curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(head_history["eval_accuracy"]) + 1)
        
        # Validation accuracy plot
        ax1.plot(epochs, head_history["eval_accuracy"], 'b-', label='Head-Only Validation')
        ax1.plot(epochs, lora_history["eval_accuracy"], 'r-', label='LoRA Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Validation Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training accuracy plot
        if head_history["train_accuracy"] and lora_history["train_accuracy"]:
            ax2.plot(epochs, head_history["train_accuracy"], 'b--', label='Head-Only Training')
            ax2.plot(epochs, lora_history["train_accuracy"], 'r--', label='LoRA Training')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results_table(self, head_history: Dict, lora_history: Dict, 
                          head_params: int, lora_params: int):
        """Save results in table format."""
        results = {
            "Head-Only": {
                "test_accuracy": head_history["test_accuracy"],
                "validation_accuracy": head_history["best_eval_accuracy"],
                "trainable_parameters": head_params
            },
            "LoRA": {
                "test_accuracy": lora_history["test_accuracy"],
                "validation_accuracy": lora_history["best_eval_accuracy"],
                "trainable_parameters": lora_params
            }
        }
        
        with open('results/results_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        with open('results/results_table.txt', 'w') as f:
            f.write("StrategyQA Classification Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"{'Approach':<15} {'Test Acc':<10} {'Val Acc':<10} {'Params':<12}\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Head-Only':<15} {results['Head-Only']['test_accuracy']:<10.4f} "
                   f"{results['Head-Only']['validation_accuracy']:<10.4f} "
                   f"{results['Head-Only']['trainable_parameters']:<12}\n")
            f.write(f"{'LoRA':<15} {results['LoRA']['test_accuracy']:<10.4f} "
                   f"{results['LoRA']['validation_accuracy']:<10.4f} "
                   f"{results['LoRA']['trainable_parameters']:<12}\n")
        
        print("\nResults Summary:")
        print(f"Head-Only - Test: {results['Head-Only']['test_accuracy']:.4f}, "
              f"Val: {results['Head-Only']['validation_accuracy']:.4f}, "
              f"Params: {results['Head-Only']['trainable_parameters']}")
        print(f"LoRA - Test: {results['LoRA']['test_accuracy']:.4f}, "
              f"Val: {results['LoRA']['validation_accuracy']:.4f}, "
              f"Params: {results['LoRA']['trainable_parameters']}")


def main():
    """Run both fine-tuning experiments."""
    config = Config()
    trainer = StrategyQATrainer(config)
    
    head_history, head_params = trainer.run_head_only_experiment()
    lora_history, lora_params = trainer.run_lora_experiment(head_params)
    
    trainer.plot_training_curves(head_history, lora_history)
    trainer.save_results_table(head_history, lora_history, head_params, lora_params)
    
    print("All experiments completed!")


if __name__ == "__main__":
    main()