# strategyqa_modernbert.py

import os
import json
from typing import Dict, Tuple, List, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    TrainerCallback, EvalPrediction, set_seed
)
from peft import get_peft_model, LoraConfig, TaskType


# ----------------------------
# Config
# ----------------------------

class Config:
    model_name = "answerdotai/ModernBERT-base"
    dataset_name = "wics/strategy-qa"

    num_epochs = 5
    batch_size = 16
    learning_rate = 2e-5
    max_length = 512

    lora_alpha = 16
    lora_dropout = 0.1

    random_seed = 42
    save_strategy = "epoch"
    evaluation_strategy = "epoch"

    output_dir_head = "results/head_only"
    output_dir_lora = "results/lora"


# ----------------------------
# Helpers
# ----------------------------

class TrainEvalLogger(TrainerCallback):
    """
    Logs validation accuracy from HF eval and computes train accuracy per epoch
    on the CURRENT model at that epoch (before best checkpoint is reloaded).
    """
    def __init__(self, get_trainer_fn, train_dataset):
        self._get_trainer = get_trainer_fn
        self._train_ds = train_dataset
        self.eval_acc: List[float] = []
        self.train_acc: List[float] = []

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        if "eval_accuracy" in metrics:
            self.eval_acc.append(float(metrics["eval_accuracy"]))

        trainer: Trainer = self._get_trainer()
        pred = trainer.predict(self._train_ds)
        # HF prefixes predict metrics with 'test_'
        self.train_acc.append(float(pred.metrics.get("test_accuracy", float("nan"))))


# ----------------------------
# Main trainer class
# ----------------------------

class StrategyQATrainer:
    """Fine-tune ModernBERT on StrategyQA with head-only and LoRA approaches."""

    def __init__(self, config: Config):
        self.config = config
        set_seed(config.random_seed)

        os.makedirs("results", exist_ok=True)
        os.makedirs(self.config.output_dir_head, exist_ok=True)
        os.makedirs(self.config.output_dir_lora, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.dataset = self.load_and_preprocess_data()

    # ---------- Data ----------

    def load_and_preprocess_data(self) -> DatasetDict:
        """
        Load StrategyQA. If HF dataset lacks explicit train/val splits,
        create them from the available split (often 'test' in this repo).
        Convert boolean 'answer' -> int label and tokenize 'question'.
        Keep only model-required columns.
        """
        raw = load_dataset(self.config.dataset_name)

        # Normalize to train/validation/test
        if set(raw.keys()) >= {"train", "validation", "test"}:
            splits = {
                "train": raw["train"],
                "validation": raw["validation"],
                "test": raw["test"]
            }
        else:
            # Pick a present split to re-split (prefer 'train', else 'test')
            base_split_name = "train" if "train" in raw else "test"
            base_split = raw[base_split_name]
            # 80/10/10 split
            tmp = base_split.train_test_split(test_size=0.2, seed=self.config.random_seed)
            val_test = tmp["test"].train_test_split(test_size=0.5, seed=self.config.random_seed)
            splits = {
                "train": tmp["train"],
                "validation": val_test["train"],
                "test": val_test["test"]
            }

        def extract_labels(examples: Dict[str, Any]):
            # StrategyQA has boolean 'answer'
            examples["labels"] = [1 if bool(a) else 0 for a in examples["answer"]]
            return examples

        def tokenize_fn(examples: Dict[str, Any]):
            return self.tokenizer(
                examples["question"],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
            )

        # map and keep only the required columns
        for split_name in list(splits.keys()):
            ds = splits[split_name]
            ds = ds.map(extract_labels, batched=True)
            ds = ds.map(tokenize_fn, batched=True)
            keep_cols = {"input_ids", "attention_mask", "labels"}
            remove_cols = [c for c in ds.column_names if c not in keep_cols]
            ds = ds.remove_columns(remove_cols)
            splits[split_name] = ds

        return DatasetDict(splits)

    # ---------- Models ----------

    def create_base_model(self) -> AutoModelForSequenceClassification:
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
            id2label={0: "False", 1: "True"},
            label2id={"False": 0, "True": 1}
        )
        return model

    def count_trainable_parameters(self, model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def freeze_base_model_head_only(self, model: torch.nn.Module) -> int:
        """Freeze everything except the classification head (robust to naming)."""
        for p in model.parameters():
            p.requires_grad = False

        # try common naming first
        head = getattr(model, "classifier", None) or getattr(model, "score", None) or getattr(model, "classification_head", None)
        if head is not None:
            for p in head.parameters():
                p.requires_grad = True
        else:
            # fallback: unfreeze any Linear whose out_features == num_labels
            num_labels = getattr(model.config, "num_labels", 2)
            unfrozen = False
            for m in model.modules():
                if isinstance(m, torch.nn.Linear) and m.out_features == num_labels:
                    for p in m.parameters():
                        p.requires_grad = True
                    unfrozen = True
            if not unfrozen:
                raise RuntimeError("Could not locate classification head to unfreeze.")

        head_params = self.count_trainable_parameters(model)
        print(f"[Head-only] trainable parameters: {head_params}")
        return head_params

    def _classifier_dims(self, model: torch.nn.Module) -> Tuple[int, int]:
        """Return (in_features=d, out_features=c) for classifier Linear."""
        head = getattr(model, "classifier", None) or getattr(model, "score", None) or getattr(model, "classification_head", None)

        candidates = []
        if head is not None:
            for m in head.modules():
                if isinstance(m, torch.nn.Linear) and m.out_features == model.config.num_labels:
                    candidates.append(m)

        if not candidates:
            for m in model.modules():
                if isinstance(m, torch.nn.Linear) and m.out_features == model.config.num_labels:
                    candidates.append(m)

        if not candidates:
            raise RuntimeError("Could not infer classifier Linear dims.")
        # Pick the first match
        lin = candidates[0]
        return lin.in_features, lin.out_features

    def create_lora_model(self, head_param_budget: int) -> Tuple[torch.nn.Module, int]:
        """
        LoRA only on the classifier module.
        Freeze all base weights (including classifier base weights).
        Choose rank r so that LoRA trainable params ~ head-only budget.
        """
        base_model = self.create_base_model()

        # Freeze EVERYTHING; LoRA adapters will be the only trainable params
        for p in base_model.parameters():
            p.requires_grad = False

        d, c = self._classifier_dims(base_model)
        # head-only params: weights (c*d) + bias (c)
        p_head = c * (d + 1)

        # ideal real-valued r* to match c(d+1) ≈ r(d+c)
        r_star = (c * (d + 1)) / (d + c)
        r = max(1, int(round(r_star)))

        # Debug: print module names to find correct target
        print("Available modules:")
        for name, module in base_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"  {name}: {type(module).__name__} ({module.in_features} -> {module.out_features})")
        
        # Try common target module names for ModernBERT
        possible_targets = ["classifier", "score", "classification_head", "cls.predictions", "pooler.dense"]
        target_modules = []
        
        for target in possible_targets:
            if any(target in name for name, _ in base_model.named_modules()):
                target_modules.append(target)
        
        if not target_modules:
            # Fallback: target the final linear layer by finding module with num_labels output
            for name, module in base_model.named_modules():
                if isinstance(module, torch.nn.Linear) and module.out_features == base_model.config.num_labels:
                    target_modules.append(name)
                    break
        
        print(f"Using target_modules: {target_modules}")
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none"
        )

        model = get_peft_model(base_model, peft_config)

        lora_params = self.count_trainable_parameters(model)
        print(f"[LoRA] budget(head)={p_head} | chosen r={r} | trainable(lora)={lora_params} | delta={lora_params - p_head}")
        return model, lora_params

    # ---------- Metrics & training ----------

    @staticmethod
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        # Handle both tuple and EvalPrediction
        if isinstance(eval_pred, tuple):
            preds, labels = eval_pred
        else:
            preds, labels = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(preds, axis=1)
        acc = (preds == labels).mean()
        return {"accuracy": float(acc)}

    def train_model(self, model: torch.nn.Module, output_dir: str, run_name: str) -> Dict[str, Any]:
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_strategy="epoch",
            eval_strategy=self.config.evaluation_strategy,  # "epoch"
            save_strategy=self.config.save_strategy,              # "epoch"
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            run_name=run_name,
            seed=self.config.random_seed,
            report_to=[],  # keep local; add "wandb" if desired
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

        # Attach callback to capture per-epoch train/val accuracy
        logger_cb = TrainEvalLogger(lambda: trainer, self.dataset["train"])
        trainer.add_callback(logger_cb)

        trainer.train()

        # Evaluate the best-on-dev model on held-out test once
        test_results = trainer.evaluate(eval_dataset=self.dataset["test"])

        history = {
            "train_accuracy": logger_cb.train_acc,
            "eval_accuracy": logger_cb.eval_acc,
            "test_accuracy": float(test_results["eval_accuracy"]),
            "best_eval_accuracy": max(logger_cb.eval_acc) if logger_cb.eval_acc else 0.0
        }
        return history

    # ---------- Experiments ----------

    def run_head_only_experiment(self) -> Tuple[Dict[str, Any], int]:
        print("==> Starting Head-Only Fine-tuning Experiment...")
        model = self.create_base_model()
        head_params = self.freeze_base_model_head_only(model)
        history = self.train_model(model, self.config.output_dir_head, "head_only")
        return history, head_params

    def run_lora_experiment(self, target_params: int) -> Tuple[Dict[str, Any], int]:
        print("==> Starting LoRA Fine-tuning Experiment...")
        model, lora_params = self.create_lora_model(target_params)
        history = self.train_model(model, self.config.output_dir_lora, "lora")
        return history, lora_params

    # ---------- Reporting ----------

    @staticmethod
    def plot_training_curves(head_history: Dict[str, Any], lora_history: Dict[str, Any]) -> None:
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Validation
        xh = np.arange(1, len(head_history["eval_accuracy"]) + 1)
        xl = np.arange(1, len(lora_history["eval_accuracy"]) + 1)
        ax1.plot(xh, head_history["eval_accuracy"], label='Head-Only Val')
        ax1.plot(xl, lora_history["eval_accuracy"], label='LoRA Val')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.set_title('Validation Accuracy')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # Training
        xh_t = np.arange(1, len(head_history["train_accuracy"]) + 1)
        xl_t = np.arange(1, len(lora_history["train_accuracy"]) + 1)
        ax2.plot(xh_t, head_history["train_accuracy"], '--', label='Head-Only Train')
        ax2.plot(xl_t, lora_history["train_accuracy"], '--', label='LoRA Train')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_title('Training Accuracy')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_results_table(head_history: Dict[str, Any], lora_history: Dict[str, Any],
                           head_params: int, lora_params: int) -> None:
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

        os.makedirs("results", exist_ok=True)
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
              f"Val(best): {results['Head-Only']['validation_accuracy']:.4f}, "
              f"Params: {results['Head-Only']['trainable_parameters']}")
        print(f"LoRA - Test: {results['LoRA']['test_accuracy']:.4f}, "
              f"Val(best): {results['LoRA']['validation_accuracy']:.4f}, "
              f"Params: {results['LoRA']['trainable_parameters']}")


# ----------------------------
# Entry point
# ----------------------------

def main():
    config = Config()
    trainer = StrategyQATrainer(config)

    head_history, head_params = trainer.run_head_only_experiment()
    lora_history, lora_params = trainer.run_lora_experiment(head_params)

    trainer.plot_training_curves(head_history, lora_history)
    trainer.save_results_table(head_history, lora_history, head_params, lora_params)

    print("\nAll experiments completed!")
    print("Plot saved to results/training_curves.png")
    print("Table saved to results/results_table.txt\n")


if __name__ == "__main__":
    main()
