#!/usr/bin/env python3
"""BinPVD training script for Chapter 3 experiments.

Usage:
    # ====== Main experiments ======

    # 1. BinPVD with prefix tuning (default):
    python -m binpvd.train

    # 2. Full fine-tuning (no prefix tuning):
    python -m binpvd.train --prefix_enabled False

    # 3. Train on O0, evaluate on O1 (cross-optimization):
    python -m binpvd.train \\
        --data_train_file datasets/O0.csv \\
        --data_eval_file datasets/O1.csv \\
        --experiment_name binpvd_cross_o0_o1

    # ====== Hyperparameter ablation ======

    # 4. Vary prefix length:
    python -m binpvd.train --prefix_prefix_length 10
    python -m binpvd.train --prefix_prefix_length 50
    python -m binpvd.train --prefix_prefix_length 100

    # 5. Vary learning rate:
    python -m binpvd.train --training_learning_rate 5e-6
    python -m binpvd.train --training_learning_rate 5e-5

    # 6. Vary batch size:
    python -m binpvd.train --training_batch_size 4
    python -m binpvd.train --training_batch_size 16

    # 7. More epochs:
    python -m binpvd.train --training_num_epochs 10

    # ====== Data options ======

    # 8. Different model checkpoint:
    python -m binpvd.train --model_model_name_or_path "microsoft/codebert-base"

    # 9. Longer/shorter max sequence length:
    python -m binpvd.train --data_max_seq_length 128
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binpvd.config import ExperimentConfig
from binpvd.utils import (
    set_seed, get_timestamp, ensure_dir, save_config,
    compute_metrics, setup_logging, count_parameters,
)
from binpvd.data import load_tokenizer, AssemblyDataset
from binpvd.models.binpvd_model import BinPVDModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BinPVD Training")
    # Experiment meta
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/data/yangx/binpvd/outputs")
    parser.add_argument("--device", type=str, default=None)

    # Model
    parser.add_argument("--model_model_name_or_path", type=str, default="/data/yangx/PBVD/checkpoints/roberta-MLM-50/")
    parser.add_argument("--model_num_labels", type=int, default=1)
    parser.add_argument("--model_hidden_size", type=int, default=768)
    parser.add_argument("--model_num_hidden_layers", type=int, default=12)
    parser.add_argument("--model_num_attention_heads", type=int, default=12)
    parser.add_argument("--model_dropout", type=float, default=0.1)

    # Prefix
    parser.add_argument("--prefix_enabled", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--prefix_prefix_length", type=int, default=10)
    parser.add_argument("--prefix_prefix_projection", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--prefix_encoder_hidden_size", type=int, default=768)

    # Data
    parser.add_argument("--data_tokenizer_path", type=str, default="/data/yangx/PBVD/datasets/O2/tokenizer1.json")
    parser.add_argument("--data_max_seq_length", type=int, default=500)
    parser.add_argument("--data_train_file", type=str, default="/data/yangx/PBVD/datasets/O0-O1-O2-O3-train_data.csv")
    parser.add_argument("--data_eval_file", type=str, default="/data/yangx/PBVD/datasets/O0-O1-O2-O3-val_data.csv")

    # Training
    parser.add_argument("--training_batch_size", type=int, default=32)
    parser.add_argument("--training_learning_rate", type=float, default=2e-5)
    parser.add_argument("--training_num_epochs", type=int, default=20)
    # parser.add_argument("--training_weight_decay", type=float, default=None)
    # parser.add_argument("--training_warmup_ratio", type=float, default=None)
    # parser.add_argument("--training_adam_epsilon", type=float, default=None)
    # parser.add_argument("--training_max_grad_norm", type=float, default=None)
    # parser.add_argument("--training_eval_threshold", type=float, default=None)
    # parser.add_argument("--training_logging_steps", type=int, default=None)
    parser.add_argument("--training_seed", type=int, default=42)

    return parser.parse_args()


def train_epoch(
    model: BinPVDModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    logging_steps: int = 50,
    max_grad_norm: float = 1.0,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_steps = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epochs}")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_steps += 1

        if (step + 1) % logging_steps == 0:
            pbar.set_postfix({"loss": f"{total_loss / total_steps:.4f}"})

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate(
    model: BinPVDModel,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Evaluate the model on a dataloader. Returns metrics dict."""
    model.eval()
    all_targets = []
    all_preds = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).squeeze(-1)
        preds = (probs >= threshold).int()

        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return compute_metrics(all_targets, all_preds)


def main():
    args = parse_args()
    cfg = ExperimentConfig.from_args(args)

    # Use experiment name for output dir
    if cfg.experiment_name == "binpvd_default":
        cfg.experiment_name = f"binpvd_{get_timestamp()}"
    output_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    ensure_dir(output_dir)

    # Setup
    logger = setup_logging(output_dir, "train")
    set_seed(cfg.training.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {output_dir}")

    # Save config
    save_config(cfg, os.path.join(output_dir, "config.json"))
    logger.info(f"Config: {cfg.to_dict()}")

    # Tokenizer
    tokenizer = load_tokenizer(cfg.data.tokenizer_path)
    never_split = list(cfg.prefix.never_split_tokens)
    tokenizer.add_tokens(never_split, special_tokens=True)
    logger.info(f"Tokenizer loaded from {cfg.data.tokenizer_path}")

    # Datasets
    train_dataset = AssemblyDataset(
        tokenizer, cfg.data.train_file, cfg.data.max_seq_length,
    )
    eval_dataset = AssemblyDataset(
        tokenizer, cfg.data.eval_file, cfg.data.max_seq_length,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.training.batch_size, shuffle=False,
    )
    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # Model
    model = BinPVDModel(
        model_name_or_path=cfg.model.model_name_or_path,
        num_labels=cfg.model.num_labels,
        hidden_size=cfg.model.hidden_size,
        dropout=cfg.model.dropout,
        prefix_enabled=cfg.prefix.enabled,
        prefix_length=cfg.prefix.prefix_length,
    ).to(device)

    # Freeze base model if prefix tuning
    if cfg.prefix.enabled:
        model.freeze_base_model()
    else:
        model.unfreeze_all()

    # Log parameter counts
    param_counts = count_parameters(model)
    logger.info(f"Total params: {param_counts['total']:,}, "
                f"Trainable: {param_counts['trainable']:,}")

    # Optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped, lr=cfg.training.learning_rate, eps=cfg.training.adam_epsilon)

    total_steps = len(train_loader) * cfg.training.num_epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Training loop
    best_f1 = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pth")
    logger.info("Starting training...")

    for epoch in range(cfg.training.num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, logger,
            epoch, cfg.training.num_epochs, cfg.training.logging_steps,
            cfg.training.max_grad_norm,
        )

        metrics = evaluate(model, eval_loader, device, cfg.training.eval_threshold)
        logger.info(
            f"Epoch {epoch + 1}/{cfg.training.num_epochs} - "
            f"Train Loss: {train_loss:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Rec: {metrics['recall']:.4f}"
        )

        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  → New best model (F1={best_f1:.4f}) saved to {best_model_path}")

    # Final evaluation with best model
    logger.info(f"\nTraining complete. Best F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(best_model_path))
    final_metrics = evaluate(model, eval_loader, device, cfg.training.eval_threshold)
    logger.info(f"Final best model metrics: {final_metrics}")
    print(f"\n{'='*50}")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Results: {final_metrics}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
