#!/usr/bin/env python3
"""BinPVD cross-optimization-level evaluation script.

Evaluates a trained model across O0, O1, O2, O3 optimization levels
as described in the Chapter 3 experimental setup.

Usage:
    # Evaluate best model from a training run
    python -m binpvd.evaluate --checkpoint binpvd_output/binpvd_20240101-120000/best_model.pth

    # Evaluate with a specific config (loads corresponding config.json)
    python -m binpvd.evaluate --experiment_dir binpvd_output/binpvd_20240101-120000

    # Evaluate all O-levels and produce a comparison table
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binpvd.config import ExperimentConfig
from binpvd.utils import set_seed, compute_metrics, setup_logging
from binpvd.data import load_tokenizer, MultiOptimizationDataset
from binpvd.models.binpvd_model import BinPVDModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BinPVD Cross-O Evaluation")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Path to experiment directory containing config.json and best_model.pth")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Direct path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json (required if --checkpoint is used without --experiment_dir)")
    parser.add_argument("--output_dir", type=str, default="binpvd_output/eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


@torch.no_grad()
def evaluate_on_loader(
    model: BinPVDModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model on a single data loader."""
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve config and checkpoint paths
    if args.experiment_dir:
        config_path = os.path.join(args.experiment_dir, "config.json")
        checkpoint_path = args.checkpoint or os.path.join(args.experiment_dir, "best_model.pth")
    elif args.config and args.checkpoint:
        config_path = args.config
        checkpoint_path = args.checkpoint
    else:
        print("Error: Provide either --experiment_dir or both --config and --checkpoint")
        sys.exit(1)

    # Load config
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config_dict = json.load(f)
    cfg = ExperimentConfig.from_dict(config_dict)
    print(f"Loaded config from {config_path}")

    # Setup
    set_seed(cfg.training.seed)
    logger = setup_logging(args.output_dir, "evaluate")

    # Tokenizer
    tokenizer = load_tokenizer(cfg.data.tokenizer_path)
    tokenizer.add_tokens(list(cfg.prefix.never_split_tokens), special_tokens=True)

    # Model
    model = BinPVDModel(
        model_name_or_path=cfg.model.model_name_or_path,
        num_labels=cfg.model.num_labels,
        hidden_size=cfg.model.hidden_size,
        dropout=cfg.model.dropout,
        prefix_enabled=cfg.prefix.enabled,
        prefix_length=cfg.prefix.prefix_length,
    ).to(device)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Multi-O evaluation
    multi_dataset = MultiOptimizationDataset(
        tokenizer, cfg.data, cfg.data.max_seq_length,
    )

    results = {}
    for o_name in ["O0", "O1", "O2", "O3"]:
        loader = multi_dataset.get_loader(o_name, args.batch_size, shuffle=False)
        if len(loader.dataset) == 0:
            print(f"  {o_name}: no data, skipping")
            continue
        print(f"\nEvaluating on {o_name} ({len(loader.dataset)} samples)...")
        metrics = evaluate_on_loader(model, loader, device, args.threshold)
        results[o_name] = metrics
        print(f"  {o_name} → Acc: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1']:.4f}, "
              f"Prec: {metrics['precision']:.4f}, "
              f"Rec: {metrics['recall']:.4f}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Cross-O Evaluation Results':^70}")
    print(f"{'='*70}")
    print(f"{'Level':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for o_name in ["O0", "O1", "O2", "O3"]:
        if o_name in results:
            m = results[o_name]
            print(f"{o_name:<8} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
                  f"{m['recall']:<12.4f} {m['f1']:<12.4f}")
    print(f"{'='*70}")

    # Log results
    logger.info(f"Cross-O results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
