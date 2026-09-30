"""Data loading and dataset classes for BinPVD experiments.

Supports two CSV formats:
1. train.csv / test.csv: columns [idx, Example, Label]
2. O0.csv / O1.csv / O2.csv / O3.csv: columns [FunctionName, Example, GoodOrBad]

Label types supported: int (0/1) or str ("Good"/"Bad").
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """Load a PreTrainedTokenizerFast from a trained tokenizer JSON file."""
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(str(tokenizer_path))
    )
    tokenizer.mask_token = "<mask>"
    tokenizer.pad_token = "<pad>"
    tokenizer.cls_token = "<s>"
    tokenizer.sep_token = "</s>"
    tokenizer.model_max_length = 512
    return tokenizer


def _parse_label(raw_label) -> int:
    """Convert label to int (0/1). Handles both int and string formats."""
    if isinstance(raw_label, str):
        return 1 if raw_label.strip().lower() == "bad" else 0
    return int(raw_label)


class AssemblyDataset(Dataset):
    """Dataset for assembly code vulnerability classification.

    Reads CSV files with columns containing assembly code and binary labels.
    Supports both `train.csv` (columns: idx, Example, Label) and
    `O*.csv` (columns: FunctionName, Example, GoodOrBad) formats.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        data_path: str,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        df = pd.read_csv(data_path)

        # Detect column format
        if "Label" in df.columns:
            label_col = "Label"
        elif "GoodOrBad" in df.columns:
            label_col = "GoodOrBad"
        else:
            raise ValueError(
                f"Unknown label column in {data_path}. "
                f"Expected 'Label' or 'GoodOrBad', got {list(df.columns)}"
            )

        for _, row in df.iterrows():
            text = row["Example"]
            label = _parse_label(row[label_col])
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            self.samples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.float),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def build_dataloader(
    tokenizer: PreTrainedTokenizerFast,
    data_path: str,
    batch_size: int,
    max_length: int = 512,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Convenience function to create a DataLoader for assembly data."""
    dataset = AssemblyDataset(tokenizer, data_path, max_length)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


class MultiOptimizationDataset:
    """Container for cross-optimization-level evaluation datasets."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        data_config,
        max_length: int = 512,
    ):
        self.datasets = {}
        o_files = {
            "O0": data_config.o0_file,
            "O1": data_config.o1_file,
            "O2": data_config.o2_file,
            "O3": data_config.o3_file,
        }
        for name, path in o_files.items():
            if path:
                self.datasets[name] = AssemblyDataset(tokenizer, path, max_length)

    def get_loader(self, name: str, batch_size: int, shuffle: bool = False):
        ds = self.datasets[name]
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
