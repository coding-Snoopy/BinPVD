"""Configuration management for BinPVD experiments.

Supports all experiment types:
- Main vulnerability classification
- Cross-optimization-level evaluation
- Prefix length / projection ablations
- Learning rate sweeps
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Base model configuration (RoBERTa/CodeBERT)."""
    model_name_or_path: str = "/data/yangx/PBVD/checkpoints/roberta-MLM-50/"
    num_labels: int = 1
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    dropout: float = 0.1


@dataclass
class PrefixConfig:
    """Prefix tuning configuration."""
    enabled: bool = True
    prefix_length: int = 10
    prefix_projection: bool = True
    encoder_hidden_size: int = 768
    init_prompt: str = (
        "The following assembly instructions Does software network vulnerability exist?"
    )   #Only answer yes as 1 or no is 0.
    never_split_tokens: tuple = (
        "The", "following", "assembly", "instructions", "network",
        "Does", "software", "vulnerability", "exist", "Only",
        "answer", "yes", "as", "or", "no", "is",
    )


@dataclass
class DataConfig:
    """Data configuration."""
    tokenizer_path: str = "/data/yangx/PBVD/datasets/O2/tokenizer1.json"
    max_seq_length: int = 512
    train_file: str = "F:/PBVD/datasets/train.csv"
    eval_file: str = "F:/PBVD/datasets/test.csv"
    # Cross-optimization-level evaluation files
    o0_file: str = "F:/PBVD/datasets/O0.csv"
    o1_file: str = "F:/PBVD/datasets/O1.csv"
    o2_file: str = "F:/PBVD/datasets/O2.csv"
    o3_file: str = "F:/PBVD/datasets/O3.csv"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 8
    learning_rate: float = 1.1e-5
    num_epochs: int = 5
    weight_decay: float = 3e-4
    warmup_ratio: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    eval_threshold: float = 0.5
    logging_steps: int = 50
    save_best: bool = True
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    prefix: PrefixConfig = field(default_factory=PrefixConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda:0"
    output_dir: str = "binpvd_output"
    experiment_name: str = "binpvd_default"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def default(cls) -> "ExperimentConfig":
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Reconstruct ExperimentConfig from a nested dict (e.g. loaded from JSON)."""
        return cls(
            model=ModelConfig(**d.get("model", {})),
            prefix=PrefixConfig(**d.get("prefix", {})),
            data=DataConfig(**d.get("data", {})),
            training=TrainingConfig(**d.get("training", {})),
            device=d.get("device", "cuda:0"),
            output_dir=d.get("output_dir", "binpvd_output"),
            experiment_name=d.get("experiment_name", "binpvd_default"),
        )

    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        """Populate config from argparse Namespace (overrides only)."""
        cfg = cls.default()
        overrides = {k: v for k, v in vars(args).items() if v is not None}
        for section_key, section_dc in [("model", cfg.model), ("prefix", cfg.prefix),
                                         ("data", cfg.data), ("training", cfg.training)]:
            for field_name in section_dc.__dataclass_fields__:
                cli_key = f"{section_key}_{field_name}"
                if cli_key in overrides:
                    setattr(section_dc, field_name, overrides[cli_key])
        for attr in ("device", "output_dir", "experiment_name"):
            if attr in overrides:
                setattr(cfg, attr, overrides[attr])
        return cfg
