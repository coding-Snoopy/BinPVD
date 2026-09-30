"""BinPVD model: RoBERTa/CodeBERT + Prefix Tuning + Classification Head.

The model implements prefix tuning by prepending virtual token embeddings
directly to the input embedding sequence, avoiding dependence on the
transformers DynamicCache internal format.

Three modes:
1. Full fine-tuning (prefix tuning disabled)
2. Prefix tuning (only prefix encoder + classifier trainable)
3. Evaluation with pretrained prefix tuning weights
"""

from typing import Optional
import torch
from torch import nn
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaClassificationHead(nn.Module):
    """Classification head for sentence-level tasks."""

    def __init__(self, hidden_size: int = 768, num_labels: int = 1, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features[:, 0, :]  # <s> token (CLS-equivalent)
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BinPVDModel(nn.Module):
    """BinPVD model: masked LM backbone + optional prefix tuning + classifier.

    Prefix tuning works by generating virtual prefix embeddings that are
    prepended to the input token embeddings. The extended embedding sequence
    is then passed through the full transformer without past_key_values.

    Args:
        model_name_or_path: Path or HF name for the base RoBERTa/CodeBERT model.
        num_labels: Number of classification labels (default 1 for binary).
        hidden_size: Transformer hidden size.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads per layer.
        dropout: Dropout rate for the classification head.
        prefix_enabled: Whether to use prefix tuning.
        prefix_length: Number of virtual prefix tokens.
        prefix_projection: Whether to use MLP projection in prefix encoder.
        encoder_hidden_size: Hidden size for prefix encoder MLP.
    """

    def __init__(
        self,
        model_name_or_path: str = "F:/PBVD/checkpoints/roberta-MLM-50",
        num_labels: int = 1,
        hidden_size: int = 768,
        dropout: float = 0.1,
        prefix_enabled: bool = True,
        prefix_length: int = 20,
        **kwargs,  # Accept unused params for config compatibility
    ):
        super().__init__()

        # Load base model
        try:
            self.base_model = RobertaModel.from_pretrained(
                model_name_or_path, add_pooling_layer=False,
            )
        except OSError:
            self.base_model = RobertaModel.from_pretrained(
                "microsoft/codebert-base", add_pooling_layer=False,
            )

        self.config = self.base_model.config
        self.hidden_size = hidden_size
        self.prefix_enabled = prefix_enabled
        self.prefix_length = prefix_length

        # Prefix tuning module: generates [prefix_length, hidden_size] embeddings
        if prefix_enabled:
            self.prefix_embeddings = nn.Parameter(
                torch.randn(1, prefix_length, hidden_size) * 0.02,
            )
            # Extend position embeddings to accommodate prefix + input
            embed_layer = self.base_model.embeddings
            old_max_len = self.base_model.config.max_position_embeddings
            self._effective_max_len = old_max_len + prefix_length
            hidden = embed_layer.position_embeddings.embedding_dim

            old_weight = embed_layer.position_embeddings.weight.data
            new_emb = nn.Embedding(self._effective_max_len, hidden)
            new_emb.weight.data[:old_max_len] = old_weight
            nn.init.normal_(new_emb.weight.data[old_max_len:], mean=0.0, std=0.02)
            embed_layer.position_embeddings = new_emb
            self.base_model.config.max_position_embeddings = self._effective_max_len
        else:
            self.prefix_embeddings = None

        # Classification head
        self.classifier = RobertaClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

    def freeze_base_model(self):
        """Freeze base model parameters; keep prefix embeddings + classifier trainable."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        if self.prefix_embeddings is not None:
            self.prefix_embeddings.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        labels: Optional[torch.FloatTensor] = None,
    ) -> SequenceClassifierOutput:
        batch_size = input_ids.shape[0]

        # --- Prefix tuning: prepend virtual embeddings ---
        if self.prefix_enabled and self.prefix_embeddings is not None:
            prefix_length = self.prefix_embeddings.size(1)
            seq_len = input_ids.size(1)

            # Get token-only embeddings (no position info yet)
            token_embeds = self.base_model.embeddings.word_embeddings(input_ids)
            # token_embeds shape: [batch_size, seq_len, hidden_size]

            # Expand prefix embeddings to batch
            prefix_embeds = self.prefix_embeddings.expand(batch_size, -1, -1)

            # Concatenate: [batch_size, prefix_length + seq_len, hidden_size]
            combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            total_len = prefix_length + seq_len

            # Adjust attention mask: prefix tokens are always attended to
            prefix_mask = torch.ones(
                batch_size, prefix_length,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # Explicit position_ids for the full sequence (bypass buffer size limits)
            position_ids = torch.arange(total_len, device=input_ids.device).unsqueeze(0)
            token_type_ids = torch.zeros(batch_size, total_len, dtype=torch.long, device=input_ids.device)

            # Forward with combined embeddings and explicit position/token_type IDs
            outputs = self.base_model(
                inputs_embeds=combined_embeds,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )
        else:
            # Standard forward pass
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # --- Classification ---
        # [CLS] token position: at `prefix_length` with prefix enabled (shifted by prepended tokens),
        # otherwise at position 0.
        sequence_output = outputs.last_hidden_state
        cls_pos = self.prefix_length if (self.prefix_enabled and self.prefix_embeddings is not None) else 0
        cls_hidden = sequence_output[:, cls_pos, :]  # [batch_size, hidden_size]
        logits = self.classifier.fc1(cls_hidden)
        logits = self.classifier.dropout(logits)
        logits = torch.tanh(logits)
        logits = self.classifier.dropout(logits)
        logits = self.classifier.fc2(logits)  # [batch_size, num_labels]

        # Loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.squeeze(-1), labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
