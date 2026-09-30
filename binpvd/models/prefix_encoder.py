"""Prefix encoder module for BinPVD.

Implements the prompt-based prefix tuning mechanism:
1. Embed prefix tokens via a learnable embedding table
2. Optionally project through a 2-layer MLP (tanh)
3. Reshape into past_key_values for transformer attention layers
"""

import torch
from torch import nn


class PrefixEncoder(nn.Module):
    """Encoder that generates prefix (past_key_values) for transformer layers.

    Args:
        prefix_length: Number of virtual prefix tokens.
        hidden_size: Transformer hidden size (token embedding dim).
        encoder_hidden_size: MLP intermediate size.
        num_hidden_layers: Number of transformer layers (each gets 2 keys).
        num_attention_heads: Number of attention heads.
        prefix_projection: Whether to use MLP projection after embedding.
    """

    def __init__(
        self,
        prefix_length: int = 20,
        hidden_size: int = 768,
        encoder_hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        prefix_projection: bool = True,
    ):
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        # Learnable prefix embeddings
        self.embedding = nn.Embedding(prefix_length, hidden_size)

        # Optional MLP projection
        if prefix_projection:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(encoder_hidden_size, num_hidden_layers * 2 * hidden_size),
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate prefix past_key_values.

        Returns:
            Tensor of shape
            (num_hidden_layers * 2, batch_size, num_attention_heads, prefix_length, head_dim)
            split into list of 2 tensors per layer for past_key_values format.
        """
        # Create prefix token indices [prefix_length]
        prefix_indices = torch.arange(self.prefix_length, device=device)
        # Expand to [batch_size, prefix_length]
        prefix_tokens = prefix_indices.unsqueeze(0).expand(batch_size, -1)

        # Embed: [batch_size, prefix_length, hidden_size]
        prefix_embeds = self.embedding(prefix_tokens)
        # Project: [batch_size, prefix_length, num_hidden_layers * 2 * hidden_size]
        past_key_values = self.mlp(prefix_embeds)

        # Reshape to transformer past_key_values format
        # [batch_size, prefix_length, num_hidden_layers * 2, num_attention_heads, head_dim]
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_length,
            self.num_hidden_layers * 2,
            self.num_attention_heads,
            self.head_dim,
        )
        # Permute to: [num_hidden_layers * 2, batch_size, num_attention_heads, prefix_length, head_dim]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
        # Split into list of 2 per layer: each (batch_size, num_attention_heads, prefix_length, head_dim)
        return past_key_values.split(2)
