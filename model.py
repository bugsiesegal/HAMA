import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from routing import CrossAttentionRouter, CrossAttentionGather


class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for transformer models.

    This module adds positional information to input embeddings using sine and cosine
    functions of different frequencies, allowing the model to learn about token positions
    in a sequence without requiring training.

    Args:
        d_model (int): The dimension of the model's embeddings
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def _get_positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates positional encodings for a sequence.

        Args:
            seq_len (int): Length of the sequence
            device (torch.device): Device to create the encodings on

        Returns:
            torch.Tensor: Positional encodings of shape [seq_len, d_model]
        """
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=device)
            * (-math.log(10000.0) / self.d_model)
        )

        pos_encoding = torch.zeros(seq_len, self.d_model, device=device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Input with positional encodings added
        """
        seq_len = x.size(1)
        pos_encoding = self._get_positional_encoding(seq_len, x.device)
        return x + pos_encoding.unsqueeze(0)

class BabyAutoencoder(nn.Module):
    """
    A simple transformer-based autoencoder with optional token masking.

    This autoencoder uses transformer encoder layers for both encoding and decoding.
    If `use_masking=True`, it can mask out tokens based on their importance scores.

    Args:
        d_model (int): Dimension of the model
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        n_mask (int): Number of tokens to mask (ignored if use_masking=False)
        use_masking (bool): Whether to apply importance-based token masking
        norm (nn.Module, optional): Normalization layer
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        n_mask: int,
        use_masking: bool = True,
        norm: nn.Module = None
    ):
        super(BabyAutoencoder, self).__init__()

        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        self.encoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True
            ),
            num_layers=num_layers,
            norm=norm
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True
            ),
            num_layers=num_layers,
            norm=norm
        )

        self.n_mask = n_mask
        self.use_masking = use_masking

        self.activation = nn.Identity()

        self.masking_token = nn.Parameter(torch.randn(1, n_mask, d_model), requires_grad=True) if use_masking else None

    def forward(self, x: torch.Tensor):
        """
        Performs full autoencoding cycle (encode + decode).

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Reconstructed input
        """
        # --- Encoding ---
        x = self.pos_encoder(x)
        x = self.encoder(x, x)  # Transformer decoder expects two inputs
        x = self.activation(x)

        if self.use_masking and self.n_mask > 0:
            x[:, :self.n_mask, :] = self.masking_token

        # --- Decoding ---
        x = self.decoder(x, x)

        return x

    def encode(self, x: torch.Tensor):
        """
        Encodes the input, including optional token masking.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Encoded representation
        """
        x = self.pos_encoder(x)
        x = self.encoder(x, x)
        x = self.activation(x)

        if self.use_masking and self.n_mask > 0:
            x[:, :self.n_mask, :] = self.masking_token

        return x

    def decode(self, x: torch.Tensor):
        """
        Decodes the encoded representation.

        Args:
            x (torch.Tensor): Encoded tensor [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Decoded output
        """
        return self.decoder(x, x)


class HAMABlock(nn.Module):
    """
    Hierarchical Attention-based Masked Autoencoder Block.

    Combines fission, node-specific processing, and fusion operations into a single
    block. Each block can optionally mask tokens in its parallel autoencoders,
    or skip masking entirely. The fusion step can optionally compress the fused
    output (via chunk-based average pooling).

    Args:
        num_nodes (int): Number of parallel processing nodes
        partition_length (int): Length of each partition (for fission queries)
        embedding_dim (int): Dimension of embeddings
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        num_layers (int): Number of transformer layers in each BabyAutoencoder
        n_mask (int): Number of tokens to mask (if use_masking=True)
        use_masking (bool): Whether to apply token masking in BabyAutoencoders
        compression_factor (int): Factor to compress the fused output
        norm (nn.Module, optional): Normalization layer
    """
    def __init__(self, *, num_nodes, partition_length, embedding_dim,
                 num_heads, dropout, num_layers, n_mask, use_masking,
                 compression_factor, norm):
        super().__init__()

        self.router = CrossAttentionRouter(
            num_nodes=num_nodes,
            partition_length=partition_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.gather = CrossAttentionGather(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Baby AEs stay unchanged
        self.autoencoders = nn.ModuleList([
            BabyAutoencoder(
                d_model=embedding_dim,
                nhead=num_heads,
                num_layers=num_layers,
                n_mask=n_mask,
                use_masking=use_masking,
                norm=norm,
            )
            for _ in range(num_nodes)
        ])

    # ------------------------------------------------------------------
    def _run_nodes(self, node_repr: torch.Tensor) -> torch.Tensor:
        # node_repr: [B, N, P, D]
        B, N, P, D = node_repr.shape
        for i in range(N):
            node_repr[:, i] = self.autoencoders[i](node_repr[:, i])
        return node_repr

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor):
        node_repr, _ = self.router(x)
        node_repr = self._run_nodes(node_repr)
        fused = self.gather(node_repr, seq_len=x.size(1))

        if self.compression_factor > 1:
            fused = fused[:, :: self.compression_factor, :]
        return fused

    # ------------------------------------------------------------------
    def decode(self, x: torch.Tensor):
        # up‑sample if compression was applied
        if self.compression_factor > 1:
            x = torch.repeat_interleave(x, repeats=self.compression_factor, dim=1)

        node_repr, _ = self.router(x)
        for i in range(self.num_nodes):
            node_repr[:, i] = self.autoencoders[i].decode(node_repr[:, i])
        fused = self.gather(node_repr, seq_len=x.size(1))
        return fused

    # --------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Full forward pass: encode and decode the input.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embedding_dim]
        Returns:
            torch.Tensor: Reconstructed output
        """
        # Encoding
        x = self.encode(x)

        # Decoding
        x = self.decode(x)

        return x


class HAMA(nn.Module):
    """
    Hierarchical Attention-based Masked Autoencoder (HAMA) with optional masking and compression.

    Stacks multiple HAMABlock layers. Each block can:
      - optionally mask tokens in parallel baby autoencoders,
      - optionally compress the fused representation in the fusion step.

    Args:
        input_dim (int): Dimension of input features
        embedding_dim (int): Dimension of embeddings used throughout the model
        num_heads (int): Number of attention heads in transformer layers
        dropout (float): Dropout rate for regularization
        num_layers (int): Number of hierarchical HAMA layers
        transformer_layers (int): Number of transformer layers per autoencoder
        norm (nn.Module, optional): Normalization layer for transformers
        initial_num_nodes (int): Initial number of parallel nodes
        initial_partition_length (int): Initial sequence length per partition
        initial_n_mask (int): Number of tokens to mask in the first layer (if use_masking=True)
        use_masking (bool): Whether to apply token masking in all baby autoencoders
        compression_factor (int): Factor to compress fused outputs in each block
        ...
    """
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        transformer_layers: int,
        norm: nn.Module = None,
        initial_num_nodes: int = None,
        initial_partition_length: int = None,
        initial_n_mask: int = None,
        use_masking: bool = True,
        compression_factor: int = 1,
        num_nodes_scaling_factor: float = 0.5,
        partition_length_scaling_factor: float = 2.0,
        n_mask_scaling_factor: float = 1.0,
        nodes_per_layer: List[int] = None,
        partition_lengths: List[int] = None,
        n_masks: List[int] = None,
        activation: str = "relu",
    ):
        super(HAMA, self).__init__()

        # Validate that at least one approach is given
        if initial_num_nodes is None and nodes_per_layer is None:
            raise ValueError("Either initial_num_nodes or nodes_per_layer must be provided.")
        if initial_partition_length is None and partition_lengths is None:
            raise ValueError("Either initial_partition_length or partition_lengths must be provided.")
        if initial_n_mask is None and n_masks is None and use_masking:
            raise ValueError("Either initial_n_mask or n_masks must be provided when use_masking=True.")

        # If not masking, we can default all n_mask to 0 to skip validation
        if not use_masking:
            initial_n_mask = 0
            n_masks = [0]*num_layers if n_masks is None else n_masks

        # Check contradictory arguments
        if nodes_per_layer is not None and initial_num_nodes is not None:
            raise ValueError("Cannot provide both nodes_per_layer and initial_num_nodes.")
        if partition_lengths is not None and initial_partition_length is not None:
            raise ValueError("Cannot provide both partition_lengths and initial_partition_length.")
        if n_masks is not None and initial_n_mask is not None and use_masking:
            # not strictly an error if they match, but typically you'd pick only one approach
            pass

        # If user gave a per-layer list, check lengths
        if nodes_per_layer is not None and len(nodes_per_layer) != num_layers:
            raise ValueError("nodes_per_layer must have the same length as num_layers.")
        if partition_lengths is not None and len(partition_lengths) != num_layers:
            raise ValueError("partition_lengths must have the same length as num_layers.")
        if n_masks is not None and len(n_masks) != num_layers:
            raise ValueError("n_masks must have the same length as num_layers.")

        # Additional validation
        if num_nodes_scaling_factor <= 0:
            raise ValueError("num_nodes_scaling_factor must be positive")
        if partition_length_scaling_factor <= 0:
            raise ValueError("partition_length_scaling_factor must be positive")
        if n_mask_scaling_factor <= 0:
            raise ValueError("n_mask_scaling_factor must be positive")

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if transformer_layers < 1:
            raise ValueError("transformer_layers must be at least 1")

        # Derive per-layer configs
        if nodes_per_layer is None:
            nodes_per_layer = [
                max(1, int(initial_num_nodes * (num_nodes_scaling_factor ** i)))
                for i in range(num_layers)
            ]
        if partition_lengths is None:
            partition_lengths = [
                max(1, int(initial_partition_length * (partition_length_scaling_factor ** i)))
                for i in range(num_layers)
            ]

        if n_masks is None:
            # We only do this if use_masking=True
            n_masks = [
                max(0, int(initial_n_mask * (n_mask_scaling_factor ** i)))
                for i in range(num_layers)
            ]

        # If we are using masking, validate partition sizes
        if use_masking:
            for i in range(num_layers):
                # sum all masked tokens up to layer i
                total_masked_tokens = sum(n_masks[j] for j in range(i + 1))
                remaining_tokens = partition_lengths[i] - total_masked_tokens
                if remaining_tokens <= 0:
                    raise ValueError(
                        f"Layer {i}: After masking {total_masked_tokens} tokens, "
                        f"partition length {partition_lengths[i]} is too small. "
                        f"Need at least {total_masked_tokens + 1} tokens."
                    )

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.nodes_per_layer = nodes_per_layer
        self.partition_lengths = partition_lengths
        self.n_masks = n_masks
        self.use_masking = use_masking
        self.compression_factor = compression_factor
        self.input_dim = input_dim

        # Create HAMA blocks
        self.layers = nn.ModuleList([
            HAMABlock(
                num_nodes=nodes_per_layer[i],
                partition_length=partition_lengths[i],
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_layers=transformer_layers,
                n_mask=n_masks[i],
                use_masking=use_masking,
                compression_factor=compression_factor,
                norm=norm
            )
            for i in range(num_layers)
        ])

        # Input projection layer
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        # Output projection layer
        self.output_projection = nn.Linear(embedding_dim, input_dim)

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation == "identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x: torch.Tensor):
        """
        Full forward: encode up each layer, then decode down.

        x: [B, seq_len, embedding_dim]
        """
        # Input projection
        x = self.input_projection(x)  # [B, seq_len, embedding_dim]
        # Up (encode)
        for layer in self.layers:
            x = layer.encode(x)

        x = self.activation(x)

        # Down (decode)
        for layer in reversed(self.layers):
            x = layer.decode(x)

        # Final activation
        x = self.activation(x)
        # Output projection
        x = self.output_projection(x)

        return x

    def encode(self, x: torch.Tensor):
        """Encodes input through all HAMA layers."""
        # Input projection
        x = self.input_projection(x)  # [B, seq_len, embedding_dim]
        for layer in self.layers:
            x = layer.encode(x)
        # Final activation
        x = self.activation(x)
        return x

    def decode(self, x: torch.Tensor):
        """Decodes input through all HAMA layers in reverse order."""
        for layer in reversed(self.layers):
            x = layer.decode(x)
        # Final activation
        x = self.activation(x)
        # Output projection
        x = self.output_projection(x)
        return x


class HAMAForLanguageModeling(nn.Module):
    """
    HAMA model wrapped for language modeling tasks.

    This wrapper manages both the input embedding and output projection,
    keeping the base HAMA model focused purely on representation learning.

    You can toggle:
    - use_masking: bool
    - compression_factor: int
    etc. in the base model to determine the behavior.

    Example:
        hama_base = HAMA(
            embedding_dim=128,
            num_heads=8,
            dropout=0.1,
            num_layers=4,
            transformer_layers=2,
            initial_num_nodes=4,
            initial_partition_length=32,
            initial_n_mask=4,
            use_masking=False,              # no masking
            compression_factor=2,          # compress by factor 2 in fusion
        )
        model = HAMAForLanguageModeling(
            hama_base=hama_base,
            vocab_size=30522,
            embedding_dim=128
        )
    """

    def __init__(self, hama_base: HAMA, vocab_size: int, embedding_dim: int):
        super(HAMAForLanguageModeling, self).__init__()
        # Basic attributes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # The underlying HAMA model
        self.hama_base = hama_base
        self.layers = hama_base.layers  # direct reference

        # Language-model embedding & head
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

        # Copy other attributes for convenience
        self.num_layers = getattr(hama_base, 'num_layers', None)
        self.num_heads = getattr(hama_base, 'num_heads', None)
        self.transformer_layers = getattr(hama_base, 'transformer_layers', None)
        self.nodes_per_layer = getattr(hama_base, 'nodes_per_layer', None)
        self.partition_lengths = getattr(hama_base, 'partition_lengths', None)
        self.n_masks = getattr(hama_base, 'n_masks', None)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for language modeling.

        input_ids: [B, seq_len]
        Returns: logits [B, final_seq_len, vocab_size]
        """
        embedded = self.embedding(input_ids)
        hidden_states = self.hama_base(embedded)
        logits = self.lm_head(hidden_states)
        return logits

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode input tokens to hidden states."""
        embedded = self.embedding(input_ids)
        return self.hama_base.encode(embedded)

    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Decode hidden states to output logits."""
        decoded_repr = self.hama_base.decode(hidden_states)
        return self.lm_head(decoded_repr)