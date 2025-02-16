import math
from typing import List

import torch
import torch.nn as nn


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
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) *
                             (-math.log(10000.0) / self.d_model))

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
    A simple transformer-based autoencoder with token masking capability.

    This autoencoder uses transformer encoder layers for both encoding and decoding,
    with the ability to mask out tokens based on their importance scores.

    Args:
        d_model (int): Dimension of the model
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        n_mask (int): Number of tokens to mask
        norm (nn.Module, optional): Normalization layer
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 n_mask: int,
                 norm: nn.Module = None
                 ):
        super(BabyAutoencoder, self).__init__()

        self.pos_encoder = SinusoidalPositionalEncoding(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True
            ),
            num_layers=num_layers,
            norm=norm
        )
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True
            ),
            num_layers=num_layers,
            norm=norm
        )

        self.n_mask = n_mask

        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor):
        """
        Performs full autoencoding cycle (encode + decode).

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Reconstructed input
        """
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.activation(x)

        if self.n_mask > 0:
            # shape: [B, L]
            scores = x.norm(dim=2)  # or x.mean(dim=2), etc.

            # topk along dimension=1 (the length dimension)
            _, token_indices = torch.topk(scores, self.n_mask, dim=1, largest=False)  # shape [B, n_mask]

            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)  # shape [B, L]
            mask.scatter_(1, token_indices, True)
            mask = mask.unsqueeze(-1)  # shape [B, L, 1] => broadcastable to [B, L, D]
            x = x.masked_fill(mask, 0.0)

        x = self.decoder(x)

        return x

    def encode(self, x: torch.Tensor):
        """
        Encodes the input, including token masking.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Encoded representation
        """
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.activation(x)

        if self.n_mask > 0:
            # shape: [B, L]
            scores = x.norm(dim=2)  # or x.mean(dim=2), etc.

            # topk along dimension=1 (the length dimension)
            _, token_indices = torch.topk(scores, self.n_mask, dim=1, largest=False)  # shape [B, n_mask]

            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)  # shape [B, L]
            mask.scatter_(1, token_indices, True)
            mask = mask.unsqueeze(-1)  # shape [B, L, 1] => broadcastable to [B, L, D]
            x = x.masked_fill(mask, 0.0)

        return x

    def decode(self, x: torch.Tensor):
        """
        Decodes the encoded representation.

        Args:
            x (torch.Tensor): Encoded tensor [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Decoded output
        """
        x = self.decoder(x)
        return x


class FissionModule(nn.Module):
    """
    Module that splits input sequence into multiple node-specific representations.

    Uses learned queries and multi-head attention to create different views of the
    input sequence for each node in the network.

    Args:
        num_nodes (int): Number of output nodes
        par_length (int): Length of each partition
        embedding_dim (int): Dimension of embeddings
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    def __init__(self, num_nodes: int, par_length: int, embedding_dim: int, num_heads: int, dropout: float):
        super(FissionModule, self).__init__()
        self.num_nodes = num_nodes

        # Initialize learned queries for each node
        self.learned_queries = nn.Parameter(torch.randn(num_nodes, par_length, embedding_dim))

        # Layer normalization before attention
        self.norm = nn.LayerNorm(embedding_dim)

        # Projections for Q, K, V
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        # Single multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection and normalization
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits input sequence into multiple node-specific representations.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            torch.Tensor: Node-specific representations [batch_size, num_nodes, seq_len, embedding_dim]
        """
        batch_size, seq_len, dim = x.shape

        # Normalize input
        x = self.norm(x)

        # Project keys and values
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Prepare queries for all nodes at once
        # Shape: [B * N, L_q, D]
        queries = self.learned_queries.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        ).reshape(batch_size * self.num_nodes, -1, dim)

        # Expand keys and values for parallel processing
        # Shape: [B * N, L, D]
        expanded_key = key.unsqueeze(1).expand(
            -1, self.num_nodes, -1, -1
        ).reshape(batch_size * self.num_nodes, seq_len, dim)

        expanded_value = value.unsqueeze(1).expand(
            -1, self.num_nodes, -1, -1
        ).reshape(batch_size * self.num_nodes, seq_len, dim)

        # Compute attention for all nodes in parallel
        attn_output, _ = self.attention(
            query=queries,
            key=expanded_key,
            value=expanded_value
        )

        # Process output
        attn_output = self.output_norm(queries + attn_output)

        # Reshape back to [B, N, L_q, D]
        output = attn_output.reshape(
            batch_size, self.num_nodes, -1, dim
        )

        return output


class FusionModule(nn.Module):
    """
    Splits input sequence into multiple node-specific representations.

    Args:
        x (torch.Tensor): Input tensor [batch_size, seq_len, embedding_dim]

    Returns:
        torch.Tensor: Node-specific representations [batch_size, num_nodes, seq_len, embedding_dim]
    """
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super(FusionModule, self).__init__()

        # Learned query for fusion
        self.learned_query = nn.Parameter(torch.randn(embedding_dim))

        # Input normalization
        self.norm = nn.LayerNorm(embedding_dim)

        # Projections for K, V
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output processing
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Combines multiple node representations into a single sequence.

        Args:
            x (torch.Tensor): Input tensor [batch_size, num_nodes, seq_len, embedding_dim]
            seq_len (int): Desired output sequence length

        Returns:
            torch.Tensor: Fused representation [batch_size, seq_len, embedding_dim]
        """
        batch_size, num_nodes, node_seq_len, dim = x.shape

        # Normalize and reshape input
        x = x.view(batch_size, -1, dim)

        x = self.norm(x)

        # Project keys and values
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Prepare query
        query = self.learned_query.view(1, 1, -1).expand(batch_size, seq_len, -1)

        # Apply attention
        attn_output, _ = self.attention(
            query=query,
            key=key,
            value=value
        )

        # Process output
        output = self.output_proj(attn_output)
        output = self.dropout(output)
        output = self.output_norm(output + query)  # Residual connection

        return output




class HAMABlock(nn.Module):
    """
    Hierarchical Attention-based Masked Autoencoder Block.

    Combines fission, node-specific processing, and fusion operations into a single
    processing block. Each block can mask different numbers of tokens and process
    them through multiple parallel autoencoders.

    Args:
        num_nodes (int): Number of parallel processing nodes
        partition_length (int): Length of each partition
        embedding_dim (int): Dimension of embeddings
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        num_layers (int): Number of transformer layers
        n_mask (int): Number of tokens to mask
        norm (nn.Module, optional): Normalization layer
    """
    def __init__(self, num_nodes: int, partition_length: int, embedding_dim: int, num_heads: int, dropout: float, num_layers: int, n_mask: int, norm: nn.Module = None):
        super(HAMABlock, self).__init__()
        self.n_mask = n_mask
        self.num_nodes = num_nodes

        self.autoencoders: nn.ModuleList = nn.ModuleList([
            BabyAutoencoder(
                d_model=embedding_dim,
                nhead=num_heads,
                num_layers=num_layers,
                n_mask=n_mask,
                norm=norm
            )
            for _ in range(num_nodes)
        ])

        self.fission_module = FissionModule(
            num_nodes=num_nodes,
            par_length=partition_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fusion_module = FusionModule(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor):
        """
        Performs complete block processing (encode + decode).

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            torch.Tensor: Processed output
        """
        # x has shape [B, L, D]
        B = x.shape[0]
        L = x.shape[1]
        D = x.shape[2]

        x = self.fission_module(x)
        for i in range(self.num_nodes):
            x[:, i] = self.autoencoders[i](x[:, i])
        x = self.fusion_module(x, L)
        return x

    def encode(self, x: torch.Tensor):
        """
        Encodes input through fission, node processing, and fusion.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            torch.Tensor: Encoded representation
        """
        # x has shape [B, L, D]
        B = x.shape[0]
        L = x.shape[1]
        D = x.shape[2]

        x = self.fission_module(x)
        for i in range(self.num_nodes):
            x[:, i] = self.autoencoders[i].encode(x[:, i])
        x = self.fusion_module(x, L - self.num_nodes * self.n_mask)
        return x

    def decode(self, x: torch.Tensor):
        """
        Decodes representation through fission, node processing, and fusion.

        Args:
            x (torch.Tensor): Encoded tensor [batch_size, seq_len, embedding_dim]

        Returns:
            torch.Tensor: Decoded output
        """
        # x has shape [B, L, D]
        B = x.shape[0]
        L = x.shape[1]
        D = x.shape[2]

        x = self.fission_module(x)
        for i in range(self.num_nodes):
            x[:, i] = self.autoencoders[i].decode(x[:, i])
        x = self.fusion_module(x, L + self.num_nodes * self.n_mask)
        return x


class HAMA(nn.Module):
    """
    "Hama early, Hama often." - Joseph Anderson, 2021

    Hierarchical Attention-based Masked Autoencoder (HAMA).

    HAMA is an advanced transformer-based architecture that combines hierarchical processing,
    parallel masked autoencoders, and attention mechanisms to learn robust representations
    of sequential data. The architecture is designed to capture information at multiple
    scales while maintaining computational efficiency.

    Key Architectural Components:
    1. Hierarchical Processing:
        - Multiple layers process data at different scales
        - Each layer can operate on different sequence lengths and feature granularities
        - Information flows both up (encoding) and down (decoding) the hierarchy

    2. Parallel Processing Nodes:
        - Each layer contains multiple parallel processing nodes
        - Nodes can specialize in different aspects of the input
        - Node count typically decreases in higher layers (controlled by scaling factor)

    3. Masked Autoencoding:
        - Uses learned masking strategies to force robust feature learning
        - Mask tokens based on their importance scores
        - Progressive masking through layers helps build hierarchical representations

    4. Fission-Fusion Mechanism:
        - Fission: Splits input into multiple parallel streams using learned queries
        - Node Processing: Each stream processed by specialized autoencoder
        - Fusion: Recombines processed streams using attention mechanism

    Training Dynamics:
        - Bottom layers learn fine-grained features with more parallel processors
        - Middle layers learn intermediate representations
        - Top layers learn high-level abstract features with fewer processors
        - Masking forces the model to learn robust and redundant representations
        - Bidirectional information flow allows for both compression and reconstruction

    Advantages:
        - Scalable to long sequences through hierarchical processing
        - Robust to noise through masked training
        - Flexible architecture through configurable scaling factors
        - Efficient parallel processing through node specialization
        - Capable of both lossy and lossless reconstruction depending on configuration

    Args:
        embedding_dim (int): Dimension of embeddings used throughout the model
        num_heads (int): Number of attention heads in transformer layers
        dropout (float): Dropout rate for regularization
        num_layers (int): Number of hierarchical HAMA layers
        transformer_layers (int): Number of transformer layers per autoencoder
        norm (nn.Module, optional): Normalization layer for transformers
        initial_num_nodes (int, optional): Initial number of parallel processing nodes
        initial_partition_length (int, optional): Initial sequence length per partition
        initial_n_mask (int, optional): Initial number of tokens to mask
        num_nodes_scaling_factor (float, optional): Factor to scale node count between layers
            (typically < 1 to reduce nodes in higher layers)
        partition_length_scaling_factor (float, optional): Factor to scale partition length
            (typically > 1 to increase receptive field in higher layers)
        n_mask_scaling_factor (float, optional): Factor to scale masking between layers
        nodes_per_layer (List[int], optional): Explicit number of nodes for each layer
        partition_lengths (List[int], optional): Explicit partition lengths for each layer
        n_masks (List[int], optional): Explicit number of masks for each layer

    Note:
        The architecture can be configured either through:
        1. Initial values and scaling factors (for geometric progression)
        2. Explicit per-layer lists (for custom progression)
        But not both simultaneously.

    Example scaling configuration:
        For a 3-layer HAMA with:
        - initial_num_nodes = 8
        - num_nodes_scaling_factor = 0.5
        - initial_partition_length = 64
        - partition_length_scaling_factor = 2.0

        The resulting structure would be:
        Layer 0: 8 nodes, partition length 64
        Layer 1: 4 nodes, partition length 128
        Layer 2: 2 nodes, partition length 256

    Implementation Note:
        This implementation follows the principle of "divide-and-conquer" through its
        fission-fusion mechanism, allowing parallel processing while maintaining the
        ability to capture long-range dependencies through the hierarchy.

    Example usage:
    >>> model = HAMA(
    ...     embedding_dim=128,
    ...     num_heads=8,
    ...     dropout=0.1,
    ...     num_layers=4,
    ...     transformer_layers=2,
    ...     initial_num_nodes=4,
    ...     initial_partition_length=32,
    ...     initial_n_mask=4,
    ...     num_nodes_scaling_factor=0.5,
    ...     partition_length_scaling_factor=2.0,
    ...     n_mask_scaling_factor=1.0
    ... )
    >>> input_tensor = torch.randn(2, 64, 128)  # [batch_size, seq_len, embedding_dim]
    >>> output = model(input_tensor)
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            dropout: float,
            num_layers: int,
            transformer_layers: int,
            norm: nn.Module = None,
            initial_num_nodes: int = None,
            initial_partition_length: int = None,
            initial_n_mask: int = None,
            num_nodes_scaling_factor: float = 0.5,
            partition_length_scaling_factor: float = 2.0,
            n_mask_scaling_factor: float = 1.0,
            nodes_per_layer: List[int] = None,
            partition_lengths: List[int] = None,
            n_masks: List[int] = None
    ):
        super(HAMA, self).__init__()

        # Validation
        if initial_num_nodes is None and nodes_per_layer is None:
            raise ValueError("Either initial_num_nodes or nodes_per_layer must be provided.")
        if initial_partition_length is None and partition_lengths is None:
            raise ValueError("Either initial_partition_length or partition_lengths must be provided.")
        if initial_n_mask is None and n_masks is None:
            raise ValueError("Either initial_n_mask or n_masks must be provided.")

        # Check that if one of the scaling factors is provided, the corresponding parameter is not provided and vice versa
        if nodes_per_layer is not None and initial_num_nodes is not None:
            raise ValueError("Cannot provide both nodes_per_layer and initial_num_nodes.")
        if partition_lengths is not None and initial_partition_length is not None:
            raise ValueError("Cannot provide both partition_lengths and initial_partition_length.")
        if n_masks is not None and initial_n_mask is not None:
            raise ValueError("Cannot provide both n_masks and initial_n_mask.")

        # Check that nodes_per_layer, partition_lengths, and n_masks have the length of num_layers
        if nodes_per_layer is not None and len(nodes_per_layer) != num_layers:
            raise ValueError("nodes_per_layer must have the same length as num_layers.")
        if partition_lengths is not None and len(partition_lengths) != num_layers:
            raise ValueError("partition_lengths must have the same length as num_layers.")
        if n_masks is not None and len(n_masks) != num_layers:
            raise ValueError("n_masks must have the same length as num_layers.")

        # Additional validation for scaling factors
        if num_nodes_scaling_factor <= 0:
            raise ValueError("num_nodes_scaling_factor must be positive")
        if partition_length_scaling_factor <= 0:
            raise ValueError("partition_length_scaling_factor must be positive")
        if n_mask_scaling_factor <= 0:
            raise ValueError("n_mask_scaling_factor must be positive")

        # Validate embedding dimension and heads
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        # Validate transformer layers
        if transformer_layers < 1:
            raise ValueError("transformer_layers must be at least 1")

        # Calculate per-layer parameters
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
            n_masks = [
                max(1, int(initial_n_mask * (n_mask_scaling_factor ** i)))
                for i in range(num_layers)
            ]

        # Validate partition sizes against masking
        for i in range(num_layers):
            total_masked_tokens = sum(n_masks[j] for j in range(i + 1))
            remaining_tokens = partition_lengths[i] - total_masked_tokens
            if remaining_tokens <= 0:
                raise ValueError(
                    f"Layer {i}: After masking {total_masked_tokens} tokens, "
                    f"partition length {partition_lengths[i]} is too small. "
                    f"Need at least {total_masked_tokens + 1} tokens."
                )

        # Store configuration
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.nodes_per_layer = nodes_per_layer
        self.partition_lengths = partition_lengths
        self.n_masks = n_masks

        # Create layers
        self.layers = nn.ModuleList([
            HAMABlock(
                num_nodes=nodes_per_layer[i],
                partition_length=partition_lengths[i],
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_layers=transformer_layers,  # Now using the configurable parameter
                n_mask=n_masks[i],
                norm=norm
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        """
        Performs complete forward pass through all HAMA layers.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            torch.Tensor: Reconstructed output
        """
        for layer in self.layers:
            x = layer.encode(x)
        for layer in reversed(self.layers):
            x = layer.decode(x)
        return x

    def encode(self, x: torch.Tensor):
        """
        Encodes input through all HAMA layers.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            torch.Tensor: Encoded representation
        """
        for layer in self.layers:
            x = layer.encode(x)
        return x

    def decode(self, x: torch.Tensor):
        """
        Decodes representation through all HAMA layers in reverse order.

        Args:
            x (torch.Tensor): Encoded tensor [batch_size, seq_len, embedding_dim]

        Returns:
            torch.Tensor: Decoded output
        """
        for layer in reversed(self.layers):
            x = layer.decode(x)
        return x
