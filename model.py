import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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

def build_1d_sin_encoding(seq_len, d_model, device):
    """
    Dynamically builds a 1D sinusoidal position encoding of shape [seq_len, d_model].
    """
    if d_model % 2 != 0:
        raise ValueError("d_model should be even.")

    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )

    # [seq_len, d_model]
    position = torch.arange(seq_len, device=device).float().unsqueeze(1)
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def build_2d_sin_encoding(H, W, d_model, device):
    """
    Dynamically builds a 2D sinusoidal position encoding of shape [H, W, d_model].
    Splits d_model in half for rows, half for columns, then merges them.
    """
    if d_model % 2 != 0:
        raise ValueError("d_model should be even.")

    d_h = d_model // 2
    d_w = d_model // 2

    row_positions = torch.arange(H, device=device).unsqueeze(1)  # [H,1]
    col_positions = torch.arange(W, device=device).unsqueeze(1)  # [W,1]

    # Frequencies
    div_term_h = torch.exp(
        torch.arange(0, d_h, 2, device=device).float() * (-math.log(10000.0) / d_h)
    )
    div_term_w = torch.exp(
        torch.arange(0, d_w, 2, device=device).float() * (-math.log(10000.0) / d_w)
    )

    # [H, d_h]
    row_enc = torch.zeros(H, d_h, device=device)
    row_enc[:, 0::2] = torch.sin(row_positions * div_term_h)
    row_enc[:, 1::2] = torch.cos(row_positions * div_term_h)

    # [W, d_w]
    col_enc = torch.zeros(W, d_w, device=device)
    col_enc[:, 0::2] = torch.sin(col_positions * div_term_w)
    col_enc[:, 1::2] = torch.cos(col_positions * div_term_w)

    # Expand to [H, W, d_model]
    # First half is row, second half is col
    # row_enc => [H, d_h], col_enc => [W, d_w]
    # We'll broadcast row_enc along W, col_enc along H
    row_enc = row_enc.unsqueeze(1)  # [H,1,d_h]
    col_enc = col_enc.unsqueeze(0)  # [1,W,d_w]

    # final: [H, W, d_model]
    pe = torch.zeros(H, W, d_model, device=device)
    pe[:, :, :d_h] = row_enc  # broadcast along width dimension
    pe[:, :, d_h:] = col_enc  # broadcast along height dimension
    return pe

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
        self.use_masking = use_masking

        self.activation = nn.Identity()

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
        x = self.encoder(x)
        x = self.activation(x)

        if self.use_masking and self.n_mask > 0:
            # shape: [B, L]
            scores = x.norm(dim=2)  # or x.mean(dim=2), etc.

            # topk along dimension=1 (the length dimension)
            _, token_indices = torch.topk(scores, self.n_mask, dim=1, largest=False)  # shape [B, n_mask]

            mask = torch.zeros(x.shape[:2], dtype=torch.int, device=x.device)
            mask.scatter_(1, token_indices, 1)
            mask = mask.unsqueeze(-1).to(x.dtype)
            x = x.masked_fill(mask.bool(), 0.0)

        # --- Decoding ---
        x = self.decoder(x)

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
        x = self.encoder(x)
        x = self.activation(x)

        if self.use_masking and self.n_mask > 0:
            scores = x.norm(dim=2)
            _, token_indices = torch.topk(scores, self.n_mask, dim=1, largest=False)
            mask = torch.zeros(x.shape[:2], dtype=torch.int, device=x.device)
            mask.scatter_(1, token_indices, 1)
            mask = mask.unsqueeze(-1).to(x.dtype)
            x = x.masked_fill(mask.bool(), 0.0)

        return x

    def decode(self, x: torch.Tensor):
        """
        Decodes the encoded representation.

        Args:
            x (torch.Tensor): Encoded tensor [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Decoded output
        """
        return self.decoder(x)


class FissionModule(nn.Module):
    """
    Splits input sequence into multiple node-specific overlapping representations using unfold.
    Calculates step size dynamically to distribute num_nodes patches across the sequence length.

    Args:
        num_nodes (int): Number of output nodes/patches.
        par_length (int): Length of each partition/patch.
        embedding_dim (int): Dimension of embeddings.
    """
    def __init__(
        self,
        num_nodes: int,
        par_length: int,
        embedding_dim: int,
    ):
        super(FissionModule, self).__init__()
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if par_length <= 0:
             raise ValueError("par_length must be positive")

        self.num_nodes = num_nodes
        self.par_length = par_length
        self.embedding_dim = embedding_dim

        self.norm = nn.LayerNorm(embedding_dim)
        # Optional: Add PE here or ensure it's added before passing to HAMA block
        self.pos_encoder = SinusoidalPositionalEncoding(embedding_dim)


    def calculate_step(self, seq_len: int) -> int:
        """Calculates step size to spread num_nodes patches over seq_len."""
        if self.num_nodes == 1:
            # Only one patch needed, step doesn't really matter but should be >= 1
            return 1
        else:
            # Calculate ideal step: step = (L - P) / (N - 1)
            # Ensure seq_len > par_length for meaningful overlap calculation
            if seq_len <= self.par_length:
                # Sequence is too short for overlap with multiple nodes based on formula.
                # Default to step=1, unfold will generate fewer patches than num_nodes.
                # Or potentially raise an error if this case is invalid for the model logic.
                # For now, let's use step=1. Fusion needs to handle this.
                 return 1
            else:
                # Use max(1, ...) to prevent step=0 if L=P and N>1
                step = max(1, int(math.floor((seq_len - self.par_length) / (self.num_nodes - 1))))
                return step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, L, D]
        batch_size, seq_len, dim = x.shape

        # --- Preprocessing ---
        x_norm = self.norm(x)
        x_processed = self.pos_encoder(x_norm) # Apply PE *after* norm? Or before? Check convention. Let's assume after.

        # --- Calculate Dynamic Parameters ---
        # Adjust par_length if sequence is too short
        effective_par_length = min(seq_len, self.par_length)
        if effective_par_length <= 0: # Handle empty sequence case
             # Return empty tensor matching expected dims but with 0 nodes/patches? Or error?
             # Returning tensor of shape [B, num_nodes, par_length, D] filled with zeros might be safest downstream.
             print(f"Warning: Input seq_len is {seq_len}, cannot create patches. Returning zeros.")
             return torch.zeros(batch_size, self.num_nodes, self.par_length, dim, device=x.device, dtype=x.dtype)


        step = self.calculate_step(seq_len)

        # --- Unfold ---
        # unfold needs contiguous tensor
        patches = x_processed.contiguous().unfold(dimension=1, size=effective_par_length, step=step)
        # Output shape: [B, num_generated_patches, D, effective_par_length]

        num_generated_patches = patches.shape[1]

        # Permute D to the end: [B, num_generated_patches, effective_par_length, D]
        patches = patches.permute(0, 1, 3, 2)

        # --- Adjust Number of Patches to Match num_nodes ---
        output_tensor = torch.zeros(batch_size, self.num_nodes, self.par_length, dim,
                                     dtype=patches.dtype, device=patches.device)

        if num_generated_patches == 0 and self.num_nodes > 0:
             # Should be caught by effective_par_length check, but as safeguard
             print(f"Warning: unfold generated 0 patches for seq_len {seq_len}, size {effective_par_length}, step {step}.")
             return output_tensor # Return zeros

        # Copy generated patches, truncating or padding as needed
        num_to_copy = min(num_generated_patches, self.num_nodes)
        len_to_copy = min(effective_par_length, self.par_length) # Should usually be par_length unless L was short

        output_tensor[:, :num_to_copy, :len_to_copy, :] = patches[:, :num_to_copy, :len_to_copy, :]

        # output_tensor shape: [B, num_nodes, par_length, D]
        # Note: If num_generated_patches < num_nodes, trailing nodes will be zeros.
        # Note: If L < par_length, trailing sequence dim will be zeros.
        return output_tensor


class FusionModule(nn.Module):
    """
    Fuses multiple overlapping node-specific representations back into
    a single sequence using overlap-add logic.

    Args:
        num_nodes (int): Number of input nodes/patches.
        par_length (int): Length of each input partition/patch.
        embedding_dim (int): Dimension of embeddings.
    """
    def __init__(
        self,
        num_nodes: int, # Need num_nodes and par_length to recalculate step
        par_length: int,
        embedding_dim: int,
    ):
        super(FusionModule, self).__init__()
        self.num_nodes = num_nodes
        self.par_length = par_length
        self.embedding_dim = embedding_dim # Store if needed

        self.norm = nn.LayerNorm(embedding_dim)
        # No PE needed here - applied before Fission

    def calculate_step(self, seq_len: int) -> int:
        """Recalculates step size used by Fission."""
        # This MUST match the logic in FissionModule!
        if self.num_nodes == 1:
            return 1
        else:
            if seq_len <= self.par_length:
                 return 1
            else:
                step = max(1, int(math.floor((seq_len - self.par_length) / (self.num_nodes - 1))))
                return step

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Combines overlapping node representations using overlap-add.

        Args:
            x (torch.Tensor): Output from BabyAEs [B, num_nodes, par_length, D]
            seq_len (int): Original input sequence length (L)

        Returns:
            fused (torch.Tensor): [B, L, D]
        """
        batch_size, num_nodes_in, node_seq_len, dim = x.shape
        output_len = seq_len # L

        if num_nodes_in != self.num_nodes:
             raise ValueError(f"Input num_nodes ({num_nodes_in}) does not match module's num_nodes ({self.num_nodes})")
        if node_seq_len != self.par_length:
             raise ValueError(f"Input node_seq_len ({node_seq_len}) does not match module's par_length ({self.par_length})")

        # If original sequence was empty or too short
        if output_len == 0:
             return torch.zeros(batch_size, 0, dim, dtype=x.dtype, device=x.device)

        step = self.calculate_step(output_len)

        output = torch.zeros(batch_size, output_len, dim, dtype=x.dtype, device=x.device)
        counts = torch.zeros(batch_size, output_len, 1, dtype=x.dtype, device=x.device)

        effective_par_length = min(output_len, self.par_length) # Max length of any real chunk

        for i in range(self.num_nodes):
            start_idx = i * step
            end_idx = start_idx + self.par_length # Use full par_length for potential window size

            # Clamp indices to the actual output sequence length
            actual_start_idx = min(start_idx, output_len)
            actual_end_idx = min(end_idx, output_len)

            # Calculate the length of the chunk to actually use from input x
            # and the slice in the output tensor
            length_in_output = actual_end_idx - actual_start_idx
            if length_in_output <= 0:
                continue # This chunk starts beyond the output length

            # The corresponding length in the input chunk x might be shorter
            # if the original sequence L was shorter than par_length,
            # or if this patch goes past L.
            length_from_input = min(length_in_output, effective_par_length)

            # Get the relevant part of the processed chunk from BabyAE 'i'
            # We only take up to length_from_input from the potentially zero-padded input x
            chunk_to_add = x[:, i, :length_from_input, :]

            if chunk_to_add.shape[1] != length_in_output:
                 # This case can happen if start_idx > 0 and end_idx > output_len
                 # We need to take the correct slice of chunk_to_add
                 if start_idx < output_len:
                     chunk_to_add = chunk_to_add[:, :(output_len - start_idx) , :]
                     length_in_output = chunk_to_add.shape[1] # Adjust length
                     if length_in_output <= 0: continue
                 else:
                     continue # Start index is already out of bounds


            # Add contributions using slicing for broadcasting
            output[:, actual_start_idx:actual_end_idx, :] += chunk_to_add
            counts[:, actual_start_idx:actual_end_idx, :] += 1.0

        # Avoid division by zero for areas with potentially no contribution
        # (e.g., if L is very short or step logic leads to gaps)
        counts = torch.clamp(counts, min=1e-8) # Use epsilon instead of 1.0 to avoid changing scale

        fused = output / counts

        # Optional final normalization
        fused = self.norm(fused)

        return fused


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
    def __init__(
        self,
        num_nodes: int,
        partition_length: int,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        n_mask: int,
        use_masking: bool = True,
        compression_factor: int = 1,
        norm: nn.Module = None
    ):
        super(HAMABlock, self).__init__()
        self.n_mask = n_mask
        self.num_nodes = num_nodes
        self.compression_factor = compression_factor

        # Parallel baby autoencoders
        self.autoencoders = nn.ModuleList([
            BabyAutoencoder(
                d_model=embedding_dim,
                nhead=num_heads,
                num_layers=num_layers,
                n_mask=n_mask,
                use_masking=use_masking,
                norm=norm
            )
            for _ in range(num_nodes)
        ])

        # Fission => produce multiple node-specific streams
        self.fission_module = FissionModule(
            num_nodes=num_nodes,
            par_length=partition_length,
            embedding_dim=embedding_dim,
        )

        # Fusion => combine node streams, optionally compress
        self.fusion_module = FusionModule(
            embedding_dim=embedding_dim,
            num_nodes=num_nodes,
            par_length=partition_length,
        )

    def forward(self, x: torch.Tensor):
        """
        Performs complete block processing (encode + decode).

        Args:
            x (torch.Tensor): [batch_size, seq_len, embedding_dim]

        Returns:
            [batch_size, fused_len, embedding_dim]
        """
        B, L, D = x.shape

        # x = self.fission_module(x)  # => [B, num_nodes, par_length, D]
        # fused = self.fusion_module(x, L)  # => [B, fused_len, D]
        # return fused

        # 1) Fission
        node_repr = self.fission_module(x)  # => [B, num_nodes, par_length, D]

        # 2) Parallel autoencoders
        for i in range(self.num_nodes):
            node_repr[:, i] = self.autoencoders[i](node_repr[:, i])

        # 3) Fusion
        fused = self.fusion_module(node_repr, L)  # => [B, fused_len, D]
        return fused

    def encode(self, x: torch.Tensor):
        """
        Encodes input (through fission + parallel AE encodes) => fuse => compress if chosen.

        Args:
            x: [B, L, D]
        Returns:
            [B, fused_len, D]
        """
        B, L, D = x.shape

        # x = self.fission_module(x)
        # fused = self.fusion_module(x, L)
        # return fused

        node_repr = self.fission_module(x)
        for i in range(self.num_nodes):
            node_repr[:, i] = self.autoencoders[i].encode(node_repr[:, i])

        # During encoding, we might produce fewer tokens if compression_factor > 1
        fused = self.fusion_module(node_repr, int((L - self.n_mask*self.num_nodes) / self.compression_factor))
        return fused

    def decode(self, x: torch.Tensor):
        """
        Decodes representation (fission + parallel AE decode) => fuse => compress if chosen.

        Args:
            x: [B, L, D]
        Returns:
            [B, fused_len, D]
        """
        B, L, D = x.shape

        # x = self.fission_module(x)
        # fused = self.fusion_module(x, L)
        # return fused

        node_repr = self.fission_module(x)
        for i in range(self.num_nodes):
            node_repr[:, i] = self.autoencoders[i].decode(node_repr[:, i])

        fused = self.fusion_module(node_repr, int((L + self.n_mask * self.num_nodes) * self.compression_factor))
        return fused


class HAMA(nn.Module):
    """
    Hierarchical Attention-based Masked Autoencoder (HAMA) with optional masking and compression.

    Stacks multiple HAMABlock layers. Each block can:
      - optionally mask tokens in parallel baby autoencoders,
      - optionally compress the fused representation in the fusion step.

    Args:
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
        # Up (encode)
        for layer in self.layers:
            x = layer.encode(x)

        x = self.activation(x)

        # Down (decode)
        for layer in reversed(self.layers):
            x = layer.decode(x)

        # Final activation
        x = self.activation(x)

        return x

    def encode(self, x: torch.Tensor):
        """Encodes input through all HAMA layers."""
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