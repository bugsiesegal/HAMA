import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _entropy(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Shannon entropy over the last dimension."""
    return -(p * (p + eps).log()).sum(-1)


# -----------------------------------------------------------------------------
# Cross‑attention router (Fission replacement)
# -----------------------------------------------------------------------------

class CrossAttentionRouter(nn.Module):
    """Route a full sequence [B, L, D] into *num_nodes* sequences of length *P*.

    The router owns a *bank* of learnable queries of shape [num_nodes, P, D].
    Each query competes (via cross‑attention) for tokens in the input sequence,
    producing a compact ordered slice that downstream BabyAutoencoders can mask.

    Args
    ----
    num_nodes          : how many parallel nodes / slices
    partition_length   : P; number of query vectors per node (== output length)
    embedding_dim      : D
    num_heads          : attention heads for the internal MultiHeadAttention
    dropout            : dropout applied within the attention module
    temperature        : softmax temperature (>1 → softer, <1 → sharper)
    entropy_weight     : coefficient for the optional entropy regulariser
    window_mask        : if True, force node *k* to look only at a contiguous
                         window of length P centred at stride*k (like unfold)
    """

    def __init__(
        self,
        num_nodes: int,
        partition_length: int,
        embedding_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0,
        entropy_weight: float = 1e-3,
        window_mask: bool = False,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.P = partition_length
        self.temp = temperature
        self.entropy_weight = entropy_weight
        self.window_mask = window_mask

        # Learnable query bank  [N, P, D]
        self.queries = nn.Parameter(
            torch.randn(num_nodes, partition_length, embedding_dim)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embedding_dim)
        self.norm_x = nn.LayerNorm(embedding_dim)

    # ---------------------------------------------------------------------
    # Utility ‑ contiguous window mask (optional)
    # ---------------------------------------------------------------------
    def _build_window_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.window_mask:
            return None
        stride = max(1, (seq_len - self.P) // max(1, self.num_nodes - 1))
        mask = torch.full((self.num_nodes, seq_len), float("-inf"), device=device)
        for n in range(self.num_nodes):
            start = n * stride
            end = min(start + self.P, seq_len)
            mask[n, start:end] = 0.0
        return mask  # [N, L]

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Route *x* into node‑specific sequences.

        Parameters
        ----------
        x           : [B, L, D]
        return_attn : whether to also return the attention map

        Returns
        -------
        node_repr : [B, N, P, D]
        attn      : [B, N*P, L]  (optional)
        """
        B, L, D = x.shape
        N, P = self.num_nodes, self.P

        # ---- build queries ------------------------------------------------
        q = self.queries.unsqueeze(0).expand(B, -1, -1, -1)  # [B, N, P, D]
        q = q.reshape(B, N * P, D)                          # flatten bank
        q = self.norm_q(q)

        x_norm = self.norm_x(x)

        # ---- optional window mask ----------------------------------------
        attn_mask = self._build_window_mask(L, x.device)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(B, 1, 1)  # [B, N, L]
            attn_mask = attn_mask.reshape(B, N * P, L)

        # ---- cross‑attention ---------------------------------------------
        out, attn = self.attn(
            query=q,
            key=x_norm,
            value=x_norm,
            need_weights=return_attn or self.entropy_weight > 0,
            attn_mask=attn_mask,
        )  # out: [B, N*P, D]; attn: [B, N*P, L]

        # optional temperature sharpening ----------------------------------
        if self.temp != 1.0:
            attn = F.softmax((attn.log() + 1e-9) / self.temp, dim=-1)
            out = torch.bmm(attn, x_norm)  # recompute weighted sum

        # optional entropy regularisation ----------------------------------
        if self.entropy_weight and self.training:
            ent = _entropy(attn)  # [B, N*P]
            self.entropy_loss = self.entropy_weight * ent.mean()
        else:
            self.entropy_loss = None

        node_repr = out.view(B, N, P, D)
        return (node_repr, attn if return_attn else None)


# -----------------------------------------------------------------------------
# Cross‑attention gatherer (Fusion replacement)
# -----------------------------------------------------------------------------

class CrossAttentionGather(nn.Module):
    """Rebuild a full sequence from node representations.

    Input : [B, N, P, D]
    Output: [B, seq_len, D]
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_nodes = nn.LayerNorm(embedding_dim)
        self.norm_queries = nn.LayerNorm(embedding_dim)

    # ---------------------------------------------------------------------
    def _build_positional_queries(self, seq_len: int, D: int, device) -> torch.Tensor:
        """Classic sine‑cosine positional queries with shape [1, L, D]."""
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=device) * (-math.log(10000.0) / D))
        pe = torch.zeros(seq_len, D, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, L, D]

    # ---------------------------------------------------------------------
    def forward(self, node_repr: torch.Tensor, seq_len: int) -> torch.Tensor:
        B, N, P, D = node_repr.shape
        flat_nodes = node_repr.reshape(B, N * P, D)

        k = self.norm_nodes(flat_nodes)
        v = k

        q = self._build_positional_queries(seq_len, D, node_repr.device)
        q = q.expand(B, -1, -1)  # [B, L, D]
        q = self.norm_queries(q)

        fused, _ = self.attn(query=q, key=k, value=v, need_weights=False)
        return fused
