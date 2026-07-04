from __future__ import annotations

import math

import torch


def scaled_dot_product_attention_manual(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Compute scaled dot-product attention explicitly.

    Args:
        q: query tensor,  shape [B, H, S_q, D].
        k: key tensor,    shape [B, H, S_k, D].
        v: value tensor,  shape [B, H, S_k, D].
        attn_mask: optional mask over the key dimension. Two conventions are
            supported, matching ``torch.nn.functional.scaled_dot_product_attention``:
              * bool tensor  -> True = "allowed to attend", False = "blocked".
              * float tensor -> added directly to the scores (use ``-inf`` to block).
            Must be broadcastable to shape [B, H, S_q, S_k].
        is_causal: if True, apply a causal (lower-triangular) mask so that query
            position i cannot attend to key position j > i. Assumes S_q == S_k.

    Returns:
        Attention output, shape [B, H, S_q, D] — the same shape as ``q``.
    """
    # --- Validate shapes
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "Expected 4D tensors shaped [B, H, S, D], got ndims "
            f"q={q.ndim}, k={k.ndim}, v={v.ndim}."
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"Query and key head_dim must match, got {q.shape[-1]} vs {k.shape[-1]}."
        )
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(
            "Key and value must share a sequence length, "
            f"got {k.shape[-2]} vs {v.shape[-2]}."
        )

    head_dim = q.shape[-1]

    # --- Similarity scores
    # k.transpose(-2, -1) turns K from [B, H, S_k, D] into [B, H, D, S_k], so
    # the matrixt mult contracts over D (the head dim) and leaves one score per
    # (query, key) pair:
    #
    #     q @ kᵀ =  scores
    #     [B, H, S_q, D] @ [B, H, D, S_k] = [B, H, S_q, S_k]
    #
    # scores[b, h, i, j] = dot( q[b, h, i, :], k[b, h, j, :] )
    scores = q @ k.transpose(-2, -1)

    # --- Scale by sqrt(head_dim) 
    # Dot products grow with D. Left unscaled, they push softmax toward a
    # near one-hot distribution (vanishing gradients). Dividing by sqrt(D)
    # keeps the score variance roughly constant regardless of head dim.
    scores = scores / math.sqrt(head_dim)

    # --- Masking 
    """
    Similarly, self-attention layers in the decoder allow each position in the decoder to attend to
    all positions in the decoder up to and including that position. We need to prevent leftward
    information flow in the decoder to preserve the auto-regressive property. We implement this
    inside of scaled dot-product attention by masking out (setting to −∞) all values in the input
    of the softmax which correspond to illegal connections. See Figure 2
    (Attention Is All You Need, 2023)
    """
    # We push blocked positions to -inf so that exp(-inf) = 0 gives them
    # exactly zero attention weight after softmax.
    if is_causal:
        s_q, s_k = q.shape[-2], k.shape[-2]
        # tril() keeps the lower triangle (allowed); the strict upper triangle
        # is the "future" and gets blocked.
        allowed = torch.ones(s_q, s_k, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    # -- Softmax over the KEY dimension (dim=-1 = S_k)
    # For each query i, this turns its row of scores into a probability
    # distribution over the keys: every weight >= 0 and the row sums to 1.
    weights = torch.softmax(scores, dim=-1)

    # --- Weighted sum of the values 
    #     weights          @  v               =  output
    #     [B, H, S_q, S_k] @  [B, H, S_k, D]  =  [B, H, S_q, D]
    #
    # output[b, h, i, :] = sum_j weights[b, h, i, j] * v[b, h, j, :]
    output = weights @ v
    return output
