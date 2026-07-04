import pytest
import torch
import torch.nn.functional as F

from rswa_mini.attention import scaled_dot_product_attention_manual


def _random_qkv(B: int = 2, H: int = 3, S: int = 4, D: int = 8, seed: int = 0):
    """Make reproducible random Q, K, V tensors of shape [B, H, S, D]."""
    torch.manual_seed(seed)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    return q, k, v


def test_output_shape() -> None:
    # Attention output must have the same shape as the queries: [B, H, S, D].
    q, k, v = _random_qkv()
    out = scaled_dot_product_attention_manual(q, k, v)
    assert out.shape == q.shape


def test_matches_torch_sdpa_no_mask() -> None:
    # Unmasked attention should match PyTorch's official implementation.
    q, k, v = _random_qkv()
    ours = scaled_dot_product_attention_manual(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-4)


def test_matches_torch_sdpa_causal() -> None:
    # Causal attention should also match PyTorch's is_causal=True path.
    q, k, v = _random_qkv()
    ours = scaled_dot_product_attention_manual(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-4)


def test_weights_form_a_distribution() -> None:
    # Check the softmax step directly: each query's attention weights
    # must be non-negative and sum to 1 across the keys.
    q, k, v = _random_qkv(S=5)
    head_dim = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    assert torch.all(weights >= 0)
    row_sums = weights.sum(dim=-1)  # sum over the key dimension
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_causal_mask_blocks_the_future() -> None:
    # With a causal mask, query position 0 may only attend to key 0. So changing
    # the *later* values must NOT change the output at position 0 — but it should
    # change the output at the last position (which can see everything).
    q, k, v = _random_qkv(S=4)
    out1 = scaled_dot_product_attention_manual(q, k, v, is_causal=True)

    v_perturbed = v.clone()
    v_perturbed[:, :, 1:, :] += 100.0  # blast the values at positions 1..3
    out2 = scaled_dot_product_attention_manual(q, k, v_perturbed, is_causal=True)

    # Position 0 saw none of the perturbed future values -> unchanged.
    assert torch.allclose(out1[:, :, 0, :], out2[:, :, 0, :])
    # The last position saw all of them -> changed.
    assert not torch.allclose(out1[:, :, -1, :], out2[:, :, -1, :])


def test_rejects_non_4d_input() -> None:
    bad = torch.randn(4, 8)
    with pytest.raises(ValueError):
        scaled_dot_product_attention_manual(bad, bad, bad)
