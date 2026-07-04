from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache


@torch.no_grad()
def naive_greedy_generate(
    model, input_ids: torch.Tensor, max_new_tokens: int
) -> tuple[torch.Tensor, list[int]]:
    """Greedy decode WITHOUT a KV cache

    At every step, feed the entire sequence-so-far back through the model with
    ``use_cache=False``, take the last position's logits, pick the argmax token,
    and append it.

    Returns:
        sequences: [1, prompt_len + max_new_tokens] token ids.
        positions_fed: number of token-positions fed to the model at each step.
            It climbs by one each step, so ``sum(positions_fed)`` grows
            quadratically with length.
    """
    model.eval()
    sequences = input_ids
    positions_fed: list[int] = []
    for _ in range(max_new_tokens):
        out = model(
            input_ids=sequences,
            attention_mask=torch.ones_like(sequences),
            use_cache=False, # no cache, the whole sequence is reprocessed each step
        )
        positions_fed.append(sequences.shape[1])
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sequences = torch.cat([sequences, next_token], dim=1)
    return sequences, positions_fed


def kv_cache_num_bytes(cache: DynamicCache) -> int:
    """Total bytes currently held by all Keys and Values across every layer."""
    total = 0
    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return total


@dataclass
class GenerationTrace:
    """The result of `greedy_generate`, plus a record of cache growth.

    Attributes:
        sequences: token ids, shape [1, prompt_len + n_new].
        cache_lengths: cache sequence length recorded after the prefill and then
            after each decode step (so you can watch it grow).
        cache_bytes: cache size in bytes at each of those same points.
    """
    sequences: torch.Tensor
    cache_lengths: list[int]
    cache_bytes: list[int]


@torch.no_grad()
def greedy_generate(model, input_ids: torch.Tensor, max_new_tokens: int) -> GenerationTrace:
    """Greedy-decode with a KV cache
    Assumes a batch size of 1 for clarity.
    """
    model.eval()
    cache = DynamicCache()
    attention_mask = torch.ones_like(input_ids)

    # --- PREFILL: process the entire prompt in one forward pass 
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    cache_lengths = [cache.get_seq_length()]
    cache_bytes = [kv_cache_num_bytes(cache)]

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
    sequences = torch.cat([input_ids, next_token], dim=1)

    # --- DECODE: one new token at a time, appending to the cache 
    for _ in range(max_new_tokens - 1):
        # The attention mask must cover the whole sequence so far (cache + new).
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        out = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        cache_lengths.append(cache.get_seq_length())
        cache_bytes.append(kv_cache_num_bytes(cache))

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sequences = torch.cat([sequences, next_token], dim=1)

    return GenerationTrace(
        sequences=sequences,
        cache_lengths=cache_lengths,
        cache_bytes=cache_bytes,
    )
