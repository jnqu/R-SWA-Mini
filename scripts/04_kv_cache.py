from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rswa_mini.generation import greedy_generate, naive_greedy_generate

MODEL_NAME = "gpt2"
PROMPT = "The best way to learn about transformers is"
COMPARE_TOKENS = 40
GROWTH_TOKENS = 12


def _best_time(fn, repeats: int = 3) -> float:
    """Best (fastest) wall time over a few repeats — least polluted by noise."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    # ---- Part 1: naive vs cached
    naive_sequences, positions_fed = naive_greedy_generate(model, input_ids, COMPARE_TOKENS)
    cached = greedy_generate(model, input_ids, COMPARE_TOKENS)
    identical = torch.equal(naive_sequences, cached.sequences)

    naive_positions = sum(positions_fed) # re-processes the past each step
    cached_positions = prompt_len + (COMPARE_TOKENS - 1) # prompt once + 1 per decode step

    # Warm up once (exclude first-call overhead), then time.
    naive_greedy_generate(model, input_ids, 4)
    greedy_generate(model, input_ids, 4)
    naive_ms = _best_time(lambda: naive_greedy_generate(model, input_ids, COMPARE_TOKENS)) * 1e3
    cached_ms = _best_time(lambda: greedy_generate(model, input_ids, COMPARE_TOKENS)) * 1e3

    print(f"=== naive vs cached, generating {COMPARE_TOKENS} tokens ===")
    print(f"  identical output tokens?  {identical}")
    print(
        f"  token-positions processed: naive {naive_positions:>5}   cached {cached_positions:>5}"
        f"   -> {naive_positions / cached_positions:.1f}x fewer"
    )
    print(
        f"  wall time (best of 3)     : naive {naive_ms:7.1f} ms   cached {cached_ms:7.1f} ms"
        f"   -> {naive_ms / cached_ms:.1f}x faster"
    )
    print("  (CPU + GPT-2: timing is rough; see notes.md Caveat 4)\n")

    # ---- Part 2: watch the cache grow
    trace = greedy_generate(model, input_ids, GROWTH_TOKENS)
    new_tokens = trace.sequences[0, prompt_len:].tolist()
    print(f"=== KV cache growth (generating {GROWTH_TOKENS} tokens) ===")
    print(f"  {'phase':<9}{'produces token':<18}{'cache len':<12}{'cache size':<10}")
    print("  " + "-" * 47)
    for i, (clen, cbytes) in enumerate(zip(trace.cache_lengths, trace.cache_bytes)):
        phase = "prefill" if i == 0 else "decode"
        tok_text = repr(tokenizer.decode([new_tokens[i]])) if i < len(new_tokens) else "-"
        print(f"  {phase:<9}{tok_text:<18}{clen:<12}{cbytes / 1024:>6.0f} KB")
    grew = trace.cache_lengths[-1] - trace.cache_lengths[0]
    print(
        f"\n  cache grew {trace.cache_lengths[0]} -> {trace.cache_lengths[-1]} (+{grew}) and, "
        "unbounded, keeps growing with length. R-SWA caps it."
    )


if __name__ == "__main__":
    main()
