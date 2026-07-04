from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rswa_mini.generation import naive_greedy_generate

MODEL_NAME = "gpt2"
PROMPT = "The best way to learn about transformers is"
MAX_NEW_TOKENS = 12


def main() -> None:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    sequences, positions_fed = naive_greedy_generate(model, input_ids, MAX_NEW_TOKENS)
    new_tokens = sequences[0, prompt_len:].tolist()

    print(f"prompt: {PROMPT!r}  ({prompt_len} tokens)\n")
    print(f"{'step':<6}{'produces token':<18}{'tokens fed to model this step':<32}")
    print("-" * 56)
    total_work = 0
    for i, (fed, tok_id) in enumerate(zip(positions_fed, new_tokens)):
        total_work += fed
        print(f"{i:<6}{repr(tokenizer.decode([tok_id])):<18}{fed:<32}")

    print(f"\nfinal text: {tokenizer.decode(sequences[0], skip_special_tokens=True)!r}")
    print(
        f"\nTotal token-positions processed across all {MAX_NEW_TOKENS} steps: {total_work}"
    )
    print(
        f"If each new token were processed only once, it would be ~{MAX_NEW_TOKENS}. "
        "Every step re-processed the whole past from scratch —"
    )
    print("that waste is exactly what the KV cache (Milestone 4) eliminates.")


if __name__ == "__main__":
    main()
