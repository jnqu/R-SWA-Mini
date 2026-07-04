from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"
PROMPT = "The best way to learn about transformers is"


def main() -> None:
    device = "cpu"
    torch.manual_seed(0)

    # --- Load tokenizer + model
    print(f"Loading '{MODEL_NAME}'")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval() 
    model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Tokenize the prompt: text -> integer token ids
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    print("\n[tokenization]")
    print(f"  prompt    : {PROMPT!r}")
    print(f"  token ids : {input_ids[0].tolist()}")
    print(f"  #tokens   : {input_ids.shape[1]}  (shape {tuple(input_ids.shape)} = [batch, seq])")
    pieces = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    print(f"  pieces    : {pieces}")

    # --- Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # shape [batch, seq, vocab]
    print("\n[single forward pass]")

    # a score for every one of 50,257 vocab tokens, at every position
    print(f"  logits shape : {tuple(logits.shape)}  = [batch, seq, vocab]")

    # The prediction for the NEXT token comes from the LAST position's logits.
    next_token_logits = logits[0, -1]
    probs = torch.softmax(next_token_logits, dim=-1)
    top = torch.topk(probs, k=5)
    print("  top-5 predicted next words:")
    for prob, tid in zip(top.values.tolist(), top.indices.tolist()):
        print(f"      {prob:6.2%}  {tokenizer.decode([tid])!r}")

    # --- Full autoregressive generation
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("\n[generated text]")
    print(f"  {text!r}")
    print(f"\n  (started with {input_ids.shape[1]} tokens, ended with {generated.shape[1]})")


if __name__ == "__main__":
    main()
