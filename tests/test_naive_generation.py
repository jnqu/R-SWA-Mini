import pytest
import torch

from rswa_mini.generation import greedy_generate, naive_greedy_generate


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return tokenizer, model


def test_naive_matches_generate(gpt2) -> None:
    tokenizer, model = gpt2
    ids = tokenizer(
        "The best way to learn about transformers is", return_tensors="pt"
    ).input_ids
    n_new = 10

    sequences, _ = naive_greedy_generate(model, ids, max_new_tokens=n_new)
    reference = model.generate(
        ids, max_new_tokens=n_new, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    assert torch.equal(sequences, reference)


def test_naive_matches_cached(gpt2) -> None:
    tokenizer, model = gpt2
    ids = tokenizer("Hello there world", return_tensors="pt").input_ids
    n_new = 8

    naive_sequences, positions_fed = naive_greedy_generate(model, ids, max_new_tokens=n_new)
    cached = greedy_generate(model, ids, max_new_tokens=n_new)

    # The optimization must not change the output.
    assert torch.equal(naive_sequences, cached.sequences)
    # And the naive loop really did re-feed a growing sequence each step.
    assert positions_fed == [ids.shape[1] + i for i in range(n_new)]
