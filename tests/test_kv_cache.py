import pytest
import torch

from rswa_mini.generation import greedy_generate


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return tokenizer, model


def test_manual_loop_matches_generate(gpt2) -> None:
    tokenizer, model = gpt2
    ids = tokenizer(
        "The best way to learn about transformers is", return_tensors="pt"
    ).input_ids
    n_new = 10

    trace = greedy_generate(model, ids, max_new_tokens=n_new)
    reference = model.generate(
        ids, max_new_tokens=n_new, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    assert torch.equal(trace.sequences, reference)


def test_cache_grows_by_one_per_step(gpt2) -> None:
    tokenizer, model = gpt2
    ids = tokenizer("Hello there world", return_tensors="pt").input_ids
    prompt_len = ids.shape[1]
    n_new = 6

    trace = greedy_generate(model, ids, max_new_tokens=n_new)

    # After prefill, the cache holds exactly the prompt tokens.
    assert trace.cache_lengths[0] == prompt_len
    # We recorded one length per generated token.
    assert len(trace.cache_lengths) == n_new
    # Each decode step adds exactly one position.
    for earlier, later in zip(trace.cache_lengths, trace.cache_lengths[1:]):
        assert later == earlier + 1
    # Memory grows strictly with it.
    assert trace.cache_bytes == sorted(trace.cache_bytes)
    assert trace.cache_bytes[-1] > trace.cache_bytes[0]
