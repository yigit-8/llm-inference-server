import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.engine import Sequence

# A ~2 MB randomly initialised GPT-2. Its text is meaningless, which does not
# matter: every assertion here is about scheduling and cache bookkeeping, and a
# random model exercises those exactly as a trained one would.
TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(TEST_MODEL)


@pytest.fixture(scope="session")
def model():
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL)
    model.eval()
    return model


@pytest.fixture()
def make_sequence():
    def _make(seq_id: str, prompt: list[int], max_new_tokens: int):
        return Sequence(
            id=seq_id,
            prompt_ids=torch.tensor([prompt]),
            max_new_tokens=max_new_tokens,
            eos_token_id=None,
        )

    return _make
