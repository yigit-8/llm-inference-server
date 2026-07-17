import torch
from transformers.cache_utils import DynamicCache

from src.cache_ops import build_cache, cache_tensors, concat_caches, select_rows


def fake_cache(batch: int, seq: int, layers: int = 2, heads: int = 2, dim: int = 4) -> DynamicCache:
    return build_cache(
        [
            (torch.randn(batch, heads, seq, dim), torch.randn(batch, heads, seq, dim))
            for _ in range(layers)
        ]
    )


def test_build_cache_round_trips_tensors():
    cache = fake_cache(batch=2, seq=3)
    keys, values = cache_tensors(cache)[0]
    assert keys.shape == (2, 2, 3, 4)
    assert values.shape == (2, 2, 3, 4)
    assert cache.get_seq_length() == 3


def test_concat_pads_the_shorter_cache_on_the_left():
    """Left padding keeps the newest token of every sequence at the same index,
    which is what lets one decode step serve the whole batch."""
    long, short = fake_cache(batch=1, seq=5), fake_cache(batch=1, seq=2)
    original_short_keys = cache_tensors(short)[0][0].clone()

    merged = concat_caches([long, short])
    keys = cache_tensors(merged)[0][0]

    assert keys.shape == (2, 2, 5, 4)
    assert torch.equal(keys[1, :, 3:, :], original_short_keys[0])  # real tokens sit at the right
    assert torch.all(keys[1, :, :3, :] == 0)  # padding on the left


def test_concat_of_a_single_cache_is_a_no_op():
    cache = fake_cache(batch=1, seq=3)
    assert concat_caches([cache]) is cache


def test_select_rows_keeps_the_requested_sequences():
    cache = fake_cache(batch=3, seq=4)
    original = cache_tensors(cache)[0][0].clone()

    selected = select_rows(cache, [0, 2], keep_len=4)
    keys = cache_tensors(selected)[0][0]

    assert keys.shape == (2, 2, 4, 4)
    assert torch.equal(keys[0], original[0])
    assert torch.equal(keys[1], original[2])


def test_select_rows_trims_padding_that_no_surviving_sequence_needs():
    """After a long sequence leaves, the columns it forced everyone else to pad
    with are dead weight, and every later decode step would attend over them."""
    cache = fake_cache(batch=2, seq=6)
    original = cache_tensors(cache)[0][0].clone()

    selected = select_rows(cache, [1], keep_len=2)
    keys = cache_tensors(selected)[0][0]

    assert keys.shape == (1, 2, 2, 4)
    assert torch.equal(keys[0], original[1, :, 4:, :])  # the two newest columns survive
