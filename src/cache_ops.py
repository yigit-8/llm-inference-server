"""
Batch-dimension surgery on the KV cache.

Continuous batching means the set of sequences sharing a forward pass changes
while generation is in flight. Every time it changes, the key/value cache has to
change with it: a newly admitted sequence brings its own cache and must be
concatenated onto the running one, and a finished sequence must be cut out.

Sequences hold different numbers of tokens, so they are aligned by padding the
cache on the *left*. Left padding keeps the newest token of every sequence at the
same index, which is what lets a single decode step append one token to all of
them at once. The padded positions are pure garbage; the attention mask is what
stops the model from ever looking at them.

`transformers` 5 removed `DynamicCache.from_legacy_cache`, so a cache is rebuilt
by feeding tensors back through `update()` one layer at a time.
"""

import torch
from transformers.cache_utils import DynamicCache

LayerKV = tuple[torch.Tensor, torch.Tensor]


def cache_tensors(cache: DynamicCache) -> list[LayerKV]:
    return [(layer.keys, layer.values) for layer in cache.layers]


def build_cache(layers: list[LayerKV]) -> DynamicCache:
    cache = DynamicCache()
    for index, (keys, values) in enumerate(layers):
        cache.update(keys, values, index)
    return cache


def _pad_left(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad a [batch, heads, seq, dim] tensor to `target_len` along seq, on the left."""
    missing = target_len - tensor.shape[2]
    if missing <= 0:
        return tensor
    padding = torch.zeros(
        tensor.shape[0],
        tensor.shape[1],
        missing,
        tensor.shape[3],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([padding, tensor], dim=2)


def concat_caches(caches: list[DynamicCache]) -> DynamicCache:
    """Join caches along the batch dimension, left-padding to the longest."""
    if len(caches) == 1:
        return caches[0]

    target = max(c.get_seq_length() for c in caches)
    per_cache = [cache_tensors(c) for c in caches]
    num_layers = len(per_cache[0])

    merged: list[LayerKV] = []
    for layer in range(num_layers):
        keys = torch.cat([_pad_left(kv[layer][0], target) for kv in per_cache], dim=0)
        values = torch.cat([_pad_left(kv[layer][1], target) for kv in per_cache], dim=0)
        merged.append((keys, values))
    return build_cache(merged)


def select_rows(cache: DynamicCache, rows: list[int], keep_len: int) -> DynamicCache:
    """Keep only `rows`, and drop left padding beyond `keep_len` real tokens.

    Trimming matters: without it the cache keeps carrying the padding of a long
    sequence that has already finished, and every later decode step pays for
    attention over columns that are masked out anyway.
    """
    index = torch.tensor(rows, dtype=torch.long)
    trimmed: list[LayerKV] = []
    for keys, values in cache_tensors(cache):
        start = keys.shape[2] - keep_len
        trimmed.append((keys[index, :, start:, :], values[index, :, start:, :]))
    return build_cache(trimmed)
