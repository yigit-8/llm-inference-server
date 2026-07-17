"""
Measures what continuous batching actually buys.

Both arms run the same model on the same prompts and decode greedily, so they
produce byte-identical output; the only difference is whether decode steps are
shared. The comparison is in-process on purpose: putting HTTP in the middle would
measure the web server as much as the engine.

    python -m src.benchmark --requests 16 --batch-size 8 --max-new-tokens 32
"""

import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import settings
from src.engine import ContinuousBatchingEngine, Sequence

PROMPTS = [
    "The future of machine learning infrastructure",
    "A shared GPU cluster is",
    "In distributed systems, the hardest problem",
    "Continuous batching improves",
    "The key insight behind paged attention",
    "When a model is served at scale",
    "Latency and throughput are",
    "An inference server should",
]


def make_sequence(index: int, tokenizer, max_new_tokens: int) -> Sequence:
    prompt = PROMPTS[index % len(PROMPTS)]
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return Sequence(
        id=str(index),
        prompt_ids=ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=None,  # fixed length, so both arms do identical work
    )


def run(model, tokenizer, sequences: list[Sequence], batch_size: int) -> dict:
    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=batch_size)
    for sequence in sequences:
        engine.add(sequence)

    started = time.perf_counter()
    steps = 0
    while engine.has_work():
        engine.step()
        steps += 1
    elapsed = time.perf_counter() - started

    tokens = sum(len(s.tokens) for s in sequences)
    ttfts = [
        s.time_to_first_token()
        for s in sequences
        if s.time_to_first_token() is not None
    ]
    return {
        "seconds": elapsed,
        "steps": steps,
        "tokens": tokens,
        "tokens_per_second": tokens / elapsed,
        "ttft_p50": statistics.median(ttfts) if ttfts else 0.0,
        "ttft_max": max(ttfts) if ttfts else 0.0,
        "outputs": [s.tokens for s in sequences],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential vs continuous batching.")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=settings.MAX_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=settings.MAX_NEW_TOKENS)
    parser.add_argument("--model", default=settings.MODEL_NAME)
    args = parser.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    print(
        f"model={args.model} requests={args.requests} max_new_tokens={args.max_new_tokens}"
    )

    baseline = run(
        model,
        tokenizer,
        [
            make_sequence(i, tokenizer, args.max_new_tokens)
            for i in range(args.requests)
        ],
        batch_size=1,
    )
    batched = run(
        model,
        tokenizer,
        [
            make_sequence(i, tokenizer, args.max_new_tokens)
            for i in range(args.requests)
        ],
        batch_size=args.batch_size,
    )

    identical = baseline["outputs"] == batched["outputs"]
    speedup = batched["tokens_per_second"] / baseline["tokens_per_second"]

    print(f"\n{'':<22}{'sequential':>14}{'batched':>14}")
    print(f"{'batch size':<22}{1:>14}{args.batch_size:>14}")
    print(
        f"{'wall seconds':<22}{baseline['seconds']:>14.2f}{batched['seconds']:>14.2f}"
    )
    print(f"{'forward passes':<22}{baseline['steps']:>14}{batched['steps']:>14}")
    print(
        f"{'tokens/second':<22}{baseline['tokens_per_second']:>14.1f}"
        f"{batched['tokens_per_second']:>14.1f}"
    )
    print(
        f"{'TTFT p50 (s)':<22}{baseline['ttft_p50']:>14.3f}{batched['ttft_p50']:>14.3f}"
    )
    print(
        f"{'TTFT max (s)':<22}{baseline['ttft_max']:>14.3f}{batched['ttft_max']:>14.3f}"
    )
    print(f"\nthroughput speedup: {speedup:.2f}x")
    print(f"identical output:   {identical}")


if __name__ == "__main__":
    main()
