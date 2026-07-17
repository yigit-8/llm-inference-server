"""
Prometheus metrics for an LLM inference server.

Request latency is the wrong headline number for generation: a request that
streams its first token in 40 ms and then produces 200 more feels fast, and a
request that takes the same total time but shows nothing for two seconds feels
broken. The two numbers that matter are therefore time-to-first-token, which is
dominated by queueing and prefill, and inter-token latency, which is dominated by
how large the running batch is.

Batch size is exported too, because it is the knob that trades one against the
other: a fuller batch raises throughput and inter-token latency together.
"""

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUESTS = Counter("llm_requests_total", "Generation requests accepted.")
TOKENS_GENERATED = Counter("llm_tokens_generated_total", "Tokens produced by the engine.")

RUNNING_BATCH = Gauge("llm_running_batch_size", "Sequences currently in the running batch.")
QUEUE_DEPTH = Gauge("llm_queue_depth", "Requests waiting to be admitted.")

TIME_TO_FIRST_TOKEN = Histogram(
    "llm_time_to_first_token_seconds",
    "Queue wait plus prefill, per request.",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)

INTER_TOKEN_LATENCY = Histogram(
    "llm_inter_token_latency_seconds",
    "Mean gap between tokens within a request.",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
)

DECODE_STEP = Histogram(
    "llm_decode_step_seconds",
    "Wall time of one decode iteration for the whole batch.",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
)


def observe_request() -> None:
    REQUESTS.inc()


def observe_completion(ttft: float | None, tokens: int, total_seconds: float) -> None:
    TOKENS_GENERATED.inc(tokens)
    if ttft is not None:
        TIME_TO_FIRST_TOKEN.observe(ttft)
    # Only the tokens after the first one carry an inter-token gap.
    if tokens > 1 and ttft is not None:
        INTER_TOKEN_LATENCY.observe((total_seconds - ttft) / (tokens - 1))


def set_engine_state(running: int, queued: int) -> None:
    RUNNING_BATCH.set(running)
    QUEUE_DEPTH.set(queued)


def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
