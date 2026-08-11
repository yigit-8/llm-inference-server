"""
Open-loop load test: static batching against continuous batching.

`src/benchmark.py` answers a closed-loop question: given N requests that are all
present at t=0, how fast does the engine chew through them. That flatters
batching and says nothing about latency under load, because nothing ever waits in
a queue that is filling while the GPU works.

This script asks the question a served system actually faces: requests arrive on
their own schedule, at a rate the server does not control, and the interesting
number is what a *single* request experiences while the batch churns around it.
Arrivals are Poisson, so the load is bursty in the way real traffic is bursty
rather than evenly spaced.

The two arms differ only in when a waiting request is allowed into the batch:

  static      a batch is admitted, run to completion, and only then is the next
              batch admitted. A request that arrives one iteration too late waits
              for every member of the current batch, including the slowest.
  continuous  a request joins at the next iteration after it arrives, and leaves
              the batch the moment it finishes.

Everything else is held identical (same model, same prompts, same output
lengths, same arrival times, same seed), so the gap between the columns is the
scheduling policy and nothing else.

    python -m benchmarks.load                        # default sweep, fp16 on GPU
    python -m benchmarks.load --rps 4 8 --requests 96
    python -m benchmarks.load --model distilgpt2     # small enough for CPU

Run it as a module from the repo root, not as `python benchmarks/load.py`, so
that `src` is importable.
"""

import argparse
import asyncio
import math
import platform
import random
import threading
import time
from collections import deque
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.engine import ContinuousBatchingEngine, Sequence

# Prompt and output lengths are mixed on purpose. A batching scheduler is only
# interesting when the members of a batch disagree about how much work they
# need: with uniform lengths static batching loses almost nothing, because every
# sequence in the batch finishes on the same iteration anyway. Weights lean
# short with a long tail, which is roughly what chat traffic looks like.
PROMPT_LENGTHS = ((0.50, 32), (0.30, 96), (0.15, 192), (0.05, 256))
OUTPUT_LENGTHS = ((0.40, 16), (0.35, 48), (0.20, 96), (0.05, 192))

# Sliced into windows to build prompts of an exact token length. The content is
# irrelevant to the measurement (only the token count is), but real text keeps
# the tokenizer honest about how many tokens a window actually holds.
CORPUS = (
    "Continuous batching schedules work at the granularity of a single decode "
    "iteration rather than a whole request. A shared GPU cluster spends most of "
    "each decode step streaming model weights out of memory, and that cost is "
    "paid once per step no matter how many sequences ride along, which is the "
    "reason an inference server batches at all. The hard part is not the idea "
    "but keeping the key value cache consistent while the batch composition "
    "changes underneath it. "
) * 60


@dataclass
class ModelBundle:
    model: object
    tokenizer: object
    prompt_corpus: list[int]
    device: str
    dtype: torch.dtype
    hardware: str


@dataclass
class Arrival:
    """One planned request: when it shows up and how much work it brings."""

    index: int
    offset: float  # seconds after the run starts
    prompt_len: int
    max_new_tokens: int
    warmup: bool


@dataclass
class Record:
    """A planned arrival once it has actually been submitted."""

    arrival: Arrival
    sequence: Sequence
    scheduled_at: float
    submitted_at: float

    @property
    def client_lag(self) -> float:
        """How late the load generator was. This is the script's own overhead."""
        return self.submitted_at - self.scheduled_at


@dataclass
class RunConfig:
    mode: str
    rps: float
    batch_size: int
    timeout: float


@dataclass
class RunResult:
    config: RunConfig
    completed: int
    submitted: int
    ttft_p50: float
    ttft_p95: float
    latency_p50: float
    latency_p95: float
    output_tokens_per_second: float
    lag_p50_ms: float
    lag_p95_ms: float
    timed_out: bool


# --------------------------------------------------------------------- helpers


def percentile(values: list[float], quantile: float) -> float:
    """Nearest-rank percentile.

    `statistics.quantiles` interpolates and needs at least two points, which is a
    poor fit here: a run can legitimately complete one request, and an
    interpolated p95 over a handful of samples invents a number that no request
    experienced.
    """
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[rank]


def sample_length(rng: random.Random, table: tuple[tuple[float, int], ...]) -> int:
    weights = [weight for weight, _ in table]
    lengths = [length for _, length in table]
    return rng.choices(lengths, weights=weights, k=1)[0]


def plan_arrivals(rps: float, requests: int, warmup: int, seed: int) -> list[Arrival]:
    """Poisson arrivals at `rps`, plus a warmup head that is not measured.

    Seeded per sweep level rather than per run, so both modes see byte-identical
    traffic: same arrival times, same prompt lengths, same output lengths. The
    warmup requests arrive on the same process so the engine, the allocator and
    the CUDA kernels are all warm before the first measured request lands.
    """
    rng = random.Random(seed)
    arrivals: list[Arrival] = []
    clock = 0.0
    for index in range(warmup + requests):
        # Exponential gaps are what make the arrival process Poisson.
        clock += rng.expovariate(rps)
        arrivals.append(
            Arrival(
                index=index,
                offset=clock,
                prompt_len=sample_length(rng, PROMPT_LENGTHS),
                max_new_tokens=sample_length(rng, OUTPUT_LENGTHS),
                warmup=index < warmup,
            )
        )
    return arrivals


def load_model(name: str, device: str, dtype: torch.dtype) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.to(device)
    model.eval()

    hardware = (
        torch.cuda.get_device_name(0)
        if device == "cuda"
        else (platform.processor() or platform.machine() or "cpu")
    )
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        # verbose=False: the corpus is deliberately longer than the model's
        # context, because it is only ever sliced into windows, and the
        # "sequence longer than maximum" warning is noise here.
        prompt_corpus=tokenizer(CORPUS, verbose=False)["input_ids"],
        device=device,
        dtype=dtype,
        hardware=hardware,
    )


def build_sequence(
    bundle: ModelBundle, arrival: Arrival, rng: random.Random
) -> Sequence:
    """Build a request. `Sequence.queued_at` is stamped here, at arrival time.

    That is deliberate: TTFT has to include the time a request spends waiting to
    be admitted, which is the entire cost static batching imposes.
    """
    corpus = bundle.prompt_corpus
    start = rng.randrange(0, max(1, len(corpus) - arrival.prompt_len))
    window = corpus[start : start + arrival.prompt_len]
    prompt_ids = torch.tensor([window], dtype=torch.long, device=bundle.device)

    return Sequence(
        id=f"req-{arrival.index}",
        prompt_ids=prompt_ids,
        max_new_tokens=arrival.max_new_tokens,
        # No EOS: every request generates exactly the sampled number of tokens,
        # so both arms perform identical work and the comparison cannot be moved
        # by one arm happening to stop early.
        eos_token_id=None,
    )


# ---------------------------------------------------------------------- engine


class Harness:
    """Runs the engine on its own thread under one of the two admission policies.

    The engine is single-threaded by design and owns the model and the cache, so
    nothing here touches it except the engine thread. Arrivals cross the thread
    boundary through `_pending` and are pulled in by the policy.
    """

    def __init__(self, bundle: ModelBundle, config: RunConfig) -> None:
        self.engine = ContinuousBatchingEngine(
            bundle.model, bundle.tokenizer, max_batch_size=config.batch_size
        )
        self.config = config
        self._pending: deque[Sequence] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="harness")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def submit(self, sequence: Sequence) -> None:
        with self._lock:
            self._pending.append(sequence)

    def _admit(self) -> None:
        with self._lock:
            pending = self._pending

            if self.config.mode == "continuous":
                # Hand everything over; the engine pulls a waiting request into
                # the running batch at the next iteration that has capacity.
                while pending:
                    self.engine.add(pending.popleft())
                return

            # Static: nothing is admitted until the current batch has fully
            # drained, so a request that misses a batch waits for the slowest
            # member of it. This is the behaviour continuous batching removes.
            if self.engine.running or self.engine.waiting:
                return
            for _ in range(min(self.config.batch_size, len(pending))):
                self.engine.add(pending.popleft())

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._admit()
            if self.engine.has_work():
                self.engine.step()
            else:
                time.sleep(0.001)


# --------------------------------------------------------------------- driving


async def drive(
    bundle: ModelBundle, harness: Harness, plan: list[Arrival], seed: int
) -> list[Record]:
    """Submit arrivals on their scheduled offsets, asynchronously.

    Sleeping on the event loop rather than blocking means the generator keeps its
    own schedule regardless of how slow the engine is; a closed-loop client would
    quietly throttle the arrival rate to whatever the server could absorb and the
    static arm would look far better than it is.
    """
    rng = random.Random(seed)
    records: list[Record] = []
    start = time.perf_counter()

    for arrival in plan:
        scheduled_at = start + arrival.offset
        delay = scheduled_at - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        sequence = build_sequence(bundle, arrival, rng)
        harness.submit(sequence)
        records.append(
            Record(
                arrival=arrival,
                sequence=sequence,
                scheduled_at=scheduled_at,
                submitted_at=time.perf_counter(),
            )
        )
    return records


async def await_completion(records: list[Record], timeout: float) -> bool:
    """Wait for every submitted request to finish. True if the run timed out."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if all(record.sequence.finished_at is not None for record in records):
            return False
        await asyncio.sleep(0.02)
    return True


def summarise(records: list[Record], config: RunConfig, timed_out: bool) -> RunResult:
    measured = [r for r in records if not r.arrival.warmup]
    done = [r for r in measured if r.sequence.finished_at is not None]

    ttfts = [r.sequence.time_to_first_token() for r in done]
    ttfts = [value for value in ttfts if value is not None]
    latencies = [r.sequence.finished_at - r.sequence.queued_at for r in done]
    lags = [r.client_lag * 1000 for r in measured]

    tokens = sum(len(r.sequence.tokens) for r in done)
    if done:
        window = max(r.sequence.finished_at for r in done) - min(
            r.sequence.queued_at for r in done
        )
    else:
        window = 0.0

    return RunResult(
        config=config,
        completed=len(done),
        submitted=len(measured),
        ttft_p50=percentile(ttfts, 0.50),
        ttft_p95=percentile(ttfts, 0.95),
        latency_p50=percentile(latencies, 0.50),
        latency_p95=percentile(latencies, 0.95),
        output_tokens_per_second=tokens / window if window > 0 else math.nan,
        lag_p50_ms=percentile(lags, 0.50),
        lag_p95_ms=percentile(lags, 0.95),
        timed_out=timed_out,
    )


async def run_once(
    bundle: ModelBundle, plan: list[Arrival], config: RunConfig, seed: int
) -> RunResult:
    harness = Harness(bundle, config)
    harness.start()
    try:
        # Same seed for both modes, so the prompt windows are identical too and
        # not just their lengths.
        records = await drive(bundle, harness, plan, seed)
        timed_out = await await_completion(records, config.timeout)
    finally:
        harness.stop()
    return summarise(records, config, timed_out)


def warm_up_kernels(bundle: ModelBundle) -> None:
    """One throwaway generation so the first measured run is not paying for
    lazily initialised CUDA kernels and allocator growth."""
    window = bundle.prompt_corpus[:32]
    sequence = Sequence(
        id="warmup",
        prompt_ids=torch.tensor([window], dtype=torch.long, device=bundle.device),
        max_new_tokens=8,
        eos_token_id=None,
    )
    engine = ContinuousBatchingEngine(bundle.model, bundle.tokenizer, max_batch_size=2)
    engine.add(sequence)
    while engine.has_work():
        engine.step()
    if bundle.device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------- output


def cell(value: float, digits: int = 3) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def markdown_tables(
    results: dict[tuple[float, str], RunResult], rps_levels: list[float]
) -> str:
    def get(rps: float, mode: str) -> RunResult:
        return results[(rps, mode)]

    lines = ["### Latency", ""]
    lines.append(
        "| rps | TTFT p50 s | TTFT p50 c | TTFT p95 s | TTFT p95 c "
        "| e2e p50 s | e2e p50 c | e2e p95 s | e2e p95 c |"
    )
    lines.append("|" + "---|" * 9)
    for rps in rps_levels:
        static, cont = get(rps, "static"), get(rps, "continuous")
        lines.append(
            f"| {rps:g} | {cell(static.ttft_p50)} | {cell(cont.ttft_p50)} "
            f"| {cell(static.ttft_p95)} | {cell(cont.ttft_p95)} "
            f"| {cell(static.latency_p50)} | {cell(cont.latency_p50)} "
            f"| {cell(static.latency_p95)} | {cell(cont.latency_p95)} |"
        )

    lines += ["", "### Throughput and completion", ""]
    lines.append(
        "| rps | tok/s static | tok/s cont | completed static | completed cont | submitted |"
    )
    lines.append("|" + "---|" * 6)
    for rps in rps_levels:
        static, cont = get(rps, "static"), get(rps, "continuous")
        lines.append(
            f"| {rps:g} | {cell(static.output_tokens_per_second, 1)} "
            f"| {cell(cont.output_tokens_per_second, 1)} "
            f"| {static.completed}{' (timeout)' if static.timed_out else ''} "
            f"| {cont.completed}{' (timeout)' if cont.timed_out else ''} "
            f"| {static.submitted} |"
        )

    # Reported separately so the numbers above can be read with the right amount
    # of trust: if the load generator itself is late by anything comparable to
    # the latencies it reports, the run measured this script, not the engine.
    # A couple of milliseconds is the event loop's timer granularity and is
    # fine; tens of milliseconds means the arrival schedule is not being kept.
    lines += ["", "### Load generator overhead (should stay near zero)", ""]
    lines.append("| rps | lag p50 ms s | lag p50 ms c | lag p95 ms s | lag p95 ms c |")
    lines.append("|" + "---|" * 5)
    for rps in rps_levels:
        static, cont = get(rps, "static"), get(rps, "continuous")
        lines.append(
            f"| {rps:g} | {cell(static.lag_p50_ms, 1)} | {cell(cont.lag_p50_ms, 1)} "
            f"| {cell(static.lag_p95_ms, 1)} | {cell(cont.lag_p95_ms, 1)} |"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------------ main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static vs continuous batching under load."
    )
    parser.add_argument("--rps", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--requests", type=int, default=64, help="measured requests per level"
    )
    parser.add_argument(
        "--warmup", type=int, default=8, help="unmeasured requests per level"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per run")
    parser.add_argument(
        "--device", default=None, help="cuda or cpu (default: cuda if present)"
    )
    return parser.parse_args()


async def sweep(args: argparse.Namespace, bundle: ModelBundle) -> None:
    results: dict[tuple[float, str], RunResult] = {}
    for rps in args.rps:
        # One plan per level, shared by both modes: identical traffic in, so the
        # only thing that differs between the columns is the admission policy.
        plan = plan_arrivals(rps, args.requests, args.warmup, args.seed)
        for mode in ("static", "continuous"):
            config = RunConfig(
                mode=mode, rps=rps, batch_size=args.batch_size, timeout=args.timeout
            )
            print(f"  running rps={rps:g} mode={mode} ...", flush=True)
            results[(rps, mode)] = await run_once(bundle, plan, config, args.seed)

    print()
    print(markdown_tables(results, args.rps))


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    # The engine builds its decode-step tensors without naming a device, which is
    # fine when everything is on CPU. Setting the process default before the
    # model is built is what puts those tensors next to the weights instead of
    # crashing the first decode step with a device mismatch.
    torch.set_default_device(device)

    bundle = load_model(args.model, device, dtype)
    warm_up_kernels(bundle)

    print(f"hardware: {bundle.hardware}")
    print(f"model:    {args.model}")
    print(f"dtype:    {str(dtype).removeprefix('torch.')}")
    print(
        f"batch:    {args.batch_size}   requests/level: {args.requests}"
        f"   warmup/level: {args.warmup}   seed: {args.seed}"
    )
    print()

    asyncio.run(sweep(args, bundle))


if __name__ == "__main__":
    main()
