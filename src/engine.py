"""
Continuous batching engine.

A naive server generates for one request at a time. The GPU (or CPU) spends most
of a decode step streaming model weights from memory, and that cost is paid once
per step regardless of how many sequences ride along. Batching therefore buys
throughput almost for free, which is why an inference server batches.

Static batching collects N requests, generates for all of them, and only then
accepts more. The batch runs until its *slowest* member finishes, so a request
asking for 4 tokens waits behind one asking for 200, and freed slots sit idle.

Continuous batching schedules at the granularity of a single decode iteration
instead of a whole request. Finished sequences leave the batch immediately, and
waiting requests join it at the very next iteration. That is the entire idea; the
work is in keeping the KV cache consistent while the batch composition churns,
which is what `cache_ops` handles.

Decoding is greedy. That makes the output a deterministic function of the input,
which is what allows the tests to assert that batching a request changes nothing
about the tokens it receives.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache

from src.cache_ops import concat_caches, select_rows


@dataclass
class Sequence:
    id: str
    prompt_ids: torch.Tensor  # [1, prompt_len]
    max_new_tokens: int
    eos_token_id: int | None = None

    tokens: list[int] = field(default_factory=list)
    # Real (non-padding) tokens this sequence currently occupies in the KV cache.
    cached_len: int = 0

    queued_at: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    finished_at: float | None = None

    @property
    def prompt_len(self) -> int:
        return int(self.prompt_ids.shape[1])

    @property
    def done(self) -> bool:
        if len(self.tokens) >= self.max_new_tokens:
            return True
        return bool(self.tokens) and self.tokens[-1] == self.eos_token_id

    def time_to_first_token(self) -> float | None:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.queued_at


class ContinuousBatchingEngine:
    def __init__(self, model, tokenizer, max_batch_size: int = 8) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.cache: DynamicCache | None = None
        self.cache_len: int = 0

    # ------------------------------------------------------------------ public

    def add(self, sequence: Sequence) -> None:
        self.waiting.append(sequence)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def cancel(self, sequence_id: str) -> bool:
        """Drop a sequence nobody is waiting for any more. Unknown ids are a no-op.

        A client that has given up (timed out, disconnected) must not keep a batch
        slot and its KV cache rows, so a cancelled sequence leaves the engine the
        same way a finished one does. Like every other method here, this runs under
        the runner's lock, on whichever thread called it.
        """
        for sequence in self.waiting:
            if sequence.id == sequence_id:
                self.waiting.remove(sequence)
                return True

        for index, sequence in enumerate(self.running):
            if sequence.id == sequence_id:
                self._keep_rows([i for i in range(len(self.running)) if i != index])
                return True
        return False

    def step(self) -> list[Sequence]:
        """Run exactly one iteration. Returns sequences that finished in it.

        An iteration is either a prefill (admitting new arrivals) or a decode
        (advancing everyone by one token). Prefill wins when there is capacity,
        because a request's time-to-first-token is dominated by how long it sits
        in the queue, and one decode step for the running batch is cheap.
        """
        self._evict_finished()

        if self.waiting and len(self.running) < self.max_batch_size:
            self._prefill()
            return self._collect_finished()

        if self.running:
            self._decode()
        return self._collect_finished()

    # ----------------------------------------------------------------- private

    def _collect_finished(self) -> list[Sequence]:
        now = time.perf_counter()
        finished = []
        for sequence in self.running:
            if sequence.done and sequence.finished_at is None:
                sequence.finished_at = now
                finished.append(sequence)
        return finished

    def _evict_finished(self) -> None:
        if not self.running:
            return
        keep = [i for i, s in enumerate(self.running) if not s.done]
        if len(keep) == len(self.running):
            return
        self._keep_rows(keep)

    def _keep_rows(self, keep: list[int]) -> None:
        """Cut the running batch (and its cache) down to the given rows."""
        if not keep:
            self.running = []
            self.cache = None
            self.cache_len = 0
            return

        survivors = [self.running[i] for i in keep]
        keep_len = max(s.cached_len for s in survivors)
        assert self.cache is not None
        self.cache = select_rows(self.cache, keep, keep_len)
        self.cache_len = keep_len
        self.running = survivors

    def _prefill(self) -> None:
        capacity = self.max_batch_size - len(self.running)
        admitted: list[Sequence] = []
        caches: list[DynamicCache] = []

        while self.waiting and capacity > 0:
            sequence = self.waiting.popleft()
            logits, cache = self._forward_prompt(sequence.prompt_ids)

            token = int(logits.argmax(dim=-1).item())
            sequence.tokens.append(token)
            sequence.cached_len = sequence.prompt_len
            sequence.first_token_at = time.perf_counter()

            admitted.append(sequence)
            caches.append(cache)
            capacity -= 1

        if not admitted:
            return

        existing = [self.cache] if self.cache is not None else []
        self.cache = concat_caches(existing + caches)
        self.cache_len = self.cache.get_seq_length()
        self.running.extend(admitted)

    def _forward_prompt(self, prompt_ids: torch.Tensor):
        attention_mask = torch.ones_like(prompt_ids)
        position_ids = torch.arange(prompt_ids.shape[1]).unsqueeze(0)
        with torch.no_grad():
            out = self.model(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
            )
        return out.logits[:, -1, :], out.past_key_values

    def _decode(self) -> None:
        assert self.cache is not None
        batch = self.running
        input_ids = torch.tensor([[s.tokens[-1]] for s in batch], dtype=torch.long)
        position_ids = torch.tensor([[s.cached_len] for s in batch], dtype=torch.long)

        # One column per cached position plus the token being fed. Each row only
        # attends to its own real tokens, which sit at the right-hand end because
        # the cache is left-padded.
        attention_mask = torch.zeros((len(batch), self.cache_len + 1), dtype=torch.long)
        for row, sequence in enumerate(batch):
            attention_mask[row, -(sequence.cached_len + 1) :] = 1

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.cache,
                use_cache=True,
            )

        self.cache = out.past_key_values
        self.cache_len += 1

        next_tokens = out.logits[:, -1, :].argmax(dim=-1)
        for row, sequence in enumerate(batch):
            sequence.cached_len += 1
            sequence.tokens.append(int(next_tokens[row].item()))


def generate_sequentially(model, tokenizer, sequence: Sequence) -> list[int]:
    """The baseline: one request at a time, no batching.

    Kept in the codebase rather than in a benchmark script because it is also the
    oracle the batched engine is tested against.
    """
    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=1)
    engine.add(sequence)
    while engine.has_work():
        engine.step()
    return sequence.tokens
