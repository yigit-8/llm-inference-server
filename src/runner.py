"""
Drives the engine on its own thread.

The engine is deliberately synchronous and single-threaded: one thread owns the
model and the KV cache, and nothing else touches them. Requests arrive from the
web server's threads, get handed over under a lock, and their callers block on an
event until the engine finishes them. Keeping all tensor work on one thread is
what makes the cache surgery in `cache_ops` safe without any locking of its own.
"""

import threading
import time
import uuid

import torch
from loguru import logger

from src import metrics
from src.config import settings
from src.engine import ContinuousBatchingEngine, Sequence


class EngineRunner:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.engine = ContinuousBatchingEngine(
            model, tokenizer, settings.MAX_BATCH_SIZE
        )

        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="engine")
        self._thread.start()
        logger.info(
            f"engine started: {settings.MODEL_NAME}, max batch {settings.MAX_BATCH_SIZE}"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def submit(
        self, prompt: str, max_new_tokens: int
    ) -> tuple[Sequence, threading.Event]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = encoded["input_ids"][:, -settings.MAX_PROMPT_TOKENS :]
        if prompt_ids.shape[1] == 0:
            prompt_ids = torch.tensor([[self.tokenizer.eos_token_id or 0]])

        sequence = Sequence(
            id=str(uuid.uuid4()),
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        event = threading.Event()
        with self._lock:
            self._events[sequence.id] = event
            self.engine.add(sequence)
        metrics.observe_request()
        return sequence, event

    def wait(self, sequence: Sequence, event: threading.Event, timeout: float) -> bool:
        finished = event.wait(timeout)
        with self._lock:
            self._events.pop(sequence.id, None)
        return finished

    def stats(self) -> dict:
        with self._lock:
            return {
                "model": settings.MODEL_NAME,
                "max_batch_size": settings.MAX_BATCH_SIZE,
                "running": len(self.engine.running),
                "queued": len(self.engine.waiting),
            }

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                has_work = self.engine.has_work()
                if has_work:
                    started = time.perf_counter()
                    finished = self.engine.step()
                    step_seconds = time.perf_counter() - started
                    running, queued = len(self.engine.running), len(self.engine.waiting)
                else:
                    finished = []
                    running = queued = 0

            if not has_work:
                metrics.set_engine_state(0, 0)
                time.sleep(settings.IDLE_SLEEP_SECONDS)
                continue

            metrics.DECODE_STEP.observe(step_seconds)
            metrics.set_engine_state(running, queued)

            for sequence in finished:
                total = (
                    sequence.finished_at or time.perf_counter()
                ) - sequence.queued_at
                metrics.observe_completion(
                    sequence.time_to_first_token(), len(sequence.tokens), total
                )
                with self._lock:
                    event = self._events.get(sequence.id)
                if event is not None:
                    event.set()
