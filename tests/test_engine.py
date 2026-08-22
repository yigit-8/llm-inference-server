"""
The contract of continuous batching: sharing a forward pass with other requests
must not change a single token you receive. Everything else the engine does is
an optimisation, and these tests exist so an optimisation cannot quietly become
a wrong answer.
"""

import torch

from src.cache_ops import cache_tensors
from src.engine import ContinuousBatchingEngine, Sequence, generate_sequentially

PROMPTS = [[5, 6, 7, 8], [11, 12], [20, 21, 22, 23, 24, 25], [3]]
LENGTHS = [6, 4, 8, 5]


def drain(engine: ContinuousBatchingEngine, limit: int = 500) -> int:
    steps = 0
    while engine.has_work() and steps < limit:
        engine.step()
        steps += 1
    return steps


def oracle(model, tokenizer, make_sequence) -> list[list[int]]:
    return [
        generate_sequentially(model, tokenizer, make_sequence(str(i), p, n))
        for i, (p, n) in enumerate(zip(PROMPTS, LENGTHS, strict=True))
    ]


def test_batching_does_not_change_the_tokens_a_request_receives(
    model, tokenizer, make_sequence
):
    expected = oracle(model, tokenizer, make_sequence)

    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=4)
    sequences = [
        make_sequence(str(i), p, n)
        for i, (p, n) in enumerate(zip(PROMPTS, LENGTHS, strict=True))
    ]
    for sequence in sequences:
        engine.add(sequence)
    drain(engine)

    assert [s.tokens for s in sequences] == expected


def test_requests_admitted_mid_flight_get_the_same_tokens(
    model, tokenizer, make_sequence
):
    """Joining a batch that is already decoding exercises cache merging, left
    padding and eviction all at once. The tokens must still be untouched."""
    expected = oracle(model, tokenizer, make_sequence)

    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=3)
    sequences = [
        make_sequence(str(i), p, n)
        for i, (p, n) in enumerate(zip(PROMPTS, LENGTHS, strict=True))
    ]
    pending = list(sequences)
    engine.add(pending.pop(0))

    step = 0
    while (engine.has_work() or pending) and step < 500:
        if pending and step in (2, 5, 9):
            engine.add(pending.pop(0))
        engine.step()
        step += 1

    assert [s.tokens for s in sequences] == expected


def test_engine_never_runs_more_sequences_than_the_batch_allows(
    model, tokenizer, make_sequence
):
    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=2)
    for i, (p, n) in enumerate(zip(PROMPTS, LENGTHS, strict=True)):
        engine.add(make_sequence(str(i), p, n))

    while engine.has_work():
        engine.step()
        assert len(engine.running) <= 2


def test_finished_sequences_leave_the_batch_immediately(
    model, tokenizer, make_sequence
):
    """A short request must not be held hostage by a long one sharing its batch,
    which is precisely what static batching gets wrong."""
    short = make_sequence("short", [5, 6], 2)
    long = make_sequence("long", [5, 6], 20)

    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=2)
    engine.add(short)
    engine.add(long)

    while not short.done:
        engine.step()
        assert len(long.tokens) < long.max_new_tokens

    engine.step()  # the eviction happens at the top of the next iteration
    assert short not in engine.running
    assert long in engine.running


def test_step_reports_the_sequences_that_finished_in_it(
    model, tokenizer, make_sequence
):
    sequence = make_sequence("a", [5, 6], 3)
    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=1)
    engine.add(sequence)

    finished: list[Sequence] = []
    while engine.has_work():
        finished.extend(engine.step())

    assert [s.id for s in finished] == ["a"]
    assert sequence.finished_at is not None
    assert sequence.time_to_first_token() is not None


def test_cache_is_released_once_everything_finishes(model, tokenizer, make_sequence):
    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=2)
    engine.add(make_sequence("a", [5, 6], 2))
    drain(engine)

    assert engine.running == []
    assert engine.cache is None
    assert engine.cache_len == 0


# ------------------------------------------------------------------ cancellation
#
# `cancel` is the only place other than eviction that performs cache row surgery,
# and it runs on a client-facing path (the runner's timeout). These tests pin down
# both halves of it: the bookkeeping, and the property that matters -- cancelling
# one member of a batch must not disturb what the others generate.


def test_cancelling_the_only_running_sequence_releases_the_cache(
    model, tokenizer, make_sequence
):
    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=2)
    sequence = make_sequence("a", [5, 6], 20)
    engine.add(sequence)
    engine.step()  # prefill; the sequence is now the single cache row

    assert engine.running == [sequence]
    assert engine.cancel("a") is True

    assert engine.running == []
    assert engine.cache is None
    assert engine.cache_len == 0
    assert not engine.has_work()


def test_cancelling_one_row_matches_a_normal_eviction_of_that_row(
    model, tokenizer, make_sequence
):
    """Cancellation must leave the cache byte-identical to ordinary eviction.

    The reference arm drives the same batch to the same point and then removes the
    same row the way a finished sequence is removed, so any divergence is `cancel`'s
    own row surgery and nothing else.
    """

    def primed():
        engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=3)
        sequences = [
            make_sequence(str(i), PROMPTS[i], 12) for i in (0, 1, 2)
        ]
        for sequence in sequences:
            engine.add(sequence)
        for _ in range(4):
            engine.step()
        return engine, sequences

    cancelled, _ = primed()
    assert cancelled.cancel("1") is True

    evicted, evicted_sequences = primed()
    victim = evicted_sequences[1]
    victim.max_new_tokens = len(victim.tokens)  # it is now `done`
    assert victim.done
    evicted._evict_finished()

    assert [s.id for s in cancelled.running] == ["0", "2"]
    assert [s.id for s in evicted.running] == ["0", "2"]
    assert cancelled.cache_len == evicted.cache_len
    assert [s.cached_len for s in cancelled.running] == [
        s.cached_len for s in evicted.running
    ]

    cancelled_layers = cache_tensors(cancelled.cache)
    evicted_layers = cache_tensors(evicted.cache)
    assert len(cancelled_layers) == len(evicted_layers)
    for (keys, values), (ref_keys, ref_values) in zip(
        cancelled_layers, evicted_layers, strict=True
    ):
        assert keys.shape == ref_keys.shape
        assert torch.equal(keys, ref_keys)
        assert torch.equal(values, ref_values)


def test_cancelling_a_row_does_not_change_what_the_others_generate(
    model, tokenizer, make_sequence
):
    """The property that matters: a neighbour giving up is invisible to you."""
    expected = {
        str(i): generate_sequentially(
            model, tokenizer, make_sequence(str(i), PROMPTS[i], LENGTHS[i])
        )
        for i in (0, 2)
    }

    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=3)
    sequences = [make_sequence(str(i), PROMPTS[i], LENGTHS[i]) for i in (0, 1, 2)]
    for sequence in sequences:
        engine.add(sequence)

    engine.step()  # prefill: one token each
    engine.step()  # decode:  two tokens each
    assert [len(s.tokens) for s in sequences] == [2, 2, 2]

    assert engine.cancel("1") is True
    assert [s.id for s in engine.running] == ["0", "2"]
    drain(engine)

    assert sequences[0].tokens == expected["0"]
    assert sequences[2].tokens == expected["2"]
    # The cancelled request keeps what it had and is never advanced again.
    assert len(sequences[1].tokens) == 2


def test_cancelling_a_waiting_sequence_leaves_the_running_batch_alone(
    model, tokenizer, make_sequence
):
    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=1)
    running = make_sequence("running", PROMPTS[0], 8)
    queued = make_sequence("queued", PROMPTS[1], 8)
    engine.add(running)
    engine.add(queued)

    engine.step()  # the batch holds one, so "queued" is still in `waiting`
    assert engine.running == [running]
    assert [s.id for s in engine.waiting] == ["queued"]
    cache_before, cache_len_before = engine.cache, engine.cache_len

    assert engine.cancel("queued") is True

    assert list(engine.waiting) == []
    assert engine.running == [running]
    assert engine.cache is cache_before
    assert engine.cache_len == cache_len_before
    assert queued.tokens == []


def test_cancelling_an_unknown_id_is_a_no_op(model, tokenizer, make_sequence):
    engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=2)
    assert engine.cancel("never-existed") is False  # nothing running at all

    sequence = make_sequence("0", PROMPTS[0], LENGTHS[0])
    engine.add(sequence)
    engine.step()
    cache_before, cache_len_before = engine.cache, engine.cache_len
    tokens_before = list(sequence.tokens)

    assert engine.cancel("never-existed") is False
    assert engine.running == [sequence]
    assert list(engine.waiting) == []
    assert engine.cache is cache_before
    assert engine.cache_len == cache_len_before
    assert sequence.tokens == tokens_before

    drain(engine)
    assert sequence.tokens == generate_sequentially(
        model, tokenizer, make_sequence("0", PROMPTS[0], LENGTHS[0])
    )
