"""
The contract of continuous batching: sharing a forward pass with other requests
must not change a single token you receive. Everything else the engine does is
an optimisation, and these tests exist so an optimisation cannot quietly become
a wrong answer.
"""

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
