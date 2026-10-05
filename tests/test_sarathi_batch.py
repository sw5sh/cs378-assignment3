#!/usr/bin/env python3
"""Part 3: Sarathi batch building, one test per rule, grouped by function.
Run with ``pytest tests/test_sarathi_batch.py``."""
from __future__ import annotations

import signal
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidur.entities import Request  # noqa: E402
from vidur.scheduler.replica_scheduler.sarathi_replica_scheduler import (  # noqa: E402
    BatchInProgress,
    SarathiReplicaScheduler,
)


def _prompt(prompt_len: int, done: int = 0, arrived_at: float = 0.0) -> Request:
    """A request still on its prompt, `done` prompt tokens already processed."""
    return Request(arrived_at=arrived_at, num_prefill_tokens=prompt_len,
                   num_decode_tokens=8, num_processed_tokens=done)


def _decoding(arrived_at: float = 0.0) -> Request:
    """A request whose prompt is done and that is generating output."""
    r = Request(arrived_at=arrived_at, num_prefill_tokens=8,
                num_decode_tokens=8, num_processed_tokens=9)
    r._is_prefill_complete = True
    return r


def _sched(budget=512, max_batch=8, batch_size_cap=32, no_memory=(), preempt=()):
    """A Sarathi scheduler with memory faked out.

    no_memory: requests that _can_allocate_request refuses.
    preempt:   decoding requests that _reserve_kv_or_preempt preempts.
    """
    s = object.__new__(SarathiReplicaScheduler)
    s._config = SimpleNamespace(chunk_size=budget, batch_size_cap=batch_size_cap)
    s._max_micro_batch_size = max_batch
    s._replica_id = 0
    s._preempted_requests = []
    s._request_queue = []
    s._allocation_map = {}
    s.reserved = []
    no_memory_ids = {r.id for r in no_memory}
    preempt_ids = {r.id for r in preempt}

    def allocate(req):
        s._allocation_map.setdefault(req.id, 1)

    def reserve(req):
        s.reserved.append(req.id)
        if req.id in preempt_ids:
            return False
        allocate(req)
        return True

    s._can_allocate_request = lambda req: req.id not in no_memory_ids
    s._allocate_request = allocate
    s._reserve_kv_or_preempt = reserve
    return s


def _guard(fn, *args, seconds=5):
    """Call fn(*args), failing instead of hanging on an endless loop."""
    if not hasattr(signal, "SIGALRM"):  # Windows: no timeout available
        return fn(*args)

    def timeout(signum, frame):
        raise TimeoutError(
            f"{fn.__name__} did not return within {seconds} s: an endless loop? "
            "(e.g. a request that is never popped, or `continue` where you must stop)"
        )

    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(seconds)
    try:
        return fn(*args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _batch_with(*tokens) -> BatchInProgress:
    """A batch that already holds requests adding these token counts."""
    b = BatchInProgress()
    for t in tokens:
        b.add(_decoding(), t)
    return b


def _ids(requests):
    return [r.id for r in requests]


# --- step 1: _add_running_decodes ---------------------------------------------

def test_step1_decodes_get_one_token_and_prompts_are_set_aside():
    p0, d1, p2, d3 = _prompt(100, arrived_at=0.0), _decoding(1.0), _prompt(50, arrived_at=2.0), _decoding(3.0)
    s = _sched()
    s._preempted_requests = [p0, d1, p2, d3]
    batch = BatchInProgress()
    prompts = _guard(s._add_running_decodes, batch)
    assert _ids(batch.requests) == _ids([d1, d3]), "only decoding requests go into the batch in step 1"
    assert batch.num_tokens == [1, 1], f"one token each: got {batch.num_tokens}"
    assert _ids(prompts or []) == _ids([p0, p2]), "return the requests still on their prompt, in order"
    assert s._preempted_requests == [], "every running request is popped"


def test_step1_decodes_reserve_memory_through_the_helper():
    d1, d2 = _decoding(0.0), _decoding(1.0)
    s = _sched()
    s._preempted_requests = [d1, d2]
    _guard(s._add_running_decodes, BatchInProgress())
    assert s.reserved == _ids([d1, d2]), "each decoding request must call self._reserve_kv_or_preempt(request)"


def test_step1_preempted_decode_is_left_out():
    d1, d2 = _decoding(0.0), _decoding(1.0)
    s = _sched(preempt=[d1])
    s._preempted_requests = [d1, d2]
    batch = BatchInProgress()
    _guard(s._add_running_decodes, batch)
    assert _ids(batch.requests) == _ids([d2]), (
        "_reserve_kv_or_preempt returned False for the first request: leave it out, keep going"
    )


def test_step1_stops_at_max_batch():
    ds = [_decoding(float(i)) for i in range(3)]
    s = _sched(max_batch=2)
    s._preempted_requests = list(ds)
    batch = BatchInProgress()
    _guard(s._add_running_decodes, batch)
    assert _ids(batch.requests) == _ids(ds[:2]), "stop once the batch holds self._max_micro_batch_size"
    assert _ids(s._preempted_requests) == _ids(ds[2:]), "the request not taken stays in self._preempted_requests"


# --- step 2: _add_unfinished_prompts ------------------------------------------

def test_step2_prompt_gets_the_rest_of_the_budget():
    p = _prompt(1000, done=100)
    s = _sched(budget=512)
    batch = _batch_with(1, 1)
    kept = s._add_unfinished_prompts(batch, [p])
    assert batch.num_tokens == [1, 1, 510] and kept == [], (
        f"2 tokens already in, so the prompt gets 510 of 512: got {batch.num_tokens}"
    )


def test_step2_decode_tokens_count_against_the_budget():
    p = _prompt(100)
    s = _sched(budget=4)
    batch = _batch_with(1, 1, 1)
    s._add_unfinished_prompts(batch, [p])
    assert batch.num_tokens == [1, 1, 1, 1], f"3 tokens already in leave 1 of 4: got {batch.num_tokens}"


def test_step2_prompt_waits_when_budget_used_up():
    first, second = _prompt(64, arrived_at=0.0), _prompt(64, arrived_at=1.0)
    s = _sched(budget=64)
    batch = BatchInProgress()
    kept = s._add_unfinished_prompts(batch, [first, second])
    assert _ids(batch.requests) == _ids([first]) and batch.num_tokens == [64], (
        "the first prompt takes the whole budget"
    )
    assert _ids(kept or []) == _ids([second]), "return the prompt that got 0 tokens; do not add it"


# --- step 3: _return_kept_prompts ---------------------------------------------

def test_step3_kept_prompts_go_back_in_arrival_order():
    p1 = _prompt(64, arrived_at=1.0)
    d0, d4 = _decoding(0.0), _decoding(4.0)
    s = _sched()
    s._preempted_requests = [d0, d4]
    s._return_kept_prompts([p1])
    got = [r.arrived_at for r in s._preempted_requests]
    assert got == [0.0, 1.0, 4.0], f"merge and sort by arrived_at: got arrivals {got}, expected [0.0, 1.0, 4.0]"


def test_step3_kept_prompts_go_ahead_of_equal_arrivals():
    kept, other = _prompt(64, arrived_at=1.0), _decoding(1.0)
    s = _sched()
    s._preempted_requests = [other]
    s._return_kept_prompts([kept])
    assert _ids(s._preempted_requests) == _ids([kept, other]), (
        "kept prompts go ahead of the requests still there (sorting keeps that order for ties)"
    )


# --- step 4: _add_new_requests ------------------------------------------------

def test_step4_starts_oldest_first_while_budget_left():
    a, b = _prompt(20, arrived_at=0.0), _prompt(20, arrived_at=1.0)
    s = _sched(budget=512)
    s._request_queue = [a, b]
    batch = BatchInProgress()
    _guard(s._add_new_requests, batch)
    assert _ids(batch.requests) == _ids([a, b]) and batch.num_tokens == [20, 20], "both fit: both start"
    assert s._request_queue == [], "started requests leave the waiting queue"
    assert set(s._allocation_map) == {a.id, b.id}, "reserve memory with self._allocate_request"


def test_step4_new_requests_share_the_budget():
    a, b = _prompt(50, arrived_at=0.0), _prompt(50, arrived_at=1.0)
    s = _sched(budget=64)
    s._request_queue = [a, b]
    batch = BatchInProgress()
    _guard(s._add_new_requests, batch)
    assert batch.num_tokens == [50, 14], (
        f"the second request gets only the 14 tokens left of 64: got {batch.num_tokens}"
    )


def test_step4_stops_when_budget_used_up():
    a, b = _prompt(50, arrived_at=1.0), _prompt(10, arrived_at=2.0)
    s = _sched(budget=64)
    s._request_queue = [a, b]
    batch = _batch_with(64)
    _guard(s._add_new_requests, batch)
    assert len(batch) == 1 and _ids(s._request_queue) == _ids([a, b]), (
        "no budget left: stop at the oldest waiting request; a later one must not start ahead of it"
    )


def test_step4_stops_when_memory_full():
    a, b = _prompt(50, arrived_at=0.0), _prompt(10, arrived_at=1.0)
    s = _sched(no_memory=[a])
    s._request_queue = [a, b]
    batch = BatchInProgress()
    _guard(s._add_new_requests, batch)
    assert len(batch) == 0 and _ids(s._request_queue) == _ids([a, b]), (
        "the oldest waiting request does not fit in memory: stop, do not start a later one"
    )


def test_step4_stops_at_batch_size_cap():
    a, b = _prompt(10, arrived_at=1.0), _prompt(10, arrived_at=2.0)
    s = _sched(batch_size_cap=2)
    s._allocation_map = {-1: 1}  # one request already holds memory
    s._request_queue = [a, b]
    batch = BatchInProgress()
    _guard(s._add_new_requests, batch)
    assert _ids(batch.requests) == _ids([a]), (
        "stop when len(self._allocation_map) == self._config.batch_size_cap (2 here)"
    )


def test_step4_stops_at_max_batch():
    queue = [_prompt(10, arrived_at=float(i)) for i in range(3)]
    s = _sched(max_batch=2)
    s._request_queue = list(queue)
    batch = BatchInProgress()
    _guard(s._add_new_requests, batch)
    assert _ids(batch.requests) == _ids(queue[:2]), "stop when the batch holds self._max_micro_batch_size"


# --- all four steps through _get_next_batch -----------------------------------

def test_decodes_go_before_an_older_prompt():
    p, d1, d2 = _prompt(100, arrived_at=0.0), _decoding(1.0), _decoding(2.0)
    s = _sched()
    s._preempted_requests = [p, d1, d2]
    batch = _guard(s._get_next_batch)
    assert _ids(batch.requests) == _ids([d1, d2, p]) and list(batch.num_tokens) == [1, 1, 100], (
        "decoding requests go in before a prompt in progress, even an older one"
    )


def test_empty_batch_is_none():
    assert _guard(_sched()._get_next_batch) is None, "nothing to run: return None"


def test_worked_example_two_iterations():
    # Budget 512, three requests decoding, a prompt with 900 tokens left,
    # a new 300-token prompt waiting.
    decoding = [_decoding(float(t)) for t in (0, 1, 2)]
    long_prompt = _prompt(1000, done=100, arrived_at=3.0)
    new_prompt = _prompt(300, arrived_at=4.0)
    s = _sched(budget=512)
    s._preempted_requests = decoding + [long_prompt]
    s._request_queue = [new_prompt]

    first = _guard(s._get_next_batch)
    toks = list(first.num_tokens)
    assert toks == [1, 1, 1, 509], f"iteration 1: got {toks}, expected [1, 1, 1, 509]"
    assert s._request_queue == [new_prompt], "iteration 1: the new prompt must still be waiting"

    for request, n in zip(first.requests, first.num_tokens):  # finish it as the replica does
        request.on_batch_end(1.0, n)
    s._preempted_requests = list(first.requests)
    toks = list(_guard(s._get_next_batch).num_tokens)
    assert toks == [1, 1, 1, 391, 118], f"iteration 2: got {toks}, expected [1, 1, 1, 391, 118]"
