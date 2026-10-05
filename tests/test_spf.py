#!/usr/bin/env python3
"""Part 4: shortest prompt first, with aging. Run with ``pytest tests/test_spf.py``."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidur.entities import Request  # noqa: E402
from vidur.scheduler.replica_scheduler.spf_replica_scheduler import (  # noqa: E402
    SpfReplicaScheduler,
)


def _req(prompt: int, done: int = 0, arrived_at: float = 0.0) -> Request:
    return Request(arrived_at=arrived_at, num_prefill_tokens=prompt,
                   num_decode_tokens=8, num_processed_tokens=done)


def _sched(aging: float = 0.0, now: float = 0.0, queue=()):
    s = object.__new__(SpfReplicaScheduler)
    s._config = SimpleNamespace(aging_tokens_per_s=aging)
    s._now = now
    s._request_queue = list(queue)
    return s


def _prompts(queue):
    return [r.num_prefill_tokens for r in queue]


# --- _priority ------------------------------------------------------------------

def test_priority_is_prompt_tokens_left_without_aging():
    s = _sched(aging=0.0, now=10.0)
    got = s._priority(_req(1000, done=300, arrived_at=2.0), 10.0)
    assert got == 700, f"1000-token prompt with 300 done: 700 tokens left, got {got}"


def test_priority_subtracts_aging_times_seconds_waited():
    s = _sched(aging=50.0, now=10.0)
    got = s._priority(_req(1000, arrived_at=4.0), 10.0)
    assert got == 700, f"1000 tokens - 50 tokens/s * 6 s waited = 700, got {got}"


# --- _order_waiting_queue -------------------------------------------------------

def test_order_shortest_prompt_first():
    s = _sched(queue=[_req(1000), _req(16), _req(400)])
    s._order_waiting_queue()
    assert _prompts(s._request_queue) == [16, 400, 1000], (
        f"shortest first: got {_prompts(s._request_queue)}"
    )


def test_order_sorts_in_place_without_dropping():
    queue = [_req(1000), _req(16), _req(400)]
    s = _sched(queue=queue)
    original = s._request_queue
    s._order_waiting_queue()
    assert s._request_queue is original, "sort self._request_queue in place (do not replace the list)"
    assert sorted(r.id for r in s._request_queue) == sorted(r.id for r in queue), "no request dropped or copied"


def test_order_ties_keep_arrival_order():
    a, b, c = _req(100, arrived_at=0.0), _req(100, arrived_at=1.0), _req(50, arrived_at=2.0)
    s = _sched(queue=[a, b, c])
    s._order_waiting_queue()
    assert [r.id for r in s._request_queue] == [c.id, a.id, b.id], (
        "equal priority keeps the current order (Python's sort is stable)"
    )


def test_aging_lets_an_old_long_prompt_overtake():
    # At t = 100 s with aging 50: the long prompt waited 100 s -> 4000 - 5000 = -1000;
    # the short one waited 1 s -> 100 - 50 = 50. The long prompt goes first.
    long_, short = _req(4000, arrived_at=0.0), _req(100, arrived_at=99.0)
    s = _sched(aging=50.0, now=100.0, queue=[short, long_])
    s._order_waiting_queue()
    assert s._request_queue[0] is long_, "after 100 s of waiting, the long prompt overtakes (use self._now)"


def test_aging_has_not_caught_up_yet():
    # At t = 10 s: long prompt 4000 - 500 = 3500; short 100 - 50 = 50. Short first.
    long_, short = _req(4000, arrived_at=0.0), _req(100, arrived_at=9.0)
    s = _sched(aging=50.0, now=10.0, queue=[long_, short])
    s._order_waiting_queue()
    assert s._request_queue[0] is short, "after only 10 s, the short prompt still goes first"
