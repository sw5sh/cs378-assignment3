#!/usr/bin/env python3
"""Part 5: LOR and LOT routers. Run with ``pytest tests/test_routing.py``."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidur.entities import Request  # noqa: E402
from vidur.scheduler.global_scheduler.lor_global_scheduler import (  # noqa: E402
    LORGlobalScheduler,
)
from vidur.scheduler.global_scheduler.lot_global_scheduler import (  # noqa: E402
    LOTGlobalScheduler,
)


def _req(prompt=100, output=10, done=0, arrived_at=0.0, finished=False) -> Request:
    """done: tokens already processed (a running request has done > 0)."""
    r = Request(arrived_at=arrived_at, num_prefill_tokens=prompt,
                num_decode_tokens=output, num_processed_tokens=done)
    if finished:
        r._completed = True
    return r


def _router(cls, num_replicas=2, in_flight=None, new=()):
    """in_flight: {replica_id: [requests already sent there]}; new: the
    requests to route now."""
    s = object.__new__(cls)
    s._replica_schedulers = {rid: None for rid in range(num_replicas)}
    s._assigned = {rid: list((in_flight or {}).get(rid, [])) for rid in range(num_replicas)}
    s._request_queue = list(new)
    return s


def _route(s):
    mapping = s.schedule()
    assert s._request_queue == [], "assign every waiting request (pop them all)"
    return [rid for rid, _ in mapping], [r.id for _, r in mapping]


# --- LOR: fewest requests in flight -------------------------------------------

def test_lor_picks_fewest_in_flight():
    s = _router(LORGlobalScheduler, in_flight={0: [_req(), _req()], 1: [_req()]}, new=[_req()])
    replicas, _ = _route(s)
    assert replicas == [1], f"replica 0 has 2 in flight, replica 1 has 1: got {replicas}"


def test_lor_counts_running_requests_not_just_waiting():
    running = [_req(done=50), _req(done=50), _req(done=50)]
    s = _router(LORGlobalScheduler, in_flight={0: running, 1: [_req()]}, new=[_req()])
    replicas, _ = _route(s)
    assert replicas == [1], (
        f"replica 0 runs 3 requests, replica 1 has 1 waiting: running requests are in flight too. got {replicas}"
    )


def test_lor_ignores_finished_requests():
    done = [_req(finished=True), _req(finished=True)]
    s = _router(LORGlobalScheduler, in_flight={0: done, 1: [_req()]}, new=[_req()])
    replicas, _ = _route(s)
    assert replicas == [0], f"finished requests are not in flight: got {replicas}"


def test_lor_counts_each_assignment_right_away():
    burst = [_req(arrived_at=0.0) for _ in range(4)]
    replicas, _ = _route(_router(LORGlobalScheduler, new=burst))
    assert replicas == [0, 1, 0, 1], f"a burst of 4 on 2 empty replicas alternates: got {replicas}"


def test_lor_tie_goes_to_lowest_replica():
    s = _router(LORGlobalScheduler, num_replicas=3, in_flight={0: [_req()]}, new=[_req()])
    replicas, _ = _route(s)
    assert replicas == [1], f"replicas 1 and 2 tie at 0: pick the lowest id, got {replicas}"


def test_lor_assigns_oldest_first():
    late, early = _req(arrived_at=2.0), _req(arrived_at=1.0)
    _, ids = _route(_router(LORGlobalScheduler, new=[late, early]))
    assert ids == [early.id, late.id], "requests are assigned oldest first"


# --- LOT: least tokens in flight ----------------------------------------------

def test_lot_picks_least_work():
    s = _router(LOTGlobalScheduler, in_flight={0: [_req(prompt=990)], 1: [_req(prompt=90), _req(prompt=90)]},
                new=[_req()])
    replicas, _ = _route(s)
    assert replicas == [1], f"replica 0 has 1000 tokens left, replica 1 has 200: got {replicas}"


def test_lot_work_is_tokens_left():
    almost_done = _req(prompt=4990, output=10, done=4900)  # 100 tokens left
    s = _router(LOTGlobalScheduler, in_flight={0: [almost_done], 1: [_req(prompt=290)]}, new=[_req()])
    replicas, _ = _route(s)
    assert replicas == [0], (
        f"work is total_tokens - num_processed_tokens: 100 left on replica 0, 300 on replica 1. got {replicas}"
    )


def test_lot_ignores_finished_requests():
    s = _router(LOTGlobalScheduler, in_flight={0: [_req(prompt=5000, finished=True)], 1: [_req()]},
                new=[_req()])
    replicas, _ = _route(s)
    assert replicas == [0], f"finished requests have no work left: got {replicas}"


def test_lot_balances_a_burst_by_work():
    # The handout's worked example: huge, tiny, huge, tiny on 2 empty replicas.
    burst = [_req(prompt=3900, output=8), _req(prompt=16, output=8),
             _req(prompt=3900, output=8), _req(prompt=16, output=8)]
    replicas, _ = _route(_router(LOTGlobalScheduler, new=burst))
    assert replicas == [0, 1, 1, 0], (
        f"expected [0, 1, 1, 0]: one huge and one tiny prompt per replica. got {replicas}"
    )


def test_lot_tie_goes_to_lowest_replica():
    replicas, _ = _route(_router(LOTGlobalScheduler, num_replicas=3, new=[_req()]))
    assert replicas == [0], f"all replicas empty: pick the lowest id, got {replicas}"
