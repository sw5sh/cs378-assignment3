#!/usr/bin/env python3
"""Part 3: _get_request_next_num_tokens, the tokens a request adds to one step.
Run with ``pytest tests/test_sarathi_chunk.py``."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidur.entities import Request
from vidur.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)


def _sched(chunk_size: int = 512):
    sched = object.__new__(SarathiReplicaScheduler)
    sched._config = SimpleNamespace(chunk_size=chunk_size)
    return sched


def _req(prefill: int, processed: int = 0, decode_done: bool = False) -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=prefill,
        num_decode_tokens=8,
        num_processed_tokens=processed,
    )
    if decode_done:
        request._is_prefill_complete = True
    return request


# (label, request, contains_prefill, num_batch_tokens, budget C, expected)
CASES = [
    ("decode -> 1", _req(128, processed=128, decode_done=True), False, 0, 512, 1),
    ("full remaining prefill fits", _req(100, processed=0), True, 0, 512, 100),
    ("chunk leftover smaller than remaining", _req(1000, processed=100), True, 0, 512, 512),
    ("partial leftover", _req(100, processed=80), True, 500, 512, 12),
    ("clamp at 0 when batch already full", _req(100, processed=50), True, 512, 512, 0),
    # Decode tokens always get in: 100 requests decoding with C = 64 puts the
    # iteration 36 tokens over budget before any prompt is considered.
    ("clamp at 0 when decodes overflow the budget (C=64, 100 decoding)",
     _req(100, processed=0), True, 100, 64, 0),
]


def _make_test(label, request, contains_prefill, num_batch_tokens, chunk_size, expected):
    def test():
        got = _sched(chunk_size)._get_request_next_num_tokens(
            request, contains_prefill, num_batch_tokens
        )
        assert got == expected, (
            f"{label}: got {got}, expected {expected} "
            f"(budget {chunk_size}, {num_batch_tokens} tokens already in the iteration)"
        )
    return test


for _case in CASES:
    globals()["test_" + re.sub(r"\W+", "_", _case[0]).strip("_")] = _make_test(*_case)
