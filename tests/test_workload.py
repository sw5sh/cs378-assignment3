#!/usr/bin/env python3
"""Part 1 workload functions. Run with ``pytest tests/test_workload.py`` or
``python tests/test_workload.py``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.workload import (  # noqa: E402
    arrival_rates,
    kv_bytes_per_token,
    kv_in_use,
    predict_queue_wait,
)


def _trace(rows):
    return pd.DataFrame(rows, columns=["arrived_at", "num_prefill_tokens", "num_decode_tokens"])


# Worked example from the handout: 1000 prompt tokens/s, 100 output tokens/s.
WORKED = _trace([
    (0.0, 1000, 100),  # 1.0 + 1.0 = 2.0 s of work
    (0.5, 500, 50),    # 0.5 + 0.5 = 1.0 s of work
    (4.0, 200, 10),    # 0.2 + 0.1 = 0.3 s of work
])


def _close(got, want, what):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{what}: got shape {got.shape}, expected {want.shape}"
    assert np.allclose(got, want, atol=1e-9), f"{what}: got {got.tolist()}, expected {want.tolist()}"


# --- arrival_rates -----------------------------------------------------------

def test_arrival_rates_1s_bins():
    df = _trace([(0.0, 100, 10), (0.5, 300, 30), (1.2, 50, 5), (3.0, 7, 1)])
    out = arrival_rates(df, 1.0)
    assert len(out) == 4, (
        f"expected 4 bins [0,1) [1,2) [2,3) [3,4) (empty bin [2,3) kept), got {len(out)}"
    )
    _close(out["bin_start"], [0, 1, 2, 3], "bin_start")
    _close(out["req_per_s"], [2, 1, 0, 1], "req_per_s")
    _close(out["prompt_tok_per_s"], [400, 50, 0, 7], "prompt_tok_per_s")
    _close(out["output_tok_per_s"], [40, 5, 0, 1], "output_tok_per_s")


def test_arrival_rates_divides_by_bin_width():
    df = _trace([(0.0, 100, 10), (0.5, 300, 30), (1.2, 50, 5), (3.0, 7, 1)])
    out = arrival_rates(df, 2.0)
    _close(out["bin_start"], [0, 2], "bin_start with bin_s=2")
    _close(out["req_per_s"], [1.5, 0.5], "req_per_s with bin_s=2 (count / bin_s)")
    _close(out["prompt_tok_per_s"], [225, 3.5], "prompt_tok_per_s with bin_s=2")


# --- predict_queue_wait ------------------------------------------------------

def test_predict_queue_wait_worked_example():
    # A at 0.0: GPU free -> start 0.0, wait 0;     busy until 0.0 + 2.0 = 2.0
    # B at 0.5: GPU free at 2.0 -> start 2.0, wait 1.5; busy until 3.0
    # C at 4.0: GPU free at 3.0 -> start 4.0, wait 0;   busy until 4.3
    got = predict_queue_wait(WORKED, prefill_tok_per_s=1000, decode_tok_per_s=100)
    _close(got, [0.0, 1.5, 0.0], "worked example")


def test_predict_queue_wait_same_time_arrivals():
    # Two requests at t=100 after an idle GPU: the second waits for the first.
    df = _trace([(0.0, 100, 1), (100.0, 100, 1), (100.0, 100, 1)])
    got = predict_queue_wait(df, prefill_tok_per_s=100, decode_tok_per_s=float("inf"))
    _close(got, [0.0, 0.0, 1.0], "idle gap, then two arrivals at the same time")


# --- KV cache ----------------------------------------------------------------

def test_kv_bytes_per_token():
    got = kv_bytes_per_token()
    assert got == 131072, (
        f"got {got}; expected 2 (key and value) * 32 layers * 8 KV heads * 128 dims * 2 bytes = 131072"
    )


# 1000 prompt tokens/s, decode step = 10 ms. Waits given, so done times are:
#   A: 0.0 + 0.0 + 0.100 + 0.10 = 0.2      (110 tokens)
#   B: 0.1 + 0.1 + 1.000 + 1.00 = 2.2      (1100 tokens)
#   C: 0.5 + 1.6 + 0.010 + 0.01 = 2.12     (11 tokens)
#   D: 2.0 + 0.0 + 0.001 + 0.01 = 2.011    (2 tokens)
KV_TRACE = _trace([(0.0, 100, 10), (0.1, 1000, 100), (0.5, 10, 1), (2.0, 1, 1)])
KV_WAITS = np.array([0.0, 0.1, 1.6, 0.0])


def test_kv_in_use():
    # t=0: A        t=1: B, C        t=2: B, C, D
    out = kv_in_use(KV_TRACE, KV_WAITS, 1.0, prefill_tok_per_s=1000,
                    decode_step_s=0.01, max_batch_size=128)
    _close(out["bin_start"], [0, 1, 2], "bin_start (t = 0, 1, 2 up to the last arrival)")
    _close(out["running"], [1, 2, 3], "running (arrived_at <= t < done, earliest max_batch_size)")
    _close(out["kv_gb"], np.array([110, 1111, 1113]) * 131072 / 1e9,
           "kv_gb (running requests' prompt + output tokens * kv_bytes_per_token() / 1e9)")


def test_kv_in_use_caps_batch_size():
    # Only the earliest max_batch_size requests in the system hold KV.
    out = kv_in_use(KV_TRACE, KV_WAITS, 1.0, prefill_tok_per_s=1000,
                    decode_step_s=0.01, max_batch_size=1)
    _close(out["running"], [1, 1, 1], "running with max_batch_size=1")
    _close(out["kv_gb"], np.array([110, 1100, 1100]) * 131072 / 1e9,
           "kv_gb with max_batch_size=1 (earliest arrival in the system holds KV)")


def main() -> None:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # NotImplementedError, AssertionError, or a bug
            msg = str(exc) if isinstance(exc, AssertionError) else f"{type(exc).__name__}: {exc}"
            print(f"FAIL {name}: {msg}")
            fails += 1
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
