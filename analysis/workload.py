"""Part 1: workload characterization and back-of-the-envelope estimates.

You fill the four functions marked ``# [Part 1] TODO``. Everything else is
provided; do not modify it.

    load_trace          -> the trace as Vidur will see it (provided)
    arrival_rates       -> what arrives per time bin
    predict_queue_wait  -> predicted wait of every request on one GPU, no Vidur
    kv_bytes_per_token  -> KV-cache memory one token needs
    kv_in_use           -> KV-cache memory in use over time
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

MAX_TOKENS = 4096  # --trace_request_generator_config_max_tokens in the wrappers

# How fast one GPU works, Llama-3-8B on one A100, measured in Vidur.
#
# Prompts: a prompt is processed in one pass, about 12,000 prompt tokens per
# second (a 2048-token prompt takes about 0.17 s).
PREFILL_TOK_PER_S = 12_000

# Output: one decode step runs the model once and makes one token for each
# request in the batch, up to MAX_BATCH_SIZE requests (vLLM's max_num_seqs,
# Vidur's batch_size_cap). A request gets one token per step. A full batch
# of 128 at this trace's context lengths takes about 30 ms per step.
MAX_BATCH_SIZE = 128
DECODE_STEP_S = 30e-3
# So one GPU makes at most 128 output tokens every 30 ms.
DECODE_TOK_PER_S = MAX_BATCH_SIZE / DECODE_STEP_S

# Llama-3-8B shape, for the KV cache. Every token stores a key and a value
# vector in every layer, for each KV head.
NUM_LAYERS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
BYTES_PER_VALUE = 2  # fp16
MODEL_WEIGHTS_GB = 16.06  # 8.03B parameters * 2 bytes

PathLike = Union[str, Path]


def load_trace(path: PathLike, max_tokens: int = MAX_TOKENS) -> pd.DataFrame:
    """Read a trace CSV and apply the same clipping Vidur's trace replay does.

    Each request keeps at least one prompt and one output token. If
    prompt + output exceeds ``max_tokens``, the prompt is shortened. Rows are
    sorted by ``arrived_at``. Provided; do not modify.
    """
    df = pd.read_csv(path)
    df["num_prefill_tokens"] = df["num_prefill_tokens"].astype(int).clip(lower=1)
    df["num_decode_tokens"] = df["num_decode_tokens"].astype(int).clip(lower=1)
    excess = (df["num_prefill_tokens"] + df["num_decode_tokens"] - max_tokens).clip(lower=0)
    df["num_prefill_tokens"] = df["num_prefill_tokens"] - excess
    return df.sort_values("arrived_at", kind="stable").reset_index(drop=True)


def arrival_rates(df: pd.DataFrame, bin_s: float) -> pd.DataFrame:
    """Per-bin arrival rates.

    df:    trace from load_trace, one row per request, sorted by arrived_at:
               arrived_at          arrival time, seconds of trace time
               num_prefill_tokens  prompt length (tokens)
               num_decode_tokens   output length (tokens)
    bin_s: bin width in seconds (the notebook uses 30.0)

    Bin k covers ``[k * bin_s, (k + 1) * bin_s)`` in trace time, for
    k = 0 .. floor(max(arrived_at) / bin_s). Empty bins are kept (all zeros).

    Returns a DataFrame with one row per bin, in order, and columns
        bin_start         k * bin_s
        req_per_s         requests arriving in the bin / bin_s
        prompt_tok_per_s  sum of num_prefill_tokens in the bin / bin_s
        output_tok_per_s  sum of num_decode_tokens in the bin / bin_s
    """
    # [Part 1] TODO: bin the arrivals and return the four columns above.
    # HINT: no Python loop needed; see np.bincount.
    raise NotImplementedError("Part 1: fill arrival_rates")
    # END OF YOUR CODE


def predict_queue_wait(
    df: pd.DataFrame,
    prefill_tok_per_s: float = PREFILL_TOK_PER_S,
    decode_tok_per_s: float = DECODE_TOK_PER_S,
) -> np.ndarray:
    """Estimate how long each request waits before one GPU starts on it.

    Treat the GPU as serving requests one at a time, in arrival order, at its
    full prompt and output throughput. Each request then takes
        work = num_prefill_tokens / prefill_tok_per_s
               + num_decode_tokens / decode_tok_per_s   (seconds)
    A request starts when it has arrived and the GPU is free:
        start = max(arrived_at, time the GPU is next free)
        wait  = start - arrived_at
    and the GPU is next free at start + work.

    df: trace from load_trace, one row per request, sorted by arrived_at:
            arrived_at          arrival time, seconds of trace time
            num_prefill_tokens  prompt length (tokens)
            num_decode_tokens   output length (tokens)

    Returns one wait in seconds per row of df, as a numpy array.
    """
    # [Part 1] TODO: apply the rule above to each request in order.
    raise NotImplementedError("Part 1: fill predict_queue_wait")
    # END OF YOUR CODE


def kv_bytes_per_token() -> int:
    """Bytes of KV cache one token needs, using the model shape constants
    above (NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, BYTES_PER_VALUE).
    """
    # [Part 1] TODO: one key and one value vector per layer, per KV head.
    raise NotImplementedError("Part 1: fill kv_bytes_per_token")
    # END OF YOUR CODE


def kv_in_use(
    df: pd.DataFrame,
    waits: np.ndarray,
    bin_s: float,
    prefill_tok_per_s: float = PREFILL_TOK_PER_S,
    decode_step_s: float = DECODE_STEP_S,
    max_batch_size: int = MAX_BATCH_SIZE,
) -> pd.DataFrame:
    """KV-cache memory in use over time, on one GPU.

    A request holds KV for all its tokens (prompt + output) from when it
    arrives until it is done:
        done = arrived_at + wait
               + num_prefill_tokens / prefill_tok_per_s
               + decode_step_s * num_decode_tokens   (one step per output token)
    Only requests the GPU is running hold KV. vLLM runs at most max_batch_size
    at once, admitted in arrival order; the rest wait in the queue and hold
    no KV yet. So at each t, only the earliest max_batch_size requests in the
    system count.

    df:    trace from load_trace (arrived_at, num_prefill_tokens,
           num_decode_tokens), sorted by arrived_at
    waits: predict_queue_wait(df), one wait per row of df
    bin_s: evaluate at t = 0, bin_s, 2 * bin_s, ... up to the last arrival
           (the same bin starts as arrival_rates)

    A request is in the system at time t if arrived_at <= t < done.

    Returns a DataFrame with one row per t and columns
        bin_start  t
        running    requests the GPU is running (holding KV) at t
        kv_gb      their prompt + output tokens * kv_bytes_per_token() / 1e9
    """
    # [Part 1] TODO: for each t, find the requests in the system, keep the
    # first max_batch_size of them, and sum their KV.
    raise NotImplementedError("Part 1: fill kv_in_use")
    # END OF YOUR CODE

