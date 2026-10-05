"""Read Vidur runs (Parts 2-5). Provided; nothing to fill in."""

import json
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from analysis.workload import kv_bytes_per_token

BLOCK_SIZE = 16  # tokens per KV-cache block in Vidur

PathLike = Union[str, Path]


def load_run(run_dir: PathLike, trace: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Read one Vidur run (a simulator_output/<timestamp>/ directory).

    trace: the trace the run replayed, from load_trace. Request i in Vidur is
    row i of the trace.

    Returns three tables:
        requests  one row per request, in trace order: the trace columns plus,
                  in seconds, wait (request_scheduling_delay), ttft
                  (prefill_e2e_time), and e2e (request_e2e_time)
        tbt       time between output tokens, over every token of every
                  request: percentile (0-100) and seconds
        kv        KV-cache memory over time: time, kv_gb
    """
    run_dir = Path(run_dir)
    metrics = pd.read_csv(run_dir / "request_metrics.csv").sort_values("Request Id")
    requests = trace.reset_index(drop=True).copy()
    requests["wait"] = metrics["request_scheduling_delay"].to_numpy()
    requests["ttft"] = metrics["prefill_e2e_time"].to_numpy()
    requests["e2e"] = metrics["request_e2e_time"].to_numpy()

    cdf = pd.read_csv(run_dir / "plots" / "decode_token_execution_plus_preemption_time.csv")
    tbt = pd.DataFrame({
        "percentile": (cdf["cdf"] * 100).round().astype(int),
        "seconds": cdf["decode_token_execution_plus_preemption_time"],
    })

    kv = pd.read_csv(run_dir / "replica_1_kv_usage.csv")
    kv["kv_gb"] = kv["blocks_used"] * BLOCK_SIZE * kv_bytes_per_token() / 1e9
    return {"requests": requests, "tbt": tbt, "kv": kv[["time", "kv_gb"]]}


def describe_run(run_dir: PathLike) -> str:
    """How a run was configured, e.g. "vllm", "sarathi C=512",
    "spf C=512 aging=100", or "sarathi C=512, 2 replicas, lor"."""
    config = json.loads((Path(run_dir) / "config.json").read_text())
    cluster = config["cluster_config"]
    scheduler = cluster["replica_scheduler_config"]
    if "aging_tokens_per_s" in scheduler:
        label = f"{scheduler['name']} C={scheduler['chunk_size']} aging={scheduler['aging_tokens_per_s']:g}"
    elif "chunk_size" in scheduler:
        label = f"{scheduler['name']} C={scheduler['chunk_size']}"
    else:
        label = scheduler["name"]
    if cluster["num_replicas"] > 1:
        label += f", {cluster['num_replicas']} replicas, {cluster['global_scheduler_config']['name']}"
    return label


def busy_percent(run_dir: PathLike) -> list:
    """Percent of the run each replica spent running batches, one value per
    replica, in replica order."""
    plots = Path(run_dir) / "plots"
    values = []
    for i in range(1, len(list(plots.glob("replica_*_stage_1_busy_time_percent.json"))) + 1):
        stats = json.loads((plots / f"replica_{i}_stage_1_busy_time_percent.json").read_text())
        values.append(stats[f"replica_{i}_stage_1_busy_time_percent_weighted_mean"])
    return values
