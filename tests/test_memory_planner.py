#!/usr/bin/env python3
"""Warmup: MemoryPlanner byte counts and capacity for several models and GPUs.
Run with ``pytest tests/test_memory_planner.py``."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidur.config import ReplicaConfig, TraceRequestGeneratorConfig
from vidur.entities.replica import Replica
from vidur.scheduler.utils.memory_planner import MemoryPlanner


def _make_replica(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    device: str = "a100",
    tensor_parallel_size: int = 1,
    num_pipeline_stages: int = 1,
    max_tokens: int = 4096,
):
    replica_config = ReplicaConfig(
        model_name=model_name,
        device=device,
        tensor_parallel_size=tensor_parallel_size,
        num_pipeline_stages=num_pipeline_stages,
    )
    generator_config = TraceRequestGeneratorConfig(max_tokens=max_tokens)
    replica = Replica(replica_config, generator_config)
    return replica_config, replica


# kv_per_layer, param_per_device, kv_per_device, max_batch, max_slots
EXPECTED = {
    "llama2-7b/a100/tp1/pp1/4k": (67108864, 12952010752, 2147483648, 29, 29),
    "a40": (67108864, 12952010752, 2147483648, 14, 14),
    "tp2": (33554432, 6476005376, 1073741824, 65, 65),
    "pp2": (67108864, 6476005376, 2147483648, 32, 64),
    "max_tokens_2048": (33554432, 12952010752, 1073741824, 59, 59),
    "llama3-8b": (16777216, 13958643712, 536870912, 118, 118),
}

CASES = [
    ("llama2-7b/a100/tp1/pp1/4k", {}),
    ("a40", {"device": "a40"}),
    ("tp2", {"tensor_parallel_size": 2}),
    ("pp2", {"num_pipeline_stages": 2}),
    ("max_tokens_2048", {"max_tokens": 2048}),
    ("llama3-8b", {"model_name": "meta-llama/Meta-Llama-3-8B"}),
]

FIELDS = (
    "_get_kv_cache_memory_per_layer_per_request (bytes)",
    "_get_parameter_memory_per_device (bytes)",
    "_get_kv_cache_memory_per_device_per_request (bytes)",
    "get_max_batch_size (requests)",
    "get_max_request_slots (requests)",
)


def _check(label, kwargs):
    replica_config, replica = _make_replica(**kwargs)
    planner = MemoryPlanner(replica_config, replica)
    got = (
        planner._get_kv_cache_memory_per_layer_per_request(),
        planner._get_parameter_memory_per_device(),
        planner._get_kv_cache_memory_per_device_per_request(),
        planner.get_max_batch_size(),
        planner.get_max_request_slots(),
    )
    wrong = [
        f"{field}: got {g}, expected {w}"
        for field, g, w in zip(FIELDS, got, EXPECTED[label])
        if g != w
    ]
    assert not wrong, f"{label}: " + "; ".join(wrong)


def _make_test(label, kwargs):
    def test():
        _check(label, kwargs)
    return test


for _label, _kwargs in CASES:
    globals()["test_" + re.sub(r"\W+", "_", _label).strip("_")] = _make_test(_label, _kwargs)
