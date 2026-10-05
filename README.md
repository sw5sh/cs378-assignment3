# CS 378 Serving Assignment (Vidur)

A slim [Vidur](https://github.com/microsoft/vidur) fork for the language-model
serving assignment. See **`assignment.md`** for the full handout: setup, every
part, questions, submission, and grading.

The assignment follows one hour of real chat traffic
(`traces/azure_conversation.csv`) on Llama-3-8B and one A100-80GB GPU:
estimate where one GPU falls behind (Part 1), check it in the simulator
(Part 2), fix stalled streaming with a per-step token budget (Part 3), decide
who waits (Part 4), then add replicas and route between them (Part 5).

## Setup

Python **3.10+** (3.10 is the course default). From this directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Files you edit

| File | Part |
|---|---|
| `vidur/scheduler/utils/memory_planner.py` | Warmup — the full `MemoryPlanner` |
| `analysis/workload.py` | Part 1 — arrival rates, queue wait, KV cache in use |
| `vidur/scheduler/replica_scheduler/sarathi_replica_scheduler.py` | Part 3 — tokens per request and the four batch-building steps |
| `vidur/scheduler/replica_scheduler/spf_replica_scheduler.py` | Part 4 — priority with aging, and queue order |
| `vidur/scheduler/global_scheduler/lor_global_scheduler.py` | Part 5 — least outstanding requests router |
| `vidur/scheduler/global_scheduler/lot_global_scheduler.py` | Part 5 — least outstanding tokens router |

Fill only the `# [Warmup] TODO` / `# [Part N] TODO` blocks. Notebooks for each
part are in `notebooks/`.

## Running

```bash
bash scripts/run_tests.sh                       # every test (pytest)
QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv               # vLLM
SCHED=sarathi CHUNK=512 QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
SCHED=spf CHUNK=512 AGING=100 QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
REPLICAS=2 GLOBAL=lot SCHED=sarathi CHUNK=512 QPS=8 bash scripts/run_default.sh traces/long_short_bursts.csv
```

Each replay of `azure_conversation.csv` takes about 3 minutes and a few GB of
memory (two replicas: about 5 minutes and more memory); run them one at a
time. `traces/long_short_bursts.csv` replays in seconds. Results land in
`simulator_output/<timestamp>/`.

The wrappers always use `--execution_time_predictor_config_type
linear_regression`. Vidur's default random-forest predictor can spend tens of
minutes training on a laptop before it produces anything.

## Acknowledgments

Simulator code is from Microsoft / Georgia Tech [Vidur](https://github.com/microsoft/vidur)
(MLSys 2024). Course scripts and TODOs are CS 378 only.
