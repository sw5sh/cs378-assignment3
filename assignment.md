---
layout: page
permalink: /assignments/assignment3
title: "Assignment 3: LLM Serving with Vidur"
---
#### **Released:** TBD <br/> **Due:** TBD
{: .no_toc}

* (The list will be replaced with the table of contents.)
{:toc}

### Part 0: Overview and Setup

#### Overview

Serving a language model well means two things for every user: their request
starts soon, and once it starts, the answer streams smoothly. This assignment
follows one hour of real chat traffic through one GPU and asks how far
scheduling can take you, and when you need more hardware instead:

* **Part 1** estimates, with arithmetic alone, where one GPU falls behind.
* **Part 2** replays the trace in a simulator and checks those estimates.
* **Part 3** fixes stalled streaming by limiting the work in each step.
* **Part 4** decides who waits when the GPU cannot keep up.
* **Part 5** adds GPUs and decides where each request goes.

You will use a slim fork of Microsoft's **Vidur**, a simulator of language
model serving. It needs no GPU: it replays a trace of requests, times each
step from measurements of real GPU kernels, and reports how long requests
wait, the time to their first token, the time between tokens, and how much
KV-cache memory is in use.

Model and hardware for the whole assignment: **Llama-3-8B on one A100-80GB
GPU** (no tensor or pipeline parallelism). A **replica** is one copy of the
model serving requests on its own GPU; Parts 1–4 use one replica.

You write code in the Warmup and in Parts 1, 3, 4, and 5; Part 2 is analysis
only:

1. **Warmup** — the full `MemoryPlanner` (how many requests fit in GPU memory)
2. **Part 1** — back-of-the-envelope estimates of one GPU's queue wait and
   KV-cache memory from the trace alone
3. **Part 3** — Sarathi: a token budget for each step (chunked prefill)
4. **Part 4** — shortest prompt first, with aging: who waits when the GPU falls
   behind
5. **Part 5** — two routers for more replicas: least outstanding requests
   (LOR) and least outstanding tokens (LOT)

Traces, all shipped in the repo:

| Part | Trace | Replica(s) |
|---|---|---|
| 1 | `traces/azure_conversation.csv` (Azure conversation service, n = 19,366) | 1 (estimates, no Vidur) |
| 2–4 | same trace, replayed in Vidur | 1 |
| 5 | same trace to size the number of replicas; `traces/long_short_bursts.csv` (synthetic) to compare routers | 2 |

This assignment is **individual**. A laptop CPU is enough.

#### Setup

Python **3.10+**. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Finish the Warmup before any Vidur replay (Part 2 onward): every replica
scheduler calls `MemoryPlanner` when it starts. Activate the venv every
session.

#### Files you edit

| File | What |
|---|---|
| `vidur/scheduler/utils/memory_planner.py` | Warmup — entire `MemoryPlanner` |
| `analysis/workload.py` | Part 1 — arrival rates, queue wait, KV cache in use |
| `vidur/scheduler/replica_scheduler/sarathi_replica_scheduler.py` | Part 3 — tokens per request and the four batch-building steps |
| `vidur/scheduler/replica_scheduler/spf_replica_scheduler.py` | Part 4 — priority with aging, and queue order |
| `vidur/scheduler/global_scheduler/lor_global_scheduler.py` | Part 5 — least outstanding requests |
| `vidur/scheduler/global_scheduler/lot_global_scheduler.py` | Part 5 — least outstanding tokens |

Read (do not edit): `round_robin_global_scheduler.py`, `vllm_replica_scheduler.py`.
In the Sarathi file, leave KV allocation and preemption (`_reserve_kv_or_preempt`)
as shipped. You fill the blocks marked `# [Part 3] TODO`.

Useful paths: `scripts/run_default.sh` (one replica and vLLM by default;
`SCHED` / `CHUNK` / `AGING` choose the replica scheduler, `REPLICAS` / `GLOBAL`
the number of replicas and the router), `scripts/run_tests.sh`, `tests/`,
`traces/azure_conversation.csv`, `traces/long_short_bursts.csv`,
`notebooks/workload.ipynb`, `notebooks/replay.ipynb`, `notebooks/chunking.ipynb`,
`notebooks/spf.ipynb`, `notebooks/replicas.ipynb`.

Run Vidur through `scripts/run_default.sh`, which uses one replica and vLLM
unless you set `SCHED` / `REPLICAS`. If you call Vidur directly, always use
`--execution_time_predictor_config_type linear_regression` (or the wrappers).

#### Deliverables

| Item | Graded how |
|---|---|
| Warmup `MemoryPlanner` | `tests/test_memory_planner.py` |
| Part 1 workload estimates + Qs | `tests/test_workload.py` + notebook Q1–Q4 |
| Part 2 comparison + Qs | notebook Q1–Q4 |
| Part 3 Sarathi + budget sweep + Qs | `tests/test_sarathi_chunk.py`, `tests/test_sarathi_batch.py` + notebook Q1–Q3 |
| Part 4 SPF + aging + Qs | `tests/test_spf.py` + notebook Q1–Q3 |
| Part 5 replicas + routers + Qs | `tests/test_routing.py` + notebook Q1–Q4 |

Submit a zip of the six edited Python files plus a **PDF report** (see
Submission).

---

### Warmup: MemoryPlanner

#### Why Vidur needs a memory planner

Vidur never allocates real GPU memory. It still has to answer the same question
a real serving stack answers at admission time: **how many concurrent requests
fit on this device?**

That answer drives everything downstream. At replica init,
`BaseReplicaScheduler` builds a `MemoryPlanner` and uses it to set:

* the KV block budget (`get_max_request_slots` × blocks per 4096-token
  request) — how much KV-cache memory the scheduler may hand out, in
  fixed-size blocks of 16 tokens
* `max_batch_size` — how many requests may share a batch, if each reserves
  its full 4096 tokens up front (used by schedulers that reserve memory up
  front)

vLLM, Sarathi, and SPF instead give each request blocks only as its tokens
need them (this is called paging), and admit up to 128 requests as long as
the block budget has room. If the memory math is wrong, that budget is wrong,
and every wait and memory number you measure later is wrong too. So before
any replay, you implement the planner that turns **GPU memory − model weights
− safety margin** into a **request capacity**.

#### What to implement

Implement the **entire** class in
`vidur/scheduler/utils/memory_planner.py` (every method below `__init__`):

1. **`_get_kv_cache_memory_per_layer_per_request`** — bytes of KV for **one**
   request, **one** layer. fp16 is 2 bytes; store both K and V. Use
   `self._replica.attention_head_dim`,
   `self._replica.kv_heads_per_tensor_parallel_worker`, and
   `self._replica.max_request_tokens`.
2. **`_get_parameter_memory_per_device`** — bytes of model parameters on this
   device (fp16): `2 * self._param_counter.get_num_parameters_per_device()`.
3. **`_get_kv_cache_memory_per_device_per_request`** — per-layer KV ×
   `self._replica.num_layers`.
4. **`get_max_batch_size`** — available memory is
   `total_memory_gb * 1024**3 * (1 - memory_margin_fraction)`. Leftover for KV
   is that minus parameter memory. Integer-divide by per-request KV bytes.
   Assert at least one request fits; return the count.
5. **`get_max_request_slots`** —
   `get_max_batch_size() * self._replica.num_pipeline_stages`.

```bash
pytest tests/test_memory_planner.py
```

---

### Part 1: Know the workload (`azure_conversation.csv`)

#### Why this part

Before you run a simulator or buy hardware, you should be able to predict the
answer with a few lines of arithmetic: can one GPU keep up with this traffic,
how long do requests queue when it can't, and how much memory does it need?
These are **back-of-the-envelope calculations**. They size hardware, and they
tell you what numbers to expect, so you notice when a measurement is wrong.
In this part you write them for one GPU. Part 2 replays the same trace in
Vidur and checks your predictions.

The trace, `traces/azure_conversation.csv`, is one hour of requests to Azure's
LLM chat service: 19,366 rows of arrival time, prompt length, and output
length (in tokens).

#### How a GPU serves a request

The GPU works in **steps** (also called iterations or passes): each step runs
the model once over a **batch** of requests. A request runs in two phases.
**Prefill** processes its prompt, all of it in one step (Part 3 changes this).
**Decode** then generates the output, one token per step: each step makes one
token for every decoding request in the batch. While a request runs, the GPU
keeps the attention keys and values of all its tokens in memory, the **KV
cache**, so it never recomputes them.

These numbers are constants in `analysis/workload.py`, measured in Vidur for
Llama-3-8B on one A100-80GB:

| | Value | |
|---|---|---|
| Prefill throughput | 12,000 tokens/s | a 2048-token prompt takes about 0.17 s; a prompt's tokens are processed in parallel, so this is far faster than generating |
| Decode step | 30 ms | with a full batch at this trace's lengths |
| Max batch size | 128 requests | |
| Decode throughput | 4,267 tokens/s | 128 tokens every 30 ms |
| Model weights | 16.06 GB | 8B parameters × 2 bytes |

#### What to implement

Fill the four `# [Part 1] TODO` blocks in `analysis/workload.py`. The
docstrings give the exact inputs and outputs.

1. **`arrival_rates(df, bin_s)`** — requests/s, prompt tokens/s, and output
   tokens/s in each time bin of `bin_s` seconds. Keep empty bins as zeros.

2. **`predict_queue_wait(df, prefill_tok_per_s, decode_tok_per_s)`** — how long
   each request waits before the GPU starts on it. Model the GPU as one
   first-come, first-served queue that works at full throughput. For each
   request in arrival order:
   * `work = prompt / prefill_tok_per_s + output / decode_tok_per_s`
   * `start = max(arrival, time the GPU is next free)`, `wait = start − arrival`
   * the GPU is next free at `start + work`

3. **`kv_bytes_per_token()`** — KV-cache bytes for one token: a key and a value
   vector for every layer and KV head.

4. **`kv_in_use(df, waits, bin_s)`** — KV-cache memory in use at the start of
   each bin. A request holds memory for all its tokens from arrival until
   `done = arrival + wait + prompt / prefill_tok_per_s + output × decode_step_s`.
   The GPU runs at most 128 requests at once, in arrival order; the rest wait
   in the queue and hold no memory. So count only the earliest 128.

**Worked example** (`predict_queue_wait`): 1,000 prompt tokens/s, 100 output
tokens/s.

| Request | Arrives | Prompt / output | Work | GPU next free | Start | Wait |
|---|---|---|---|---|---|---|
| A | 0.0 s | 1000 / 100 | 2.0 s | 0 | 0.0 | **0** |
| B | 0.5 s | 500 / 50 | 1.0 s | 2.0 | 2.0 | **1.5 s** |
| C | 4.0 s | 200 / 10 | 0.3 s | 3.0 | 4.0 | **0** |

**Worked example** (`kv_in_use`): 1,000 prompt tokens/s, 10 ms decode step,
`bin_s = 1`.

| Request | Arrives | Wait | Prompt / output | Done | Holding memory at |
|---|---|---|---|---|---|
| A | 0.0 | 0.0 | 100 / 10 | 0.2 | t = 0 |
| B | 0.1 | 0.1 | 1000 / 100 | 2.2 | t = 1, 2 |
| C | 0.5 | 1.6 | 10 / 1 | 2.12 | t = 1, 2 |
| D | 2.0 | 0.0 | 1 / 1 | 2.011 | t = 2 |

Tokens in memory: 110 at t = 0, 1111 at t = 1, 1113 at t = 2. With at most
one request running: 110, 1100, 1100.

```bash
pytest tests/test_workload.py
```

#### Notebook and questions

```bash
jupyter notebook notebooks/workload.ipynb
```

The notebook plots arrivals, predicted wait, and KV-cache memory over time.
The wait target is **P99 under 2 s**: 99% of requests should wait less than
2 s. Answer in the notebook.

1. When does your model predict one GPU misses the 2 s target? Is that stretch
   driven by more requests, longer prompts, or both? Cite the arrival plot.
2. What are the average and the peak KV-cache memory in use, and when is the
   peak? Which of these GPUs has enough memory for this trace: 24 GB (L4),
   40 GB (A100), 48 GB (L40S), 80 GB (A100 / H100)? Count the model weights
   and keep 10% of memory free. Which would you pick if you sized by the
   average instead? Then rerun `kv_in_use` with `max_batch_size` set very
   large: what peak do you get, and why is the capped number the right one?
3. At 128 tokens per 30 ms step, one GPU decodes at most 4,267 tokens/s. What
   is that throughput if a step takes 20 ms or 50 ms, or if the batch holds
   only 64? Rerun `predict_queue_wait` with each. How much does the P99 wait
   move, and why is it so sensitive? What step time does one GPU need for this
   trace?
4. Record your predictions for Part 2: the P99 wait, the time range where the
   wait misses the target, and the peak KV-cache memory.

Write down **native QPS** (printed by the notebook): the trace's own rate, in
requests per second. Part 2 replays the trace at that rate.

---

### Part 2: Check the estimates (`azure_conversation.csv`)

#### Why this part

Part 1 predicted the queue and the memory with arithmetic. Now you replay the
same trace in Vidur and check those predictions. Then you look at what the
arithmetic cannot see: how smoothly each answer streams. A user reading a chat
answer notices when tokens stop arriving, even if the average speed looks
fine. The measure for this is the **time between tokens**: the gap between
consecutive output tokens of a request.

There is no code to write in this part.

#### Run and compare

```bash
QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
```

One run, about 3 minutes; Vidur logs its progress every 10% of the trace.
Paste its output directory into `notebooks/replay.ipynb`. The notebook puts
your Part 1 predictions next to Vidur's numbers, plots wait and KV-cache
memory over time, and plots the time between tokens. The target for time
between tokens is **P99 under 100 ms**.

#### Questions

1. Compare your Part 1 predictions with Vidur: P99 wait, the longest stretch
   above the 2 s target, and peak KV-cache memory. Which held and which did
   not? For any that missed, say what the Part 1 model left out.
2. The target for time between tokens is P99 under 100 ms. Does vLLM meet it?
   Report the P50, P99, and longest time between tokens. What does the longest
   one mean for a user reading the answer?
3. Why do running requests get no new tokens for whole seconds during the
   surge? Read `_get_next_batch` in
   `vidur/scheduler/replica_scheduler/vllm_replica_scheduler.py`: when new
   requests are waiting, what goes into the next iteration, and what has to
   wait?
4. Part 3 fixes this by capping each iteration at a token budget C: every
   running request gets its one token, and waiting prompts fill the rest of the
   budget in chunks. One pass with C tokens takes about 6 ms + C × 78 µs
   (Vidur's measured prefill time). Predict the P99 time between tokens for C =
   256, 512, and 2048. Which meet the 100 ms target? A smaller C is not free:
   what happens to the fixed 6 ms per pass, and so to throughput and the queue,
   when the same prompts take more passes?

---

### Part 3: Chunked prefill (Sarathi)

#### Why this part

Part 2 showed vLLM freezing running requests behind whole prompts, and you
predicted that capping each iteration at a token budget fixes it at some
cost. Now you build that scheduler, Sarathi (from the Sarathi-Serve paper,
OSDI 2024), and check your prediction.

Each iteration has a token budget C. It fills in this order:

1. every running request that is decoding gets its one token;
2. a prompt already in progress gets as much of the rest of its prompt as
   fits in the budget left;
3. waiting requests start, oldest first, with as much of their prompt as fits.

Decode tokens always get in, even past the budget: skipping one would stall
that request. So the budget limits how many **prompt** tokens join the decodes
in one pass. When it runs out, an unfinished prompt waits for the next
iteration, and no waiting request starts ahead of an older one.

#### What to implement

In `vidur/scheduler/replica_scheduler/sarathi_replica_scheduler.py`. Reserving
KV-cache memory, and preemption (when memory runs out, stopping a running
request and restarting it later), are provided as `_reserve_kv_or_preempt`
and `_allocate_request`.

1. **`_get_request_next_num_tokens`** — how many tokens a request adds to this
   iteration: one if it is decoding; otherwise as much of the rest of its
   prompt as fits in the budget left. The budget left can be negative when
   decodes alone overflow it; then the prompt adds 0.
2. **The four steps of building a batch**, one function each, in the order
   above. `_get_next_batch` (provided) calls them in turn, passing a
   `BatchInProgress` that tracks the requests and tokens so far:
   * `_add_running_decodes` — decoding requests get their one token; set
     prompts in progress aside;
   * `_add_unfinished_prompts` — those prompts get what fits; keep the ones
     that get 0;
   * `_return_kept_prompts` — put the kept prompts back, oldest first;
   * `_add_new_requests` — start waiting requests, oldest first, until the
     budget, KV memory, or the batch-size limit stops you.

   Each docstring gives the exact rules. The tests are grouped by function,
   so you can finish one step at a time.

**Worked example.** C = 512. Three requests are decoding, one prompt has 900
tokens left, and a new 300-token prompt is waiting.

| Iteration | Decoding | Prompt in progress | New prompt | Tokens |
|---|---|---|---|---|
| 1 | 3 × 1 | 509 (391 left) | waits: no budget left | 512 |
| 2 | 3 × 1 | 391 (done) | starts with 118 | 512 |

```bash
pytest tests/test_sarathi_chunk.py
pytest tests/test_sarathi_batch.py
```

#### Run and compare

Run Sarathi at the three budgets from Part 2 Q4:

```bash
for C in 256 512 2048; do
  SCHED=sarathi CHUNK=$C QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
done
```

The runs go one after another, about 3 minutes each. With plenty of RAM you
can run them in the background instead (add `&` after the command, then
`wait`), but each run needs a few GB: four at once ran out of memory on a
16 GB laptop.

Paste the three output directories and your Part 2 vLLM run into
`notebooks/chunking.ipynb`. It tabulates your predictions next to the measured
time between tokens and wait, and plots both.

#### Questions

1. Fill in the table. Which budgets meet the 100 ms target?

   | C | Predicted P99 time between tokens (Part 2 Q4) | Measured | Meets 100 ms? |
   |---|---|---|---|
   | 256 | | | |
   | 512 | | | |
   | 2048 | | | |

2. Each pass pays about 6 ms fixed plus 78 µs per prompt token, so prompt
   throughput is C / (6 ms + C × 78 µs). Fill in the table, then pick the
   budget that meets the 100 ms target with the lowest P99 wait. In one
   sentence: why does a smaller C cost throughput?

   | C | Prompt throughput (tokens/s) | Measured P99 wait | Median time to first token |
   |---|---|---|---|
   | 256 | | | |
   | 512 | | | |
   | 2048 | | | |

3. What is the lowest P99 wait across your four runs, and what P99 wait did
   Part 1 predict for one GPU at full throughput? Can any scheduler meet the
   2 s target on one GPU? Answer yes or no, with one sentence on why.

---

### Part 4: Which request goes first? (shortest prompt first, with aging)

#### Why this part

Part 3 showed that during the surge (the stretch from about t = 1350 s to
2230 s where one GPU falls behind) the wait cannot stay under 2 s, however the
GPU batches. Someone has to wait. This part is about **who**: the order in
which waiting requests start. Serving the shortest prompt first (SPF) makes
most requests wait far less, because most prompts are short. But a long prompt
can be overtaken by every newcomer and **starve**. **Aging**, the textbook fix
for starvation, gives a waiting request more priority for every second it
waits.

#### What to implement

`SpfReplicaScheduler` in `vidur/scheduler/replica_scheduler/spf_replica_scheduler.py`
extends your Part 3 Sarathi scheduler. Before each iteration, Sarathi calls
`_order_waiting_queue()`, and `_add_new_requests` then starts requests from
the front of the queue.

1. **`_priority(request, now)`** — smaller starts first:
   `prompt tokens left − aging × seconds waited`, where `aging` is
   `self._config.aging_tokens_per_s` (tokens of priority gained per second).
2. **`_order_waiting_queue()`** — sort `self._request_queue` in place by
   `_priority(request, self._now)`; equal priorities keep their order.

**Worked example.** Aging 100 tokens/s. A 4000-token prompt has waited since
t = 0; a 100-token prompt arrives just before each decision.

| At | 4000-token prompt | Newly arrived 100-token prompt | Goes first |
|---|---|---|---|
| t = 10 | 4000 − 100 × 10 = 3000 | 100 | the short prompt |
| t = 45 | 4000 − 100 × 45 = −500 | 100 | the long prompt |

The long prompt stops losing to fresh short ones after (4000 − 100) / 100 =
39 s of waiting.

```bash
pytest tests/test_spf.py
```

#### Run and compare

```bash
for A in 0 25 50 100 200; do
  SCHED=spf CHUNK=512 AGING=$A QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
done
```

Five runs, about 15 minutes; see Part 3 for running in the background.

`AGING=0` is plain shortest-prompt-first. Paste the five output directories
and your Part 3 Sarathi run at C = 512 (arrival order) into
`notebooks/spf.ipynb`.

#### Questions

1. Fill in the table. The surge is the requests that arrive between t = 1350 s
   and t = 2230 s. Which order gives the lowest surge median wait, and which
   gives the lowest worst wait? Does any order meet the 2 s P99 target?

   | Order | Surge median wait | P99 wait | Worst wait | Requests over 60 s |
   |---|---|---|---|---|
   | Arrival order (Part 3) | | | | |
   | Aging 0 (plain shortest-first) | | | | |
   | Aging 25 | | | | |
   | Aging 50 | | | | |
   | Aging 100 | | | | |
   | Aging 200 | | | | |

2. Under shortest-prompt-first (aging 0), how long are the prompts of the
   requests that wait longest, and when did they arrive? In one sentence, why
   do they wait so long? P99 summarizes the slowest 1% of requests, 194 of
   19,366: compare aging 0's count of requests over 60 s with 194 to explain
   why its P99 still looks better than arrival order's.
3. From your table: which is the smallest aging that keeps the surge median
   wait under 1 s and has at most 5 requests waiting over 60 s? What does it
   cost against arrival order in P99 and worst wait? Aging 25 has a higher P99
   than plain SPF: use the two counts of requests over 60 s and the 194 from Q2
   to explain why.

---

### Part 5: More replicas, and where each request goes

#### Why this part

Parts 3 and 4 showed that one GPU cannot keep the P99 wait under 2 s during
the surge, however it schedules. The fix is more replicas. That raises two
questions: **how many**, and **which replica gets each request**. The second
is the job of the **router** (Vidur calls it the global scheduler).

#### How many replicas

Estimate first, with your Part 1 model: N replicas together process N times
the prefill and decode throughput of one. The notebook predicts the P99 wait
for N = 1, 2, and 3. Then check the smallest N that your model says meets the
2 s target, with round-robin routing (each request goes to the next replica in
turn). This run takes about 5 minutes and more memory than one replica: close
other programs and run it on its own.

```bash
REPLICAS=2 GLOBAL=round_robin SCHED=sarathi CHUNK=512 QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
```

#### What to implement

Round-robin is provided; it ignores how busy each replica is. You write two
routers that look at what each replica has **outstanding**: the requests sent
to it that are still waiting or running. `self.outstanding_requests(replica_id)`
(provided) returns them.

1. **`LORGlobalScheduler.schedule`** in
   `vidur/scheduler/global_scheduler/lor_global_scheduler.py` — least
   outstanding requests: send each request to the replica with the fewest
   outstanding requests.
2. **`LOTGlobalScheduler.schedule`** in
   `vidur/scheduler/global_scheduler/lot_global_scheduler.py` — least
   outstanding tokens: send each request to the replica with the least work
   outstanding, where a request's work is the tokens it still has to process,
   `request.total_tokens - request.num_processed_tokens`.

Both assign waiting requests oldest first, count each assignment on its
replica right away (so a burst of requests that arrive together spreads out),
and break ties toward the lowest replica id.

**Worked example.** Two empty replicas; four requests arrive together: a
3,908-token request, a 24-token one, another 3,908, another 24.

| Request | Round-robin | LOR (requests on 0 / 1 before) | LOT (tokens on 0 / 1 before) |
|---|---|---|---|
| 3,908 | 0 | 0 (0 / 0) | 0 (0 / 0) |
| 24 | 1 | 1 (1 / 0) | 1 (3,908 / 0) |
| 3,908 | 0 | 0 (1 / 1) | 1 (3,908 / 24) |
| 24 | 1 | 1 (2 / 1) | 0 (3,908 / 3,932) |

Round-robin and LOR put both long requests on replica 0; LOT gives each replica
one long and one short.

```bash
pytest tests/test_routing.py
```

#### Run and compare

`traces/long_short_bursts.csv` is built to stress routers: 128 requests in
bursts of 8 that arrive together, alternating 3,900-token and 16-token prompts.
Each run takes seconds:

```bash
for G in round_robin lor lot; do
  REPLICAS=2 GLOBAL=$G SCHED=sarathi CHUNK=512 QPS=8 bash scripts/run_default.sh traces/long_short_bursts.csv
done
```

Then LOT on the real trace, to compare with your round-robin run above. A
two-replica run of the full trace takes about 5 minutes and more memory than
one replica: close other programs and run it on its own.

```bash
REPLICAS=2 GLOBAL=lot SCHED=sarathi CHUNK=512 QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
```

Paste the output directories into `notebooks/replicas.ipynb`.

#### Questions

1. Fill in the table. What is the smallest number of replicas that meets both
   targets (P99 wait under 2 s, P99 time between tokens under 100 ms)? How
   close was your Part 1 model to Vidur?

   | Replicas | Predicted P99 wait (Part 1 model) | Vidur P99 wait | Vidur P99 time between tokens |
   |---|---|---|---|
   | 1 | | (Part 3, C = 512) | |
   | 2 | | | |
   | 3 | | — | — |

2. Fill in the table for the long-short bursts trace. Why does round-robin
   leave one replica almost idle? Why does LOT beat LOR? Use the worked
   example.

   | Router | Wait P99 | End-to-end P99 | Busy % (replica 1 / 2) |
   |---|---|---|---|
   | Round-robin | | | |
   | LOR | | | |
   | LOT | | | |

3. Vidur's original LOR counted only requests **waiting** in each replica's
   queue, not running ones. On the real trace with 2 replicas, those queues are
   almost always empty. Which replica would that router pick for almost every
   request, and why? Why is counting running requests the right definition?
4. On the real trace with 2 replicas, compare round-robin with LOT: P99 wait
   and P99 time between tokens. Does the router matter here? In one sentence:
   when is a load-aware router worth it?

---

### How to test

```bash
bash scripts/run_tests.sh          # every test (same as `pytest`)
pytest tests/test_sarathi_batch.py # one file
pytest tests/test_sarathi_batch.py -k step4   # tests whose name contains "step4"
pytest -x                          # stop at the first failure
```

Every test is named after the rule it checks, for example
`test_step4_stops_when_memory_full`, and its failure message says what was
expected.

---

### What NOT to modify

* Anything outside a `# [Part N] TODO` block in the files you edit. In
  particular, these are provided: `load_trace` (Part 1); `BatchInProgress`,
  `_reserve_kv_or_preempt`, `_order_waiting_queue`, and `_get_next_batch` in the
  Sarathi file (Part 3)
* `analysis/replay.py`, `vidur/simulator.py`, `vidur/metrics/`,
  `vidur/events/`, `base_replica_scheduler.py`
* `round_robin_global_scheduler.py`, and `record_assignment` /
  `outstanding_requests` in `base_global_scheduler.py`
* `vllm_replica_scheduler.py`, `orca_replica_scheduler.py`
* `vidur/types/`, `vidur/config/config.py`, `*_registry.py`
* `tests/`, `scripts/`, `data/`, `traces/`
* Flags inside `run_default.sh` other than `QPS`,
  `SCHED`, `CHUNK`, `AGING`, `GLOBAL`, `REPLICAS`, and the trace path

---

### Submission

Zip these six files:

* `vidur/scheduler/utils/memory_planner.py`
* `analysis/workload.py`
* `vidur/scheduler/replica_scheduler/sarathi_replica_scheduler.py`
* `vidur/scheduler/replica_scheduler/spf_replica_scheduler.py`
* `vidur/scheduler/global_scheduler/lor_global_scheduler.py`
* `vidur/scheduler/global_scheduler/lot_global_scheduler.py`

Submit `submission.zip` and a **PDF report** with:

* name, EID
* `bash scripts/run_tests.sh` output (the pytest summary)
* Part 1 arrivals, predicted-wait, and KV-in-use figures; answers Q1–Q4
* Part 2 predicted-vs-measured table and plots, time-between-tokens plot, answers Q1–Q4
* Part 3 time-between-tokens and wait plots, answers Q1–Q3 (with the two
  tables)
* Part 4 wait plot, the top-waits table, answers Q1–Q3
* Part 5 the two tables, answers Q1–Q4

---

### Grading

* Warmup — `test_memory_planner.py`: **10%**
* Part 1 — workload estimates: **15%** (`test_workload.py` 8%, notebook Q1–Q4 7%)
* Part 2 — check the estimates: **10%** (notebook Q1–Q4)
* Part 3 — Sarathi: **25%** (`test_sarathi_chunk.py` + `test_sarathi_batch.py` 15%, notebook Q1–Q3 10%)
* Part 4 — shortest prompt first + aging: **10%** (`test_spf.py` 4%, notebook Q1–Q3 6%)
* Part 5 — replicas + routers: **30%** (`test_routing.py` 10%, notebook Q1–Q4 20%)

---

### FAQ

* **`NotImplementedError` as soon as Vidur starts** — Warmup unfilled.
* **Run hangs 20+ minutes, only `config.json`** — random forest predictor. Kill
  it; use `linear_regression` (or the wrappers).
* **Sarathi `NotImplementedError` in Part 2** — leave `SCHED` unset (the
  wrapper uses vLLM).
* **`test_sarathi_batch.py` failures** — each test is named after its step and
  one rule (for example `test_step4_stops_when_memory_full`); read its
  message. A `TimeoutError` means an endless loop, usually a `continue` where
  you must stop.
* **SPF `NotImplementedError`** — only pass `SCHED=spf` after Part 4 is filled.
  SPF extends your Part 3 Sarathi, so Part 3 must pass its tests first.
* **`NotImplementedError` in Part 5** — fill both routers before using
  `GLOBAL=lor` or `GLOBAL=lot`. Round-robin runs without them.
* **`CHUNK=` with `SCHED=vllm`** — the token budget is a Sarathi-only flag.
  Use `SCHED=sarathi CHUNK=512`.
* **Full replay time** — `azure_conversation.csv` takes about 3 minutes per
  run once the linear-regression cache is warm; Vidur logs its progress every
  10% of the trace. Do not downsample the CSV.
* **A run is killed, or the machine freezes** — out of memory. Run one replay
  at a time; each needs a few GB.

---

### Acknowledgments

Simulator code from Microsoft / Georgia Tech
[Vidur](https://github.com/microsoft/vidur) (MLSys 2024). Course scripts,
TODOs, staff traces, and this handout are CS 378 only.
