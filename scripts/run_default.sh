#!/usr/bin/env bash
# Run Vidur on one trace: Llama-3-8B on one A100 (no tensor or pipeline
# parallelism), vLLM on one replica unless SCHED / REPLICAS say otherwise.
#
# Usage:
#   QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
#   SCHED=sarathi CHUNK=512 QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
#   SCHED=spf CHUNK=512 AGING=100 QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
#   REPLICAS=2 GLOBAL=round_robin SCHED=sarathi CHUNK=512 QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
#   REPLICAS=2 GLOBAL=lot SCHED=sarathi CHUNK=512 QPS=8 bash scripts/run_default.sh traces/long_short_bursts.csv
#
# QPS sets the replay rate in requests per second: Vidur multiplies every
# arrival time by time_scale_factor = (N - 1) / (trace span * QPS).
# Always uses the linear_regression predictor: Vidur's default random forest
# can spend tens of minutes training before the first run produces anything.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TRACE="${1:-traces/azure_conversation.csv}"
QPS="${QPS:-5.53}"
SCHED="${SCHED:-vllm}"
CHUNK="${CHUNK:-}"
AGING="${AGING:-}"
REPLICAS="${REPLICAS:-1}"
GLOBAL="${GLOBAL:-round_robin}"
export WANDB_MODE=disabled

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

if [[ ! -f "$TRACE" ]]; then
  echo "trace not found: $TRACE" >&2
  exit 1
fi

TIME_SCALE="$("$PYTHON" - "$TRACE" "$QPS" <<'PY'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
t = df["arrived_at"].to_numpy(float)
qps = float(sys.argv[2])
span = float(t.max() - t.min()) if len(t) else 0.0
n = len(t)
if n < 2 or span <= 0 or qps <= 0:
    print("1.0")
else:
    print(f"{(n - 1) / (span * qps):.8f}")
PY
)"

echo "trace=$TRACE  target_QPS=$QPS  time_scale_factor=$TIME_SCALE"
echo "WANDB_MODE=disabled  model=meta-llama/Meta-Llama-3-8B  device=a100  scheduler=$SCHED  chunk=${CHUNK:-default}  aging=${AGING:-0}  replicas=$REPLICAS  global=$GLOBAL  TP=1 PP=1"

EXTRA=()
if [[ -n "$CHUNK" ]]; then
  if [[ "$SCHED" != "sarathi" && "$SCHED" != "spf" ]]; then
    echo "CHUNK= only applies with SCHED=sarathi or SCHED=spf" >&2
    exit 1
  fi
  EXTRA+=(--${SCHED}_scheduler_config_chunk_size "$CHUNK")
fi
if [[ -n "$AGING" ]]; then
  if [[ "$SCHED" != "spf" ]]; then
    echo "AGING= only applies with SCHED=spf" >&2
    exit 1
  fi
  EXTRA+=(--spf_scheduler_config_aging_tokens_per_s "$AGING")
fi

set -x
"$PYTHON" -m vidur.main \
  --replica_config_device a100 \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --cluster_config_num_replicas "$REPLICAS" \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --request_generator_config_type trace_replay \
  --trace_request_generator_config_trace_file "$TRACE" \
  --trace_request_generator_config_time_scale_factor "$TIME_SCALE" \
  --trace_request_generator_config_max_tokens 4096 \
  --replica_scheduler_config_type "$SCHED" \
  --global_scheduler_config_type "$GLOBAL" \
  --execution_time_predictor_config_type linear_regression \
  --no-metrics_config_enable_chrome_trace \
  --metrics_config_store_token_completion_metrics \
  "${EXTRA[@]}"
