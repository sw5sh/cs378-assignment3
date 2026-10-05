#!/usr/bin/env bash
# Runs every test. Unfilled TODOs fail until you implement them.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi
"$PYTHON" tests/test_memory_planner.py
"$PYTHON" tests/test_workload.py
"$PYTHON" tests/test_sarathi_chunk.py
"$PYTHON" tests/test_sarathi_batch.py
"$PYTHON" tests/test_spf.py
"$PYTHON" tests/test_routing.py
