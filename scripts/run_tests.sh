#!/usr/bin/env bash
# Runs every test with pytest. Unfilled TODOs fail until you implement them.
# Extra arguments go to pytest, e.g. `bash scripts/run_tests.sh -k step4`.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi
"$PYTHON" -m pytest tests/ "$@"
