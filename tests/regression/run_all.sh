#!/usr/bin/env bash
# Run the full offline regression suite for glm-rev from the repo root.
# Usage: bash tests/regression/run_all.sh
set -u
cd "$(dirname "$0")/../.." || exit 1

echo "=== py_compile ==="
python3 -m py_compile glm_rev/*.py server.py run.py || exit 1
echo "OK"

FAILED=0

run() {
  local label="$1" envs="${2:-}"
  local out
  out=$(timeout 120 env PYTHONPATH=. $envs python3 "$1" 2>&1)
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "PASS  $label"
  else
    echo "FAIL  $label (exit $rc)"
    echo "$out" | tail -15
    FAILED=1
  fi
}

for f in tests/regression/test_fix1.py tests/regression/test_fix2.py \
         tests/regression/test_fix3.py tests/regression/test_fix4.py \
         tests/regression/test_fix5.py tests/regression/test_fix6.py \
         tests/regression/test_parser_edge_cases.py; do
  run "$f"
done
run tests/regression/test_contract.py
run tests/regression/test_client_tools.py "GLM_CLIENT_TOOLS=1"
run tests/regression/test_refusal.py
run tests/regression/test_history_commit.py
run tests/regression/test_inflight_hist.py
run tests/regression/test_memory.py
run tests/regression/test_jitter.py
run tests/regression/test_masking.py

if [ $FAILED -ne 0 ]; then
  echo "=== SOME TESTS FAILED ==="
  exit 1
fi
echo "=== ALL REGRESSION TESTS PASSED ==="