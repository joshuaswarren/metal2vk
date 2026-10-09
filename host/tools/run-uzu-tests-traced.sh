#!/bin/bash
# Run ONE uzu metal unit test per process with the kernel trace enabled, and
# write one JSON line for gate-from-tests.py. GPU host only (the uzu metal
# tests need Apple silicon Metal); wire it into the GPU guard queue.
# Usage: run-uzu-tests-traced.sh TEST_NAME OUT.jsonl [extra cargo args...]
set -euo pipefail
TEST=${1:?test name}
OUT=${2:?output jsonl file}
shift 2
TRACE=$(mktemp)
trap 'rm -f "$TRACE"' EXIT
RECEIPT=${RECEIPT:-}
STATUS=pass
M2V_KERNEL_TRACE="$TRACE" \
  cargo test -p uzu-engine --test '*' --features metal "$TEST" "$@" \
  > "$TRACE.log" 2>&1 || STATUS=fail
python3 - "$OUT" "$TEST" "$STATUS" "$RECEIPT" "$TRACE" <<'PY'
import json, sys
out, test, status, receipt, trace = sys.argv[1:6]
with open(trace) as f:
    kernels = sorted({line.strip() for line in f if line.strip()})
entry = {"test": test, "status": status, "kernels": kernels, "receipt": receipt}
with open(out, "a") as f:
    f.write(json.dumps(entry) + "\n")
PY
echo "recorded: $TEST -> $STATUS"
tail -5 "$TRACE.log"
