#!/bin/bash
# Decide the gate state for the add.m2vlib kernels from the G13C receipt.
# Usage: add-gate-args.sh OUTDIR
# Writes $OUTDIR/add.gates.json when the receipt proves add_kernel_end_to_end
# (its PASS line covers the `add` kernel); prints the make-m2vlib.py extra args.
# The `add_offset` kernel stays unverified: no GPU test has run it yet.
set -euo pipefail
OUT=${1:?outdir}
H=$(cd "$(dirname "$0")/.." && pwd)
RECEIPT="$H/receipts/G13C/run-g13c.txt"
KERNEL=add
GATES="$OUT/add.gates.json"
if [ -f "$RECEIPT" ] && grep -q "^PASS add_kernel_end_to_end" "$RECEIPT"; then
  printf '{"%s": {"state": "verified", "evidence": "host/receipts/G13C/run-g13c.txt"}}\n' "$KERNEL" > "$GATES"
  echo "--gates $GATES"
else
  rm -f "$GATES"
  echo ""
fi
