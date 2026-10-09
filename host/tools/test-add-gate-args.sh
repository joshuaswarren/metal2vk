#!/bin/bash
# Prove both gate branches of add-gate-args.sh without any toolchain:
# receipt with a PASS line -> verified gate file; missing or PASS-less receipt
# -> no gate file (kernel stays unverified).
set -euo pipefail
H=$(cd "$(dirname "$0")/.." && pwd)
GATE=$H/tools/add-gate-args.sh
TMP=$(mktemp -d)
PRISTINE="$TMP/receipt.pristine"
cp "$H/receipts/G13C/run-g13c.txt" "$PRISTINE"
trap 'cp "$PRISTINE" "$H/receipts/G13C/run-g13c.txt"; rm -rf "$TMP"' EXIT
fail() { echo "FAIL: $1" >&2; exit 1; }

# branch 1: real receipt (has "PASS add_kernel_end_to_end")
OUT="$TMP/with-receipt"; mkdir -p "$OUT"
ARGS=$("$GATE" "$OUT")
[ -f "$OUT/add.gates.json" ] || fail "receipt present: gate file missing"
grep -q '"state": "verified"' "$OUT/add.gates.json" || fail "receipt present: state not verified"
grep -q 'run-g13c.txt' "$OUT/add.gates.json" || fail "receipt present: evidence path missing"
case "$ARGS" in *"--gates $OUT/add.gates.json"*) ;; *) fail "receipt present: wrong args '$ARGS'";; esac

# branch 2a: receipt absent
OUT="$TMP/no-receipt"; mkdir -p "$OUT"
rm "$H/receipts/G13C/run-g13c.txt"
ARGS=$("$GATE" "$OUT")
[ ! -f "$OUT/add.gates.json" ] || fail "receipt absent: gate file exists"
[ -z "$ARGS" ] || fail "receipt absent: args not empty '$ARGS'"
cp "$PRISTINE" "$H/receipts/G13C/run-g13c.txt"

# branch 2b: receipt without a PASS line
OUT="$TMP/passless"; mkdir -p "$OUT"
grep -v '^PASS ' "$PRISTINE" > "$H/receipts/G13C/run-g13c.txt"
ARGS=$("$GATE" "$OUT")
[ ! -f "$OUT/add.gates.json" ] || fail "pass-less receipt: gate file exists"
[ -z "$ARGS" ] || fail "pass-less receipt: args not empty '$ARGS'"

cp "$PRISTINE" "$H/receipts/G13C/run-g13c.txt"
# restored byte-identical (the EXIT trap is the safety net)
diff -q "$PRISTINE" "$H/receipts/G13C/run-g13c.txt" >/dev/null || fail "receipt not restored"

echo "PASS add-gate-args: verified when the receipt has its PASS line, unverified otherwise"
