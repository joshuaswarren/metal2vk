#!/bin/bash
# Usage: tools-pass-trace.sh IR.ll [seconds]  -> prints the last passes started before the time cap (or the end)
LL=$1
CAP=${2:-60}
CLSPV=${CLSPV:-$HOME/scratch/metal2vk/clspv/build/bin/clspv}
OUT=$(mktemp -d "${TMPDIR:-$HOME/scratch}/m2v-trace.XXXXXX")
timeout "$CAP" "$CLSPV" -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points --spv-version=1.5 --debug-pass=Executions "$LL" -o "$OUT/o.spv" > "$OUT/log" 2>&1
echo "rc=$? lines=$(wc -l < "$OUT/log")"
grep -E "Executing Pass" "$OUT/log" | tail -14 | cut -c1-150
echo "-- most frequent passes"
grep -E "Executing Pass" "$OUT/log" | sed -E "s/ on .*//" | sort | uniq -c | sort -rn | head -6
