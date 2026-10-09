#!/bin/bash
# Run clspv on an IR file in the background and print a few sampled stacks (function names only). Usage: tools-sample-stack.sh IR.ll [samples] [interval_s]
LL=$1
N=${2:-4}
GAP=${3:-15}
CLSPV=${CLSPV:-$HOME/scratch/metal2vk/clspv/build/bin/clspv}
"$CLSPV" -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points --spv-version=1.5 "$LL" -o /dev/null > /dev/null 2>&1 &
P=$!
sleep 10
for i in $(seq "$N"); do
  echo "== sample $i (+$((10 + (i - 1) * GAP)) s)"
  gdb -p "$P" -batch -ex 'bt 18' 2>/dev/null | grep -E '^#' | sed -E 's/^#([0-9]+) +0x[0-9a-f]+ in /#\1 /' | cut -c1-170
  sleep "$GAP"
done
kill "$P" 2>/dev/null
