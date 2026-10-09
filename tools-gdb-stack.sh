#!/bin/bash
# Usage: tools-gdb-stack.sh IR.ll [seconds]   (clspv is a child of gdb, so ptrace works under yama ptrace_scope=1)
LL=$1
SEC=${2:-45}
CLSPV=${CLSPV:-$HOME/scratch/metal2vk/clspv/build/bin/clspv}
timeout -s INT "$SEC" gdb -batch -ex run -ex 'bt 30' --args "$CLSPV" -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points \
  --spv-version=1.5 "$LL" -o /dev/null 2>&1 | grep -E '^#' | sed -E 's/^#([0-9]+) +0x[0-9a-f]+ in /#\1 /; s/\(this=.*//' | cut -c1-170
