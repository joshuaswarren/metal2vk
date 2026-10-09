#!/bin/bash
# One command to reproduce the bench on the GPU host, under the GPU queue wrapper:
#   PATH=$HOME/bin:$PATH <gpu-queue-wrapper> -m 25 -- bash bench/run-all.sh OUTDIR [--device g14c] [--skip-hand]
# Env: UZU_ROOT (uzu checkout), CLSPV (patched clspv), PYTHON (python with mlx for the hand side),
#      M2V_ICD (fork ICD json for VK_DRIVER_FILES), M2V_OPT (compile route; default is the
#      typed-GEP route, see docs/bench-status.md). bench/env-local.sh (untracked; template:
#      bench/env-local.sh.template) is sourced when present.
# Compiles the registered cases, prints the translated-vs-hand table, runs the tuning sweep.
# Trailing args go to m2v-bench.py only.
set -eu
OUT=${1:?outdir}
shift || true
H=$(cd "$(dirname "$0")/.." && pwd)
[ -r "$H/bench/env-local.sh" ] && . "$H/bench/env-local.sh"
PY=${PYTHON:-python3}
export UZU_ROOT=${UZU_ROOT:?set UZU_ROOT}
export CLSPV=${CLSPV:-$HOME/scratch/metal2vk/clspv/build/bin/clspv}
SPV=$OUT/spv
mkdir -p "$OUT" "$SPV"
# post-fix route (typed GEPs; see docs/bench-status.md): clang stops before InstCombine's
# i8-GEP canonicalisation, clspv's own pipeline optimises. Override with M2V_OPT to compare.
export M2V_OPT=${M2V_OPT:--O0 -Xclang -disable-O0-optnone}
[ -n "${M2V_ICD:-}" ] && export VK_DRIVER_FILES=${VK_DRIVER_FILES:-$M2V_ICD}
bash "$H/compile.sh" "$SPV" activation softmax tilematmul tilematmul_rt
"$PY" "$H/bench/m2v-bench.py" "$OUT" --spvdir "$SPV" --mlx-python "$PY" "$@"
"$PY" "$H/bench/sweep.py" "$OUT" --spvroot "$OUT/spv-sweep"
# the receipts land in bench/ (results-<device>-<date>.json, sweep-<device>-<date>.json)
cp -f "$OUT"/results-*.json "$OUT"/sweep-*.json "$H/bench/" 2>/dev/null || true
