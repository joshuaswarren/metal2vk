#!/bin/bash
# One command to reproduce the bench on the GPU host (jw16), under gpu-turn:
#   PATH=$HOME/bin:$PATH gpu-turn -m 25 -- bash bench/run-all.sh OUTDIR [--device g14c] [--skip-hand]
# Env: UZU_ROOT (uzu checkout), CLSPV (patched clspv), PYTHON (python with mlx for the hand side),
#      M2V_OPT (compile route; default is the typed-GEP route, see docs/bench-status.md).
# Compiles the registered cases, prints the translated-vs-hand table, runs the tuning sweep.
# Trailing args go to m2v-bench.py only.
set -eu
OUT=${1:?outdir}
shift || true
H=$(cd "$(dirname "$0")/.." && pwd)
# host-specific paths (uzu, clspv, mlx python, ICD) live in bench/env-<host>.sh
case "$(hostname -s)" in
  jw16*) [ -r "$H/bench/env-jw16.sh" ] && . "$H/bench/env-jw16.sh" ;;
esac
PY=${PYTHON:-python3}
export UZU_ROOT=${UZU_ROOT:?set UZU_ROOT}
export CLSPV=${CLSPV:-$HOME/scratch/metal2vk/clspv/build/bin/clspv}
SPV=$OUT/spv
mkdir -p "$OUT" "$SPV"
# post-fix route (typed GEPs; see docs/bench-status.md): clang stops before InstCombine's
# i8-GEP canonicalisation, clspv's own pipeline optimises. Override with M2V_OPT to compare.
export M2V_OPT=${M2V_OPT:--O0 -Xclang -disable-O0-optnone}
bash "$H/compile.sh" "$SPV" activation softmax tilematmul tilematmul_rt
"$PY" "$H/bench/m2v-bench.py" "$OUT" --spvdir "$SPV" --mlx-python "$PY" "$@"
"$PY" "$H/bench/sweep.py" "$OUT" --spvroot "$OUT/spv-sweep"
