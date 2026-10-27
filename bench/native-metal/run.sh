#!/bin/bash
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-"$DIR/results"}
fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }

command -v swiftc >/dev/null 2>&1 || fail "swiftc is missing; install Xcode Command Line Tools or Xcode."
command -v xcode-select >/dev/null 2>&1 || fail "xcode-select is missing; install Xcode Command Line Tools or Xcode."
xcode-select -p >/dev/null 2>&1 || fail "the active developer directory is unavailable; run xcode-select --install or select Xcode."
command -v sw_vers >/dev/null 2>&1 || fail "sw_vers is missing; this harness requires macOS."
command -v sysctl >/dev/null 2>&1 || fail "sysctl is missing; unable to report load average."
MACOS=$(sw_vers -productVersion) || fail "unable to read macOS version."
LOAD=$(sysctl -n vm.loadavg 2>/dev/null) || fail "unable to read system load average."
CHIP=$(system_profiler -detailLevel mini SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip/ {print $2; exit}')
CHIP=${CHIP:-$(sysctl -n hw.model 2>/dev/null || printf 'unknown')}
printf 'Native Metal timing harness\nChip: %s\nmacOS: %s\nLoad average: %s\n' "$CHIP" "$MACOS" "$LOAD"

BIN=${NATIVE_BENCH_BIN:-"$DIR/.native-bench"}
if ! swiftc -O -framework Metal -framework Foundation "$DIR/native-bench.swift" -o "$BIN"; then
  fail "swiftc could not build the harness. Check the Xcode Command Line Tools installation."
fi
if [ "${1:-}" = "--rsqrt" ]; then exec "$BIN" "$@"; fi  # precise-rsqrt capture: KERNEL.metal INPUT.u32le.bin OUTPREFIX
"$BIN" "$OUT"
