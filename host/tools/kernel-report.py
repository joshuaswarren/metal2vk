#!/usr/bin/env python3
"""Summarise the Linux/vulkan-host kernel translation results.

Reads the JSONL diagnostics the uzu build script's m2v toolchain writes to
OUT_DIR/m2v-kernels.json (one line per kernel: run, source, kernel,
translated, diagnostic) and emits host/KERNELS.md: translated / failed /
total counts plus the top 5 failure reasons.

Usage: kernel-report.py OUT_DIR/m2v-kernels.json [OUT_DIR/m2v-kernels.run] [> KERNELS.md]
The optional run file selects the newest run; without it the newest run in the
file is used.
"""
import collections
import json
import sys

path = sys.argv[1]
records = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

if not records:
    sys.exit(f"{path}: no records")

if len(sys.argv) > 2:
    run = int(open(sys.argv[2]).read().strip())
else:
    run = max(r["run"] for r in records)
records = [r for r in records if r["run"] == run]

total = len(records)
translated = sum(1 for r in records if r["translated"])
failed = total - translated

reasons = collections.Counter()
for r in records:
    if not r["translated"]:
        diag = (r.get("diagnostic") or "unknown").split(" | ")[0]
        # normalise the noisy parts so reasons group by cause
        for token in ("error: ", "warning: "):
            if token in diag:
                diag = diag[diag.index(token) + len(token):]
                break
        diag = diag.split("(")[0].strip()[:120] or "unknown"
        reasons[diag] += 1

sources = sorted({r["source"] for r in records})
out = []
out.append("# host/KERNELS.md - uzu kernel translation (vulkan-host pipeline)\n")
out.append(f"Run {run} of OUT_DIR/m2v-kernels.json: **{translated} translated / {failed} failed / {total} total**\n")
out.append("A kernel left untranslated is absent from its .m2vlib: creating its pipeline fails at runtime with the kernel name. Policy-only unit tests do not need the kernels.\n")
out.append("## Failure reasons (top 5)\n")
for reason, count in reasons.most_common(5):
    out.append(f"- {count} × `{reason}`")
if not reasons:
    out.append("- none")
out.append("")
out.append("## Per-source counts\n")
out.append("| source | translated | failed |")
out.append("| --- | --- | --- |")
for src in sources:
    rs = [r for r in records if r["source"] == src]
    ok = sum(1 for r in rs if r["translated"])
    out.append(f"| `{src}` | {ok} | {len(rs) - ok} |")
out.append("")
print("\n".join(out))
