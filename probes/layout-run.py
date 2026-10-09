"""Usage: layout-run.py OUTDIR RUNNER PROBE.spv   (run with VK_DRIVER_FILES pointing at an ICD that exposes coopmat)
Decodes the three 8x8 dumps (accumulator, A, B use) into lane -> (row, col) lists and checks them against Apple's
layout: row = (lane >> 4 & 1) * 4 + (lane >> 1 & 3); col0 = (lane >> 3 & 1) * 4 + (lane & 1) * 2, elements col0, col0 + 1."""
import os
import subprocess
import sys

import numpy as np

out, runner, spv = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
files = []
for n in ("acc", "usea", "useb"):
    p = os.path.join(out, n + ".in")
    np.zeros(65, np.float32).tofile(p)
    files.append(p)
cmd = [runner, spv, "main", "1", "1", "1"]
for f in files:
    cmd += ["--buf", f]
for i, n in enumerate(("acc", "usea", "useb")):
    cmd += ["--dump", f"{i}:{os.path.join(out, n + '.out')}"]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
print("rc", r.returncode, r.stdout.strip(), r.stderr.strip()[-300:])
if r.returncode:
    sys.exit(1)


def apple(lane):
    row = ((lane >> 4) & 1) * 4 + ((lane >> 1) & 3)
    col0 = ((lane >> 3) & 1) * 4 + (lane & 1) * 2
    return row, col0


for n in ("acc", "usea", "useb"):
    d = np.fromfile(os.path.join(out, n + ".out"), np.float32)
    length = int(d[64])
    mism = 0
    held = {}
    for pos in range(64):
        v = int(d[pos])
        lane, idx = v // 16, v % 16
        held.setdefault(lane, []).append((idx, pos // 8, pos % 8))
    for lane, lst in sorted(held.items()):
        lst.sort()
        if n == "acc" or True:
            ar, ac = apple(lane)
            want = [(0, ar, ac), (1, ar, ac + 1)]
            if lst[:2] != want:
                mism += 1
    print(f"{n}: elements per lane (length()) = {length}; lanes whose (row, col) differ from the Apple map: {mism} of {len(held)}")
    for lane in (0, 1, 2, 3, 8, 16, 31):
        if lane in held:
            print(f"  lane {lane:2d}: coopmat {sorted(held[lane])}  apple row,col0 = {apple(lane)}")
