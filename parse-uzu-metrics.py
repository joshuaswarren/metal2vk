"""Extract uzu's published benchmark numbers (trymirai.com/metrics payload) for one device key.
Usage: parse-uzu-metrics.py PAGE.html DEVICE_KEY   e.g. macos-m1-pool"""
import json
import re
import sys

s = open(sys.argv[1]).read().replace('\\"', '"')
dev = sys.argv[2]
rows = []
for m in re.finditer(r'\{"key":"([^"]+)","participant_key":"([^"]+)","role":"[^"]*","engine":"([^"]+)","version":"([^"]+)"', s):
    start = m.start()
    depth = 0
    end = None
    for i in range(start, min(len(s), start + 200000)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        continue
    try:
        o = json.loads(s[start:end])
    except json.JSONDecodeError:
        continue
    b = o.get("benchmarks", {}).get(dev)
    if not b:
        continue
    sm = b["summary"]
    rows.append((o["model"]["name"], o["participant_key"], sm["output"].get("autoregressive"), sm.get("input"), sm.get("memory"), b.get("source")))
rows.sort(key=lambda r: (r[0], r[1]))
print(f"{dev}: output tok/s (autoregressive), input tok/s, resident GiB, checkpoint")
for r in rows:
    print(f"{r[0]:32s} {r[1]:10s} out={r[2] if r[2] is None else round(r[2], 1)!s:>7} in={r[3] if r[3] is None else round(r[3], 1)!s:>8} mem={'' if r[4] is None else round(r[4] / 2**30, 2)!s:>6}  {r[5]}")
