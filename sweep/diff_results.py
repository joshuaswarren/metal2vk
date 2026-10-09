#!/usr/bin/env python3
"""Diff two sweep result JSONs row by row, keyed by (set, file, entry, variant)."""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
K = lambda r: (r["set"], r["file"], r["entry"], r["variant"])
da, db = {K(r): r for r in a}, {K(r): r for r in b}
dropped = [k for k in da if k not in db]
added = [k for k in db if k not in da]
changed = []
for k in da:
    if k in db:
        diffs = {f: (da[k].get(f), db[k].get(f)) for f in set(da[k]) | set(db[k]) if da[k].get(f) != db[k].get(f)}
        if diffs:
            changed.append((k, diffs))
print(f"base {len(da)} rows, after {len(db)} rows")
for k in dropped:
    print("DROPPED:", k)
for k in added:
    print("ADDED:", k, {f: db[k].get(f) for f in ("parse", "error")})
for k, diffs in changed:
    print("CHANGED:", k)
    for f, (va, vb) in diffs.items():
        print(f"    {f}: {str(va)[:90]!r} -> {str(vb)[:90]!r}")
print(f"{len(dropped)} dropped, {len(added)} added, {len(changed)} changed")
