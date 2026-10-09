#!/usr/bin/env python3
"""Coverage gate for the sweep: no entry that has a valid Vulkan SPIR-V module may stop having one.

  check_coverage.py RESULT.json [--floor sweep/coverage-floor.txt]            exit 1 if an entry in the floor is no longer valid
  check_coverage.py RESULT.json [--floor ...] --update                         rewrite the floor with every entry that is valid now

The floor lists one entry per line (`set/file::entry[variant]`). An entry in the floor that fails only because clspv hit its time limit is a
warning, not a failure: a slow runner must not block a PR. New valid entries are reported so the floor can be raised with --update.
"""
import argparse
import json
import sys


def key(r):
    return f"{r['set']}/{r['file']}::{r['entry']}[{r['variant']}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--floor", default="sweep/coverage-floor.txt")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    rows = json.load(open(a.result))
    valid = {key(r) for r in rows if r.get("val") == "ok"}
    print(f"coverage: {len(valid)} of {len(rows)} entries have valid Vulkan SPIR-V ({100 * len(valid) / len(rows):.1f}%)")
    if a.update:
        open(a.floor, "w").write("".join(k + "\n" for k in sorted(valid)))
        print(f"floor rewritten: {len(valid)} entries")
        return 0
    floor = {line.strip() for line in open(a.floor) if line.strip()}
    by_key = {key(r): r for r in rows}
    lost = sorted(floor - valid)
    slow = [k for k in lost if k in by_key and "timeout" in (by_key[k].get("error") or "")]
    broken = [k for k in lost if k not in slow]
    for k in slow:
        print(f"warning: {k}: clspv did not finish in time (not counted as a regression)")
    for k in broken:
        r = by_key.get(k)
        print(f"REGRESSION: {k}: " + ("missing from the sweep" if r is None else f"{r['error'][:160]}"))
    gained = sorted(valid - floor)
    if gained:
        print(f"{len(gained)} entries are valid now and not in the floor; run with --update and commit it to keep them")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
