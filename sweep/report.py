#!/usr/bin/env python3
"""Reports for m2v_sweep.py results.
Usage: report.py RESULT.json [--table OUT.tsv] [--features OUT.md] [--top N] [--compare BEFORE.json]
Table: one row per (file, entry, variant): parse, ir, spv, val, run, ref, first error.
Features: missing Metal features ranked by the number of distinct entries whose compile failed on them (clang: the identifier/type
in the diagnostics, clspv/spirv-val: the normalized message). An entry is counted once per feature."""
import argparse
import collections
import json
import re


def load(path):
    return json.load(open(path))


def rate(rows):
    out = {}
    for s in ("uzu", "tf"):
        rs = [r for r in rows if r["set"] == s and r["parse"] != "n/a"]
        n = len(rs)
        out[s] = {"entries": n, "ir": sum(r["ir"] == "ok" for r in rs), "spv": sum(r["spv"] == "ok" for r in rs),
                  "val": sum(r["val"] == "ok" for r in rs), "run": sum(r.get("run") == "ok" for r in rs),
                  "ref": sum(r.get("ref") == "ok" for r in rs)}
    return out


def features(rows, primary=False):
    cnt = collections.Counter()
    ex = {}
    for r in rows:
        if r["parse"] == "n/a" or r["val"] == "ok":
            continue
        keys = [r.get("primary") or (r["features"][0] if r["features"] else "other:?")] if primary else set(r["features"])
        for k in keys:
            cnt[k] += 1
            ex.setdefault(k, (r["file"], r["entry"], r["error"]))
    return cnt, ex


def fmt_rate(c):
    n = c["entries"] or 1
    return (f"entries {c['entries']}: IR {c['ir']} ({100 * c['ir'] / n:.1f}%), SPIR-V {c['spv']} ({100 * c['spv'] / n:.1f}%), "
            f"valid {c['val']} ({100 * c['val'] / n:.1f}%), runs {c['run']}, matches reference {c['ref']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--table")
    ap.add_argument("--features")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--compare")
    ap.add_argument("--primary", action="store_true", help="rank by the first error of each entry (what blocks it first)")
    a = ap.parse_args()
    rows = load(a.result)
    if a.table:
        with open(a.table, "w") as f:
            f.write("set\tfile\tentry\tvariant\tparse\tir\tspv\tval\trun\tref\tfirst_error\n")
            for r in sorted(rows, key=lambda r: (r["set"], r["file"], r["entry"], r["variant"])):
                f.write("\t".join([r["set"], r["file"], r["entry"], r["variant"], r["parse"], r["ir"], r["spv"], r["val"],
                                   r.get("run", ""), r.get("ref", ""), re.sub(r"\s+", " ", r["error"])[:140]]) + "\n")
    cur = rate(rows)
    print("\n".join(f"{s}: {fmt_rate(c)}" for s, c in cur.items()))
    cnt, ex = features(rows, a.primary)
    if a.features:
        before = rate(load(a.compare)) if a.compare else None
        with open(a.features, "w") as f:
            f.write("# Missing Metal features, ranked\n\n")
            f.write("Counts are distinct entries (a kernel variant is one entry) whose compile stopped on the feature, over uzu (55 files) and\n"
                    "TensorFold Zig (225 files). An entry that fails on several features counts for each of them.\n\n")
            f.write("## Compile rates\n\n| set | entries | IR | SPIR-V | valid |\n|---|---|---|---|---|\n")
            for s, c in cur.items():
                n = c["entries"] or 1
                f.write(f"| {s} | {c['entries']} | {c['ir']} ({100 * c['ir'] / n:.1f}%) | {c['spv']} ({100 * c['spv'] / n:.1f}%) | "
                        f"{c['val']} ({100 * c['val'] / n:.1f}%) |\n")
            if before:
                f.write("\nBefore the shim fixes:\n\n| set | entries | IR | SPIR-V | valid |\n|---|---|---|---|---|\n")
                for s, c in before.items():
                    n = c["entries"] or 1
                    f.write(f"| {s} | {c['entries']} | {c['ir']} ({100 * c['ir'] / n:.1f}%) | {c['spv']} ({100 * c['spv'] / n:.1f}%) | "
                            f"{c['val']} ({100 * c['val'] / n:.1f}%) |\n")
            f.write(f"\n## Top {a.top} features still missing\n\n| # | feature | entries | example (file, entry) | first error |\n|---|---|---|---|---|\n")
            for i, (k, v) in enumerate(cnt.most_common(a.top), 1):
                fl, en, er = ex[k]
                f.write(f"| {i} | `{k}` | {v} | {fl} `{en[:40]}` | {re.sub(chr(124), '/', er)[:90]} |\n")
    else:
        for k, v in cnt.most_common(a.top):
            print(f"{v:5d}  {k}   e.g. {ex[k][0]} {ex[k][1][:40]}")


if __name__ == "__main__":
    main()
