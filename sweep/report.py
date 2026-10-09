#!/usr/bin/env python3
"""Reports for m2v_sweep.py results.
Usage: report.py RESULT.json [--table OUT.tsv] [--features OUT.md] [--failures OUT.md] [--compare BEFORE.json] [--primary] [--top N]
  --table     one row per (file, entry, variant): parse, ir, spv, val, run, ref, the IR and valid columns of the --compare run, first error
  --features  compile rates (before and after), missing features grouped into families and ranked by the entries they block first
              (an entry's first error) and by the entries they touch anywhere, then the top raw diagnostics
  --failures  per-entry failure classes: the first clang, clspv or spirv-val message of every failing entry, grouped
  --primary   print the raw ranking by first error instead of writing files"""
import argparse
import collections
import json
import re

PATH = re.compile(r"(?:/[\w.+@-]+)+/([\w.+-]+(?::\d+(?::\d+)?)?)")


def scrub(text):
    """Absolute paths in compiler diagnostics (a build host's home or scratch directories) reduced to the file name and line."""
    return PATH.sub(r"\1", text)


FAMILIES = [
    ("Apple MetalPerformancePrimitives / NAX (mpp::tensor_ops, tensor, dextents, cooperative tensors, matmul2d)",
     r"mpp|dextents|tensor|execution_simd|remove_addrspace|matmul2d|cooperative|MatmulMode|matmul_op|frag|mma_16x32|type-id cannot|"
     r"simdgroup_load|simdgroup_store|row_reduce|simdgroup_bfloat|expected 'X' (?:for function|at end)|deduced type|redeclaration|"
     r"call to 'X' is ambiguous|expected external declaration"),
    ("bfloat vs float in one expression (`c ? bfloat : float` is ambiguous: bfloat converts both ways)", r"conditional expression is ambiguous"),
    ("clspv pointer passes (Invalid bitcast, undefined reference, OpPhi/OpSelect/AccessChain pointer types)",
     r"^clspv:(?:Invalid bitcast|error: undefined reference|\s*%)|OpPhi|OpPtrAccessChain|Select|AtomicLoad|AtomicIAdd|Expected input to be a pointer"),
    ("clspv does not finish (timeout, crash without a message, SimplifyPointerBitcast loop)", r"^clspv:(?:timeout|clspv exit|MNV)"),
    ("clspv: other", r"^clspv:"),
    ("spirv-val: other", r"^spirv-val:"),
    ("vector construction and conversion (mk1, C-style vector casts, scalar initializers)",
     r"mk1|C-style cast from|excess elements|functional-style cast|no viable conversion"),
    ("kernel parameter shapes the wrapper cannot express", r"kernel parameter|declared as array of references|array by value|wrapper:"),
    ("host-injected symbols (a kernel needs a -D, a typedef or a helper the host provides)", r"^ident:[A-Za-z_]*[A-Z]|^type:|FZ_|fsoftplus|sq_acc|quad_dot|simd_topk"),
    ("template kernels with no instantiation", r"without host_name"),
]


def family(key):
    for name, rx in FAMILIES:
        if re.search(rx, key):
            return name
    return "other"


def load(path):
    return json.load(open(path))


def rate(rows):
    out = {}
    for s in dict.fromkeys(r["set"] for r in rows):
        rs = [r for r in rows if r["set"] == s and r["parse"] not in ("n/a", "refused")]
        out[s] = {"entries": len(rs), "ir": sum(r["ir"] == "ok" for r in rs), "spv": sum(r["spv"] == "ok" for r in rs),
                  "val": sum(r["val"] == "ok" for r in rs), "run": sum(r.get("run") == "ok" for r in rs),
                  "ref": sum(r.get("ref") == "ok" for r in rs), "ran": sum(bool(r.get("run")) for r in rs),
                  "refused": sum(r["parse"] == "refused" for r in rows if r["set"] == s)}
    tot = {k: sum(c[k] for c in out.values()) for k in ("entries", "ir", "spv", "val", "run", "ref", "ran", "refused")}
    out["all"] = tot
    return out


def first_key(r):
    return r.get("primary") or (r["features"][0] if r["features"] else "other:?")


def rate_table(rates):
    lines = ["| set | entries | IR | SPIR-V | valid | refused | runs | matches reference |", "|---|---|---|---|---|---|---|---|"]
    for s, c in rates.items():
        n = c["entries"] or 1
        runs = f"{c['run']} of {c['ran']}" if c["ran"] else "not run here"
        lines.append(f"| {s} | {c['entries']} | {c['ir']} ({100 * c['ir'] / n:.1f}%) | {c['spv']} ({100 * c['spv'] / n:.1f}%) | "
                     f"{c['val']} ({100 * c['val'] / n:.1f}%) | {c['refused']} | {runs} | {c['ref']} |")
    return "\n".join(lines) + "\n"


def first_message(r):
    """The first diagnostic as the tool printed it, without the file path."""
    line = ((r.get("detail") or r["error"]).splitlines() or [""])[0]
    line = re.sub(r"^\S*?:\d+:\d+: ", "", line)
    return scrub(line).replace("|", "/")[:90]


def write_features(rows, path, before, top):
    cur = rate(rows)
    failing = [r for r in rows if r["parse"] not in ("n/a", "refused") and r["val"] != "ok"]
    first = collections.Counter(family(first_key(r)) for r in failing)
    anywhere = collections.Counter()
    for r in failing:
        for fam in {family(k) for k in r["features"]}:
            anywhere[fam] += 1
    raw = collections.Counter()
    ex = {}
    for r in failing:
        for k in set(r["features"]):
            raw[k] += 1
            ex.setdefault(k, r)
    with open(path, "w") as f:
        f.write("# Missing Metal features, ranked\n\n")
        f.write("Entries are kernel variants: one per uzu variant pair (first and last of each VARIANTS list) and one per TensorFold\n"
                "host_name instantiation. A count is a number of entries.\n\n## Compile rates\n\n")
        if before:
            f.write("Before (the shim and flags of main when the sweep started):\n\n" + rate_table(rate(before)) + "\nAfter:\n\n")
        f.write(rate_table(cur))
        f.write("\n## Families, ranked by the entries they block first\n\n| # | family | blocks first | touches | \n|---|---|---|---|\n")
        for i, (fam, n) in enumerate(first.most_common(), 1):
            f.write(f"| {i} | {fam} | {n} | {anywhere[fam]} |\n")
        f.write(f"\n## Top {top} raw diagnostics (entries affected, any position)\n\n| # | feature | entries | example | first message |\n"
                "|---|---|---|---|---|\n")
        for i, (k, v) in enumerate(raw.most_common(top), 1):
            r = ex[k]
            f.write(f"| {i} | `{k[:80]}` | {v} | {r['file']} `{r['entry'][:36]}` | {first_message(r)} |\n")


def write_failures(rows, path):
    groups = collections.defaultdict(list)
    for r in rows:
        if r["parse"] in ("n/a", "refused") or r["val"] == "ok":
            continue
        stage = "front end" if r["parse"] == "FAIL" else "clspv" if r["spv"] == "FAIL" else "spirv-val"
        groups[(stage, first_key(r))].append(r)
    with open(path, "w") as f:
        f.write("# Per-entry failure classes\n\nThe first message of every failing entry, grouped by stage and normalized message, largest first.\n")
        for (stage, key), rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            f.write(f"\n## {stage}: {key[:110]} ({len(rs)})\n\n")
            f.write("```\n" + scrub(rs[0].get("detail") or rs[0]["error"])[:420] + "\n```\n\n")
            f.write("Entries: " + ", ".join(f"`{r['file']}:{r['entry'][:30]}`" for r in rs[:8]) + (f" and {len(rs) - 8} more" if len(rs) > 8 else "") + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--table")
    ap.add_argument("--features")
    ap.add_argument("--failures")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--compare")
    ap.add_argument("--primary", action="store_true")
    a = ap.parse_args()
    rows = load(a.result)
    if a.table:
        base = {}
        if a.compare:
            for r in load(a.compare):
                base[(r["set"], r["file"], r["entry"], r["variant"])] = r
        with open(a.table, "w") as f:
            f.write("set\tfile\tentry\tvariant\tparse\tir\tspv\tval\trun\tref\tbefore_ir\tbefore_val\tfirst_error\n")
            for r in sorted(rows, key=lambda r: (r["set"], r["file"], r["entry"], r["variant"])):
                b = base.get((r["set"], r["file"], r["entry"], r["variant"]))
                f.write("\t".join([r["set"], r["file"], r["entry"], r["variant"], r["parse"], r["ir"], r["spv"], r["val"], r.get("run", ""),
                                   r.get("ref", ""), (b["ir"] if b else "?"), (b["val"] if b else "?"),
                                   scrub(re.sub(r"\s+", " ", r["error"]))[:140]]) + "\n")
    if a.features:
        write_features(rows, a.features, load(a.compare) if a.compare else None, a.top)
    if a.failures:
        write_failures(rows, a.failures)
    if a.primary or not (a.table or a.features or a.failures):
        for s, c in rate(rows).items():
            print(s, c)
        cnt = collections.Counter(first_key(r) for r in rows if r["parse"] not in ("n/a", "refused") and r["val"] != "ok")
        for k, v in cnt.most_common(a.top):
            print(f"{v:5d}  {k[:100]}")


if __name__ == "__main__":
    main()
