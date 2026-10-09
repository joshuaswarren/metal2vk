#!/usr/bin/env python3
"""Build the host's kernel gate table (M2V_POLICY_FILE, see m2v-host/src/policy.rs) from a sweep run.

    make-gate.py SWEEP_RUN.json OUT.json [--fallbacks FALLBACKS.json] [--tag NAME]

SWEEP_RUN.json is the list of rows written by sweep/m2v_sweep_run.py (sweep-run.json). A kernel is `verified` only when its
row has `ref` == "ok": it ran on the device and matched the CPU reference within tolerance. A row whose `ref` starts with
"FAIL" is `failed`. Everything else, including every kernel that ran without a reference, is `unverified` and is checked on
first use or routed to its fallback. Kernels that did not produce valid SPIR-V are not listed: the host sees them as not
translated. FALLBACKS.json is merged into the output as the `fallbacks` section:
{"kernel": {"hand": "omarchy-mlx kernel"} | {"cpu": "reference name"}}.
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("out")
    ap.add_argument("--fallbacks")
    ap.add_argument("--tag", default="sweep")
    a = ap.parse_args()
    with open(a.rows) as f:
        rows = json.load(f)
    kernels = {}
    for r in rows:
        if r.get("val") != "ok":
            continue
        name = r.get("kernel") or r.get("entry")
        ref = r.get("ref") or ""
        if ref == "ok":
            entry = {"state": "verified", "evidence": f"{a.tag}: {r['file']} {r['entry']} ref ok"}
        elif ref.startswith("FAIL"):
            entry = {"state": "failed", "evidence": f"{a.tag}: {r['file']} {r['entry']} {ref[:80]}"}
        else:
            entry = {"state": "unverified", "evidence": ""}
        old = kernels.get(name)
        # a name can come from several variants; the weakest state wins (failed < unverified < verified)
        order = {"failed": 0, "unverified": 1, "verified": 2}
        if old is None or order[entry["state"]] < order[old["state"]]:
            kernels[name] = entry
    fallbacks = {}
    if a.fallbacks:
        with open(a.fallbacks) as f:
            fallbacks = json.load(f)
    with open(a.out, "w") as f:
        json.dump({"kernels": kernels, "fallbacks": fallbacks}, f, indent=1, sort_keys=True)
    states = [v["state"] for v in kernels.values()]
    print(f"{a.out}: {len(kernels)} kernels, verified {states.count('verified')}, unverified {states.count('unverified')}, failed {states.count('failed')}")


if __name__ == "__main__":
    sys.exit(main())
