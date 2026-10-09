#!/usr/bin/env python3
"""Compile every tuning variant of the registered cases, validate + time it, record the winner.

Usage (on the GPU host, under the GPU queue wrapper): sweep.py OUTDIR [--spvroot DIR]
Env: UZU_ROOT, CLSPV (passed to compile.sh).
Writes OUTDIR/sweep-<device>-<date>.json. The default (no M2V_DEFS) variant is
always measured as the baseline.
"""
import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys

# m2v-bench.py has a hyphen in its name; load it by path
_spec = importlib.util.spec_from_file_location(
    "m2v_bench", os.path.join(os.path.dirname(os.path.abspath(__file__)), "m2v-bench.py"))
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)

REPO = b.REPO

# typed-GEP route: clang emits unoptimised IR with typed GEPs (no InstCombine i8
# canonicalisation), clspv's own pipeline does the optimising. Same route as the bench table.
TYPED_GEP = "-O0 -Xclang -disable-O0-optnone"

# (case, dtype, n) -> [(variant name, M2V_DEFS, M2V_OPT)]; base (no defines) is first.
VARIANTS = {
    ("activation", "f32", 1 << 23): [("base", "", TYPED_GEP), ("base_o2route", "", ""),
                                     ("ept2", "-DM2V_ACT_EPT=2", TYPED_GEP),
                                     ("ept4", "-DM2V_ACT_EPT=4", TYPED_GEP), ("wg512", "-DM2V_ACT_WG=512", TYPED_GEP)],
    ("activation", "f16", 1 << 23): [("base", "", TYPED_GEP), ("ept2", "-DM2V_ACT_EPT=2", TYPED_GEP),
                                     ("ept4", "-DM2V_ACT_EPT=4", TYPED_GEP)],
    ("softmax", "f32", 4096): [("base", "", TYPED_GEP)],  # wg + elements/thread are baked into softmax.metal
    ("tilematmul", "f32", 1024): [("base_o2route", "", ""), ("base", "", TYPED_GEP),
                                  ("chains2", "-DM2V_TM_CHAINS=2", TYPED_GEP),
                                  ("chains4", "-DM2V_TM_CHAINS=4", TYPED_GEP)],
}


def parse_defs(defs):
    return dict(p.split("=", 1) for p in defs.split() if "=" in p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--spvroot", default=None)
    ap.add_argument("--runner", default=os.environ.get("M2V_RUNNER", "m2v-run"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    spvroot = args.spvroot or os.path.join(args.outdir, "spv-sweep")
    env = dict(os.environ)
    if not env.get("UZU_ROOT"):
        sys.exit("sweep: set UZU_ROOT")

    tag, devname, drv = b.device_info()
    date = datetime.date.today().isoformat()
    # receipts carry device name, driver string and relative paths only (no host names or home paths)
    report = {"device": tag, "device_name": devname, "driver": drv, "date": date, "kernels": {}}

    for (case, dt, n), variants in VARIANTS.items():
        rows = []
        for name, defs, opt in variants:
            spvdir = os.path.join(spvroot, f"{case}-{name}")
            os.makedirs(spvdir, exist_ok=True)
            r = subprocess.run(["bash", os.path.join(REPO, "compile.sh"), spvdir, case],
                               capture_output=True, text=True, timeout=600,
                               env={**env, "M2V_DEFS": defs, "M2V_OPT": opt})
            spv = os.path.join(spvdir, f"{case}.spv")
            if not os.path.exists(spv):
                rows.append({"variant": name, "defs": defs, "opt": opt, "ok": False,
                             "error": f"compile failed: {(r.stdout + r.stderr)[-300:]}"})
                continue
            # non-default workgroup / elements-per-thread need their own grid + local
            grid = local = None
            if case == "activation":
                d = parse_defs(defs)
                wg = int(d.get("-DM2V_ACT_WG", 256))
                ept = int(d.get("-DM2V_ACT_EPT", 1))
                grid, local = ((n + ept * wg - 1) // (ept * wg), 1, 1), (wg, 1, 1)
            try:
                row = b.translated(args.outdir, args.runner, spvdir, case, dt, n, grid=grid, local=local)
                row.update(variant=name, defs=defs, opt=opt)
            except Exception as e:  # keep going; record the failure
                row = {"variant": name, "defs": defs, "opt": opt, "ok": False, "error": repr(e)}
            rows.append(row)
        good = [r for r in rows if r.get("ok")]
        winner = min(good, key=lambda r: r["us"])["variant"] if good else None
        report["kernels"][f"{case}_{dt}_{n}"] = {"variants": rows, "winner": winner}
        print(f"{case}_{dt}_{n}  winner={winner}")
        for r in rows:
            us = r.get("us")
            print(f"  {r['variant']:<12} us={us if us is None else round(us, 1)} ok={r.get('ok')} err={r.get('err')}")

    out = os.path.join(args.outdir, f"sweep-{tag}-{date}.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"# wrote {out}")


if __name__ == "__main__":
    main()
