#!/usr/bin/env python3
"""Host check for precise_rsqrt_scoped_math_unchanged. No GPU.

Stand-ins replace the device dispatch. The three results are the contract:

  native-FMA misses Irvine -> parent pass, nested status unsupported
  native-FMA matches Irvine -> parent fail
  missing precise module -> parent is not pass

  python3 m2v_rsqrt_scoped_status_check.py
"""
import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent


def load_driver():
    sys.argv = ["m2v-test.py", "/tmp/m2v-scoped-status-unused", "/bin/false", "/tmp/no-such-spv"]
    spec = importlib.util.spec_from_file_location("m2v_test_scoped_status", ROOT / "m2v-test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scope_dispatch(spv, entry, ins, n_out, n):
    """Base and current agree. The native-FMA fma lanes differ, so the selection is visible."""
    native = "nativefma" in spv
    rec = {"spv_sha256": "scope", "status": "ran", "output_sha256": ["h"] * n_out}
    cols = []
    for i in range(n_out):
        values = np.zeros(n, np.uint32)
        if native and i >= 3:
            values[:] = 1
        cols.append(values)
    return rec, cols


def run(mod, layer):
    mod.rsqrt_common = lambda: {"n": 8, "in_path": "in.bin"}
    mod.rsqrt_dispatch = scope_dispatch
    mod.rsqrt_layer = layer
    return mod.precise_rsqrt_scoped_math_unchanged()


def main():
    mod = load_driver()

    def miss(c, stem, entry, strict):
        return {"spv_sha256": "precise-nf", "status": "ran",
                "compatibility": {"verdict": "fail", "finite_mismatches": 1814}}, None

    def match(c, stem, entry, strict):
        return {"spv_sha256": "precise-nf", "status": "ran",
                "compatibility": {"verdict": "pass", "finite_mismatches": 0}}, None

    def missing(c, stem, entry, strict):
        return {"status": "incomplete: module not built"}, None

    missed = run(mod, miss)
    if missed["status"] != "pass" or missed["precise_under_native_fma"]["status"] != "unsupported":
        raise SystemExit("miss: parent %s nested %s" % (missed["status"], missed["precise_under_native_fma"]["status"]))
    matched = run(mod, match)
    if matched["status"] != "fail":
        raise SystemExit("match: parent %s" % matched["status"])
    absent = run(mod, missing)
    if absent["status"] == "pass":
        raise SystemExit("missing module returned pass")
    print("scoped status check: ok")
    print("miss", missed["status"], missed["precise_under_native_fma"]["status"])
    print("match", matched["status"], matched["precise_under_native_fma"]["status"])
    print("missing", absent["status"])


if __name__ == "__main__":
    main()
