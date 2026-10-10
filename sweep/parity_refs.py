#!/usr/bin/env python3
"""numpy references for the sibling kernels the omarchy path refuses.

Each refs_* module registers one function per kernel in its REFS dict:
``fn(kern, inputs) -> {output name: numpy array of output bits}``. ``kern``
is the parity_common table entry and ``inputs`` maps buffer name to the numpy
array exactly as saved by parity_mlx (bf16 as uint16 bit patterns, others
their numpy dtype). The references simulate the arithmetic of the assembled
.metal source op for op; see refs_common for the rounding conventions and
refs_router for a fully worked example.
"""
import parity_common as pc  # noqa: F401  (table access for check()/callers)

REGISTRY = ("refs_router", "refs_moe_expert", "refs_gdn_prework",
            "refs_gdn_verify", "refs_moe_attn", "refs_attention")


def _load():
    refs = {}
    secondary = {}
    judges = {}
    missing = []
    for mod in REGISTRY:
        try:
            m = __import__(mod)
            refs.update(m.REFS)
            secondary.update(getattr(m, "SECONDARY", {}))
            judges.update(getattr(m, "JUDGES", {}))
        except ImportError:
            missing.append(mod)
    return refs, missing, secondary, judges


REFS, _MISSING, SECONDARY, JUDGES = _load()


def check_all(seed=1234):
    """Generate every input from the seeded table and run every reference.
    Catches shape/dtype errors and non-finite results without any dispatch."""
    import numpy as np
    pc.check_constant_bytes(seed)
    bad = []
    for name, fn in sorted(REFS.items()):
        kern = pc.BY_NAME[name]
        inputs = {}
        for i, spec in enumerate(kern["inputs"]):
            inputs[spec[0]] = np.frombuffer(pc.gen_input(name, i, spec, seed),
                                            pc.DTYPES[spec[1]][1] or np.uint16
                                            ).reshape(spec[2])
        outs = fn(kern, inputs)
        for o in kern["outputs"]:
            oname, ocode, shape = o[0], o[1], o[2]
            if oname not in outs:
                bad.append(f"{name}: no reference output {oname}")
                continue
            arr = np.asarray(outs[oname])
            want_dt = pc.DTYPES[ocode][1] or np.uint16
            if arr.shape != tuple(shape):
                bad.append(f"{name}.{oname}: shape {arr.shape} != {tuple(shape)}")
            if arr.dtype != want_dt:
                bad.append(f"{name}.{oname}: dtype {arr.dtype} != {want_dt}")
            f = pc.bits_to_float(ocode, arr.reshape(-1).astype(want_dt))
            if np.isnan(f).any():
                bad.append(f"{name}.{oname}: NaN values")
            # +-inf can be kernel-legitimate (masked softmax rows store
            # -INFINITY), so only NaN is a generation-side failure
    if _MISSING:
        print(f"registry: modules absent (kernels without references): {_MISSING}")
    if bad:
        raise SystemExit("reference check failed:\n  " + "\n  ".join(bad))
    print(f"reference check ok: {len(REFS)} kernels")


if __name__ == "__main__":
    check_all()
