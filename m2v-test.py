#!/usr/bin/env python3
"""metal2vk spike test driver. Run on the GPU host (one small pilot submit, then a timed batch under 20 s).
Usage: m2v-test.py OUTDIR RUNNER SPVDIR [case ...]   cases: activation softmax tilematmul
precise_rsqrt_*: SPVDIR is the output of cases/precise_rsqrt/lower.sh; see docs/precise-rsqrt.md for the five cases.
Writes OUTDIR/results.json. CPU reference is numpy float64 -> float32/16 compare."""
import json, subprocess, sys, os, struct, time
import numpy as np

out, runner, spvdir = sys.argv[1:4]
RSQRT_CASES = ("precise_rsqrt_fp32_boundaries", "precise_rsqrt_fp32_old_fail_new_pass", "precise_rsqrt_fp32_special_values",
               "precise_rsqrt_fp32_rounding_boundaries", "precise_rsqrt_scoped_math_unchanged")
ALL = ["activation", "softmax", "gemm", "tilematmul_rt", "tilematmul_h", "tilematmul", *RSQRT_CASES]
cases = sys.argv[4:] or ["activation", "softmax", "tilematmul"]
rng = np.random.default_rng(7)
res = {}


LOCAL = {"activation": "256 1 1", "softmax": "256 1 1", "tile_matmul": "32 1 1", "gemm_f32_t8x": "32 1 1", "gemm_f32_t32x": "32 2 2",
         "gemm_f32_t64x": "32 2 2", "rsqrt_": "64 1 1"}


_SEEN = set()


def run(spv, entry, grid, bufs, push=None, iters=1, dump=None):
    loc = next(v for k, v in LOCAL.items() if entry.startswith(k))
    cmd = [runner, spv, entry, *map(str, grid), "--iters", str(iters), "--local", *loc.split()]
    if push is not None:
        pf = os.path.join(out, entry + ".push")
        open(pf, "wb").write(push)
        cmd += ["--push", pf]
    for b in bufs:
        cmd += ["--buf", b]
    for d in ([dump] if isinstance(dump, str) else dump or []):
        cmd += ["--dump", d]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if os.environ.get("M2V_STDERR_FILE") and entry not in _SEEN:
        _SEEN.add(entry)
        with open(os.environ["M2V_STDERR_FILE"], "a") as f:
            f.write(f"=== {entry} grid={grid}\n{r.stderr}\n")
    if r.returncode:
        return None, r.stderr.strip()[-400:]
    line = [l for l in r.stdout.splitlines() if l.startswith("iters=")][0]
    return (float(line.split("per_dispatch_us=")[1].split()[0]), float(line.split("wall_us=")[1])), r.stderr.strip()


def timed(spv, entry, grid, bufs, push=None):
    """pilot with 1 dispatch, then size the batch so the submit stays near 1 s (well under the 20 s rule)."""
    p1, err = run(spv, entry, grid, bufs, push, 1)
    if p1 is None:
        return None, err
    wall1 = p1[1]  # wall clock of a 1-dispatch submit: includes launch latency, so it over-estimates the cost
    iters = int(max(1, min(500, 2e6 / max(wall1, 1.0))))  # target about 2 s per submit, hard ceiling well under the 20 s rule
    t, err = run(spv, entry, grid, bufs, push, iters)
    if t is None:
        return None, err
    ts_per, wall_total = t
    wall_per = wall_total / iters
    # GPU timestamps can under-read (seen on the fork ICD); the wall clock of the whole submit is the conservative figure.
    per = max(ts_per, wall_per) if iters < 50 else wall_per
    return per, (f"iters={iters} est_submit_s={iters * wall1 / 1e6:.2f} actual_submit_s={wall_total / 1e6:.2f} "
                 f"ts_per_us={ts_per:.2f} wall_per_us={wall_per:.2f}")


def w(name, arr):
    p = os.path.join(out, name)
    arr.tofile(p)
    return p


# Frozen Irvine M1 capture (macOS 27.0.1 26A434, Metal 3.0 safe math): sha256 of the whole 8,448-word file. These bits are
# the expected output of the compatibility verdict; any other native file is refused, not scored.
RSQRT_IRVINE_SHA256 = "104d7c9bdd0881ee6e47d401d7defaa35e6d9d5b5506eab14b92fd2958547e4c"
RSQRT_IRVINE_PATH = os.path.join("docs", "repro", "precise-rsqrt", "native-irvine", "native.safe_msl30.out.u32le.bin")


def rsqrt_special_rule(b):
    """Frozen special-value predicate for one non-positive-normal input (docs/precise-rsqrt.md, "Frozen verdict rules").
    Returns (rule, check) or None when the input is only characterized. Positive normals use the exact-bits rule instead.
    NaN payload and sign are never required: neither Metal nor Vulkan specifies them."""
    sign, exp, man = b >> 31, (b >> 23) & 0xFF, b & 0x7FFFFF
    def quiet_nan(o):
        return (o >> 23) & 0xFF == 0xFF and o & 0x400000 != 0
    if exp == 0xFF and man:
        return "nan -> quiet nan", quiet_nan
    if exp == 0xFF:
        return ("-inf -> quiet nan", quiet_nan) if sign else ("+inf -> +0", lambda o: o == 0x00000000)
    if exp == 0 and man == 0:
        return ("-0 -> -inf", lambda o: o == 0xFF800000) if sign else ("+0 -> +inf", lambda o: o == 0x7F800000)
    if exp == 0:
        return None if sign else ("+subnormal -> +inf (flushed)", lambda o: o == 0x7F800000)
    return ("negative -> quiet nan", quiet_nan) if sign else None


def rsqrt_verdict(inputs, primary, got, expected):
    """Compatibility verdict against the frozen Irvine bits: every primary output bit-exact (no tolerance, no allowed ULP)
    and every required special predicate true. Mathematical error is reported elsewhere and never enters this verdict."""
    finite = [i for i in range(primary) if int(got[i]) != int(expected[i])]
    failed, characterized = [], {}
    for i in range(primary, len(inputs)):
        rule = rsqrt_special_rule(inputs[i])
        if rule is None:
            characterized[f"{inputs[i]:08x}"] = f"{int(got[i]):08x}"
        elif not rule[1](int(got[i])):
            failed.append({"input_bits": f"{inputs[i]:08x}", "rule": rule[0], "got": f"{int(got[i]):08x}"})
    first = None if not finite else {"input_bits": f"{inputs[finite[0]]:08x}", "got": f"{int(got[finite[0]]):08x}",
                                     "expected": f"{int(expected[finite[0]]):08x}"}
    return {"verdict": "pass" if not finite and not failed else "fail", "finite_mismatches": len(finite), "first_finite_mismatch": first,
            "special_failures": failed, "characterized_only": characterized}


# sha256 of lower.sh's base/a_shim/rsqrt_precise.spv: the unmodified shim, byte-identical to the Phase 0 module in
# docs/repro/precise-rsqrt/lowering/a_shim. The old half of precise_rsqrt_fp32_old_fail_new_pass runs only this module.
RSQRT_OLD_SHIM_SHA256 = "272624f568ed8f01e95c6e2a5d09d57df4998850d4a356e173df47534b6dec6f"
# Half-propagation fixture: xorshift32 seed and the sha256 of its padded s, xw and expected words, frozen at first build.
RSQRT_HALF_SEED = 0x124F16
RSQRT_HALF_SHA256 = "54d7f4c27808faeea5c909d7b558bd2f713af8c838fee1e45f827512a6893297"
RSQRT_SENTINEL = 0xA5A5A5A5
_RSQRT = {}  # corpus, native bits and one device result per module, shared by the precise_rsqrt cases


def rsqrt_common():
    """Frozen corpus, padded device input, frozen Irvine bits and exact references, or an "incomplete: ..." string.
    The corpus must match its generator byte for byte and the expected bits must be the frozen Irvine file."""
    if "common" in _RSQRT:
        return _RSQRT["common"]
    import hashlib
    import m2v_rsqrt_corpus as corpus
    here = os.path.dirname(os.path.abspath(__file__))
    fixture = os.path.join(here, "cases", "precise_rsqrt")
    bad = corpus.check(fixture)
    if bad:
        _RSQRT["common"] = "incomplete: frozen corpus mismatch " + ", ".join(bad)
        return _RSQRT["common"]
    inputs = corpus.load_inputs(fixture)
    manifest = json.load(open(os.path.join(fixture, "manifest.json")))
    n, primary = len(inputs), manifest["counts"]["primary"]
    padded = np.array(corpus.padded(inputs), np.uint32)
    native = os.environ.get("M2V_RSQRT_NATIVE") or os.path.join(here, RSQRT_IRVINE_PATH)
    raw = open(native, "rb").read() if os.path.exists(native) else None
    if raw is None or hashlib.sha256(raw).hexdigest() != RSQRT_IRVINE_SHA256:
        _RSQRT["common"] = "incomplete: " + ("missing" if raw is None else "refused: not the frozen Irvine capture") + " " + native
        return _RSQRT["common"]
    nat = np.frombuffer(raw, np.uint32)[:n]
    if rsqrt_verdict(inputs, primary, nat, nat)["special_failures"]:
        _RSQRT["common"] = "incomplete: frozen predicates disagree with the frozen native bits"
        return _RSQRT["common"]
    refs = np.array([corpus.exact_rsqrt_bits(int(b)) for b in inputs[:primary]], np.int64)
    d = nat[:primary].astype(np.int64) - refs
    info = {"path": native, "output_sha256": hashlib.sha256(nat.tobytes()).hexdigest(), "required_predicates_hold": True,
            "distance_from_exact_nearest_even": {str(v): int(k) for v, k in zip(*np.unique(d, return_counts=True))}}
    _RSQRT["common"] = {"inputs": inputs, "manifest": manifest, "n": n, "primary": primary, "padded": padded,
                        "in_path": w("rsqrt_in.u32le.bin", padded), "native": nat, "native_info": info, "refs": refs,
                        "inputs_sha256": hashlib.sha256(padded[:n].tobytes()).hexdigest()}
    return _RSQRT["common"]


def spv_shape(dis):
    """Static facts about a module: the defining op of the InverseSqrt operand (None if there is none), whether a word
    loaded from the binding-0 buffer is used by any instruction (a constant-folded result uses none), and the defining
    op and type of every stored value."""
    import re
    def definer(v):
        d = re.search(r"^\s*" + re.escape(v) + r" = (Op\w+)(?: (%\w+))?", dis, re.M)
        return None if not d else (d.group(1) if d.group(1) != "OpSelect" else d.group(1) + " " + d.group(2))
    m = re.search(r"InverseSqrt (%\w+)", dis)
    var = re.search(r"OpDecorate (%\w+) Binding 0\b", dis)
    chains = set(re.findall(r"^\s*(%\w+) = OpAccessChain %\w+ " + re.escape(var.group(1)) + r"\b", dis, re.M)) if var else set()
    loads = [v for v, src in re.findall(r"^\s*(%\w+) = OpLoad %\w+ (%\w+)", dis, re.M) if src in chains]
    return {"inverse_sqrt_operand_defined_by": definer(m.group(1)) if m else None,
            "loads_input": any(len(re.findall(re.escape(v) + r"\b", dis)) > 1 for v in loads),
            "store_defined_by": sorted({definer(v) for v in re.findall(r"OpStore %\w+ (%\w+)", dis)})}


def rsqrt_dispatch(spv, entry, ins, n_out, n):
    """One dispatch of spv over input files ins (bindings 0..) plus n_out sentinel-filled outputs. Returns the record and
    the first n words of each output, or None when the run failed or custody broke (an input changed, or a scored lane
    still holds the sentinel): such bits cannot be attributed to the frozen inputs. Cached per module."""
    import hashlib
    if (spv, entry) in _RSQRT:
        return _RSQRT[(spv, entry)]
    rec, bits = {"spv": spv}, None
    if not os.path.exists(spv):
        rec["status"] = "incomplete: module not built"
        _RSQRT[(spv, entry)] = rec, None
        return rec, None
    stem = spv[:-4]
    refl = json.load(open(stem + ".json"))
    regions = refl.get("push_constant_regions", {})
    push = bytes(max([r["offset"] + r["size"] for r in regions.values()], default=0)) or None
    rec.update(spv_sha256=hashlib.sha256(open(spv, "rb").read()).hexdigest(), **spv_shape(open(stem + ".dis").read()))
    tag = "rsqrt_" + os.path.relpath(stem, spvdir).replace(os.sep, "_").replace(".", "_")
    words = len(np.fromfile(ins[0], np.uint32))
    outs = [os.path.join(out, f"{tag}.out{k}.u32le.bin") for k in range(n_out)]
    backs = [os.path.join(out, f"{tag}.in{k}_after.u32le.bin") for k in range(len(ins))]
    sent = [w(f"{tag}.sentinel{k}.bin", np.full(words, RSQRT_SENTINEL, np.uint32)) for k in range(n_out)]
    dumps = [f"{len(ins) + k}:{o}" for k, o in enumerate(outs)] + [f"{k}:{b}" for k, b in enumerate(backs)]
    t, info = run(spv, entry, (words // 64, 1, 1), list(ins) + sent, push, 1, dump=dumps)
    if t is None:
        rec["status"] = "incomplete: runner failed: " + info
    else:
        got = [np.fromfile(o, np.uint32) for o in outs]
        kept = all(np.array_equal(np.fromfile(b, np.uint32), np.fromfile(i, np.uint32)) for b, i in zip(backs, ins))
        unwritten = sum(int(np.sum(g[:n] == RSQRT_SENTINEL)) for g in got)
        rec.update(runner_stderr=info, input_buffers_unchanged=kept, unwritten_lanes=unwritten,
                   output_sha256=[hashlib.sha256(g[:n].tobytes()).hexdigest() for g in got])
        rec["per_dispatch_us"], rec["wall_us"] = t  # m2v-run's existing host-wall line; not a separate timer
        if not kept or unwritten:
            rec["status"] = "incomplete: buffer custody (input changed or unwritten lanes)"
        else:
            rec["status"], bits = "ran", [g[:n] for g in got]
    _RSQRT[(spv, entry)] = rec, bits
    return rec, bits


def rsqrt_layer(c, stem, entry, strict):
    """Run one scalar rsqrt module over the corpus and score it against the frozen Irvine bits. strict keeps the Phase 0
    shape rule for the uncorrected reductions (the InverseSqrt operand is a runtime OpBitcast load); otherwise the module
    only has to use a word loaded from the input buffer, which is how a corrected shader without InverseSqrt is admitted."""
    rec, bits = rsqrt_dispatch(os.path.join(spvdir, stem + ".spv"), entry, [c["in_path"]], 1, c["n"])
    rec = dict(rec)
    if bits is not None and (rec["inverse_sqrt_operand_defined_by"] != "OpBitcast" if strict else not rec["loads_input"]):
        rec["status"] = "rejected: " + ("InverseSqrt operand is not a runtime-loaded value" if strict else
                                        "no word loaded from the input buffer is used")
        bits = None
    if bits is not None:
        b, primary = bits[0], c["primary"]
        d = b[:primary].astype(np.int64) - c["refs"]  # exact-result distance in binary32 steps; every primary result is positive normal
        rec["distance_from_exact_nearest_even"] = {str(v): int(k) for v, k in zip(*np.unique(d, return_counts=True))}
        rec["vs_native"] = rsqrt_diff(c, b, c["native"])
        rec["compatibility"] = rsqrt_verdict(c["inputs"], primary, b, c["native"])
        bits = b
    return rec, bits


def rsqrt_diff(c, a, b):
    idx = np.nonzero(a != b)[0]
    first = None if not len(idx) else {"index": int(idx[0]), "input_bits": f"{c['inputs'][idx[0]]:08x}", "a": f"{a[idx[0]]:08x}", "b": f"{b[idx[0]]:08x}"}
    p = c["primary"]
    return {"differ_primary": int(np.sum(a[:p] != b[:p])), "differ_specials": int(np.sum(a[p:] != b[p:])), "first": first}


def verdict_of(rec):
    return rec["compatibility"]["verdict"] if "compatibility" in rec else rec["status"]


def precise_rsqrt_fp32_boundaries():
    """Scalar FP32 rsqrt through the lowering layers on one device, scored by the frozen verdict rules. The case verdict
    is the shim_precise layer's (the translated Metal path). direct_opencl and glsl_inversesqrt are the uncorrected
    reductions and are reported only. clspv PR 1682 is not dispatched; its failure receipt stays under phase1/."""
    c = rsqrt_common()
    if isinstance(c, str):
        return {"status": c}
    result = {"inputs": c["n"], "primary": c["primary"], "inputs_sha256": c["inputs_sha256"], "native": c["native_info"], "layers": {}}
    bits = {}
    for name, stem, entry, strict in (("shim_precise", "a_shim/rsqrt_precise", "rsqrt_precise", False),
                                      ("direct_opencl", "b_direct/rsqrt_direct", "rsqrt_direct", True),
                                      ("glsl_inversesqrt", "c_glsl/rsqrt_inversesqrt", "rsqrt_inversesqrt", True)):
        result["layers"][name], b = rsqrt_layer(c, stem, entry, strict)
        if b is not None:
            bits[name] = b
    names = list(bits)
    result["layer_pairs"] = {f"{a}__{b}": rsqrt_diff(c, bits[a], bits[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    result["status"] = verdict_of(result["layers"]["shim_precise"])
    return result


def rsqrt_half_fixture(c):
    """Public half-propagation lanes: s is a primary corpus input whose frozen Irvine inverse inv lies in [2^-6, 2^6];
    x is a positive normal half for which half(x * inv) != half(x * nextup(inv)), chosen by xorshift32; w is a half in
    [0.5, 2). Expected n = half(x * inv) and y = half(w * n) are host IEEE float32 and round-to-nearest-even half
    arithmetic on the Irvine inverse. Lanes whose n or y is not a finite normal half are skipped. No model data."""
    import hashlib
    import m2v_rsqrt_corpus as corpus
    halves = np.arange(0x0400, 0x7C00, dtype=np.uint16).view(np.float16).astype(np.float32)
    state, s, xw, exp = RSQRT_HALF_SEED, [], [], []
    for i in range(c["primary"]):
        inv = np.uint32(c["native"][i]).view(np.float32)
        if not 2.0 ** -6 <= inv <= 2.0 ** 6:
            continue
        with np.errstate(over="ignore"):  # products past 65504 become inf halves and are never chosen
            n0, n1 = (halves * inv).astype(np.float16), (halves * np.nextafter(inv, np.float32(np.inf))).astype(np.float16)
        ok = np.nonzero((n0 != n1) & (n0 >= 2.0 ** -14) & np.isfinite(n0))[0]
        if not len(ok):
            continue
        state = corpus.xorshift32(state)
        x = np.float16(halves[ok[state % len(ok)]])
        state = corpus.xorshift32(state)
        wt = np.uint16(0x3800 + state % 0x800).view(np.float16)
        n = np.float16(np.float32(x) * inv)
        y = np.float16(np.float32(wt) * np.float32(n))
        if not (np.isfinite(y) and y >= 2.0 ** -14):
            continue
        s.append(c["inputs"][i])
        xw.append(int(x.view(np.uint16)) | int(wt.view(np.uint16)) << 16)
        exp.append(int(n.view(np.uint16)) | int(y.view(np.uint16)) << 16)
    rows = len(s)
    pad = -(-rows // 64) * 64 - rows
    s, xw, exp = (np.array(a + [v] * pad, np.uint32) for a, v in ((s, 0x3F800000), (xw, 0x3C003C00), (exp, 0x3C003C00)))
    digest = hashlib.sha256(s.tobytes() + xw.tobytes() + exp.tobytes()).hexdigest()
    return rows, s, xw, exp, digest


def rsqrt_half(c):
    """The half-propagation fixture through the base-shim and corrected-shim modules."""
    rows, s, xw, exp, digest = rsqrt_half_fixture(c)
    res = {"rows": rows, "seed": RSQRT_HALF_SEED, "fixture_sha256": digest}
    if RSQRT_HALF_SHA256 and digest != RSQRT_HALF_SHA256:
        res["status"] = "incomplete: half fixture differs from its frozen digest"
        return res
    ins = [w("rsqrt_half_s.u32le.bin", s), w("rsqrt_half_xw.u32le.bin", xw)]
    for name, stem in (("old", "base/f_half/rsqrt_half_prop"), ("new", "f_half/rsqrt_half_prop")):
        rec, bits = rsqrt_dispatch(os.path.join(spvdir, stem + ".spv"), "rsqrt_half_prop", ins, 1, rows)
        rec = {k: rec[k] for k in ("spv", "status", "spv_sha256", "per_dispatch_us", "wall_us") if k in rec}
        if bits is not None:
            got = bits[0]
            bad = np.nonzero(got != exp[:rows])[0]
            rec.update(mismatches=int(len(bad)), pre_weight_mismatches=int(np.sum((got ^ exp[:rows]) & 0xFFFF != 0)),
                       first=None if not len(bad) else {"s_bits": f"{s[bad[0]]:08x}", "xw": f"{xw[bad[0]]:08x}",
                                                        "got": f"{got[bad[0]]:08x}", "expected": f"{exp[bad[0]]:08x}"})
        res[name] = rec
    return res


def precise_rsqrt_fp32_old_fail_new_pass():
    """The frozen Phase 0 module (base shim) must still fail with finite mismatches, the corrected shim must pass the
    same frozen contract, and the uncorrected reductions must still fail. FP32 bits come first; only then the
    half-propagation fixture, where the old module must differ and the corrected one must match."""
    c = rsqrt_common()
    if isinstance(c, str):
        return {"status": c}
    old, _ = rsqrt_layer(c, "base/a_shim/rsqrt_precise", "rsqrt_precise", False)
    new, _ = rsqrt_layer(c, "a_shim/rsqrt_precise", "rsqrt_precise", False)
    red = {n: rsqrt_layer(c, st, e, True)[0] for n, st, e in (("direct_opencl", "b_direct/rsqrt_direct", "rsqrt_direct"),
                                                             ("glsl_inversesqrt", "c_glsl/rsqrt_inversesqrt", "rsqrt_inversesqrt"))}
    layers = {"old_shim": old, "new_shim": new, **red}
    checks = {"old_module_is_phase0": old.get("spv_sha256") == RSQRT_OLD_SHIM_SHA256,
              "old_fails": verdict_of(old) == "fail" and old["compatibility"]["finite_mismatches"] > 0,
              "new_passes": verdict_of(new) == "pass",
              "reductions_still_fail": all(verdict_of(r) == "fail" for r in red.values())}
    result = {"layers": {k: {f: v[f] for f in ("spv_sha256", "status", "compatibility", "distance_from_exact_nearest_even",
                                               "per_dispatch_us", "wall_us") if f in v}
                         for k, v in layers.items()}, "checks": checks}
    missing = [k for k, v in layers.items() if "compatibility" not in v]
    if missing:
        result["status"] = "incomplete: " + ", ".join(f"{k} {layers[k]['status']}" for k in missing)
        return result
    if not all(checks.values()):
        result["half"], result["status"] = "not run: FP32 checks failed first", "fail"
        return result
    half = result["half"] = rsqrt_half(c)
    if "mismatches" not in half.get("old", {}) or "mismatches" not in half.get("new", {}):
        result["status"] = "incomplete: half fixture " + half.get("status", "did not run on both modules")
        return result
    checks["half_old_differs"] = half["old"]["mismatches"] > 0
    checks["half_new_exact"] = half["new"]["mismatches"] == 0
    result["status"] = "pass" if all(checks.values()) else "fail"
    return result


def precise_rsqrt_fp32_special_values():
    """Required special-value predicates on the corrected shim, and the structural reason they hold: the stored value
    is an integer OpSelect on the input bits, so no special result comes from an InverseSqrt whose specials are undefined
    without float controls. Negative subnormals are characterized only."""
    c = rsqrt_common()
    if isinstance(c, str):
        return {"status": c}
    rec, bits = rsqrt_layer(c, "a_shim/rsqrt_precise", "rsqrt_precise", False)
    if bits is None:
        return {"status": rec["status"]}
    comp, p = rec["compatibility"], c["primary"]
    lanes = {f"{c['inputs'][i]:08x}": f"{int(bits[i]):08x}" for i in range(p, c["n"])}
    required = sum(rsqrt_special_rule(b) is not None for b in c["inputs"][p:])
    checks = {"required_predicates_hold": not comp["special_failures"], "store_is_integer_select": rec["store_defined_by"] == ["OpSelect %uint"]}
    return {"required": required, "special_failures": comp["special_failures"], "characterized_only": comp["characterized_only"],
            "outputs": lanes, "store_defined_by": rec["store_defined_by"], "checks": checks,
            "status": "pass" if all(checks.values()) else "fail"}


def precise_rsqrt_fp32_rounding_boundaries():
    """Exact Irvine bits on the midpoint-neighbor inputs: for each of the 504 manifest rows, the nearest-even preimage of
    an output rounding midpoint and two neighbors each side. The base shim on the same lanes is reported only."""
    c = rsqrt_common()
    if isinstance(c, str):
        return {"status": c}
    where = {b: i for i, b in enumerate(c["inputs"][:c["primary"]])}
    want = sorted({r["nearest_even_input_bits"] + d for r in c["manifest"]["midpoints"] for d in range(-2, 3)})
    idx = np.array([where[b] for b in want if b in where], np.int64)
    res = {"rows": len(c["manifest"]["midpoints"]), "inputs": len(want), "found": len(idx)}
    for name, stem in (("new_shim", "a_shim/rsqrt_precise"), ("old_shim", "base/a_shim/rsqrt_precise")):
        rec, bits = rsqrt_layer(c, stem, "rsqrt_precise", False)
        if bits is None:
            res[name] = {"status": rec["status"]}
            continue
        bad = idx[bits[idx] != c["native"][idx]]
        d = bits[idx].astype(np.int64) - c["refs"][idx]
        res[name] = {"mismatches": int(len(bad)), "first": None if not len(bad) else f"{c['inputs'][bad[0]]:08x}",
                     "distance_from_exact_nearest_even": {str(v): int(k) for v, k in zip(*np.unique(d, return_counts=True))}}
    if "mismatches" not in res["new_shim"]:
        res["status"] = res["new_shim"]["status"]
    else:
        res["status"] = "pass" if len(idx) == len(want) and res["new_shim"]["mismatches"] == 0 else "fail"
    return res


def precise_rsqrt_scoped_math_unchanged():
    """Unqualified rsqrt (scalar and float4), fast::rsqrt, and fma (scalar and float4) must stay byte-identical
    between the base shim and the current shim under default clspv fma and under --use-native-builtins=fma.
    The case records whether that selection is observable. precise::rsqrt compiled as a_shim/rsqrt_precise.nativefma
    is a separate claim: missing the frozen Irvine bits is status unsupported, not a failure hidden inside this pass.
    The parent passes only when the selection probe passes and that status is exactly unsupported. A match against
    Irvine fails the parent, because the rejection disappeared and the contract changed."""
    c = rsqrt_common()
    if isinstance(c, str):
        return {"status": c}
    names = ("rsqrt", "rsqrt_float4", "fast_rsqrt", "fma", "fma_float4")
    res, ok, bits = {"selections": {}}, True, {}
    for sel, suffix in (("default", ""), ("native_fma", ".nativefma")):
        r = res["selections"][sel] = {}
        for side, stem in (("base", "base/e_scope/rsqrt_scope"), ("current", "e_scope/rsqrt_scope")):
            rec, b = rsqrt_dispatch(os.path.join(spvdir, stem + suffix + ".spv"), "rsqrt_scope", [c["in_path"]], 5, c["n"])
            r[side] = {k: rec[k] for k in ("spv_sha256", "status", "output_sha256") if k in rec}
            bits[(sel, side)] = b
        r["same_module"] = r["base"].get("spv_sha256") is not None and r["base"].get("spv_sha256") == r["current"].get("spv_sha256")
        if bits[(sel, "base")] is None or bits[(sel, "current")] is None:
            res["status"] = "incomplete: " + "; ".join(f"{sel} {s} {r[s]['status']}" for s in ("base", "current"))
            return res
        r["outputs_equal"] = {nm: bool(np.array_equal(x, y)) for nm, x, y in zip(names, bits[(sel, "base")], bits[(sel, "current")])}
        ok = ok and r["same_module"] and all(r["outputs_equal"].values())
    res["fma_selection_observable"] = {nm: int(np.sum(x != y)) for nm, x, y in
                                       zip(names, bits[("default", "current")], bits[("native_fma", "current")])}
    rec, _ = rsqrt_layer(c, "a_shim/rsqrt_precise.nativefma", "rsqrt_precise", False)
    claim = {k: rec[k] for k in ("spv_sha256", "status", "compatibility") if k in rec}
    comp = rec.get("compatibility")
    if comp is None:
        claim["status"] = rec.get("status", "incomplete")
    elif comp["verdict"] == "pass":
        claim["status"] = "pass"
    else:
        claim["status"] = "unsupported"  # misses Irvine: unsupported contract, not a parent pass over a nested fail
    res["precise_under_native_fma"] = claim
    if comp is None:
        res["status"] = claim["status"]
    elif ok and claim["status"] == "unsupported":
        res["status"] = "pass"
    else:
        res["status"] = "fail"
    return res


def silu(x):
    return x / (1.0 + np.exp(-x))


def _run_cases():
    bad = [c for c in cases if c not in ALL]
    if bad:
        print(f"m2v-test.py: unknown case {bad[0]!r}; choose from: {' '.join(ALL)}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(out, exist_ok=True)
    T0 = time.time()
    for case in cases:
        try:
            if case == "activation":
                n = 1 << 23  # 32768 workgroups of 256, under the 65535 dispatch limit
                for dt, npdt, tol in (("f32", np.float32, 2e-5), ("f16", np.float16, 2e-3)):
                    x = rng.standard_normal(n).astype(npdt)
                    y = np.zeros(n, npdt)
                    bufs = [w(f"act_in_{dt}.bin", x), w(f"act_out_{dt}.bin", y), w("act_n.bin", np.array([n], np.uint32)),
                            w("act_type.bin", np.array([0], np.uint32))]  # ActivationType::SILU = 0
                    push = struct.pack("<4I", 0, 0, 0, 0) + struct.pack("<I", 0)  # clspv push block: 16 bytes group/region offsets, then POD in_place = false
                    t, info = timed(os.path.join(spvdir, "activation.spv"), f"activation_{dt}", ((n + 255) // 256, 1, 1), bufs, push)
                    # dump pass (one dispatch) for correctness
                    run(os.path.join(spvdir, "activation.spv"), f"activation_{dt}", ((n + 255) // 256, 1, 1), bufs, push, 1,
                        dump=f"1:{os.path.join(out, f'act_res_{dt}.bin')}")
                    got = np.fromfile(os.path.join(out, f"act_res_{dt}.bin"), npdt)
                    ref = silu(x.astype(np.float64))
                    err = float(np.max(np.abs(got.astype(np.float64) - ref) / (1.0 + np.abs(ref))))
                    res[f"activation_{dt}"] = {"n": n, "max_rel_err": err, "tol": tol, "ok": bool(err < tol), "us": t, "info": info,
                                              "GBps": (None if t is None else 2 * n * np.dtype(npdt).itemsize / (t * 1e3))}
            elif case.startswith("precise_rsqrt_") and case in RSQRT_CASES:
                res[case] = globals()[case]()
            elif case == "softmax":
                rows, cols = 4096, 4096
                for dt, npdt, tol in (("f32", np.float32, 1e-5), ("f16", np.float16, 2e-3)):
                    x = (rng.standard_normal((rows, cols)) * 3).astype(npdt)
                    bufs = [w(f"sm_vals_{dt}.bin", x.copy()), w(f"sm_sinks_{dt}.bin", np.zeros(1, npdt)),
                            w("sm_meta.bin", np.array([cols, rows, 1], np.uint32))]
                    push = struct.pack("<4I", 0, 0, 0, 0) + struct.pack("<I", 0)  # offsets + has_sinks = false
                    t, info = timed(os.path.join(spvdir, "softmax.spv"), f"softmax_{dt}", (rows, 1, 1), bufs, push)
                    # the timed run softmaxes in place repeatedly; correctness needs a fresh single dispatch
                    w(f"sm_vals_{dt}.bin", x.copy())
                    run(os.path.join(spvdir, "softmax.spv"), f"softmax_{dt}", (rows, 1, 1), bufs, push, 1,
                        dump=f"0:{os.path.join(out, f'sm_res_{dt}.bin')}")
                    got = np.fromfile(os.path.join(out, f"sm_res_{dt}.bin"), npdt).reshape(rows, cols).astype(np.float64)
                    xr = x.astype(np.float64)
                    ref = np.exp(xr - xr.max(1, keepdims=True))
                    ref /= ref.sum(1, keepdims=True)
                    err = float(np.max(np.abs(got - ref)))
                    res[f"softmax_{dt}"] = {"rows": rows, "cols": cols, "max_abs_err": err, "tol": tol, "ok": bool(err < tol), "us": t,
                                           "info": info, "GBps": (None if t is None else 2 * rows * cols * np.dtype(npdt).itemsize / (t * 1e3))}
            elif case == "gemm":
                # uzu Gemm (cases/gemm.cl), f32, transposed B (B is [N, K]); entry from M2V_GEMM_ENTRY (default the 64x64x32 tiling), sizes from M2V_SIZES
                entry = os.environ.get("M2V_GEMM_ENTRY", "gemm_f32_t64x64x32")
                bm, bn, bk = [int(x) for x in entry.split("_t")[1].split("x")]
                for M in [int(x) for x in os.environ.get("M2V_SIZES", "256,1024").split(",")]:
                    N = K = M
                    A = rng.standard_normal((M, K)).astype(np.float32)
                    B = rng.standard_normal((N, K)).astype(np.float32)
                    tpc, tpr = (M + bm - 1) // bm, (N + bn - 1) // bn
                    params = struct.pack("<13I?3xf", M, N, K, K, K, 0, 0, 0, 0, N, tpc, tpr, K // bk, False, 1.0)
                    dummy = np.zeros(4, np.uint32)
                    refl = json.load(open(os.path.join(spvdir, "gemm.json")))
                    kern = next(k for k in refl["kernels"] if k["name"] == entry)
                    # argument ordinals of Gemm: 0 a, 1 b, 2 d, 3 scales, 4 biases, 5 zero_points, 6 output_bias, 7 rht_factors, 8 a_int8,
                    # 9 a_scales, 10 a_group_sums, 11 params, 12 counts, 13 transform_bits, 14 alignment_bits
                    files = {0: w("gm_a.bin", A), 1: w("gm_b.bin", B), 2: w("gm_c.bin", np.zeros((M, N), np.float32)),
                             11: w("gm_params.bin", np.frombuffer(params.ljust(64, b"\0"), np.uint8)),
                             12: w("gm_counts.bin", np.array([tpr, tpc, 1], np.uint32))}
                    binds = sorted((a for a in kern["args"] if a["kind"] in ("storage_buffer", "uniform_buffer")), key=lambda a: a["binding"])
                    bufs = [files.get(a["ordinal"]) or w(f"gm_dummy{a['ordinal']}.bin", dummy) for a in binds]
                    c_index = next(i for i, a in enumerate(binds) if a["ordinal"] == 2)
                    pods = {13: 0, 14: 0}  # transform_bits, alignment_bits
                    regions = refl.get("push_constant_regions", {})
                    size = max([r["offset"] + r["size"] for r in regions.values()] + [a["offset"] + a["size"] for a in kern["args"] if a["kind"] == "pod_push_constant"])
                    push_buf = bytearray(size)
                    if "PushConstantNumWorkgroups" in regions:
                        o = regions["PushConstantNumWorkgroups"]["offset"]
                        push_buf[o:o + 12] = struct.pack("<3I", tpr, tpc, 1)
                    for a in kern["args"]:
                        if a["kind"] == "pod_push_constant":
                            push_buf[a["offset"]:a["offset"] + 4] = struct.pack("<I", pods[a["ordinal"]])
                    push = bytes(push_buf)
                    grid = (tpr, tpc, 1)
                    t, info = timed(os.path.join(spvdir, "gemm.spv"), entry, grid, bufs, push)
                    run(os.path.join(spvdir, "gemm.spv"), entry, grid, bufs, push, 1, dump=f"{c_index}:{os.path.join(out, 'gm_res.bin')}")
                    got = np.fromfile(os.path.join(out, "gm_res.bin"), np.float32).reshape(M, N).astype(np.float64)
                    ref = A.astype(np.float64) @ B.astype(np.float64).T
                    err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
                    res[f"{entry}_{M}"] = {"M": M, "max_rel_err": err, "tol": 1e-5, "ok": bool(err < 1e-5), "us": t, "info": info,
                                           "GFLOPs": (None if t is None else 2.0 * M * N * K / (t * 1e3))}
            elif case == "tilematmul_rt":
                # register-tiled variants: one subgroup owns an MR x NR block of 8x8 tiles (cases/tilematmul_rt.cl)
                for name, mr, nr in (("tile_matmul_rt2x2_f32", 2, 2), ("tile_matmul_rt4x2_f32", 4, 2), ("tile_matmul_rt4x4_f32", 4, 4)):
                    for M in [int(x) for x in os.environ.get("M2V_SIZES", "256,1024").split(",")]:
                        N = K = M
                        A = rng.standard_normal((M, K)).astype(np.float32)
                        B = rng.standard_normal((K, N)).astype(np.float32)
                        bufs = [w("mm_a.bin", A), w("mm_b.bin", B), w("mm_c.bin", np.zeros((M, N), np.float32)),
                                w("mm_dims.bin", np.array([M, N, K], np.uint32))]
                        grid = (N // (8 * nr), M // (8 * mr), 1)
                        push0 = struct.pack("<4I", 0, 0, 0, 0)
                        t, info = timed(os.path.join(spvdir, "tilematmul_rt.spv"), name, grid, bufs, push0)
                        run(os.path.join(spvdir, "tilematmul_rt.spv"), name, grid, bufs, push0, 1, dump=f"2:{os.path.join(out, 'mm_res.bin')}")
                        got = np.fromfile(os.path.join(out, "mm_res.bin"), np.float32).reshape(M, N).astype(np.float64)
                        ref = A.astype(np.float64) @ B.astype(np.float64)
                        err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
                        res[f"{name}_{M}"] = {"M": M, "max_rel_err": err, "tol": 1e-5, "ok": bool(err < 1e-5), "us": t, "info": info,
                                              "GFLOPs": (None if t is None else 2.0 * M * N * K / (t * 1e3))}
            elif case == "tilematmul_h":
                # half A and B, f32 accumulate and output (cases/tilematmul_h.cl); the reference uses the half-rounded inputs
                for name, mr, nr in (("tile_matmul_h2x2", 2, 2), ("tile_matmul_h4x4", 4, 4)):
                    for M in [int(x) for x in os.environ.get("M2V_SIZES", "256,1024").split(",")]:
                        N = K = M
                        A = rng.standard_normal((M, K)).astype(np.float16)
                        B = rng.standard_normal((K, N)).astype(np.float16)
                        bufs = [w("mm_a.bin", A), w("mm_b.bin", B), w("mm_c.bin", np.zeros((M, N), np.float32)),
                                w("mm_dims.bin", np.array([M, N, K], np.uint32))]
                        grid = (N // (8 * nr), M // (8 * mr), 1)
                        push0 = struct.pack("<4I", 0, 0, 0, 0)
                        t, info = timed(os.path.join(spvdir, "tilematmul_h.spv"), name, grid, bufs, push0)
                        run(os.path.join(spvdir, "tilematmul_h.spv"), name, grid, bufs, push0, 1, dump=f"2:{os.path.join(out, 'mm_res.bin')}")
                        got = np.fromfile(os.path.join(out, "mm_res.bin"), np.float32).reshape(M, N).astype(np.float64)
                        ref = A.astype(np.float64) @ B.astype(np.float64)
                        err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
                        res[f"{name}_{M}"] = {"M": M, "max_rel_err": err, "tol": 1e-5, "ok": bool(err < 1e-5), "us": t, "info": info,
                                              "GFLOPs": (None if t is None else 2.0 * M * N * K / (t * 1e3))}
            elif case == "tilematmul":
                for M in (256, 1024):
                    N = K = M
                    A = rng.standard_normal((M, K)).astype(np.float32)
                    B = rng.standard_normal((K, N)).astype(np.float32)
                    bufs = [w("mm_a.bin", A), w("mm_b.bin", B), w("mm_c.bin", np.zeros((M, N), np.float32)),
                            w("mm_dims.bin", np.array([M, N, K], np.uint32))]
                    grid = (N // 8, M // 8, 1)
                    push0 = struct.pack("<4I", 0, 0, 0, 0)
                    t, info = timed(os.path.join(spvdir, "tilematmul.spv"), "tile_matmul_f32", grid, bufs, push0)
                    run(os.path.join(spvdir, "tilematmul.spv"), "tile_matmul_f32", grid, bufs, push0, 1, dump=f"2:{os.path.join(out, 'mm_res.bin')}")
                    got = np.fromfile(os.path.join(out, "mm_res.bin"), np.float32).reshape(M, N).astype(np.float64)
                    ref = A.astype(np.float64) @ B.astype(np.float64)
                    err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
                    res[f"tilematmul_{M}"] = {"M": M, "max_rel_err": err, "tol": 1e-5, "ok": bool(err < 1e-5), "us": t, "info": info,
                                             "GFLOPs": (None if t is None else 2.0 * M * N * K / (t * 1e3))}
        except Exception as e:  # keep going so one failure does not hide the others
            res[case] = {"error": repr(e)}

    for k, v in res.items():
        v["ok"] = (v.get("status") == "pass" if k in RSQRT_CASES else bool(v.get("ok"))) and "error" not in v
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    KEY = ("max_rel_err", "max_abs_err", "sum_rel_err", "GFLOPs", "GBps", "us")
    for k, v in res.items():
        num = next((f"{n}={v[n]:.3g}" for n in KEY if v.get(n) is not None), "")
        print(f"{'PASS' if v['ok'] else 'FAIL'} {k} {num}".rstrip())
        if not v["ok"]:
            print("  " + (v["error"] if "error" in v else f"expected {v['expected']} got {v['got']}" if "got" in v else "result differs from the reference")
                  + f"  (details: {os.path.join(out, 'results.json')})")
    failed = sum(not v["ok"] for v in res.values())
    print(f"{len(res) - failed} passed, {failed} failed in {time.time() - T0:.1f} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run_cases()
