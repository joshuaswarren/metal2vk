#!/usr/bin/env python3
"""Corpus for the m2v-test.py case precise_rsqrt_fp32_boundaries (docs/precise-rsqrt.md).

Everything here comes from bit patterns and exact rational arithmetic: no audio, transcripts, tokens, embeddings or
weights. Standard library only, so the corpus can be generated and checked on any host.

  python3 m2v_rsqrt_corpus.py write DIR   write inputs.u32le.bin, manifest.json and corpus.sha256 into DIR
  python3 m2v_rsqrt_corpus.py check DIR   regenerate and compare byte for byte with the frozen files in DIR
  python3 m2v_rsqrt_corpus.py padded DIR FILE   write the device input buffer (frozen inputs plus 1.0f padding to 64 words)

Freeze order: write, record the digest, and only then run a device or a candidate. The host oracle below
(exact_rsqrt_bits) is a characterization aid. Whether it is the final rounding judge is not decided here.
"""
import hashlib
import json
import math
import pathlib
import struct
import sys
from fractions import Fraction

SEED = 0x124
CAP = 16384
FIXED_MANTISSAS = (0, 1, 2, 0x3fffff, 0x400000, 0x400001, 0x7ffffd, 0x7ffffe, 0x7fffff)
RANDOM_PER_EXPONENT = 16
MIDPOINT_EXPONENTS = range(-63, 63)  # unbiased output exponents -63..62
MIDPOINT_MANTISSAS = (0, 0x200000, 0x400000, 0x600000)
NEIGHBORS = 2  # inputs on each side of the nearest-even input
GROUP = 64  # workgroup size of the device kernels; the device buffer is padded to a multiple of it
PAD_BITS = 0x3F800000  # 1.0f; padded lanes are never scored
TRUNCATION_RULE = ("not exercised while pre_cap_count <= cap; a binding cap would keep the first cap entries of the "
                   "canonical order (primary ascending by bit pattern, then specials in manifest order)")

# (label, bits). Labels are for the manifest only. No predicate is attached to any of them.
SPECIALS = (
    ("+0", 0x00000000), ("-0", 0x80000000),
    ("+inf", 0x7F800000), ("-inf", 0xFF800000),
    ("qnan+ canonical", 0x7FC00000), ("qnan- canonical", 0xFFC00000),
    ("qnan+ payload", 0x7FC00001), ("qnan- payload", 0xFFC12345),
    ("snan+ min payload", 0x7F800001), ("snan- min payload", 0xFF800001),
    ("snan+ max payload", 0x7FBFFFFF), ("snan- max payload", 0xFFBFFFFF),
    ("nan+ all ones", 0x7FFFFFFF), ("nan- all ones", 0xFFFFFFFF),
    ("-1", 0xBF800000), ("-4", 0xC0800000), ("-min normal", 0x80800000), ("-max finite", 0xFF7FFFFF),
    ("-min subnormal", 0x80000001), ("-max subnormal", 0x807FFFFF),
    ("+min subnormal", 0x00000001), ("+subnormal 2", 0x00000002), ("+subnormal mid", 0x00400000),
    ("+max subnormal -1", 0x007FFFFE), ("+max subnormal", 0x007FFFFF),
    ("+min normal", 0x00800000), ("+min normal +1", 0x00800001), ("+min normal +2", 0x00800002),
    ("+max finite -1", 0x7F7FFFFE), ("+max finite", 0x7F7FFFFF),
)


def xorshift32(state):
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state


def value(bits):
    """Exact value of a positive normal binary32 bit pattern."""
    exp, man = bits >> 23, bits & 0x7FFFFF
    assert 1 <= exp <= 254 and bits >> 31 == 0
    return Fraction((1 << 23) | man) * Fraction(2) ** (exp - 150)


def round_nearest_even(x):
    """Nearest-even binary32 bits of a positive rational x inside the normal range."""
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    scaled = x / Fraction(2) ** (e - 23)  # in [2^23, 2^24)
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n & 1):
        n += 1
    if n == 1 << 24:
        n >>= 1
        e += 1
    exp = e + 127
    assert 1 <= exp <= 254, "outside the normal range"
    return (exp << 23) | (n & 0x7FFFFF)


def exact_rsqrt_bits(bits):
    """Correctly rounded (nearest-even) binary32 bits of 1/sqrt(x) for a positive normal x, by exact comparison.

    A double-precision estimate only seeds the search. The decision uses x*mid^2 against 1 in rationals, which is never
    a tie because every binary32 midpoint has an odd numerator above 1."""
    x = value(bits)
    cand = struct.unpack("<I", struct.pack("<f", 1.0 / math.sqrt(float(x))))[0]
    while True:
        lo = (value(cand - 1) + value(cand)) / 2
        hi = (value(cand) + value(cand + 1)) / 2
        if x * lo * lo >= 1:  # 1/sqrt(x) <= lo: too high
            cand -= 1
        elif x * hi * hi <= 1:  # 1/sqrt(x) >= hi: too low
            cand += 1
        else:
            return cand


def fixed_and_random():
    state, out = SEED, []
    for exp in range(1, 255):
        mans = list(FIXED_MANTISSAS)
        for _ in range(RANDOM_PER_EXPONENT):
            state = xorshift32(state)
            mans.append(state & 0x7FFFFF)
        out += [(exp << 23) | m for m in mans]
    return out


def midpoint_neighbors():
    """504 output-rounding midpoints inverted exactly. Returns (rows, input_bits)."""
    rows, bits = [], []
    for e in MIDPOINT_EXPONENTS:
        for m in MIDPOINT_MANTISSAS:
            out_bits = ((e + 127) << 23) | m
            mid = (value(out_bits) + value(out_bits + 1)) / 2
            centre = round_nearest_even(1 / (mid * mid))
            around = [centre + d for d in range(-NEIGHBORS, NEIGHBORS + 1)]
            rows.append({"output_exponent": e, "output_mantissa": m, "output_bits": out_bits, "nearest_even_input_bits": centre})
            bits += around
    return rows, bits


def build():
    mid_rows, mid_bits = midpoint_neighbors()
    raw_primary = fixed_and_random() + mid_bits
    primary = sorted(set(raw_primary))
    seen = set(primary)
    specials = []
    for label, b in SPECIALS:
        specials.append({"label": label, "bits": b, "also_in_primary": b in seen})
    extra = []
    for s in specials:
        if s["bits"] not in seen:
            seen.add(s["bits"])
            extra.append(s["bits"])
    inputs = primary + extra
    pre_cap = len(inputs)
    assert pre_cap <= CAP, "cap binds: record the pre-cap count and the truncation rule before any device run"
    return inputs, {
        "case": "precise_rsqrt_fp32_boundaries",
        "seed": SEED, "rng": "xorshift32 shifts 13,17,5; one stream seeded once; 16 draws per exponent in ascending exponent order; mantissa = state & 0x7FFFFF",
        "fixed_mantissas": list(FIXED_MANTISSAS), "random_per_exponent": RANDOM_PER_EXPONENT,
        "midpoint_exponents": [MIDPOINT_EXPONENTS.start, MIDPOINT_EXPONENTS.stop - 1], "midpoint_mantissas": list(MIDPOINT_MANTISSAS),
        "neighbors_each_side": NEIGHBORS,
        "counts": {"fixed_and_random_generated": 254 * (len(FIXED_MANTISSAS) + RANDOM_PER_EXPONENT), "midpoint_rows": len(mid_rows),
                   "midpoint_inputs_generated": len(mid_bits), "primary_before_dedup": len(raw_primary), "primary": len(primary),
                   "specials_listed": len(specials), "specials_added_beyond_primary": len(extra),
                   "pre_cap": pre_cap, "final": len(inputs)},
        "cap": CAP, "cap_binds": pre_cap > CAP, "truncation_rule": TRUNCATION_RULE,
        "order": "primary ascending by bit pattern, then specials not already present, in SPECIALS order",
        "device_buffer": {"group": GROUP, "pad_bits": PAD_BITS, "padded_words": -(-len(inputs) // GROUP) * GROUP},
        "specials": specials, "midpoints": mid_rows,
        "predicates_declared": "none; special-value predicates and the acceptance verdict are frozen by a later row",
        "inputs_sha256": hashlib.sha256(pack(inputs)).hexdigest(),
        "generator_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    }


def pack(bits):
    return struct.pack("<%dI" % len(bits), *bits)


def padded(inputs):
    n = -(-len(inputs) // GROUP) * GROUP
    return list(inputs) + [PAD_BITS] * (n - len(inputs))


def frozen_files(inputs, manifest):
    mtext = json.dumps(manifest, indent=1, sort_keys=False) + "\n"
    return {
        "inputs.u32le.bin": pack(inputs),
        "manifest.json": mtext.encode(),
        "corpus.sha256": ("%s  inputs.u32le.bin\n%s  manifest.json\n" % (
            hashlib.sha256(pack(inputs)).hexdigest(), hashlib.sha256(mtext.encode()).hexdigest())).encode(),
    }


def load_inputs(d):
    raw = (pathlib.Path(d) / "inputs.u32le.bin").read_bytes()
    return list(struct.unpack("<%dI" % (len(raw) // 4), raw))


def check(d):
    """Regenerate and compare with the frozen files. Returns a list of mismatching file names."""
    inputs, manifest = build()
    bad = []
    for name, data in frozen_files(inputs, manifest).items():
        p = pathlib.Path(d) / name
        if not p.is_file() or p.read_bytes() != data:
            bad.append(name)
    return bad


if __name__ == "__main__":
    mode, d = sys.argv[1], pathlib.Path(sys.argv[2])
    if mode == "write":
        inputs, manifest = build()
        d.mkdir(parents=True, exist_ok=True)
        for name, data in frozen_files(inputs, manifest).items():
            (d / name).write_bytes(data)
        print(json.dumps(manifest["counts"]), manifest["inputs_sha256"])
    elif mode == "check":
        bad = check(d)
        print("frozen corpus matches" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    elif mode == "padded":
        pathlib.Path(sys.argv[3]).write_bytes(pack(padded(load_inputs(d))))
    else:
        sys.exit("usage: m2v_rsqrt_corpus.py write|check DIR | padded DIR FILE")
