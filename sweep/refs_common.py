#!/usr/bin/env python3
"""Shared arithmetic helpers for the per-kernel numpy references.

One function per kernel, named ``ref_<kernel>``, registered in REFS. Each
takes ``(kern, inputs)`` where ``inputs`` maps buffer name to the numpy array
exactly as saved by parity_mlx (bf16 as uint16 bit patterns, others their
numpy dtype) and returns ``{output name: numpy array of output bits}`` in the
same saved form.

The references simulate the arithmetic of the assembled .metal source op for
op: every explicit T(...) cast in the kernel is a rounding here (bfloat16_t
round-to-nearest-even via parity_common, half via numpy float16). Cross-lane
simd reductions are simulated as sequential float32 sums; that reassociation
is the only freedom, and it stays far inside the table tolerances.

Origin of each kernel's semantics: the assembled artifact MSL
(<lane>/msl/<name>.metal); the composed-op python next to the assemblers is
the semantic ground truth for what the kernel must compute.
"""
import numpy as np

import parity_common as pc

# --- bf16 helpers: bits in, bits out ---------------------------------------
def bf(x):
    """float array -> bf16 bit patterns (uint16), round-to-nearest-even."""
    return pc.bf16_round(np.atleast_1d(np.asarray(x, np.float32)))


def BF(bits):
    """bf16 bit patterns (uint16) -> float32 values."""
    return pc.bf16_to_f32(np.atleast_1d(np.asarray(bits, np.uint16)))


def h(x):
    """round to metal half (float16), value in/out."""
    return np.asarray(x, np.float32).astype(np.float16)


def H(x):
    return np.asarray(x, np.float16).astype(np.float32)


def f32sum(vals):
    """sequential float32 left-to-right sum (simd_sum approximation)."""
    s = np.float32(0.0)
    for v in vals:
        s = np.float32(s + np.float32(v))
    return s


def sigmoid_f32(g):
    """The kernels' form: y = 1/(1+exp(|g|)); g < 0 ? y : 1-y. For every g
    this is exactly 1/(1+exp(-g)) computed the way the MSL writes it."""
    g = np.asarray(g, np.float32)
    y = np.float32(1.0) / (np.float32(1.0) + np.exp(np.abs(g)))
    return np.where(g < 0, y, np.float32(1.0) - y)


def topk_keys(p_bf16_bits, ne):
    """Router top-k selection keys: ((bf16 bits + 1) << 16) | expert.

    The kernel selects the K largest keys, i.e. descending bf16 probability
    with ties broken toward the LARGER expert index.
    """
    bits = np.asarray(p_bf16_bits, np.uint32)
    return ((bits + np.uint32(1)) << np.uint32(16)) | np.arange(ne, dtype=np.uint32)


def select_topk(p_bf16_bits, ne, k):
    """-> (indices uint32 [k], sel_p float32 [k]) in selection order."""
    key = topk_keys(p_bf16_bits, ne)
    order = np.argsort(-key.astype(np.uint64), kind="stable")[:k]
    sel_p = BF(bf16_bits_from_key(key[order]))
    return order.astype(np.uint32), sel_p


def bf16_bits_from_key(key):
    return ((np.asarray(key, np.uint64) >> np.uint64(16)).astype(np.uint32)
            - np.uint32(1)).astype(np.uint16)


