#!/usr/bin/env python3
"""numpy references for the sibling kernels the omarchy path refuses.

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

from refs_common import (bf, BF, h, H, f32sum, sigmoid_f32, topk_keys,
                         select_topk, bf16_bits_from_key)  # noqa: F401


# --- the four simple kernels (corrected semantics) --------------------------
def ref_router_topk(kern, inputs):
    """Value-descending selection; exact ties resolve to the HIGHER index
    (the kernel scans with >= and reduces with simd_max over indices).
    Renormalization in f32 with one bf16 rounding at the store."""
    ne = int(dict((n, v) for n, v in kern["tmpl"])["NE"])
    k = int(dict((n, v) for n, v in kern["tmpl"])["K"])
    probs = BF(inputs["probs"].reshape(-1))
    order = np.lexsort((-np.arange(ne), -probs.astype(np.float64)))[:k]
    sel_p = probs[order]
    total = f32sum(sel_p)
    inv = np.float32(1.0) / total
    scores = bf(sel_p * inv)
    return {"indices": order.astype(np.uint32).reshape(1, k),
            "scores": scores.reshape(1, k)}


def ref_softmax_topk_row(kern, inputs):
    return _softmax_topk(kern, inputs)


def ref_softmax_topk_rows(kern, inputs):
    return _softmax_topk(kern, inputs)


def _softmax_topk(kern, inputs):
    t = dict((n, v) for n, v in kern["tmpl"])
    ne, k = int(t["NE"]), int(t["K"])
    logits = BF(inputs["logits"].reshape(-1))
    rows = logits.size // ne
    indices = np.zeros((rows, k), np.uint32)
    scores = np.zeros((rows, k), np.uint16)
    for r in range(rows):
        x = logits[r * ne:(r + 1) * ne]
        e = np.exp(x - np.float32(x.max())).astype(np.float32)
        p = (e * np.float32(np.float32(1.0) / f32sum(e))).astype(np.float32)
        pb = bf(p)
        idx, sel_p = select_topk(pb, ne, k)
        total = f32sum(sel_p)
        inv = np.float32(1.0) / total
        indices[r] = idx
        scores[r] = bf(sel_p * inv)
    return {"indices": indices, "scores": scores}


def ref_combine_row(kern, inputs):
    """omlx_qwen35_moe_combine_row: every op rounds to bf16 (MSL is explicit)."""
    t = dict((n, v) for n, v in kern["tmpl"])
    k, hv = int(t["K"]), int(t["H"])
    routed = BF(inputs["routed"].reshape(-1)).reshape(k, hv)
    scores = BF(inputs["scores"].reshape(-1))
    shared = BF(inputs["shared"].reshape(-1))
    gate = BF(inputs["gate"].reshape(-1))
    lane = np.zeros((k, hv), np.uint16)
    for j in range(k):
        p = bf(routed[j] * scores[j])
        lane[j] = bf(BF(p) + BF(lane[j]))
    acc = lane[0]
    for l in range(1, k):
        acc = bf(BF(lane[l]) + BF(acc))
    g = gate[0]
    e = bf(np.float32(1.0) + BF(bf(np.exp(np.float32(abs(g))))))[0]
    y = bf(np.float32(1.0) / BF(np.asarray([e], np.uint16))[0])[0]
    s = y if g < 0 else bf(np.float32(1.0) - BF(np.asarray([y], np.uint16))[0])[0]
    sh = bf(BF(s) * shared)
    out = bf(BF(acc) + BF(sh))
    return {"out": out.reshape(hv)}


def ref_silu(kern, inputs):
    g = inputs["gates"].reshape(-1).astype(np.float32)
    sig = sigmoid_f32(g)
    return {"out": (g * sig).astype(np.float32).reshape(g.shape)}


REFS = {}
for _mod in ("refs_router", "refs_moe_expert", "refs_gdn_prework",
              "refs_gdn_verify", "refs_moe_attn"):
    try:
        REFS.update(__import__(_mod).REFS)
    except ImportError:
        pass
REFS.update({
    "omlx_qwen35_moe_router_topk": ref_router_topk,
    "omlx_qwen35_moe_router_softmax_topk_row": ref_softmax_topk_row,
    "omlx_qwen35_moe_router_softmax_topk_rows": ref_softmax_topk_rows,
    "omlx_qwen35_moe_combine_row": ref_combine_row,
    "omlx_qwen35_moe_router_topk": ref_router_topk,
    "omlx_qwen35_moe_router_softmax_topk_row": ref_softmax_topk_row,
    "omlx_qwen35_moe_router_softmax_topk_rows": ref_softmax_topk_rows,
    "omlx_qwen35_moe_combine_row": ref_combine_row,
    "omlx_gdn_sigmoid_probe": ref_silu,
})
