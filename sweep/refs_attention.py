#!/usr/bin/env python3
"""numpy reference for omlx_verify_attn_wide_partial.

Simulates the assembled MSL op for op: per (query head, split) an online
softmax walks 32-key blocks in order; each block computes the full 256-dim
score (the four simdgroup slabs summed), masks with the verify causal rule
r < L and key >= base and key < t_end and key <= T-L+r, scales, takes the
block max, stores exp as bf16 (the threadgroup P tile), and rescales the
accumulator by exp(m_old - m_new). The one stated freedom: the 8-term
simdgroup matmul dots and the 4-slab slab sum are float32 with
hardware-defined order; the xor butterflies are simulated staged exactly.

Origin: the assembled artifact MSL
sibling-scan/msl/omlx_verify_attn_wide_partial.metal; the composed
chain-attention partial in qwen35_verify_sdpa_split.py is the semantic
ground truth for what the kernel must compute.

Open freedoms (fixed by the MSL: the block walk, the mask, the block max,
the bf16 P store, the per-block rescale, the tile order; left open by the
MSL: the 8-term simdgroup matmul dot order and the 4-slab sum order).
Against the g13g-wp dump: ml_part matches at 2.7e-7 max rel (the l sum has
no cancellation); o_part differs by at most 0.19% relative on 0.38% of
slots - cancellation-amplified reassociation of ~2048-term dots, which a
1e-5 f32 tolerance cannot absorb. Proposed for this kernel class: f32
attention-partial outputs judged at 2e-3 relative, named here rather than
widened silently in the table.
"""
import numpy as np

from refs_common import bf, BF, f32sum

REFS = {}

FLT_MAX = np.float32(np.finfo(np.float32).max)


def _xor_butterfly_sum(vals):
    """simd_shuffle_xor staged sum over 16 lanes, offs 1/2/4/8."""
    v = np.asarray(vals, np.float32).copy()
    for off in (1, 2, 4, 8):
        prev = v.copy()
        idx = np.arange(len(v))
        v[idx] = np.float32(prev[idx] + prev[idx ^ off])
    return v


def ref_omlx_verify_attn_wide_partial(kern, inputs):
    grid = kern["grid"]
    n_splits = grid[1]
    p = inputs["params"].reshape(-1).view(np.uint32)
    big_t = int(p[0])
    left = int(p[1])
    chunk = int(p[2])
    scale = np.float32(np.asarray([p[4]], np.uint32).view(np.float32)[0])
    q = BF(inputs["q"].reshape(-1))
    k = BF(inputs["k"].reshape(-1))
    v = BF(inputs["v"].reshape(-1))
    kst = [int(x) for x in inputs["k_strides"].reshape(-1)]
    vst = [int(x) for x in inputs["v_strides"].reshape(-1)]
    heads = grid[0] // 128          # one 128-thread workgroup per query head
    g = 8
    dim = 256
    rows = 8
    o_part = np.zeros((n_splits, heads, rows, dim), np.float32)
    ml_part = np.zeros((n_splits, heads, rows, 2), np.float32)
    kv_heads = k.size // (kst[1] * 1) if False else None
    n_kv = 2
    for split in range(n_splits):
        t_begin = split * chunk
        t_end = min(t_begin + chunk, big_t)
        for qh in range(heads):
            h = qh // g
            qm = q[qh * rows * dim:(qh + 1) * rows * dim].reshape(rows, dim)
            kb = k[h * kst[1]:(h + 1) * kst[1]].reshape(-1, dim)
            vb = v[h * vst[1]:(h + 1) * vst[1]].reshape(-1, dim)
            m = np.full(rows, -np.inf, np.float32)
            acc_l = np.zeros(rows, np.float32)
            o = np.zeros((rows, dim), np.float32)
            for t0 in range(t_begin, t_end, 32):
                # four 8-key tiles; the tile start clamps to T-8 and the
                # mask kills the duplicated keys
                cols = np.arange(32)
                tc = cols // 8
                base = t0 + tc * 8
                keys = np.minimum(base, big_t - 8) + (cols % 8)
                # full-256 scores: the 4 slab partials summed (fp freedom)
                val = (qm @ kb[keys].T).astype(np.float32)
                ok = ((np.arange(rows)[:, None] < left)
                      & (keys[None, :] >= base[None, :])
                      & (keys[None, :] < t_end)
                      & (keys[None, :] <= big_t - left + np.arange(rows)[:, None]))
                sv = np.where(ok, (val * scale).astype(np.float32),
                              np.float32(-np.inf))
                rmax = sv.max(axis=1)
                m_new = np.maximum(m, rmax)
                with np.errstate(invalid="ignore"):
                    p = np.where(m_new[:, None] == np.float32(-np.inf),
                                 np.float32(0.0),
                                 np.exp(sv - m_new[:, None])).astype(np.float32)
                p = np.where(np.isnan(p), np.float32(0.0), p)
                p_b = bf(p)                       # the threadgroup bf16 P tile
                # per-row rsum: each of 16 threads sums its 2 columns, then
                # the staged xor butterfly over 16 lanes
                rsum = np.zeros(rows, np.float32)
                for r in range(rows):
                    thr = np.add.reduceat(p[r], np.arange(0, 32, 2)).astype(np.float32)
                    rsum[r] = _xor_butterfly_sum(thr)[0]
                with np.errstate(invalid="ignore"):
                    alpha = np.where(m_new == np.float32(-np.inf),
                                     np.float32(1.0),
                                     np.exp(m - m_new)).astype(np.float32)
                acc_l = (acc_l * alpha + rsum).astype(np.float32)
                m = m_new
                o = (o * alpha[:, None]).astype(np.float32)
                for kt in range(4):
                    ks = keys[kt * 8:(kt + 1) * 8]
                    pv = BF(p_b[:, kt * 8:(kt + 1) * 8])          # (8 rows, 8 keys)
                    o += (pv @ vb[ks]).astype(np.float32)
            o_part[split, qh] = o
            ml_part[split, qh, :, 0] = m
            ml_part[split, qh, :, 1] = acc_l
    return {"o_part": o_part.reshape(-1), "ml_part": ml_part.reshape(-1)}


REFS["omlx_verify_attn_wide_partial"] = ref_omlx_verify_attn_wide_partial
