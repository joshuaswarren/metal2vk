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

Open freedoms. Fixed by the MSL: the block walk, the mask, the block max,
the bf16 P store, the per-block rescale, the tile order. Primary open
freedom: exp() precision - Metal's exp is not correctly rounded and differs
from numpy's by about 1 f32 ulp; the resulting bf16 flips of the P tile
propagate into o_part as sparse 1e-3-scale errors (perturbing exp by 1 ulp
on half the entries reproduces the GPU error scale; pure f32 dot
reassociation would move every slot at ~1e-6, which the dump does not
show). Secondary: the 8-term simdgroup matmul dot order and the 4-slab sum
order.

Judgment (exp-flip freedom, scale-relative): o_part passes when
|got - want| <= 5e-4 * max|want| over the same 128-wide output row
(about 2x the observed max err / row max of 2.2e-4; a single bf16 flip of
the row scale is far below the criterion's 3.9e-3 bf16-ulp bound).
ml_part stays at the 1e-5 table tolerance. Negative controls in
self_check(): skipping one 32-key block, or shifting the causal mask by
one row, must both fail the criterion.

Against the g13g-wp dump: ml_part matches at 2.7e-7 max rel; o_part max
abs 3.65e-3, max err / row max 2.2e-4 - inside the judgment, sparse slots
only (0.38%), consistent with the exp-flip freedom.
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


def ref_omlx_verify_attn_wide_partial(kern, inputs, skip_block_at=None,
                                      mask_row_shift=0):
    """skip_block_at / mask_row_shift exist only for the negative controls
    in self_check: drop one 32-key block from the walk, or shift the causal
    bound by one row. The registered reference uses the defaults."""
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
                if skip_block_at is not None and t0 == skip_block_at:
                    continue
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
                      & (keys[None, :] <= big_t - left + np.arange(rows)[:, None]
                         + mask_row_shift))
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


def judge_o_part(got, want):
    """The exp-flip judgment: per 128-wide output row, pass when every slot
    is within 5e-4 of that row's max |want|."""
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    scale = np.abs(want.reshape(-1, 128)).max(axis=1)
    err = np.abs(got.reshape(-1, 128) - want.reshape(-1, 128)).max(axis=1)
    return bool((err <= 5e-4 * scale).all())


def self_check(dump_dir, mlx_dir):
    """Negative controls: the judgment must FAIL for a one-32-key-block skip
    and for a one-row causal shift, and PASS for the faithful reference."""
    import pathlib
    import parity_common as pc

    name = "omlx_verify_attn_wide_partial"
    kern = pc.BY_NAME[name]
    inputs = {s[0]: np.load(pathlib.Path(mlx_dir) / f"{s[0]}.npy")
              for s in kern["inputs"]}
    dump = pathlib.Path(dump_dir)
    got_o = np.load(dump / "out_o_part.npy").reshape(-1)
    got_ml = np.load(dump / "out_ml_part.npy").reshape(-1)
    ref = ref_omlx_verify_attn_wide_partial(kern, inputs)
    ref_skip = ref_omlx_verify_attn_wide_partial(kern, inputs,
                                                 skip_block_at=t_begin_of(0, inputs))
    ref_shift = ref_omlx_verify_attn_wide_partial(kern, inputs,
                                                  mask_row_shift=1)
    pass_real = judge_o_part(got_o, ref["o_part"])
    fail_skip = not judge_o_part(got_o, ref_skip["o_part"])
    fail_shift = not judge_o_part(got_o, ref_shift["o_part"])
    ml_ok = pc.compare("f32", got_ml.astype(np.float32),
                       ref["ml_part"].astype(np.float32))["status"] == "match"
    return {"pass_real": pass_real and ml_ok,
            "controls_fail": fail_skip and fail_shift,
            "ml_match": ml_ok}


def t_begin_of(split, inputs):
    p = inputs["params"].reshape(-1).view(np.uint32)
    return int(p[2]) * split
