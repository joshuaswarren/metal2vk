#!/usr/bin/env python3
"""numpy references for the three GDN prework kernels.

omlx_qwen35_gdn_prework_S1 (L2=0, the Qwen3.5 RMS arm) and
omlx_qwen4_gdn_decode_prework / omlx_qwen4_gdn_prefill_prework (the L2 arm):
depthwise conv over conv_state rows + qkv, SiLU with the one T(...) rounding
at the branch, per-head q/k normalization with every rounding site of the
assembled source (omlx071_qwen35_gdn_prework.py is the semantic ground
truth), the qwen4 decode gate/beta pair, and the next conv state.

The L2 arm's cross-lane reduction is the xor butterfly the source spells out,
so it is simulated exactly; the RMS arm's simd_sum is the sequential f32 sum
(refs_common's declared freedom). omlx_log1p is copied from the kernel's
header (the xp1 == max branch is unreachable for our inputs).
"""
import numpy as np

import parity_common as pc  # noqa: F401  (table access for check()/callers)

from refs_common import bf, BF

_LANES = np.arange(32)


def _silu(conv_bits):
    """act = conv * T((conv < 0) ? sy : 1 - sy); sy = 1/(1+exp(|conv|)) in
    f32, one bf16 rounding at the T(...) branch, one at the product."""
    c = BF(conv_bits)
    sy = np.float32(1.0) / (np.float32(1.0) + np.exp(np.abs(c)))
    return bf(c * np.where(c < 0, sy, np.float32(1.0) - sy))


def _l2_inv(act_bits):
    """-> inv bits (..., 32): per-lane l2acc with a T() rounding on every
    square and add, the xor-16/8/4/2/1 butterfly in f32, then
    inv = T(rsqrt(T(T(tv) + T(1e-6f))))."""
    a = BF(act_bits)
    sqv = bf(a * a)
    acc = BF(sqv[..., 0])
    for i in (1, 2, 3):
        acc = BF(bf(acc + BF(sqv[..., i])))
    tv = acc.astype(np.float32)
    for k in (16, 8, 4, 2, 1):
        tv = tv + tv[..., _LANES ^ k]
    eps = bf(tv + BF(bf(np.float32(1e-6))))
    return bf(np.float32(1.0) / np.sqrt(BF(eps)))


def _omlx_log1p(x):
    """The kernel header's omlx_log1p (f32)."""
    xp1 = np.float32(1.0) + x
    return np.where(xp1 == np.float32(1.0), x,
                    x * (np.log(xp1) / (xp1 - np.float32(1.0)))
                    ).astype(np.float32)


def _heads(kern):
    t = dict((n, v) for n, v in kern["tmpl"])
    hk, hv = int(t["HK"]), int(t["HV"])
    return t, hk, hv, 2 * hk + hv


def _head_span(lh, hk, dk, dv):
    """-> (is_q, is_k, head, channel_base, n_ch) for one logical head."""
    is_q, is_k = lh < hk, hk <= lh < 2 * hk
    head = lh if is_q else (lh - hk if is_k else lh - 2 * hk)
    base = (head * dk if is_q else
            (hk * dk + head * dk if is_k else 2 * hk * dk + head * dv))
    return is_q, is_k, head, base, dk if is_q or is_k else dv


def ref_omlx_qwen35_gdn_prework_S1(kern, inputs):
    return _row_prework(kern, inputs)


def ref_omlx_qwen4_gdn_prefill_prework(kern, inputs):
    return _row_prework(kern, inputs)


def _row_prework(kern, inputs):
    """The shared _SOURCE body (qwen35 S=1 decode at L2=0, qwen4 prefill at
    L2=1): per (row, logical head) conv + SiLU, then the L2 or RMS arm."""
    t, hk, hv, nheads = _heads(kern)
    dk, dv = int(t["DK"]), int(t["DV"])
    c, nkeep, l2 = int(t["C"]), int(t["NKEEP"]), int(t["L2"])
    s_rows = int(t["S"]) if "S" in t else int(np.asarray(inputs["s_len"]).reshape(-1)[0])
    qkv = BF(inputs["qkv"].reshape(s_rows, c))
    state_bits = np.asarray(inputs["conv_state"]).reshape(nkeep, c)
    state = BF(state_bits)
    w = BF(inputs["conv_w"].reshape(c, 4))
    q_scale = BF(inputs["q_scale"].reshape(-1))
    k_scale = BF(inputs["k_scale"].reshape(-1))  # L2 arm never reads it
    rows = np.arange(s_rows)
    taps = np.arange(4)
    q_out = np.zeros((s_rows, hk, dk), np.uint16)
    k_out = np.zeros_like(q_out)
    v_out = np.zeros((s_rows, hv, dv), np.uint16)
    for lh in range(nheads):
        is_q, is_k, head, base, nch = _head_span(lh, hk, dk, dv)
        chs = base + np.arange(nch)
        r = rows[:, None] + taps[None, :]
        from_state = r < nkeep
        xv = np.where(from_state[..., None],
                      state[np.clip(r, 0, nkeep - 1)[..., None], chs],
                      qkv[np.clip(r - nkeep, 0, s_rows - 1)[..., None], chs])
        acc = xv[:, 0] * w[chs, 0]
        for tap in (1, 2, 3):
            acc = acc + xv[:, tap] * w[chs, tap]
        act = _silu(bf(acc))  # conv = T(acc), then SiLU
        if is_q or is_k:
            act4 = act.reshape(s_rows, 32, 4)
            if l2:
                inv = _l2_inv(act4)
                l2v = bf(BF(act4) * BF(inv)[..., None])
                val = bf(BF(l2v) * q_scale) if is_q else l2v
                (q_out if is_q else k_out)[:, head, :] = val.reshape(s_rows, nch)
            else:
                a = BF(act4)
                total = (a * a).sum(-1)[:, 0]  # per-lane f32 sums, then simd_sum
                for lane in range(1, 32):
                    total = total + (a * a).sum(-1)[:, lane]
                inv = np.float32(1.0) / np.sqrt(
                    total / np.float32(dk) + np.float32(1e-6) / np.float32(dk))
                rms = bf(a * inv[:, None, None])  # T(1) * T(a * inv)
                val = bf(BF(rms) * (q_scale if is_q else k_scale))
                (q_out if is_q else k_out)[:, head, :] = val.reshape(s_rows, nch)
        else:
            v_out[:, head, :] = act.reshape(s_rows, nch)
    conv_out = state_bits.copy()
    if s_rows < nkeep:  # the row == 0 shift of the surviving old rows
        conv_out[:nkeep - s_rows] = state_bits[s_rows:]
    qkv_bits = np.asarray(inputs["qkv"]).reshape(s_rows, c)
    for row in rows:
        if row + nkeep >= s_rows:  # tail rows move into the next state
            conv_out[row + nkeep - s_rows] = qkv_bits[row]
    return {"q_out": q_out.reshape(-1), "k_out": k_out.reshape(-1),
            "v_out": v_out.reshape(-1), "conv_out": conv_out.reshape(-1)}


def _gate_pair(b_bits, a_bits, A_log_bits, dtb_bits):
    """The shared per-head decay/beta pair: beta = T(sigmoid-like branch of
    b), g = exp(-(exp(A_log) * T(softplus-like sp))): every T() rounding of
    compute_g, with the softplus held in T before the f32 multiply."""
    bv = BF(b_bits)
    by = np.float32(1.0) / (np.float32(1.0) + np.exp(np.abs(bv)))
    beta = bf(np.where(bv < 0, by, np.float32(1.0) - by))
    apd = bf(BF(a_bits) + BF(dtb_bits))
    exp_term = bf(np.exp(-np.abs(BF(apd))))
    log_term = bf(_omlx_log1p(BF(exp_term)))
    sp = bf(np.maximum(BF(apd), np.float32(0.0)) + BF(log_term))
    g = np.exp(-(np.exp(BF(A_log_bits)) * BF(sp)))
    return g.astype(np.float32), beta


def ref_omlx_qwen4_gdn_decode_prework(kern, inputs):
    """The Qwen4 B1 decode prework: T=1 conv (three state rows + qkv), the
    state shift, the L2 arm, and lane 0's beta/gate pair per value head."""
    t, hk, hv, nheads = _heads(kern)
    dk, dv, c = int(t["DK"]), int(t["DV"]), int(t["C"])
    qkv = BF(inputs["qkv"].reshape(-1))
    qkv_bits = np.asarray(inputs["qkv"]).reshape(-1)
    state_bits = np.asarray(inputs["conv_state"]).reshape(3, c)
    state = BF(state_bits)
    w = BF(inputs["conv_w"].reshape(c, 4))
    q_scale = BF(inputs["q_scale"].reshape(-1))
    q_out = np.zeros((hk, dk), np.uint16)
    k_out = np.zeros_like(q_out)
    v_out = np.zeros((hv, dv), np.uint16)
    for lh in range(nheads):
        is_q, is_k, head, base, nch = _head_span(lh, hk, dk, dv)
        chs = base + np.arange(nch)
        acc = state[0, chs] * w[chs, 0]
        for tap in (1, 2):
            acc = acc + state[tap, chs] * w[chs, tap]
        acc = acc + qkv[chs] * w[chs, 3]
        act = _silu(bf(acc))
        if is_q or is_k:
            act4 = act.reshape(32, 4)
            inv = _l2_inv(act4)
            l2v = bf(BF(act4) * BF(inv)[..., None])
            val = bf(BF(l2v) * q_scale) if is_q else l2v
            (q_out if is_q else k_out)[head] = val.reshape(-1)
        else:
            v_out[head] = act.reshape(-1)
    # conv_out = [old1, old2, qkv], written per channel by every head
    conv_out = np.stack([state_bits[1], state_bits[2], qkv_bits])
    g_out, beta_out = _gate_pair(inputs["b_in"], inputs["a_in"],
                                 inputs["A_log"], inputs["dt_bias"])
    return {"q_out": q_out.reshape(-1), "k_out": k_out.reshape(-1),
            "v_out": v_out.reshape(-1), "conv_out": conv_out.reshape(-1),
            "g_out": g_out.astype(np.float32), "beta_out": beta_out}


def _pair_tree8(p):
    """((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7)) over a trailing axis of 8."""
    return (((p[..., 0] + p[..., 1]) + (p[..., 2] + p[..., 3]))
            + ((p[..., 4] + p[..., 5]) + (p[..., 6] + p[..., 7])))


def _xor_row(v):
    """simd_shuffle_xor(v, 1) then (v, 2) over a trailing 4-lane row; every
    lane ends holding the row total."""
    v = v + v[..., np.arange(4) ^ 1]
    return v + v[..., np.arange(4) ^ 2]


def _decode_step_row(qkv_bits, z_bits, b_bits, a_bits, state_bits, conv_w_bits,
                     q_scale_bits, A_log_bits, dtb_bits, state_in, norm_w_bits,
                     eps, hk, hv, dk, dv, hv_idx):
    """One value head of _QWEN4_DECODE_STEP_SOURCE: sg0/1/2 run the decode
    prework into threadgroup q/k/v, sg3 lane 0 the gate pair, the packed
    btree recurrence updates the f32 state, sg0 runs the norm-gate."""
    hk_idx = hv_idx // (hv // hk)
    q = np.zeros(dk, np.uint16)
    k = np.zeros(dk, np.uint16)
    v = np.zeros(dv, np.uint16)
    state_f = BF(np.asarray(state_bits).reshape(3, -1))
    w = BF(np.asarray(conv_w_bits).reshape(state_f.shape[1], 4))
    qkv_f = BF(np.asarray(qkv_bits).reshape(-1))
    qs = BF(q_scale_bits.reshape(-1))
    for sg in range(3):
        is_q, is_k = sg == 0, sg == 1
        head = hk_idx if is_q or is_k else hv_idx
        base = (head * dk if is_q else
                (hk * dk + head * dk if is_k else 2 * hk * dk + head * dv))
        nch = dk if is_q or is_k else dv
        chs = base + np.arange(nch)
        acc = state_f[0, chs] * w[chs, 0]
        for tap in (1, 2):
            acc = acc + state_f[tap, chs] * w[chs, tap]
        acc = acc + qkv_f[chs] * w[chs, 3]
        act = _silu(bf(acc))
        if is_q or is_k:
            act4 = act.reshape(32, 4)
            inv = _l2_inv(act4)
            l2v = bf(BF(act4) * BF(inv)[..., None])
            dst = bf(BF(l2v) * qs) if is_q else l2v
            (q if is_q else k)[:] = dst.reshape(-1)
        else:
            v[:] = act
    g, beta = _gate_pair(b_bits[hv_idx:hv_idx + 1], a_bits[hv_idx:hv_idx + 1],
                         A_log_bits[hv_idx:hv_idx + 1],
                         dtb_bits[hv_idx:hv_idx + 1])
    gt, beta_f = np.float32(g[0]), np.float32(BF(beta)[0])
    # packed recurrence: four lanes per value row, 32 f32 state values per lane
    st = np.asarray(state_in, np.float32).reshape(
        hv * dv, dk)[hv_idx * dv:(hv_idx + 1) * dv]
    st3 = st.reshape(dv, 4, 32) * gt
    kmat = BF(k).reshape(4, 32)
    qmat = BF(q).reshape(4, 32)
    part = (st3 * kmat[None]).reshape(dv, 4, 8, 4).sum(-1)
    kv = _xor_row(_pair_tree8(part))[..., 0]
    delta = (BF(v) - kv) * beta_f
    st3 = st3 + kmat[None] * delta[:, None, None]
    part = (st3 * qmat[None]).reshape(dv, 4, 8, 4).sum(-1)
    ty = bf(_xor_row(_pair_tree8(part))[..., 0])
    # norm-gate on simdgroup 0: lanes cover ty linearly
    xs = BF(ty).reshape(32, 4)
    total = (xs * xs).sum(-1)[0]
    for lane in range(1, 32):
        total = total + (xs * xs).sum(-1)[lane]
    invn = np.float32(1.0) / np.sqrt(total / np.float32(dv)
                                     + np.float32(eps))
    normed = bf(BF(bf(xs * invn)) * BF(norm_w_bits.reshape(32, 4)))
    zv = BF(np.asarray(z_bits).reshape(hv * dv)[hv_idx * dv:(hv_idx + 1) * dv]).reshape(32, 4)
    syz = np.float32(1.0) / (np.float32(1.0) + np.exp(np.abs(zv)))
    out_head = bf(BF(normed) * np.where(zv < 0, syz, np.float32(1.0) - syz))
    return st3.reshape(dv, dk), out_head.reshape(-1)


def _decode_step(kern, inputs):
    t = _heads(kern)[0]
    dk, dv = int(t["DK"]), int(t["DV"])
    hk, hv = int(t["HK"]), int(t["HV"])
    c = int(t["C"])
    batch = int(np.asarray(inputs["qkv"]).size) // c
    eps = float(np.asarray(inputs["eps"]).reshape(-1)[0])
    conv_out = np.zeros((batch, 3, c), np.uint16)
    state_out = np.zeros((batch, hv * dv, dk), np.float32)
    out = np.zeros((batch, hv * dv), np.uint16)
    for b in range(batch):
        conv_out[b] = np.stack([inputs["conv_state"].reshape(batch, 3, c)[b, 1],
                                inputs["conv_state"].reshape(batch, 3, c)[b, 2],
                                np.asarray(inputs["qkv"]).reshape(batch, c)[b]])
        for hv_idx in range(hv):
            st, o = _decode_step_row(
                np.asarray(inputs["qkv"]).reshape(batch, c)[b],
                np.asarray(inputs["z"]).reshape(batch, hv * dv)[b],
                np.asarray(inputs["b_in"]).reshape(batch, hv)[b],
                np.asarray(inputs["a_in"]).reshape(batch, hv)[b],
                np.asarray(inputs["conv_state"]).reshape(batch, 3, c)[b],
                inputs["conv_w"], inputs["q_scale"],
                np.asarray(inputs["A_log"]).reshape(-1),
                np.asarray(inputs["dt_bias"]).reshape(-1),
                np.asarray(inputs["state_in"], np.float32).reshape(
                    batch, hv * dv, dk)[b],
                inputs["norm_w"], eps, hk, hv, dk, dv, hv_idx)
            state_out[b, hv_idx * dv:(hv_idx + 1) * dv] = st
            out[b, hv_idx * dv:(hv_idx + 1) * dv] = o
    return {"conv_out": conv_out.reshape(-1),
            "state_out": state_out.reshape(-1),
            "out": out.reshape(-1)}


def ref_omlx_qwen4_gdn_decode_step(kern, inputs):
    return _decode_step(kern, inputs)


def ref_omlx_qwen4_gdn_batch_decode_step(kern, inputs):
    """The batch kernel rebinds every buffer to its grid-z row and runs the
    one-row source verbatim, so B independent single-row steps."""
    return _decode_step(kern, inputs)


REFS = {
    "omlx_qwen35_gdn_prework_S1": ref_omlx_qwen35_gdn_prework_S1,
    "omlx_qwen4_gdn_decode_prework": ref_omlx_qwen4_gdn_decode_prework,
    "omlx_qwen4_gdn_prefill_prework": ref_omlx_qwen4_gdn_prefill_prework,
    "omlx_qwen4_gdn_decode_step": ref_omlx_qwen4_gdn_decode_step,
    "omlx_qwen4_gdn_batch_decode_step": ref_omlx_qwen4_gdn_batch_decode_step,
}
