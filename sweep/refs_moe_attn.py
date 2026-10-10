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

Origin of these four kernels' semantics: the assembled artifact MSL
(OUT/src/moe_qwen35_gated_delta_step_*.cl, moe_qwen35_ragged_sdpa_{1p,2p1,2p2}_*.cl
in the metal2vk-sibsweep sweep output; the shim and IR passes do not change
the arithmetic). All four run everything in float32 and round to bfloat16_t
only at the final stores; the online-softmax exp is fast::exp, simulated with
numpy float32 exp. Every kernel assumes 32-lane subgroups (PARITY.md caveat).
"""
import numpy as np

import parity_common as pc

from refs_common import bf, BF, f32sum, sigmoid_f32  # noqa: F401

FLT_MAX = np.float32(np.finfo(np.float32).max)


def _lane_sum(x):
    """simd_sum over the last axis: sequential float32 left-to-right sum
    (the vectorized form of refs_common.f32sum, same reassociation)."""
    s = x[..., 0]
    for l in range(1, x.shape[-1]):
        s = np.float32(s + x[..., l])
    return s


# --- qwen35_gated_delta_step -------------------------------------------------
def ref_qwen35_gated_delta_step(kern, inputs):
    """Recurrent gated-delta step over T tokens, one thread per
    (head n, value row dv, 4-of-128 key slice). state (f32) decays by g,
    accumulates kv_mem = simd_sum(state*k), delta = (v - kv_mem)*beta,
    state += k*delta, y = bf16(simd_sum(state*q)). One threadgroup-local x
    lane owns Dk/32 = 4 contiguous key slots (s_idx = 4*dk_idx + i). The MSL
    derives hk_idx = hv_idx / (Hv/Hk) and b_idx = n / Hv with the template
    values baked in; everything except the y store stays float32."""
    t = dict((n, v) for n, v in kern["tmpl"])
    dk, dv = int(t["Dk"]), int(t["Dv"])
    hk, hv = int(t["Hk"]), int(t["Hv"])
    lanes = int(kern["tg"][0])
    npt = dk // lanes
    T = int(np.asarray(inputs["T"]).reshape(-1)[0])  # kernel reads the T buffer
    B = int(kern["grid"][2]) // hv
    q = BF(inputs["q"]).reshape(B, T, hk, dk)
    k = BF(inputs["k"]).reshape(B, T, hk, dk)
    v = BF(inputs["v"]).reshape(B, T, hv, dv)
    g = np.asarray(inputs["g"], np.float32).reshape(B, T, hv)
    beta = BF(inputs["beta"]).reshape(B, T, hv)
    # state_in[(n*Dv + dv)*Dk + 4*dk_idx + i], n = b*Hv + hv_idx (f32, no
    # rounding on the whole path)
    st = np.asarray(inputs["state_in"], np.float32) \
        .reshape(B, hv, dv, lanes, npt).copy()
    kh = hv // hk
    y = np.zeros((B, T, hv, dv), np.uint16)
    for tt in range(T):
        st *= g[:, tt][:, :, None, None, None]
        kq = np.repeat(k[:, tt], kh, axis=1).reshape(B, hv, lanes, npt)
        qe = np.repeat(q[:, tt], kh, axis=1).reshape(B, hv, lanes, npt)
        kv_acc = st[..., 0] * kq[..., 0][..., None, :]
        for i in range(1, npt):
            kv_acc = kv_acc + st[..., i] * kq[..., i][..., None, :]
        kv_mem = _lane_sum(kv_acc)  # (B, hv, dv)
        delta = (v[:, tt] - kv_mem) * beta[:, tt][:, :, None]
        out_acc = None
        for i in range(npt):
            st[..., i] = st[..., i] + kq[..., i][..., None, :] * delta[..., None]
            term = st[..., i] * qe[..., i][..., None, :]
            out_acc = term if out_acc is None else out_acc + term
        y[:, tt] = bf(_lane_sum(out_acc))
    return {"y": y.reshape(B * T * hv * dv),
            "state_out": st.reshape(B * hv * dv * dk)}


# --- qwen35_ragged_sdpa_1p / 2p1 / 2p2 ---------------------------------------
def ref_qwen35_ragged_sdpa_1p(kern, inputs):
    """Single-pass ragged SDPA over the pad-trimmed keys. Each of the BN=32
    simdgroups of a batch-head walks rows i = pad + g, g+32, ... with the
    online-softmax rescale (first iteration: max_score - new_max = -inf, so
    factor = exp(-inf) = 0 and the zero-initialized o/sum survive); lanes
    split the 256 dims 8-wide, score = simd_sum of the per-lane 8-term dots.
    The cross-simdgroup merge transposes o through threadgroup memory,
    rescales by exp(their max - global max), normalizes once and stores
    bf16(out[qh*256 + g*8 + i]) from lane 0."""
    t = dict((n, v) for n, v in kern["tmpl"])
    qh_n, kvh, gqa = (int(t["NUM_Q_HEADS"]), int(t["NUM_KV_HEADS"]),
                      int(t["GQA_FACTOR"]))
    k_size = int(np.asarray(inputs["k_size"]).reshape(-1)[0])
    pads = np.asarray(inputs["pads"], np.int32).reshape(-1)
    scale = np.float32(np.asarray(inputs["scale"], np.float32).reshape(-1)[0])
    qbh_n = int(kern["grid"][1])
    B = qbh_n // qh_n
    lanes = 32
    bn = int(kern["tg"][0]) // lanes  # simdgroups per threadgroup
    dpt = 256 // lanes
    q = (BF(inputs["queries"]).reshape(qbh_n, lanes, dpt) * scale)
    keys = BF(inputs["keys"]).reshape(B * kvh, k_size, 256)
    vals = BF(inputs["values"]).reshape(B * kvh, k_size, 256)
    qbh = np.arange(qbh_n)
    kv_idx = (qbh // qh_n) * kvh + (qbh % qh_n) // gqa
    pad = pads[qbh // qh_n].astype(np.int64)
    n_rows = k_size - pad
    mx = np.full((qbh_n, bn), -FLT_MAX, np.float32)
    sm = np.zeros((qbh_n, bn), np.float32)
    o = np.zeros((qbh_n, bn, lanes, dpt), np.float32)
    with np.errstate(over="ignore"):
        for m in range((k_size + bn - 1) // bn):
            r = pad[:, None] + np.arange(bn)[None, :] + bn * m  # (qbh, g)
            valid = r < n_rows[:, None]
            rc = np.where(valid, r, 0)
            kr = keys[kv_idx[:, None], rc] \
                .reshape(qbh_n, bn, lanes, dpt)
            vr = vals[kv_idx[:, None], rc] \
                .reshape(qbh_n, bn, lanes, dpt)
            acc = q[:, None, :, 0] * kr[..., 0]
            for j in range(1, dpt):
                acc = acc + q[:, None, :, j] * kr[..., j]
            score = _lane_sum(acc * valid[..., None])
            new_mx = np.maximum(mx, score)
            factor = np.exp(mx - new_mx)
            e = np.exp(np.where(valid, score - new_mx, np.float32(0)))
            mx = np.where(valid, new_mx, mx)
            sm = sm * factor + e * valid
            o = o * factor[..., None, None] \
                + (e * valid)[..., None, None] * vr
        new_mx = mx.max(axis=1, keepdims=True)
        f = np.exp(mx - new_mx)  # (qbh, g): each simdgroup's rescale
        sum_exp = _lane_sum(sm * f)  # (qbh,)
        # outputs[l*32 + g] = o_l[g-block]; thread (g, lane) sums slot
        # g*32 + l' over the lanes: numerator for value dims g*8 + i
        ot = o.transpose(0, 2, 1, 3)  # (qbh, g_block, g_simd, i)
        merged = ot[:, :, 0, :] * f[:, 0, None, None]
        for l in range(1, bn):
            merged = merged + ot[:, :, l, :] * f[:, l, None, None]
        out = np.where(sum_exp[:, None, None] == 0, merged,
                       merged / sum_exp[:, None, None])
    return {"out": bf(out).reshape(qbh_n * 256)}


def ref_qwen35_ragged_sdpa_2p1(kern, inputs):
    """Partial pass: threadgroup (kv_head, batch) with grid z = block walks
    every BLOCKS-th row (i = pad + block, block+4, ...), gqa_idx selects the
    query head (q_head = 8*kv + gqa). Same online softmax as 1p but NO
    cross-block merge: partials keep the UNNORMALIZED o rounded to bf16,
    sums/maxs keep this block's denominator and running max.

    Geometry is the ARTIFACT LAUNCH, not the template's full extent:
    launch_grid (threads) / threadgroup = the workgroup counts the real
    oMLX call dispatches. For the pinned template that is (2, 1, 4): x
    covers both kv heads, z all 4 blocks, but y is ONE batch workgroup, so
    batch_idx only takes 0 and q_batch_head_idx stays below 16. The kernel
    as shipped never writes partials[16384:], sums[64:] or maxs[64:] (the
    batch-1 half of the declared outputs); the reference zeros that hole
    (the verify_step_states convention) and the comparison flags it. The
    template still supports a batch axis: a wider launch would compute it."""
    t = dict((n, v) for n, v in kern["tmpl"])
    qh_n, kvh, gqa = (int(t["NUM_Q_HEADS"]), int(t["NUM_KV_HEADS"]),
                      int(t["GQA_FACTOR"]))
    launch = kern.get("launch_grid")
    wg = (tuple(int(a) // int(b) for a, b in zip(launch, kern["tg"]))
          if launch else kern["grid"])  # real workgroup counts
    blocks = wg[2]
    k_size = int(np.asarray(inputs["k_size"]).reshape(-1)[0])
    pads = np.asarray(inputs["pads"], np.int32).reshape(-1)
    scale = np.float32(np.asarray(inputs["scale"], np.float32).reshape(-1)[0])
    B = wg[1]
    lanes = int(kern["tg"][0])
    dpt = 256 // lanes
    qbh_n = B * qh_n
    # the harness buffers are sized for the template's full batch extent
    # (here 32 q-heads); the launch reads only the first B batch heads, so
    # slice that prefix and simulate the launched workgroups
    q = (BF(inputs["queries"]).reshape(-1, lanes, dpt)[:qbh_n] * scale)
    keys = BF(inputs["keys"]).reshape(-1, k_size, 256)[:B * kvh]
    vals = BF(inputs["values"]).reshape(-1, k_size, 256)[:B * kvh]
    # axes (b, kv, block, gqa, lane, ...)
    qb = ((np.arange(B) * qh_n)[:, None, None]
          + (8 * np.arange(kvh))[None, :, None]
          + np.arange(gqa)[None, None, :])  # q_batch_head index
    kv_idx = ((np.arange(B) * kvh)[:, None]
              + np.arange(kvh)[None, :])  # keys/values first index
    pad = pads[np.arange(B)][:, None, None, None]
    mx = np.full((B, kvh, blocks, gqa), -FLT_MAX, np.float32)
    sm = np.zeros((B, kvh, blocks, gqa), np.float32)
    o = np.zeros((B, kvh, blocks, gqa, lanes, dpt), np.float32)
    qs = q[qb]  # (B, kv, gqa, lanes, dpt)
    with np.errstate(over="ignore"):
        for m in range((k_size - int(pads[0]) + blocks - 1) // blocks):
            r = pad + blocks * m + np.arange(blocks)[None, None, :, None]
            valid = (r < k_size - pad).astype(np.float32)  # (B,1,block,1)
            rc = np.minimum(r, k_size - 1)
            kr = keys[kv_idx[:, :, None, None], rc] \
                .reshape(B, kvh, blocks, lanes, dpt)
            vr = vals[kv_idx[:, :, None, None], rc] \
                .reshape(B, kvh, blocks, lanes, dpt)
            qt = qs[:, :, None]  # (B, kv, 1, gqa, lanes, dpt)
            kt = kr[:, :, :, None]  # (B, kv, block, 1, lanes, dpt)
            vt = vr[:, :, :, None]
            acc = qt[..., 0] * kt[..., 0]
            for j in range(1, dpt):
                acc = acc + qt[..., j] * kt[..., j]
            score = _lane_sum(acc * valid[..., None])
            new_mx = np.maximum(mx, score)
            factor = np.exp(mx - new_mx)
            e = np.exp((score - new_mx) * valid)
            mx = np.where(valid > 0, new_mx, mx)
            sm = sm * factor + e * valid
            o = o * factor[..., None, None] \
                + (e * valid)[..., None, None] * vt
        # partials[qbh*4*256 + block*256 + lane*8 + j], qbh = b*16 + 8*kv + gqa.
        # The declared outputs cover the template's full batch extent, but the
        # launch only computes batches < B: everything at or beyond
        # q_batch_head B*qh_n stays zero here (the kernel never writes it) and
        # the comparison flags that hole against the poisoned dumps.
        outs = dict((o[0], int(np.prod(o[2]))) for o in kern["outputs"])
        pt = np.zeros(outs["partials"], np.uint16)
        pt[:B * qh_n * blocks * 256] = \
            bf(o.transpose(0, 1, 3, 2, 4, 5)).reshape(-1)
        st = np.zeros(outs["sums"], np.float32)
        st[:B * qh_n * blocks] = sm.transpose(0, 1, 3, 2).reshape(-1)
        mt = np.zeros(outs["maxs"], np.float32)
        mt[:B * qh_n * blocks] = mx.transpose(0, 1, 3, 2).reshape(-1)
    return {"partials": pt, "sums": st, "maxs": mt}


def ref_qwen35_ragged_sdpa_2p2(kern, inputs):
    """Combine pass over the 2p1 partials. The merge loops run
    BLOCKS // BN times with BN = 32: at the table's BLOCKS=4 that is zero
    passes, so max_score stays -FLT_MAX, sum_exp_score stays 0, o stays 0
    and the sum==0 guard stores bf16(0) for every element (parity_common's
    note: the degenerate combine). Simulated literally, not special-cased."""
    t = dict((n, v) for n, v in kern["tmpl"])
    blocks = int(t["BLOCKS"])
    qbh_n = int(kern["grid"][1])
    lanes = 32
    bn = 256 // lanes  # BD, and BN = 32 constexpr in the MSL
    dpt = 256 // bn
    mx = np.full((qbh_n, bn), -FLT_MAX, np.float32)
    sm = np.zeros((qbh_n, bn), np.float32)
    merged = np.zeros((qbh_n, bn, dpt), np.float32)
    with np.errstate(over="ignore", invalid="ignore"):  # 0/0 in the guard
        for b in range(blocks // bn):  # empty at BLOCKS=4
            pass
        sum_exp = _lane_sum(sm)
        out = np.where(sum_exp[:, None, None] == 0, merged,
                       merged / sum_exp[:, None, None])
    return {"out": bf(out).reshape(qbh_n * 256)}


def ref_omlx_gdn_norm_gate_eps1em06(kern, inputs):
    """omlx_gdn_norm_gate_eps1em06: RMS norm (mean of squares over the 128
    columns, eps 1e-6, precise rsqrt), norm weight multiply in bf16, gate by
    the MLX sigmoid form in f32, bf16 product store, plus per-16-lane f32
    partial sums of the stored outputs (two per row) for the out projection.
    Origin: the composed norm+gate chain in qwen35_gdn_prework.py."""
    y = BF(inputs["y"].reshape(-1))
    z = BF(inputs["z"].reshape(-1))
    nw = BF(inputs["norm_w"].reshape(-1))
    rows = int(kern["grid"][1])
    cols = y.size // rows
    out = np.zeros(rows * cols, np.uint16)
    xs = np.zeros((rows, 2), np.float32)
    for r in range(rows):
        x = y[r * cols:(r + 1) * cols]
        ss = f32sum(x * x)
        inv = np.float32(1.0) / np.sqrt(np.float32(np.float32(ss) / np.float32(cols)
                                                   + np.float32(1e-06)))
        parts = np.zeros(32, np.float32)
        for lane in range(cols // 4):
            base = r * cols + lane * 4
            for i in range(4):
                normed = bf(nw[lane * 4 + i] * BF(bf(x[lane * 4 + i] * inv))[0])[0]
                g = z[base + i]
                sig = sigmoid_f32(g)
                o = bf(np.float32(g * sig) * BF(np.asarray([normed], np.uint16))[0])[0]
                out[base + i] = o
                parts[lane] = np.float32(parts[lane] + BF(np.asarray([o], np.uint16))[0])
        for off in (1, 2, 4, 8):
            prev = parts.copy()
            for lane in range(32):
                parts[lane] = np.float32(prev[lane] + prev[lane ^ off])
        xs[r, 0] = parts[0]
        xs[r, 1] = parts[16]
    return {"out": out.reshape(rows * cols), "xs": xs.reshape(-1)}


REFS = {
    "omlx_gdn_norm_gate_eps1em06": ref_omlx_gdn_norm_gate_eps1em06,
    "qwen35_gated_delta_step": ref_qwen35_gated_delta_step,
    "qwen35_ragged_sdpa_1p": ref_qwen35_ragged_sdpa_1p,
    "qwen35_ragged_sdpa_2p1": ref_qwen35_ragged_sdpa_2p1,
    "qwen35_ragged_sdpa_2p2": ref_qwen35_ragged_sdpa_2p2,
}
