#!/usr/bin/env python3
"""numpy references for the fused Qwen3.5 MoE expert sibling kernels.

Simulates the assembled MSL op for op. Rounding rules (see refs_common):
every explicit T(...) or static_cast<T> in the kernel is one bf16
round-to-nearest-even here; plain float arithmetic stays float32; cross-lane
simd_sum reductions are sequential float32 sums over lanes in index order,
the one approved reassociation.

Origins (composed semantic ground truth whose function bodies these
references mirror where the MSL is terse):
  qwen35_moe_router.py       omlx_router_gemv_rows: the router logits gemv
  qwen35_moe_routed_decode.py
    routed_decode / _down        the decode gate+up and down+combine launches
    routed_window / _window_down the verify-window variants (every window row
                                 runs the one-token arithmetic on its own
                                 input, routing and output)
    _gate_up_topk                the folded softmax + top-k gate+up launch
  qwen35_moe_gate_up.py      the underlying gate/up/down qmv launches

Weight layout (all six kernels): 4-bit values packed two per byte, element e
in nibble e of the row (u32 words, little endian), group size 32, one bf16
scale and bias per group. The qmv traversal is the FAST path (16 inputs per
lane per 512 block); the guarded tail path of the shared-expert gate row
(FAST = 0) degenerates to the same arithmetic for K = 2048 because every
lane keeps a full 16-value tail.

Grid notes. The recorded launch grids are thread grids; divided by the threadgroup size they
give the block counts these references simulate (1 + NS/ROWS + TOPK*NI/ROWS blocks for the gate+up kernels).
The
folded top-k of gate_up_topk fills its last four slots from the
zero-probability experts outside the 4-expert weight pool; the kernel would
read past the weight table there, the reference writes zeros (their scores
are zero, so the composed down launch multiplies those rows away).
"""

import numpy as np

from refs_common import bf, BF, f32sum, select_topk

REFS = {}


def _tmpl(kern):
    return dict((n, v) for n, v in kern["tmpl"])


# --- qmv: MLX qmv_fast traversal of one 4-bit group-32 weight table ---------
_LANE = np.arange(32)
_J16 = np.arange(16)


def _nibbles(words):
    """(R, W) u32 words -> (R, W*8) raw nibble values; element e lives in
    byte e//2, low nibble for even e."""
    w = np.ascontiguousarray(words, np.uint32)
    b = w.reshape(-1, w.shape[-1]).view(np.uint8)
    out = np.empty((b.shape[0], b.shape[1] * 2), np.uint8)
    out[:, 0::2] = b & np.uint8(0x0F)
    out[:, 1::2] = b >> np.uint8(4)
    return out


class _Q:
    """One packed weight table with its bf16 scales and biases."""

    def __init__(self, w, s, b, K):
        self.nib = _nibbles(w)
        self.s = BF(s)
        self.b = BF(b)
        self.K = K

    def dot(self, rows, xf):
        """qmv_rows for `rows` against one f32 input vector.

        Each lane owns 16 consecutive inputs per 512 block. The kernel
        prescales x_thread by the nibble position and shifts each nibble up
        by the same power of two, so every product is the raw input times
        the raw nibble with one rounding; sum (the bias term) walks the raw
        inputs in load_vector order. qdot_n folds one 4-product group per
        u16 word, result accumulates one group per block, and the cross-lane
        simd_sum is the sequential f32 stand-in.
        """
        rows = np.asarray(rows, np.int64)
        nib = self.nib[rows]
        s = self.s[rows]
        b = self.b[rows]
        res = np.zeros((len(rows), 32), np.float32)
        for k0 in range(0, self.K, 512):
            e = k0 + _LANE[:, None] * 16 + _J16[None, :]
            xv = xf[e]
            ssum = np.zeros(32, np.float32)
            for i in range(4):
                g = xv[:, 4 * i] + xv[:, 4 * i + 1]
                g = g + xv[:, 4 * i + 2]
                g = g + xv[:, 4 * i + 3]
                ssum = ssum + g
            nb = nib[:, k0:k0 + 512].reshape(-1, 32, 16)
            acc = np.zeros((len(rows), 32), np.float32)
            for i in range(4):
                g = xv[None, :, 4 * i] * nb[:, :, 4 * i]
                g = g + xv[None, :, 4 * i + 1] * nb[:, :, 4 * i + 1]
                g = g + xv[None, :, 4 * i + 2] * nb[:, :, 4 * i + 2]
                g = g + xv[None, :, 4 * i + 3] * nb[:, :, 4 * i + 3]
                acc = acc + g
            sidx = k0 // 32 + _LANE // 2
            res = res + (s[:, sidx] * acc + ssum[None, :] * b[:, sidx])
        out = np.zeros(len(rows), np.float32)
        for l in range(32):
            out = out + res[:, l]
        return out


# --- the bf16 sigmoid the compiled swiglu and combine use -------------------
def _sigmoid_t(g_bits):
    """omlx_mlx_sigmoid<bfloat16_t>, spelled out with T casts in the down
    kernel: exp rounds to T, 1+exp rounds to T, the divide rounds to T,
    1-sy rounds to T; every intermediate between roundings is float32."""
    g = BF(g_bits)
    with np.errstate(over="ignore"):  # |g| > 88: exp -> inf, sigmoid -> 0
        e = bf(np.float32(1.0) + BF(bf(np.exp(np.abs(g)))))
    sy = bf(np.float32(1.0) / BF(e))
    om = bf(np.float32(1.0) - BF(sy))
    return np.where(g < 0, sy, om)


def _swiglu(gres, ures):
    """swiglu_store: gate and up results each round to T, then
    silu(gate) * up with every op in T."""
    g = bf(gres)
    u = bf(ures)
    t = bf(BF(g) * BF(_sigmoid_t(g)))
    return bf(BF(t) * BF(u))


def _gate_up_y(xf, mat, sg, su, gm, experts, ni, ns, topk):
    """One token's gate+up output: routed silu(gate)*up per slot, the shared
    expert's, then the shared gate row (routed_decode's h layout)."""
    y = np.zeros(topk * ni + ns + 1, np.uint16)
    y[topk * ni + ns:topk * ni + ns + 1] = bf(gm.dot([0], xf))
    rows = np.arange(ns)
    y[topk * ni:topk * ni + ns] = _swiglu(sg.dot(rows, xf), su.dot(rows, xf))
    pool = mat.nib.shape[0] // (2 * ni)
    for slot in range(topk):
        e = int(experts[slot])
        if e >= pool:
            continue  # zero-probability selection past the weight pool: zeros
        y[slot * ni:(slot + 1) * ni] = _swiglu(
            mat.dot(e * 2 * ni + rows, xf),
            mat.dot(e * 2 * ni + ni + rows, xf))
    return y


# --- omlx_qwen35_moe_router_gemv --------------------------------------------
def ref_router_gemv(kern, inputs):
    """omlx_router_gemv_rows: one simdgroup per expert row, each lane
    accumulating its strided 4-element slices in f32, then the explicit
    shuffle_down butterfly (16, 8, 4, 2, 1); lane 0 stores bf16."""
    t = _tmpl(kern)
    K, m_count, nsg = int(t["K"]), int(t["M"]), int(t["NSG"])
    xf = BF(inputs["x"].reshape(-1))
    wf = BF(inputs["w"].reshape(-1)).reshape(-1, K)
    s_count = kern["grid"][1] * nsg
    S = np.arange(s_count)
    experts = S // m_count
    vals = np.zeros((s_count, 32), np.float32)
    wl = wf[experts]
    for i in range(K // 128):
        idx = _LANE[:, None] * 4 + i * 128 + np.arange(4)[None, :]
        for tn in range(4):
            vals += wl[:, idx[:, tn]] * xf[idx[:, tn]][None, :]
    for sn in (16, 8, 4, 2, 1):
        vals[:, :32 - sn] += vals[:, sn:]
    return {"y": bf(vals[:, 0]).reshape(1, s_count // m_count)}


# --- omlx_qwen35_moe_gate_up_decode / _window / _topk -----------------------
def _gate_up_common(kern, inputs):
    t = _tmpl(kern)
    ni, ns, topk = int(t["NI"]), int(t["NS"]), int(t["TOPK"])
    xf = BF(inputs["x"].reshape(-1))
    mat = _Q(inputs["w"], inputs["scales"], inputs["biases"], int(t["K"]))
    sg = _Q(inputs["sg_w"], inputs["sg_s"], inputs["sg_b"], int(t["K"]))
    su = _Q(inputs["su_w"], inputs["su_s"], inputs["su_b"], int(t["K"]))
    gm = _Q(inputs["g_w"], inputs["g_s"], inputs["g_b"], int(t["K"]))
    return t, xf, mat, sg, su, gm, ni, ns, topk


def ref_gate_up_decode(kern, inputs):
    """Blocks in order: the shared gate row, NS/ROWS shared swiglu blocks,
    then TOPK*NI/ROWS blocks per selected expert (routed_decode's launch)."""
    t, xf, mat, sg, su, gm, ni, ns, topk = _gate_up_common(kern, inputs)
    rhs = inputs["rhs"].reshape(-1)
    return {"y": _gate_up_y(xf, mat, sg, su, gm, rhs, ni, ns, topk)}


def ref_gate_up_window(kern, inputs):
    """The window variant: threadgroup y = block*M + window_row, so every
    window row runs the one-token arithmetic on its own input, routing and
    output (routed_window's launch)."""
    t, xf, mat, sg, su, gm, ni, ns, topk = _gate_up_common(kern, inputs)
    m_rows = int(t["M"])
    rhs = inputs["rhs"].reshape(m_rows, -1)
    yw = topk * ni + ns + 1
    y = np.zeros(m_rows * yw, np.uint16)
    K = int(t["K"])
    for wr in range(m_rows):
        y[wr * yw:(wr + 1) * yw] = _gate_up_y(
            xf[wr * K:(wr + 1) * K], mat, sg, su, gm,
            rhs[wr], ni, ns, topk)
    return {"y": y}


def ref_gate_up_topk(kern, inputs):
    """_gate_up_topk's launch: block 0 folds softmax_topk_row (its simdgroup
    1 stores indices and scores), every routed block re-derives its slot's
    expert from the same keys via nth (descending key, ties to the higher
    expert), then runs the decode arithmetic."""
    t = _tmpl(kern)
    ne, topk = int(t["NE"]), int(t["TOPK"])
    logits = BF(inputs["logits"].reshape(-1))
    # omlx_router_softmax_row: exact f32 max, f32 exp, f32 sum stand-in, one
    # T rounding per probability
    e = np.exp(logits - np.float32(logits.max())).astype(np.float32)
    p = (e * np.float32(np.float32(1.0) / f32sum(e))).astype(np.float32)
    idx, sel_p = select_topk(bf(p), ne, topk)
    scores = bf(sel_p * (np.float32(1.0) / f32sum(sel_p)))
    t2, xf, mat, sg, su, gm, ni, ns, topk2 = _gate_up_common(kern, inputs)
    return {"y": _gate_up_y(xf, mat, sg, su, gm, idx, ni, ns, topk),
            "indices": idx, "scores": scores}


# --- omlx_qwen35_moe_down_combine_decode / _window --------------------------
def _down_y(xf_row, mat, sd, rhs, scores, gate_bits, k, n, topk, ks, rps):
    """One row's down+combine (_down's launch): simdgroup slot j < TOPK runs
    the routed expert's down rows, slot TOPK the shared expert's, then
    fused_moe_combine folds the parts with T-rounding per product and add."""
    blocks = n // rps
    part = np.zeros((topk + 1, rps, blocks), np.uint16)
    for j in range(topk):
        d = mat.dot(int(rhs[j]) * n + np.arange(n), xf_row[j * k:(j + 1) * k])
        part[j] = bf(d).reshape(blocks, rps).T
    d = sd.dot(np.arange(n), xf_row[topk * k:topk * k + ks])
    part[topk] = bf(d).reshape(blocks, rps).T
    pf = BF(part)
    sc = BF(scores)
    lane = np.zeros((8, rps, blocks), np.uint16)
    for j in range(topk):
        p = bf(pf[j] * sc[j])
        lane[j % 8] = bf(BF(p) + BF(lane[j % 8]))
    acc = lane[0]
    for l in range(1, 8):
        acc = bf(BF(lane[l]) + BF(acc))
    sg = _sigmoid_t(gate_bits)
    sh = bf(BF(sg) * pf[topk])
    out = bf(BF(acc) + BF(sh))
    return out.T.reshape(-1)


def _down_common(kern, inputs):
    t = _tmpl(kern)
    xf = BF(inputs["x"].reshape(-1))
    mat = _Q(inputs["w"], inputs["scales"], inputs["biases"], int(t["K"]))
    sd = _Q(inputs["sd_w"], inputs["sd_s"], inputs["sd_b"], int(t["KS"]))
    return t, xf, mat, sd


def ref_down_combine_decode(kern, inputs):
    t, xf, mat, sd = _down_common(kern, inputs)
    k, n = int(t["K"]), int(t["N"])
    topk, ks, rps = int(t["TOPK"]), int(t["KS"]), int(t["RPS"])
    rhs = inputs["rhs"].reshape(-1)
    scores = inputs["scores"].reshape(-1)
    gate = inputs["x"].reshape(-1)[topk * k + ks:topk * k + ks + 1]
    return {"y": _down_y(xf, mat, sd, rhs, scores, gate, k, n, topk, ks, rps)}


def ref_down_combine_window(kern, inputs):
    """The window variant: window_row = tid.y % M, out_row = tid.y / M * RPS,
    every window row folds its own routing and gate (routed_window's _down)."""
    t, xf, mat, sd = _down_common(kern, inputs)
    k, n = int(t["K"]), int(t["N"])
    topk, ks, rps = int(t["TOPK"]), int(t["KS"]), int(t["RPS"])
    m_rows = int(t["M"])
    xw = topk * k + ks + 1
    xr = inputs["x"].reshape(-1)
    rhs = inputs["rhs"].reshape(m_rows, -1)
    scores = inputs["scores"].reshape(m_rows, -1)
    y = np.zeros(m_rows * n, np.uint16)
    for wr in range(m_rows):
        base = wr * xw
        gate = xr[base + topk * k + ks:base + topk * k + ks + 1]
        y[wr * n:(wr + 1) * n] = _down_y(
            xf[base:base + xw], mat, sd, rhs[wr], scores[wr], gate,
            k, n, topk, ks, rps)
    return {"y": y}


REFS["omlx_qwen35_moe_router_gemv"] = ref_router_gemv
REFS["omlx_qwen35_moe_gate_up_decode_b4g32f_shared_b4g32f_gate_b4g32s"] = \
    ref_gate_up_decode
REFS["omlx_qwen35_moe_gate_up_window_b4g32f_shared_b4g32f_gate_b4g32s"] = \
    ref_gate_up_window
REFS["omlx_qwen35_moe_gate_up_topk_b4g32f_shared_b4g32f_gate_b4g32s"] = \
    ref_gate_up_topk
REFS["omlx_qwen35_moe_down_combine_decode_b4g32f_shared_b4g32f"] = \
    ref_down_combine_decode
REFS["omlx_qwen35_moe_down_combine_window_b4g32f_shared_b4g32f"] = \
    ref_down_combine_window


if __name__ == "__main__":
    import parity_common as pc
    for name, fn in sorted(REFS.items()):
        kern = pc.BY_NAME[name]
        inputs = {}
        for i, spec in enumerate(kern["inputs"]):
            inputs[spec[0]] = np.frombuffer(
                pc.gen_input(name, i, spec, 1234),
                pc.DTYPES[spec[1]][1] or np.uint16).reshape(spec[2])
        outs = fn(kern, inputs)
        for oname, arr in outs.items():
            oname_code = next(o for o in kern["outputs"] if o[0] == oname)
            want_dt = pc.DTYPES[oname_code[1]][1] or np.uint16
            assert np.asarray(arr).shape == tuple(oname_code[2]), oname
            assert np.asarray(arr).dtype == want_dt, oname
            f = pc.bits_to_float(oname_code[1], np.asarray(arr).reshape(-1))
            assert np.all(np.isfinite(f)), oname
            print(f"{name}.{oname}: min={f.min():+.4g} max={f.max():+.4g} "
                  f"mean={f.mean():+.4g}")
