#!/usr/bin/env python3
"""numpy references for the gdn verify family: the fused qwen4 pair from the
gdn lane, the sibling main replay, and the sibling chain attention partial.

One ref_<kernel> per kernel in REFS, per the parity_refs contract:
(kern, inputs) -> {output name: numpy array of output bits}, bf16 as uint16
bit patterns, every explicit T(...) cast a bfloat16 round-to-nearest-even,
the xor butterflies the MSL spells out simulated exactly, simd_sum as a
sequential f32 sum (refs_common's declared freedom). Shared helpers come
from refs_gdn_prework (_silu, _l2_inv, _omlx_log1p); the decay and beta
helpers here are shared by both kernels because the qwen4 gate chain and
main_replay's gdn_decay/gdn_beta round at identical sites.

Sources: the assembled verify MSL (gdn lane, S=4 fused conv+recurrence+norm
form) and the sibling-scan main replay, re-derived op for op from those
sources; the composed-op python next to the assemblers
(qwen35_gdn_verify_fused.py) is kept only as the labelled SECONDARY checks
of parity_refs (the fallback computes the conv activation without the T()
rounding at the SiLU branch, the one site where it and the assembled MSL
disagree; the MSL is the Metal-intended behaviour).

Two table facts the references pin:
- proj is one (S, P) buffer of per-token [qkv | z | b | a] rows with
  P = C + HV*DV + 2*HV = 12352, sized S*P in parity_common.
- verify_step_states never writes states[S-1] (the `t + 1 < S` guard): the
  reference zeros that row, so a comparison against a poisoned or garbage
  region flags the hole instead of papering over it.
"""
import numpy as np

import parity_common as pc

from refs_common import bf, BF, sigmoid_f32
from refs_gdn_prework import _l2_inv, _omlx_log1p, _silu, _silu_t

_LANES = np.arange(32)


def _tmpl(kern):
    return dict((n, v) for n, v in kern["tmpl"])


def _seqsum(x, axis=-1):
    """Sequential float32 left-to-right sum over one axis."""
    x = np.moveaxis(np.asarray(x, np.float32), axis, -1)
    s = x[..., 0].copy()
    for i in range(1, x.shape[-1]):
        s = s + x[..., i]
    return s


def _tree8(p):
    """((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7)) over a trailing axis of 8."""
    return (((p[..., 0] + p[..., 1]) + (p[..., 2] + p[..., 3]))
            + ((p[..., 4] + p[..., 5]) + (p[..., 6] + p[..., 7])))


def _xor_row(v):
    """simd_shuffle_xor(v, 1) then (v, 2) over a trailing 4-lane row; every
    lane ends at the same value (f32 addition is commutative), so callers
    may read lane 0."""
    v = v + v[..., [1, 0, 3, 2]]
    v = v + v[..., [2, 3, 0, 1]]
    return v


def _decay(x_f, dtb_f, a_log_f):
    """The shared decay: softplus with every InT rounding site:
    s = InT(x + dt), e = InT(exp(-|s|)), l1p = InT(omlx_log1p(e)),
    sp = InT(max(s, 0) + l1p); then g = exp(-exp(A_log) * sp) in f32."""
    s = bf(x_f + dtb_f)
    s_f = BF(s)
    e = bf(np.exp(-np.abs(s_f)))
    l1p = bf(_omlx_log1p(BF(e)))
    sp = bf(np.maximum(s_f, np.float32(0.0)) + BF(l1p))
    return np.exp(-(np.exp(a_log_f) * BF(sp)))


def _beta(x_f):
    """The shared beta: the MSL computes by = 1/(1+exp(|x|)) and takes
    (x < 0) ? by : 1 - by, which is the true sigmoid of x for both signs;
    sigmoid_f32 is exactly that expression, so the beta is its single T()
    rounding. (An earlier version wrapped sigmoid_f32 in a second
    where(x < 0, y, 1 - y), which re-inverted the branch for positive x and
    made every positive-b head diverge; the g7 dumps adjudicated the
    kernel.)"""
    return bf(sigmoid_f32(x_f))


def _channels(base, n, width):
    return base + (np.arange(n)[:, None] * width + np.arange(width)[None, :])


def _conv_act(proj_bits, proj_f, state_bits, conv_w, chans, silu=_silu_t):
    """Depthwise conv over the three state rows and the (S, P) proj rows,
    then SiLU. The step-kernel MSL rounds the branch value to T before the
    product (conv * T((conv < T(0)) ? sy : 1 - sy), _silu_t); the composed
    fallback multiplies the unrounded f32 value (_silu). acc is a sequential
    f32 tap sum, conv = T(acc), act = conv * T(selected branch)."""
    s = proj_bits.shape[0]
    rows = np.arange(s)
    acc = None
    for tap in range(4):
        if tap < 3:
            src = rows + tap
            xv = np.where(
                (src < 3)[:, None, None],
                state_bits[np.clip(src, 0, 2)[:, None, None], chans],
                proj_bits[np.clip(src - 3, 0, s - 1)[:, None, None], chans])
            term = BF(xv) * conv_w[chans, tap]
        else:
            term = proj_f[rows[:, None, None], chans] * conv_w[chans, 3]
        acc = term if acc is None else acc + term
    return silu(bf(acc))


def _qwen4_verify(kern, inputs, with_states, silu=_silu_t):
    t = _tmpl(kern)
    hk, hv, dk, dv, c, s = (int(t[k]) for k in ("HK", "HV", "DK", "DV", "C", "S"))
    b_off = c + hv * dv
    a_off = b_off + hv
    p_row = a_off + hv
    proj_bits = np.asarray(inputs["proj"]).reshape(s, p_row)
    proj_f = BF(proj_bits.reshape(-1)).reshape(s, p_row)
    state_bits = np.asarray(inputs["conv_state"]).reshape(3, c)
    conv_w = BF(inputs["conv_w"].reshape(-1)).reshape(c, 4)
    q_scale = BF(inputs["q_scale"].reshape(-1))
    state = inputs["state_in"].reshape(hv, dv, 4, dk // 4).astype(np.float32)
    norm_w = BF(inputs["norm_w"].reshape(-1))
    eps = np.float32(inputs["eps"].reshape(-1)[0])
    dtb = BF(inputs["dt_bias"].reshape(-1))
    a_log = BF(inputs["A_log"].reshape(-1))
    hk_of = np.arange(hv) // (hv // hk)

    # conv + SiLU per family; q (and k) repeat per value head pair, so the
    # two threadgroups of a key head write identical bytes and are simulated
    # once per key head.
    qk_chans = {"q": _channels(0, hk, dk),
                "k": _channels(hk * dk, hk, dk),
                "v": _channels(2 * hk * dk, hv, dv)}
    act = dict((fam, _conv_act(proj_bits, proj_f, state_bits, conv_w, ch,
                               silu=silu))
               for fam, ch in qk_chans.items())

    # window = [old state rows; this block's x_t rows], conv_out = the shift
    window = np.concatenate([state_bits, proj_bits[:, :c]], axis=0)
    conv_out = proj_bits[1:4, :c].copy()

    # q/k: l2 norm with inv = T(rsqrt(...)); v stores the raw activation
    tq = np.zeros((s, hk, dk), np.uint16)
    tk = np.zeros_like(tq)
    for fam in ("q", "k"):
        act4 = act[fam].reshape(s, hk, 32, 4)
        inv = _l2_inv(act4)
        l2v = bf(BF(act4) * BF(inv)[..., None])
        val = bf(BF(l2v) * q_scale) if fam == "q" else l2v
        (tq if fam == "q" else tk)[:] = val.reshape(s, hk, dk)
    tv = act["v"].reshape(s, hv, dv)

    # gates: one decay per (token, value head) off the fused proj a rows, one
    # beta off the b rows (the MSL reads the two columns independently)
    g = _decay(proj_f[:, a_off:a_off + hv], dtb[None, :], a_log[None, :])
    beta_f = BF(_beta(proj_f[:, b_off:b_off + hv]))

    # the recurrence: per token, decay, key match (xor over the 4-lane row),
    # delta against v, state update, q readout rounded to T on the store
    tqf = BF(tq)
    tkf = BF(tk)
    ty = np.zeros((s, hv, dv), np.uint16)
    states = (np.zeros((s, hv, dv, 4, dk // 4), np.float32)
              if with_states else None)
    for tt in range(s):
        state *= g[tt][:, None, None, None]
        krow = tkf[tt][hk_of].reshape(hv, 1, 4, dk // 4)
        part = _seqsum((state * krow).reshape(hv, dv, 4, 8, 4))
        kv = _xor_row(_tree8(part))[..., 0]
        delta = (BF(tv[tt]) - kv) * beta_f[tt][:, None]
        state += krow * delta[..., None, None]
        qrow = tqf[tt][hk_of].reshape(hv, 1, 4, dk // 4)
        part = _seqsum((state * qrow).reshape(hv, dv, 4, 8, 4))
        ty[tt] = bf(_xor_row(_tree8(part))[..., 0])
        if with_states and tt + 1 < s:
            states[tt] = state

    # gated rms norm + silu gate on ty; the written t rows of out mirror the
    # sg < S threads (lane*4 + i covers dv)
    out = np.zeros((s, hv, dv), np.uint16)
    for tt in range(s):
        xs = BF(ty[tt])
        sumsq = _seqsum(_seqsum((xs * xs).reshape(hv, 32, 4)))
        inv = np.float32(1.0) / np.sqrt(sumsq / np.float32(dv) + eps)
        normed = bf(norm_w[None, :] * BF(bf(xs * inv[:, None])))
        zv = proj_f[tt, c:c + hv * dv].reshape(hv, dv)
        # the MSL branch (zv < 0 ? sy : 1 - sy) with sy = 1/(1+exp(|zv|))
        # is the true sigmoid; sigmoid_f32 is that expression, so the gate
        # is its plain f32 value with no further branch
        sig = sigmoid_f32(zv)
        out[tt] = bf(BF(normed) * sig)

    outs = {"conv_out": conv_out.reshape(-1), "window": window.reshape(-1),
            "state_out": state.reshape(-1).astype(np.float32),
            "out": out.reshape(-1)}
    if with_states:
        outs["states"] = states.reshape(-1).astype(np.float32)
    return outs


def ref_omlx_qwen4_gdn_verify_step(kern, inputs):
    return _qwen4_verify(kern, inputs, with_states=False)


def ref_omlx_qwen4_gdn_verify_step_states(kern, inputs):
    return _qwen4_verify(kern, inputs, with_states=True)


# composed-op variants: the oMLX fallback computes the conv activation
# WITHOUT the T() rounding at the SiLU branch (the one candidate site where
# the composed fallback and the assembled MSL can disagree). Kept as
# labelled secondary checks by parity_compare; the MSL side is the
# Metal-intended behaviour. The historical composed-op references also
# wrapped sigmoid_f32 in a second sign branch, re-inverting beta and the
# norm-gate sig for positive inputs; that was a transcription error (the
# composed source selects the branch once, exactly like the MSL), corrected
# in both, so the secondary isolates the SiLU rounding.
SECONDARY = {
    "omlx_qwen4_gdn_verify_step":
        lambda kern, inputs: _qwen4_verify(kern, inputs, with_states=False,
                                           silu=_silu),
    "omlx_qwen4_gdn_verify_step_states":
        lambda kern, inputs: _qwen4_verify(kern, inputs, with_states=True,
                                           silu=_silu),
}


def ref_omlx_gdn_verify_main_replay(kern, inputs):
    """Sibling main replay: keep_rows committed rows replayed into the state,
    then the T-token block, all one simdgroup per (head, dv) row with
    simd_sum read as the sequential f32 lane sum.

    Re-derived from the assembled sibling MSL: its gdn_decay / gdn_beta are
    the composed softplus / sigmoid rounded at exactly the sites this
    reference rounds (T(s), T(exp(-|s|)), T(log1p term), T(sp), beta rounded
    to T once; lo - hi is exact because one operand is zero), so the MSL and
    composed semantics coincide op for op here and one reference is both;
    there is no separate composed-op secondary for this kernel. The
    remaining freedoms are precise::exp / precise::log vs numpy (f32,
    inside the table tolerances) and the simd_sum lane order."""
    t = _tmpl(kern)
    hk, hv, dk, dv = int(t["Hk"]), int(t["Hv"]), int(t["Dk"]), int(t["Dv"])
    tok, per = int(t["T"]), int(t["P"])
    nk = dk // 32
    st = inputs["state_in"].reshape(hv, dv, 32, nk).astype(np.float32)
    keep = int(inputs["keep_rows"].reshape(-1)[0])
    b_idx = 0  # grid z == Hv: one batch row at the parity geometry
    pk = BF(inputs["pk"]).reshape(per, hk, dk)
    pv = BF(inputs["pv"]).reshape(per, hv, dv)
    pa = BF(inputs["pa"]).reshape(per, hv)
    pb = BF(inputs["pb"]).reshape(per, hv)
    q = BF(inputs["q"]).reshape(tok, hk, dk)
    k = BF(inputs["k"]).reshape(tok, hk, dk)
    v = BF(inputs["v"]).reshape(tok, hv, dv)
    a = BF(inputs["a"]).reshape(tok, hv)
    bb = BF(inputs["b"]).reshape(tok, hv)
    hk_of = np.arange(hv) // (hv // hk)

    def pass_(g, beta, keys, vals, readout):
        """One recurrence pass: per token, decay, full-lane key match
        (simd_sum as the sequential f32 lane sum), delta against v, state
        update, optional q readout rounded to InT on the store."""
        nonlocal st
        y = np.zeros((g.shape[0], hv, dv), np.uint16) if readout else None
        for tt in range(g.shape[0]):
            st *= g[tt][:, None, None, None]
            kp = keys[tt][hk_of].reshape(hv, 1, 32, nk)
            kv = _seqsum(_seqsum(st * kp, -1), -1)
            delta = (vals[tt] - kv) * beta[tt][:, None]
            st += kp * delta[..., None, None]
            if readout:
                acc = _seqsum(_seqsum(st * q[tt][hk_of].reshape(hv, 1, 32, nk),
                                      -1), -1)
                y[tt] = bf(acc)
        return y

    dtb = BF(inputs["dt_bias"].reshape(-1))
    a_log = BF(inputs["A_log"].reshape(-1))
    # replay of the committed rows, state committed to state_out after it
    g = _decay(pa[:keep], dtb, a_log)
    pass_(g, BF(_beta(pb[:keep])), pk, pv, readout=False)
    state_out = st.copy()

    # the block itself
    g = _decay(a, dtb, a_log)
    y = pass_(g[:tok], BF(_beta(bb[:tok])), k, v, readout=True)
    return {"y": y.reshape(-1), "state_out": state_out.reshape(-1)}


def ref_omlx_chain_attn_partial(kern, inputs):
    """Sibling chain partial: online-softmax attention over n_splits k/v
    chunks plus the kt/vt tail (split == n_splits), one threadgroup per
    (query head, split), 8 simdgroups striding tokens by SGN = 8, then the
    threadgroup merge the MSL writes (w_j = 0 for tm[j] == -inf). simd_sum
    is the sequential f32 lane sum; every other op is f32."""
    t = _tmpl(kern)
    g_n, h_n = int(t["G"]), int(t["H"])
    d, sgn = 256, 8
    params = np.asarray(inputs["params"]).reshape(-1)
    p_len, tn = int(params[0]), int(params[1])
    chunk, n_splits = int(params[2]), int(params[3])
    scale = np.asarray(params[4:5]).view(np.float32)[0]
    ks = np.asarray(inputs["k_strides"]).reshape(-1)
    vs = np.asarray(inputs["v_strides"]).reshape(-1)
    qv = BF(inputs["q"]).reshape(h_n, 32, 8) * scale
    krows = BF(inputs["k"]).reshape(-1, d)
    vrows = BF(inputs["v"]).reshape(-1, d)
    ktrows = BF(inputs["kt"]).reshape(-1, d)
    vtrows = BF(inputs["vt"]).reshape(-1, d)
    h_of = np.arange(h_n) // g_n

    n_tot = n_splits + 1
    o_part = np.zeros((n_tot, h_n, d), np.float32)
    ml_part = np.zeros((n_tot, h_n, 2), np.float32)
    for split in range(n_tot):
        if split < n_splits:
            begin = split * chunk
            end = min(begin + chunk, p_len)
            kb, vb = krows, vrows
            hrow, vrow = (ks[1] // d) * h_of, (vs[1] // d) * h_of
        else:
            begin, end = 0, tn
            kb, vb = ktrows, vtrows
            hrow = vrow = tn * h_of
        toks = np.arange(begin, end)
        m_sg = np.full((sgn, h_n), -np.inf, np.float32)
        l_sg = np.zeros((sgn, h_n), np.float32)
        ta = np.zeros((sgn, h_n, 32, 8), np.float32)
        if toks.size:
            kblk = kb[hrow[:, None] + toks[None, :]].reshape(
                h_n, toks.size, 32, 8)
            vblk = vb[vrow[:, None] + toks[None, :]].reshape(
                h_n, toks.size, 32, 8)
            s = _seqsum(_seqsum(qv[:, None, :, :] * kblk[h_of], -1), -1)
            for j in range(sgn):
                if j >= toks.size:
                    continue
                m = np.full(h_n, -np.inf, np.float32)
                l = np.zeros(h_n, np.float32)
                acc = np.zeros((h_n, 32, 8), np.float32)
                for pos in range(j, toks.size, sgn):
                    s_tok = s[:, pos]
                    m_new = np.maximum(m, s_tok)
                    alpha = np.exp(m - m_new)
                    p = np.exp(s_tok - m_new)
                    l = l * alpha + p
                    acc = (acc * alpha[:, None, None]
                           + p[:, None, None] * vblk[h_of, pos])
                    m = m_new
                m_sg[j], l_sg[j], ta[j] = m, l, acc
        # threadgroup merge; w_j = 0 where the simdgroup saw no tokens
        M = m_sg.max(axis=0)
        w = np.where(np.isneginf(m_sg), np.float32(0.0),
                     np.exp(m_sg - M[None, :]))
        o = np.zeros((h_n, d), np.float32)
        L = np.zeros(h_n, np.float32)
        taf = ta.reshape(sgn, h_n, d)
        for j in range(sgn):
            o = o + w[j][:, None] * taf[j]
            L = L + w[j] * l_sg[j]
        o_part[split] = o
        ml_part[split, :, 0] = M
        ml_part[split, :, 1] = L
    return {"o_part": o_part.reshape(-1), "ml_part": ml_part.reshape(-1)}


REFS = {
    "omlx_qwen4_gdn_verify_step": ref_omlx_qwen4_gdn_verify_step,
    "omlx_qwen4_gdn_verify_step_states": ref_omlx_qwen4_gdn_verify_step_states,
    "omlx_gdn_verify_main_replay": ref_omlx_gdn_verify_main_replay,
    "omlx_chain_attn_partial": ref_omlx_chain_attn_partial,
}


if __name__ == "__main__":
    # the states variant must be byte-identical to the plain step on the four
    # shared outputs (same MSL modulo the extra snapshot write)
    kern0 = pc.BY_NAME["omlx_qwen4_gdn_verify_step"]
    kern1 = pc.BY_NAME["omlx_qwen4_gdn_verify_step_states"]

    def gen(kern):
        ins = {}
        for i, spec in enumerate(kern["inputs"]):
            ins[spec[0]] = np.frombuffer(
                pc.gen_input(kern["name"], i, spec, 1234),
                pc.DTYPES[spec[1]][1] or np.uint16).reshape(spec[2])
        return ins

    a = REFS["omlx_qwen4_gdn_verify_step"](kern0, gen(kern0))
    # same bytes for both sides: gen_input seeds by kernel name, but the two
    # tables carry identical input specs and the MSLs differ only in the
    # extra states write
    assert ([(s[0], s[1], s[2]) for s in kern0["inputs"]]
            == [(s[0], s[1], s[2]) for s in kern1["inputs"]])
    b = REFS["omlx_qwen4_gdn_verify_step_states"](kern1, gen(kern0))
    for name, arr in a.items():
        assert np.array_equal(arr, b[name]), f"{name} diverges between the pair"
    s = int(_tmpl(kern1)["S"])
    flat = b["states"].reshape(s, -1)
    assert not flat[s - 1].any(), "states[S-1] must stay the documented zero hole"
    assert flat[:s - 1].any(), "states snapshots are all zero"
    print(f"ok: verify pair agrees on {len(a)} outputs; "
          f"states[0:{s - 1}] written, states[{s - 1}] is the zero hole")
