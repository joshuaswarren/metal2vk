#!/usr/bin/env python3
"""Shared core of the sibling-kernel parity harness: the kernel table, seeded
input generation, npy exchange (bfloat16 as raw uint16), output comparison and
the parity.tsv table.

Two entry scripts use it:
  parity_mlx.py     run each kernel through omarchy-mlx mx.fast.metal_kernel
                    (host with the mlx wheel; reached through a guard ticket)
  parity_compare.py run each compiled module through m2v-run and compare
                    against the mlx outputs (host with the runner)

Every output path comes from the environment; nothing here is host-specific.
"""
import json
import os
import pathlib
import struct
import zlib

import numpy as np

# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------
# code: (itemsize bytes, numpy dtype or None for bfloat16)
DTYPES = {
    "bf16": (2, None),          # stored/loaded as uint16 bit patterns
    "f32": (4, np.float32),
    "f16": (2, np.float16),
    "u32": (4, np.uint32),
    "i32": (4, np.int32),
    "i64": (8, np.int64),
}
# relative tolerance per float dtype; integers compare exactly
TOL = {"bf16": 1e-2, "f32": 1e-5, "f16": 1e-3}


def bf16_round(x):
    """float32 array -> bfloat16 bit patterns (uint16), round-to-nearest-even."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# kernel table
# ---------------------------------------------------------------------------
# One entry per assembled kernel (32). Fields:
#   group/name       .metal file is f"{group}/{name}.metal" in the kernel root
#   grid, tg         launch geometry used by BOTH sides (what parity runs at)
#   launch_grid      geometry recorded in the artifact .launch, when the parity
#                    grid deviates from it (notes explain every deviation)
#   tmpl             [(name, value)] template arguments; dtype names as codes
#   inputs/outputs   (name, dtype-code, shape, gen) with gen one of
#                      "rand"     uniform [-1, 1) in the buffer dtype
#                      "bits"     random uint32 words (packed 4-bit weights)
#                      "idx:N"    uniform [0, N) (expert/index buffers)
#                      "logi"     router logits kept inside the small expert
#                                 pool: first 4 entries random, rest -1000
#                      "c:[...]"  exact values (scalars, strides, params)
#   scalar buffers ("c:[v]") are 1-element ref/constant buffers unless the
#   shape says otherwise; "fbits" packs a float value as its uint32 bits.
#   cpu_ref          numpy reference for kernels simple enough to check even
#                    when the omarchy path refuses them
#   m2v_only_note    why the kernel has no mlx-side comparison partner

def _fbits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


SCALE = _fbits(0.0625)
QSCALE = 128 ** -0.5
E4 = 4  # small expert pool: weights carry 4 experts, routing stays inside it

GATE_UP_W = ("w", "u32", (E4 * 2 * 512, 256), "bits")
GATE_UP_SB = [("scales", "bf16", (E4 * 2 * 512, 64), "rand"),
              ("biases", "bf16", (E4 * 2 * 512, 64), "rand")]
SHARED = [("sg_w", "u32", (512, 256), "bits"), ("sg_s", "bf16", (512, 64), "rand"),
          ("sg_b", "bf16", (512, 64), "rand"), ("su_w", "u32", (512, 256), "bits"),
          ("su_s", "bf16", (512, 64), "rand"), ("su_b", "bf16", (512, 64), "rand"),
          ("g_w", "u32", (1, 256), "bits"), ("g_s", "bf16", (1, 64), "rand"),
          ("g_b", "bf16", (1, 64), "rand")]
DOWN_W = [("w", "u32", (E4 * 2048, 64), "bits"), ("scales", "bf16", (E4 * 2048, 16), "rand"),
          ("biases", "bf16", (E4 * 2048, 16), "rand"), ("sd_w", "u32", (2048, 64), "bits"),
          ("sd_s", "bf16", (2048, 16), "rand"), ("sd_b", "bf16", (2048, 16), "rand")]
PARAMS_WIDE = ("params", "u32", (5,), f"c:[2048,4,512,4,{SCALE}]")
PARAMS_GQA = ("params", "u32", (5,), f"c:[2048,4,256,8,{SCALE}]")
K_STRIDES = ("k_strides", "i64", (4,), "c:[1048576,524288,256,1]")
V_STRIDES = ("v_strides", "i64", (4,), "c:[1048576,524288,256,1]")
# the signature interleaves strides after their input: q, k, k_strides, v, v_strides

GDN_HDR = ["T", "HK", "HV", "DK", "DV", "C"]


def K(name, group, grid, tg, tmpl=(), inputs=(), outputs=(), launch_grid=None,
      cpu_ref=None, note=None):
    return dict(name=name, group=group, grid=tuple(grid), tg=tuple(tg),
                tmpl=list(tmpl), inputs=[list(i) for i in inputs],
                outputs=[list(o) for o in outputs],
                launch_grid=tuple(launch_grid) if launch_grid else None,
                cpu_ref=cpu_ref, note=note)


KERNELS = [
    # ---- sibling: MoE router ----
    K("omlx_qwen35_moe_router_topk", "sibling", (32, 1, 1), (32, 1, 1),
      [("T", "bf16"), ("NE", 512), ("K", 8)],
      [("probs", "bf16", (1, 512), "rand")],
      [("indices", "u32", (1, 8)), ("scores", "bf16", (1, 8))],
      cpu_ref="router_topk"),
    K("omlx_qwen35_moe_router_gemv", "sibling", (32, 128, 1), (32, 4, 1),
      [("T", "bf16"), ("K", 2048), ("N", 512), ("M", 1), ("NSG", 4)],
      [("x", "bf16", (1, 2048), "rand"), ("w", "bf16", (512, 2048), "rand")],
      [("y", "bf16", (1, 512),)],
      launch_grid=(32, 512, 1),
      note="artifact grid over-launches 4x (s = tg.y*NSG covers 2048 experts "
           "for 512 weight rows); parity runs the kernel-consistent 128"),
    K("omlx_qwen35_moe_router_softmax_topk_row", "sibling", (32, 1, 1), (32, 1, 1),
      [("T", "bf16"), ("NE", 512), ("K", 8)],
      [("logits", "bf16", (1, 512), "rand")],
      [("indices", "u32", (1, 8)), ("scores", "bf16", (1, 8))],
      cpu_ref="softmax_topk"),
    K("omlx_qwen35_moe_router_softmax_topk_rows", "sibling", (32, 4, 1), (32, 1, 1),
      [("T", "bf16"), ("NE", 512), ("K", 8)],
      [("logits", "bf16", (4, 512), "rand")],
      [("indices", "u32", (4, 8)), ("scores", "bf16", (4, 8))],
      cpu_ref="softmax_topk"),
    K("omlx_qwen35_moe_combine_row", "sibling", (2048, 1, 1), (256, 1, 1),
      [("T", "bf16"), ("K", 8), ("H", 2048)],
      [("routed", "bf16", (8, 2048), "rand"), ("scores", "bf16", (8,), "rand"),
       ("shared", "bf16", (2048,), "rand"), ("gate", "bf16", (1,), "rand")],
      [("out", "bf16", (2048,))],
      cpu_ref="combine_row"),
    # ---- sibling: fused MoE expert kernels (small 4-expert weight pool) ----
    K("omlx_qwen35_moe_gate_up_decode_b4g32f_shared_b4g32f_gate_b4g32s",
      "sibling", (32, 2306, 1), (32, 2, 1),
      [("T", "bf16"), ("K", 2048), ("NI", 512), ("RPS", 2), ("NSG", 2),
       ("TOPK", 8), ("NS", 512)],
      [("x", "bf16", (2048,), "rand"), GATE_UP_W, *GATE_UP_SB,
       ("rhs", "u32", (8,), f"idx:{E4}"), *SHARED],
      [("y", "bf16", (4609,))]),
    K("omlx_qwen35_moe_down_combine_decode_b4g32f_shared_b4g32f",
      "sibling", (32, 1024, 1), (32, 9, 1),
      [("T", "bf16"), ("K", 512), ("N", 2048), ("TOPK", 8), ("KS", 512),
       ("NPART", 9), ("RPS", 2)],
      [("x", "bf16", (4609,), "rand"), *DOWN_W,
       ("rhs", "u32", (8,), f"idx:{E4}"), ("scores", "bf16", (8,), "rand")],
      [("y", "bf16", (2048,))],
      launch_grid=(32, 9216, 1),
      note="artifact grid y=9216 writes y rows up to 18432 for a 2048-row "
           "output; parity runs the covering 1024"),
    K("omlx_qwen35_moe_gate_up_window_b4g32f_shared_b4g32f_gate_b4g32s",
      "sibling", (32, 9224, 1), (32, 2, 1),
      [("T", "bf16"), ("K", 2048), ("NI", 512), ("RPS", 2), ("NSG", 2),
       ("TOPK", 8), ("NS", 512), ("M", 4)],
      [("x", "bf16", (8192,), "rand"), GATE_UP_W, *GATE_UP_SB,
       ("rhs", "u32", (32,), f"idx:{E4}"), *SHARED],
      [("y", "bf16", (18436,))]),
    K("omlx_qwen35_moe_down_combine_window_b4g32f_shared_b4g32f",
      "sibling", (32, 2048, 1), (32, 9, 1),
      [("T", "bf16"), ("K", 512), ("N", 2048), ("TOPK", 8), ("KS", 512),
       ("NPART", 9), ("RPS", 4), ("M", 4)],
      [("x", "bf16", (18436,), "rand"), *DOWN_W,
       ("rhs", "u32", (32,), f"idx:{E4}"), ("scores", "bf16", (32,), "rand")],
      [("y", "bf16", (8192,))],
      launch_grid=(32, 18432, 1),
      note="artifact grid writes y rows up to 18428 for a 4x2048-row output; "
           "parity runs the covering 2048"),
    K("omlx_qwen35_moe_gate_up_topk_b4g32f_shared_b4g32f_gate_b4g32s",
      "sibling", (32, 2306, 1), (32, 2, 1),
      [("T", "bf16"), ("K", 2048), ("NI", 512), ("RPS", 2), ("NSG", 2),
       ("TOPK", 8), ("NS", 512), ("NE", 512), ("M", 1), ("YW", 4609)],
      [("x", "bf16", (2048,), "rand"), GATE_UP_W, *GATE_UP_SB,
       ("logits", "bf16", (512,), "logi"), *SHARED],
      [("y", "bf16", (4609,)), ("indices", "u32", (8,)),
       ("scores", "bf16", (8,))],
      note="logits outside the 4-expert weight pool are pinned to -1000 so "
           "the folded top-k stays inside the small weight pool"),
    # ---- sibling: attention verify/prefix ----
    K("omlx_gdn_sigmoid_probe", "sibling", (131072, 1, 1), (256, 1, 1), [],
      [("gates", "f32", (131072,), "rand")],
      [("out", "f32", (131072,))],
      cpu_ref="silu"),
    K("omlx_gdn_norm_gate_eps1em06", "sibling", (32, 128, 1), (32, 8, 1),
      [("InT", "bf16")],
      [("y", "bf16", (16384,), "rand"), ("z", "bf16", (16384,), "rand"),
       ("norm_w", "bf16", (128,), "rand")],
      [("out", "bf16", (16384,)), ("xs", "f32", (256,))]),
    K("omlx_gdn_verify_main_replay", "sibling", (32, 128, 32), (32, 4, 1),
      [("InT", "bf16"), ("Hk", 16), ("Hv", 32), ("Dk", 128), ("Dv", 128),
       ("T", 4), ("P", 4)],
      [("state_in", "f32", (524288,), "rand"), ("A_log", "bf16", (32,), "rand"),
       ("dt_bias", "bf16", (32,), "rand"), ("pk", "bf16", (8192,), "rand"),
       ("pv", "bf16", (16384,), "rand"), ("pa", "bf16", (128,), "rand"),
       ("pb", "bf16", (128,), "rand"), ("keep_rows", "i32", (1,), "c:[2]"),
       ("q", "bf16", (8192,), "rand"), ("k", "bf16", (8192,), "rand"),
       ("v", "bf16", (16384,), "rand"), ("a", "bf16", (128,), "rand"),
       ("b", "bf16", (128,), "rand")],
      [("y", "bf16", (16384,)), ("state_out", "f32", (524288,))],
      note="metal2vk compile fails on metal::numeric_limits<half> (t1); "
           "mlx side still runs"),
    K("omlx_verify_attn_wide_partial", "sibling", (2048, 4, 1), (128, 1, 1),
      [("T_", "bf16"), ("G", 8), ("H", 16)],
      [("q", "bf16", (32768,), "rand"), ("k", "bf16", (1048576,), "rand"),
       K_STRIDES, ("v", "bf16", (1048576,), "rand"), V_STRIDES, PARAMS_WIDE],
      [("o_part", "f32", (131072,)), ("ml_part", "f32", (1024,))],
      note="metal2vk compile fails on the simdgroup_matrix store (t1); "
           "mlx side still runs"),
    K("omlx_verify_attn_wide_combine", "sibling", (1024, 16, 1), (256, 1, 1),
      [("T_", "bf16"), ("G", 8), ("H", 16)],
      [("o_part", "f32", (131072,), "rand"), ("ml_part", "f32", (1024,), "rand"),
       PARAMS_WIDE],
      [("out", "bf16", (16384,))]),
    K("omlx_verify_attn_gqa_partial", "sibling", (2, 8, 1), (256, 1, 1),
      [("T_", "bf16"), ("G", 8), ("KVH", 2)],
      [("q", "bf16", (32768,), "rand"), ("k", "bf16", (1048576,), "rand"),
       K_STRIDES, ("v", "bf16", (1048576,), "rand"), V_STRIDES, PARAMS_GQA],
      [("o_part", "f32", (262144,)), ("ml_part", "f32", (2048,))],
      launch_grid=(512, 8, 1),
      note="metal2vk compile fails on the mpp tensor constructor (t1); "
           "artifact grid x=512 over-launches kv heads 256x; parity runs x=KVH"),
    K("omlx_verify_attn_gqa_combine", "sibling", (256, 64, 2), (256, 1, 1),
      [("T_", "bf16"), ("G", 8), ("KVH", 2)],
      [("o_part", "f32", (262144,), "rand"), ("ml_part", "f32", (2048,), "rand"),
       PARAMS_GQA],
      [("out", "bf16", (16384,))]),
    K("omlx_chain_attn_partial", "sibling", (16, 5, 1), (256, 1, 1),
      [("T_", "bf16"), ("G", 8), ("H", 16)],
      [("q", "bf16", (4096,), "rand"), ("k", "bf16", (1048576,), "rand"),
       K_STRIDES, ("v", "bf16", (1048576,), "rand"), V_STRIDES,
       ("kt", "bf16", (2048,), "rand"), ("vt", "bf16", (2048,), "rand"),
       PARAMS_WIDE],
      [("o_part", "f32", (20480,)), ("ml_part", "f32", (160,))],
      launch_grid=(4096, 5, 1),
      note="artifact grid x=heads*256 indexes q[qh*D] out of bounds for 16 "
           "heads; parity runs x=heads"),
    K("omlx_chain_attn_combine", "sibling", (256, 16, 1), (256, 1, 1),
      [("T_", "bf16"), ("G", 8), ("H", 16)],
      [("o_part", "f32", (20480,), "rand"), ("ml_part", "f32", (160,), "rand"),
       PARAMS_WIDE],
      [("out", "bf16", (4096,))]),
    # ---- gdn decode/prefill/verify ----
    K("omlx_qwen35_gdn_prework_S1", "gdn", (32, 1, 64), (32, 1, 1),
      [("T", "bf16"), ("HK", 16), ("HV", 32), ("DK", 128), ("DV", 128),
       ("NKEEP", 3), ("C", 8192), ("S", 1), ("L2", 0)],
      [("qkv", "bf16", (8192,), "rand"), ("conv_state", "bf16", (24576,), "rand"),
       ("conv_w", "bf16", (32768,), "rand"),
       ("q_scale", "bf16", (1,), f"c:[{QSCALE}]"),
       ("k_scale", "bf16", (1,), "c:[1.0]")],
      [("q_out", "bf16", (2048,)), ("k_out", "bf16", (2048,)),
       ("v_out", "bf16", (4096,)), ("conv_out", "bf16", (24576,))]),
    K("omlx_qwen4_gdn_decode_prework", "gdn", (32, 1, 64), (32, 1, 1),
      [(n, v) for n, v in zip(GDN_HDR, ["bf16", 16, 32, 128, 128, 8192])],
      [("qkv", "bf16", (8192,), "rand"), ("conv_state", "bf16", (24576,), "rand"),
       ("conv_w", "bf16", (32768,), "rand"),
       ("q_scale", "bf16", (1,), f"c:[{QSCALE}]"),
       ("b_in", "bf16", (32,), "rand"), ("a_in", "bf16", (32,), "rand"),
       ("A_log", "bf16", (32,), "rand"), ("dt_bias", "bf16", (32,), "rand")],
      [("q_out", "bf16", (2048,)), ("k_out", "bf16", (2048,)),
       ("v_out", "bf16", (4096,)), ("conv_out", "bf16", (24576,)),
       ("g_out", "f32", (32,)), ("beta_out", "bf16", (32,))]),
    K("omlx_qwen4_gdn_decode_step", "gdn", (32, 16, 32), (32, 16, 1),
      [(n, v) for n, v in zip(GDN_HDR, ["bf16", 16, 32, 128, 128, 8192])],
      [("qkv", "bf16", (8192,), "rand"), ("z", "bf16", (4096,), "rand"),
       ("b_in", "bf16", (32,), "rand"), ("a_in", "bf16", (32,), "rand"),
       ("conv_state", "bf16", (24576,), "rand"), ("conv_w", "bf16", (32768,), "rand"),
       ("q_scale", "bf16", (1,), f"c:[{QSCALE}]"),
       ("A_log", "bf16", (32,), "rand"), ("dt_bias", "bf16", (32,), "rand"),
       ("state_in", "f32", (524288,), "rand"), ("norm_w", "bf16", (128,), "rand"),
       ("eps", "f32", (1,), "c:[1e-06]")],
      [("conv_out", "bf16", (24576,)), ("state_out", "f32", (524288,)),
       ("out", "bf16", (4096,))]),
    K("omlx_qwen4_gdn_batch_decode_step", "gdn", (32, 16, 128), (32, 16, 1),
      [(n, v) for n, v in zip(GDN_HDR, ["bf16", 16, 32, 128, 128, 8192])],
      [("qkv", "bf16", (32768,), "rand"), ("z", "bf16", (16384,), "rand"),
       ("b_in", "bf16", (128,), "rand"), ("a_in", "bf16", (128,), "rand"),
       ("conv_state", "bf16", (98304,), "rand"), ("conv_w", "bf16", (32768,), "rand"),
       ("q_scale", "bf16", (1,), f"c:[{QSCALE}]"),
       ("A_log", "bf16", (32,), "rand"), ("dt_bias", "bf16", (32,), "rand"),
       ("state_in", "f32", (2097152,), "rand"), ("norm_w", "bf16", (128,), "rand"),
       ("eps", "f32", (1,), "c:[1e-06]")],
      [("conv_out", "bf16", (98304,)), ("state_out", "f32", (2097152,)),
       ("out", "bf16", (16384,))],
      note="metal2vk compile fails on threadgroup scope (t1); mlx side still runs"),
    K("omlx_qwen4_gdn_decode_norm_gate", "gdn", (32, 1, 32), (32, 1, 1),
      [("T", "bf16"), ("HV", 32), ("DV", 128)],
      [("y", "bf16", (4096,), "rand"), ("z", "bf16", (4096,), "rand"),
       ("norm_w", "bf16", (128,), "rand"), ("eps", "f32", (1,), "c:[1e-06]")],
      [("out", "bf16", (4096,))]),
    K("omlx_qwen4_gdn_prefill_prework", "gdn", (32, 64, 64), (32, 1, 1),
      [("T", "bf16"), ("HK", 16), ("HV", 32), ("DK", 128), ("DV", 128),
       ("NKEEP", 3), ("C", 8192), ("L2", 1)],
      [("qkv", "bf16", (524288,), "rand"), ("conv_state", "bf16", (24576,), "rand"),
       ("conv_w", "bf16", (32768,), "rand"),
       ("q_scale", "bf16", (1,), f"c:[{QSCALE}]"),
       ("k_scale", "bf16", (1,), "c:[1.0]"),
       ("s_len", "i32", (1,), "c:[64]")],
      [("q_out", "bf16", (131072,)), ("k_out", "bf16", (131072,)),
       ("v_out", "bf16", (262144,)), ("conv_out", "bf16", (24576,))]),
    K("omlx_qwen4_gdn_prefill_norm_gate", "gdn", (32, 64, 32), (32, 1, 1),
      [("T", "bf16"), ("HV", 32), ("DV", 128)],
      [("y", "bf16", (4096,), "rand"), ("z", "bf16", (4096,), "rand"),
       ("norm_w", "bf16", (128,), "rand"), ("eps", "f32", (1,), "c:[1e-06]")],
      [("out", "bf16", (4096,))],
      note="grid y=64 re-runs the same 32 heads (the kernel has no row "
           "index); the duplicate writes are identical"),
    K("omlx_qwen4_gdn_verify_step", "gdn", (32, 16, 32), (32, 16, 1),
      [(n, v) for n, v in zip(GDN_HDR, ["bf16", 16, 32, 128, 128, 8192])]
      + [("S", 4)],
      [("proj", "bf16", (32768,), "rand"), ("conv_state", "bf16", (24576,), "rand"),
       ("conv_w", "bf16", (32768,), "rand"),
       ("q_scale", "bf16", (1,), f"c:[{QSCALE}]"),
       ("A_log", "bf16", (32,), "rand"), ("dt_bias", "bf16", (32,), "rand"),
       ("state_in", "f32", (524288,), "rand"), ("norm_w", "bf16", (128,), "rand"),
       ("eps", "f32", (1,), "c:[1e-06]")],
      [("conv_out", "bf16", (24576,)), ("window", "bf16", (57344,)),
       ("state_out", "f32", (524288,)), ("out", "bf16", (16384,))]),
    K("omlx_qwen4_gdn_verify_step_states", "gdn", (32, 16, 32), (32, 16, 1),
      [(n, v) for n, v in zip(GDN_HDR, ["bf16", 16, 32, 128, 128, 8192])]
      + [("S", 4)],
      [("proj", "bf16", (32768,), "rand"), ("conv_state", "bf16", (24576,), "rand"),
       ("conv_w", "bf16", (32768,), "rand"),
       ("q_scale", "bf16", (1,), f"c:[{QSCALE}]"),
       ("A_log", "bf16", (32,), "rand"), ("dt_bias", "bf16", (32,), "rand"),
       ("state_in", "f32", (524288,), "rand"), ("norm_w", "bf16", (128,), "rand"),
       ("eps", "f32", (1,), "c:[1e-06]")],
      [("conv_out", "bf16", (24576,)), ("window", "bf16", (57344,)),
       ("state_out", "f32", (524288,)), ("states", "f32", (2097152,)),
       ("out", "bf16", (16384,))]),
    # ---- moe/vlm: gated delta + ragged sdpa ----
    K("qwen35_gated_delta_step", "moe", (32, 128, 32), (32, 4, 1),
      [("InT", "bf16"), ("StT", "f32"), ("Dk", 128), ("Dv", 128), ("Hk", 16),
       ("Hv", 32)],
      [("q", "bf16", (65536,), "rand"), ("k", "bf16", (65536,), "rand"),
       ("v", "bf16", (131072,), "rand"), ("g", "f32", (1024,), "rand"),
       ("beta", "bf16", (1024,), "rand"), ("state_in", "f32", (524288,), "rand"),
       ("T", "i32", (1,), "c:[32]")],
      [("y", "bf16", (131072,)), ("state_out", "f32", (524288,))]),
    K("qwen35_ragged_sdpa_1p", "moe", (1024, 32, 1), (1024, 1, 1),
      [("T", "bf16"), ("D_SIZE", 256), ("V_SIZE", 256), ("NUM_Q_HEADS", 16),
       ("NUM_KV_HEADS", 2), ("GQA_FACTOR", 8)],
      [("queries", "bf16", (8192,), "rand"), ("keys", "bf16", (262144,), "rand"),
       ("values", "bf16", (262144,), "rand"), ("pads", "i32", (2,), "c:[0,0]"),
       ("scale", "f32", (1,), f"c:[{_fbits(0.0625)}]"),
       ("k_size", "i32", (1,), "c:[256]")],
      [("out", "bf16", (8192,))]),
    K("qwen35_ragged_sdpa_2p1", "moe", (2, 2, 4), (32, 8, 1),
      [("T", "bf16"), ("D_SIZE", 256), ("V_SIZE", 256), ("NUM_Q_HEADS", 16),
       ("NUM_KV_HEADS", 2), ("GQA_FACTOR", 8), ("BLOCKS", 4)],
      [("queries", "bf16", (8192,), "rand"), ("keys", "bf16", (262144,), "rand"),
       ("values", "bf16", (262144,), "rand"), ("pads", "i32", (2,), "c:[0,0]"),
       ("scale", "f32", (1,), f"c:[{_fbits(0.0625)}]"),
       ("k_size", "i32", (1,), "c:[256]")],
      [("partials", "bf16", (32768,)), ("sums", "f32", (128,)),
       ("maxs", "f32", (128,))],
      launch_grid=(64, 8, 4),
      note="artifact grid (64,8,4) indexes 64 kv heads x 8 batches against a "
           "2-kv-head template; parity runs (KVH, B, BLOCKS)"),
    K("qwen35_ragged_sdpa_2p2", "moe", (1024, 32, 1), (1024, 1, 1),
      [("T", "bf16"), ("D_SIZE", 256), ("BLOCKS", 4)],
      [("partials", "bf16", (32768,), "rand"), ("sums", "f32", (128,), "rand"),
       ("maxs", "f32", (128,), "rand")],
      [("out", "bf16", (8192,))],
      note="BLOCKS=4 < BN=32 makes the combine loop degenerate (out is 0); "
           "the artifact template pins 4"),
]

assert len(KERNELS) == 32, len(KERNELS)
BY_NAME = {k["name"]: k for k in KERNELS}


# ---------------------------------------------------------------------------
# input generation (identical bytes on both sides)
# ---------------------------------------------------------------------------
def gen_input(kname, idx, spec, seed):
    """-> uint8 buffer bytes for one input per its table spec."""
    name, code, shape, gen = spec
    n = int(np.prod(shape))
    dt, npdt = DTYPES[code]
    # per-buffer stream: stable across processes and python versions
    kern_tag = zlib.crc32(kname.encode()) if kname else 0
    rng = np.random.default_rng([seed & 0xFFFFFFFF, idx, kern_tag])
    if code == "bf16":
        return bf16_uniform(rng, n, gen).tobytes()
    if gen == "bits":
        return rng.integers(0, 2 ** 32, n, dtype=np.uint32).tobytes()
    if gen.startswith("idx:"):
        return rng.integers(0, int(gen[4:]), n, dtype=np.uint32).tobytes()
    if gen.startswith("c:["):
        vals = [v.strip() for v in gen[3:-1].split(",")]
        arr = np.array([float(v) for v in vals])
        if npdt is None:
            return bf16_round(arr).tobytes()
        return arr.astype(npdt).tobytes()
    # "rand": uniform [-1, 1)
    if npdt is None:
        return bf16_uniform(rng, n, gen).tobytes()
    return rng.uniform(-1.0, 1.0, n).astype(npdt).tobytes()


def bf16_uniform(rng, n, gen):
    x = rng.uniform(-1.0, 1.0, n)
    if gen == "logi":
        # keep the folded top-k inside the small expert pool
        x = np.concatenate([x[:4], np.full(max(0, n - 4), -1000.0)])
    return bf16_round(x)


# ---------------------------------------------------------------------------
# npy exchange (bfloat16 travels as uint16 bits + dtype in meta.json)
# ---------------------------------------------------------------------------
def save_arr(outdir, name, code, data_bytes, shape):
    dt, npdt = DTYPES[code]
    arr = np.frombuffer(data_bytes, dtype=npdt or np.uint16)
    np.save(outdir / f"{name}.npy", arr.reshape(shape))


def load_arr(outdir, name, code, shape):
    return np.load(outdir / f"{name}.npy")  # bf16 stays uint16 bits


def bits_to_float(code, arr):
    """uint16 bf16 bits -> float32; other float dtypes pass through."""
    if code == "bf16":
        return bf16_to_f32(arr)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def compare(code, got_bits, want_bits):
    """-> dict with status match/mismatch, error magnitudes, first diff.
    Integers compare exactly; floats by relative tolerance; positions where
    BOTH sides are NaN are skipped (counted separately)."""
    n = want_bits.size
    if code in ("u32", "i32", "i64"):
        diff = got_bits != want_bits
        idx = np.flatnonzero(diff)
        return dict(status="mismatch" if idx.size else "match",
                    max_abs="", max_rel="", first=int(idx[0]) if idx.size else "",
                    n_diff=int(idx.size), n_nan=0,
                    tol="exact")
    got = bits_to_float(code, got_bits).astype(np.float64)
    want = bits_to_float(code, want_bits).astype(np.float64)
    both_nan = np.isnan(got) & np.isnan(want)
    fin = np.isfinite(got) & np.isfinite(want)
    diff = np.where(fin, got - want, 0.0)
    adiff = np.abs(diff)
    rdiff = adiff / np.where(fin, 1.0 + np.abs(want), 1.0)
    tol = TOL[code]
    bad = fin & (rdiff > tol)
    # a NaN on one side only, or unequal infinities, is a difference
    bad |= ~both_nan & ~fin & ~(got == want)
    idx = np.flatnonzero(bad)
    status = "mismatch" if idx.size else "match"
    return dict(status=status,
                max_abs=f"{adiff[fin].max():.3g}" if fin.any() else "",
                max_rel=f"{rdiff[fin].max():.3g}" if fin.any() else "",
                first=int(idx[0]) if idx.size else "",
                n_diff=int(idx.size), n_nan=int((np.isnan(got) ^ np.isnan(want)).sum()),
                tol=f"rel{tol:g}")


# ---------------------------------------------------------------------------
# cpu references for the simple kernels (used when the omarchy path refuses)
# ---------------------------------------------------------------------------
def cpu_ref(kind, kern, inputs):
    """-> {output_name: float64 numpy array}; inputs: {name: float64 array}."""
    if kind == "router_topk":
        x = inputs["probs"].reshape(-1, 512)
        idx = np.argsort(-x, axis=1)[:, :8]
        rows = np.take_along_axis(x, idx, 1)
        s = rows.sum(1, keepdims=True)
        return {"indices": idx.astype(np.float64), "scores": rows / s}
    if kind == "softmax_topk":
        x = inputs["logits"].reshape(-1, 512).astype(np.float64)
        e = np.exp(x - x.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        idx = np.argsort(-p, axis=1)[:, :8]
        rows = np.take_along_axis(p, idx, 1)
        return {"indices": idx.astype(np.float64), "scores": rows / rows.sum(1, keepdims=True)}
    if kind == "combine_row":
        acc = (inputs["routed"].reshape(8, 2048) * inputs["scores"].reshape(8, 1)).sum(0)
        g = float(inputs["gate"].reshape(-1)[0])
        sig = g / (1.0 + np.exp(-abs(g))) if g >= 0 else g * (1.0 / (1.0 + np.exp(abs(g))))
        return {"out": acc + sig * inputs["shared"].reshape(2048)}
    if kind == "silu":
        g = inputs["gates"]
        sig = np.where(g < 0, 1.0 / (1.0 + np.exp(np.abs(g))),
                       1.0 - 1.0 / (1.0 + np.exp(np.abs(g))))
        return {"out": g * sig}
    raise KeyError(kind)


# ---------------------------------------------------------------------------
# fake kernel (CPU dry run): deterministic outputs from the input bytes
# ---------------------------------------------------------------------------
def fake_outputs(kern, input_bytes):
    """-> {output name: bytes}; a stand-in for a real dispatch in dry runs."""
    outs = {}
    ins = list(input_bytes.items())
    for j, (name, code, shape) in enumerate(kern["outputs"]):
        n = int(np.prod(shape)) * DTYPES[code][0]
        src = ins[j % len(ins)][1] if ins else bytes(n)
        outs[name] = (src * (n // len(src) + 1))[:n]
    return outs


# ---------------------------------------------------------------------------
# table writer
# ---------------------------------------------------------------------------
def write_tsv(path, rows):
    cols = ["kernel", "status", "max_abs", "max_rel", "first_diff", "n_diff",
            "n_nan", "tol", "mlx_est_s", "mlx_actual_s", "run_est_s",
            "run_actual_s", "note"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def env_path(var, default):
    return pathlib.Path(os.environ.get(var, os.path.expanduser(default)))
