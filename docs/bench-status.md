# Bench status

Bench harness for the translated (m2v-run) kernels against the hand kernels
(omarchy-mlx, timed through mlx) on one table, plus a tuning sweep. Receipts from
the latest run are committed next to this file
(`bench/results-g13c-2026-10-09.json`, `bench/sweep-g13c-2026-10-09.json`).

## Reproduce (one command, jw16)

```
rsync -a --delete --exclude .git <checkout>/ jw16mbp1-linux:~/scratch/m2v-bench-wt/
ssh jw16mbp1-linux 'PATH=$HOME/bin:$PATH gpu-turn -m 25 -- bash ~/scratch/m2v-bench-wt/bench/run-all.sh ~/scratch/m2v-bench-out'
```

`run-all.sh` compiles the registered cases (typed-GEP route below), runs the
table (correctness + timing per row, mlx hand side included), then the sweep,
and drops `results-<device>-<date>.json` / `sweep-<device>-<date>.json` into
`bench/`. Submits inside are sized to about 2 s (same rule as m2v-test.py).
`--device g14c` only changes the output tag and the `VK_DRIVER_FILES` env
(looked up in `bench/devices.json`).

## Compile route: typed GEPs

clang's `-O2` InstCombine canonicalises float GEPs into i8 byte-offset form, and
clspv then emits hundreds of byte-wise loads (found by the coopmat lane;
independent confirmation below). Compiling with
`M2V_OPT="-O0 -Xclang -disable-O0-optnone"` keeps GEPs typed and lets clspv's own
pipeline do the optimising. Measured on jw16 (G13C) today, same device, same ICD:

| kernel | `-O2` route | typed-GEP route |
| --- | --- | --- |
| tile matmul 1024³ f32 | 11264 us (181 GFLOP/s) | 2651 us (813 GFLOP/s) |
| activation SILU f32 8M | 397 us | 336 us |

The bench table below is all typed-GEP route; `bench/sweep.py` keeps the `-O2`
route variants (`base_o2route`) so the comparison is re-measurable.

## Table (jw16, Apple M1 Max G13C, Mesa 26.3.0-devel git-6543eeb7df fork ICD, 2026-10-09)

Hand = omarchy-mlx through mlx, wall time of a batch of independent evals, best
of 3. For activation the faster of mlx's fused `nn.silu` and unfused
`x*sigmoid(x)` is the comparator; both are in the receipts. Hand-side f16 silu
varies run to run on this driver (211 us / 158.7 GB/s in the committed receipt,
148 us / 227 GB/s in a same-day rerun) — the f16 explanation below rests on the
unfused-parity measurement, which is stable (315.5 vs 330.8 us).

| kernel | translated | hand | ratio | check |
| --- | --- | --- | --- | --- |
| activation f32, 8M | 343.5 us (195 GB/s) | 739.4 us (unfused) | 0.46x | err 3.5e-7 OK |
| activation f16, 8M | 330.8 us (101 GB/s) | 211.4 us (fused) | 1.56x | err 3.8e-4 OK |
| softmax f32 4096² | 742.3 us (181 GB/s) | 770.9 us | 0.96x | err 2.4e-7 OK |
| softmax f16 4096² | 358.1 us (187 GB/s) | 410.9 us | 0.87x | err 2.4e-4 OK |
| naive tile matmul 256³ | 38.5 us (871 GFLOP/s) | 12.3 us | 3.13x | err 6.7e-7 OK |
| naive tile matmul 1024³ | 2667.8 us (805 GFLOP/s) | 530.9 us | 5.03x | err 1.3e-6 OK |
| rt2x2 256³ | 32.6 us | 21.4 us | 1.52x | err 6.7e-7 OK |
| rt4x2 256³ | 45.3 us | 21.4 us | 2.12x | err 6.7e-7 OK |
| rt4x4 256³ | 65.1 us | 21.4 us | 3.04x | err 6.7e-7 OK |
| rt2x2 1024³ | 1323.4 us (1623 GFLOP/s) | 526.0 us | 2.52x | err 1.3e-6 OK |
| rt4x2 1024³ | 1076.3 us (1995 GFLOP/s) | 526.0 us | 2.05x | err 1.3e-6 OK |
| rt4x4 1024³ | 489.6 us (4386 GFLOP/s) | 526.0 us | 0.93x | err 1.3e-6 OK |

uzu's full `Gemm` has no row: the bench has no driver for it and the clspv stage
still does not finish (parent slice's open item).

## Ratios above 1.5x, each with its measurement

**activation f16 (1.56x vs fused mlx silu, 1.05x vs unfused).** The Metal kernel
is `x*sigmoid(x)` element-wise; mlx's unfused two-op f16 runs at 106 GB/s — the
same as ours (101 GB/s), 330.8 vs 315.5 us, i.e. parity for the same op
structure. The gap is to mlx's single fused f16 kernel (158.7 GB/s). The
translated f16 path moves half the bytes of f32 in the same wall time (101 vs
195 GB/s), so it is element-throughput-bound, not bandwidth-bound; the sweep
confirms widening per-thread work does not move it (ept2 352.6 us, ept4 338.9 vs
base 333.1). Lever: vectorised 16-bit access in the lowering — pipeline work,
not bench scope.

**naive tile matmul (3.13x at 256³, 5.03x at 1024³).** The kernel is uzu's 8x8
tile driver with no operand reuse: one subgroup streams K/8 A-fragments and
K/8 B-fragments from DRAM per 8x8 output tile — 2 flops/byte. Measured bytes:
1024³ moves 1.07 GB in 2668 us = 402 GB/s, the M1 Max DRAM ceiling, so it is
bandwidth-bound by construction. The rt rows are the controlled experiment:
same pipeline, same lowering, same device, only reuse changes — rt4x4 (16
flops/byte) reaches 4386 GFLOP/s and beats the hand GEMM (0.93x). The gap is
the kernel's operand traffic, not the Metal→Vulkan stack.

**rt shapes at 256³ (1.52x / 2.12x / 3.04x).** Occupancy: one subgroup per
MRxNR block means rt4x4 launches (256/32)² = 64 threadgroups (2048 threads)
against the naive kernel's 1024 — the GPU is underfilled and the 12-21 us hand
figure is dominated by submit overhead at this size. Measured: the same shapes
at 1024³ (grid 32x32 = 1024 threadgroups) flip to 2.52x / 2.05x / 0.93x.

## Sweep winners (bench/sweep-g13c-2026-10-09.json)

| kernel | winner | note |
| --- | --- | --- |
| activation f32 8M | base (typed-GEP) | wg512 within noise (336 vs 336); ept2/ept4 no better; `-O2` route 18% slower |
| activation f16 8M | base (typed-GEP) | ept2/ept4 no better (measured above) |
| softmax f32 4096² | base | workgroup and elements/thread are baked into softmax.metal; no wrapper knob |
| tile matmul 1024³ | chains4 (`-DM2V_TM_CHAINS=4`) | 2591 vs 2651 us — marginal; the real answer is rt4x4 |

Tuning knobs are compile-time defines in the wrappers, driven by `M2V_DEFS` /
`M2V_OPT` through compile.sh; defaults reproduce the unmodified uzu mapping
exactly (`M2V_ACT_WG=256`, `M2V_ACT_EPT=1`, `M2V_TM_CHAINS=1`).
