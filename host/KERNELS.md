# host/KERNELS.md - uzu kernel translation (vulkan-host pipeline)

Run 7 of OUT_DIR/m2v-kernels.json: **45 translated / 716 failed / 761 total**

A kernel left untranslated is absent from its .m2vlib: creating its pipeline fails at runtime with the kernel name. Policy-only unit tests do not need the kernels.

## Why entry wrappers

`sweep/m2v_sweep.py` on main gets 106 of 121 uzu entry points through clang + clspv + spirv-val; the bindgen chain adopted its per-entry layout. The mechanism, as it landed in `toolchain_m2v.rs`:

1. One kernel variant per translation unit (`split_variants`): the preprocessed front text plus one DSL wrapper. A variant that fails cannot hide its siblings, and no variant's body text leaks into the next variant's unit (the old fallback carried each chunk's body into the following chunk, putting `__local` declarations at file scope: 241 failures in the round-1 baseline).
2. The front source is preprocessed once per shard (`clang -E -P -dD` against the kernel mirror; Metal system headers stubbed empty via the sweep's stub policy, the shim `-include`d per unit). Every header is baked in exactly once, so differing include spellings can no longer double-define a type through mirror-vs-original resolution (338 failures in the baseline). Mirror-only include roots keep the `..`-laden spellings inside the mirror. `-dD` keeps the front's own `#define` lines alive: the DSL wrapper bodies pasted after the text reference the front's variant macros (`AT`, `GEMM_TGA_ELEMENTS`).
3. Function-constant slots stay lowered to `__m2v_spec_N` globals (the patched clspv makes them real SpecConstants), keyed by uzu's function-constant indices; the sweep's "SPECIALIZE as uint args" spelling is NOT used, so the kernel ABI uzu's generated Rust passes is unchanged.
4. Repeated slot blocks and typed wrapper globals in the footer are emitted once (exact-repeat dedupe): the float and half spellings of one kernel share the same `[[function_constant(N)]]` lines, and a second copy is a file-scope redefinition (47 failures in the baseline). The typed wrappers (`constant <T> __dsl_typed_<x> = <T>(expr);`) become a raw uint constant plus a constructing macro, and the DSL constexpr constructors lose their `thread` qualifier in the translate-only copy: this front end can neither construct a thread-qualified struct in the constant address space nor bind a constant object to a thread-only copy constructor.
5. `unsigned int`/`unsigned long` slot spellings parse like their fixed-width aliases (16 failures in the baseline), and the ThreadContext initializer reads the `__dsl_simd_groups_per_threadgroup` local it actually declares (8 failures).

Round-by-round on the CT (all with the sweep lane's patched clspv snapshot): baseline whole-shard pipeline 23/761, per-entry units 53/761, chunk-head fix round 36/761 (regression: shim content inlined twice through the `-I` shim root; fixed by stubbing every shim `metal_*`/`simd*` header), stub policy 43/761, `-dD` 45/761.

## Failure reasons (top 5)

- 235 × `no matching function for call to 'load_paired_vectors'`
- 144 × `clang frontend command failed with exit code 139`
- 137 × `no matching member function for call to 'decode'`
- 111 × `overriding the module target triple with spirv32-unknown-vulkan`
- 20 × `cannot convert 'const __global I4'`

`matmul/gemv/gemm.metal` (the `decode` class) does not translate on main's sweep either; `gdn/conv_scan.metal`'s 111 are clspv crashing (signal 11, no diagnostic) on the lowered IR - the build script now reports "clspv killed by a signal without a diagnostic" for that class (fix landed after run 7). The `load_paired_vectors` and `vec_sel` classes are device/bfloat address-space conversions in the DSL templates that this clang OpenCL front end rejects; they are the next wall, not regressions.

## Per-source counts

| source | translated | failed |
| --- | --- | --- |
| --- | --- | --- |
| `activation/activation.metal` | 0 | 3 |
| `activation_transform/activation_transform.metal` | 0 | 4 |
| `activation_transform/trellis.metal` | 6 | 0 |
| `attention/ancestor_attention.metal` | 0 | 1 |
| `attention/attention_fallback.metal` | 0 | 4 |
| `attention/attention_gemm.metal` | 0 | 14 |
| `attention/attention_prepare.metal` | 0 | 1 |
| `attention/attention_single_pass.metal` | 0 | 8 |
| `attention/attention_two_pass.metal` | 0 | 16 |
| `attention/gemm_grouped/attention_gemm_grouped.metal` | 0 | 4 |
| `attention/kv_cache_update.metal` | 2 | 0 |
| `attention/qkv_norm.metal` | 0 | 8 |
| `attention/sigmoid_gate.metal` | 2 | 0 |
| `convolution/separable_causal_conv.metal` | 0 | 1 |
| `embedding/input_embedding_lookup.metal` | 0 | 2 |
| `gated_act_mul/gated_act_mul.metal` | 0 | 2 |
| `gdn/chunked/a_diag_inv.metal` | 2 | 0 |
| `gdn/chunked/causal_inv.metal` | 0 | 4 |
| `gdn/chunked/cumsum.metal` | 2 | 0 |
| `gdn/chunked/gram.metal` | 0 | 2 |
| `gdn/chunked/output_and_state.metal` | 0 | 16 |
| `gdn/conv_scan.metal` | 0 | 2 |
| `gdn/conv_update.metal` | 0 | 2 |
| `gdn/norm_gate.metal` | 2 | 0 |
| `gdn/prefill.metal` | 2 | 0 |
| `gdn/prefill_prep.metal` | 0 | 4 |
| `gdn/tree_verify/conv_scan.metal` | 0 | 2 |
| `gdn/tree_verify/out.metal` | 0 | 12 |
| `gdn/tree_verify/prefix.metal` | 1 | 0 |
| `gdn/tree_verify/state_advance.metal` | 0 | 2 |
| `gdn/tree_verify/tree_gram.metal` | 0 | 4 |
| `gdn/tree_verify/tree_update_solve.metal` | 0 | 4 |
| `gdn/update.metal` | 2 | 0 |
| `logit_transform/logit_transform.metal` | 0 | 2 |
| `matmul/gemm/gemm.metal` | 0 | 334 |
| `matmul/gemm/gemm_split_k_reduce.metal` | 0 | 2 |
| `matmul/gemm/gemm_trellis.metal` | 0 | 4 |
| `matmul/gemv/gemv.metal` | 0 | 187 |
| `normalization/normalization.metal` | 0 | 27 |
| `pooling/pooling.metal` | 3 | 0 |
| `radix_top_k_small.metal` | 0 | 2 |
| `sampling/context_ring_update.metal` | 1 | 0 |
| `sampling/repetition_penalty.metal` | 2 | 0 |
| `sampling/unified_sampling.metal` | 0 | 2 |
| `short_conv/short_conv.metal` | 3 | 9 |
| `softmax/softmax.metal` | 0 | 3 |
| `ssm/conv1d.metal` | 4 | 6 |
| `ssm/split_inproj.metal` | 3 | 0 |
| `ssm/ssd_prefill.metal` | 6 | 0 |
| `ssm/ssd_update.metal` | 0 | 3 |
| `tensor_add_bias/tensor_add_bias.metal` | 0 | 9 |
| `tensor_add_scale/tensor_add_scale.metal` | 0 | 3 |
| `weaver/weaver_frontier_insert_children.metal` | 1 | 0 |
| `weaver/weaver_frontier_select.metal` | 1 | 0 |
| `weaver/weaver_top_children.metal` | 0 | 1 |

