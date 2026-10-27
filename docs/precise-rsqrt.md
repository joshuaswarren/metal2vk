# Scalar precise::rsqrt compatibility

The old scalar `precise::rsqrt(float)` calls OpenCL `rsqrt`, which the pinned clspv lowers to `GLSL.std.450 InverseSqrt`. Its allowed approximation differs from the native Metal precise result. This is a compatibility-layer correction, not a Mesa conformance bug.

The replacement refines the inverse-square-root seed with a compensated residual and handles exceptional inputs with integer bit tests. Unqualified `rsqrt` and `fast::rsqrt` are unchanged. Vector overloads are outside this change.

## Measured scope

The independently generated corpus contains 8,366 positive normal FP32 values and 25 additional specials. It includes fixed exponent/mantissa boundaries, xorshift32 seed `0x124`, and exact-rational midpoint neighbors. `m2v_rsqrt_corpus.py` regenerates every input and keeps a mathematical oracle separate from the native-compatibility result.

The frozen native control is Apple M1, macOS 27.0.1 build 26A434, Metal language 3.0, safe math with the default floating-point-function setting. The capture contains 8,448 output words including padding; SHA256 `104d7c9bdd0881ee6e47d401d7defaa35e6d9d5b5506eab14b92fd2958547e4c`. Inputs and output contain only independently generated numbers.

Linux measurement: Apple M1 G13G B1, kernel 7.2.12-2-1-ARCH, private Honeykrisp Mesa 26.3.0-devel `e7631595df`; driver-library SHA256 `ac837b1c3aa0867258f17e7c3a33842e0989d8bcba9eff7e6fd6843028abc3b1`. clspv `f2b01dd6dce7058585eb4bcdb5326997cfeebfd4` with the metal2vk patch stack, compiler SHA256 `446c1ba731ae9cfa949a25e51ca3bbc64805348d50b11319482981308714be70`.

Old shim: 1,346 finite mismatches. Corrected shim: zero. The first old mismatch is input `0x00a72f03`, old output `0x5ee00002`, native expected `0x5ee00001`. The half-propagation fixture went from 29 mismatches to zero. Khronos validation was enabled and reported no errors in the retained run. These are prior physical measurements, not GPU tests performed by CI.

**Seed-dependent rounding boundary.** The compensated refinement is not a universally correctly rounded implementation. For inputs whose exact result lies within about 2^-47 (relative) of a rounding midpoint, its rounded residual can lose the deciding correction. At those inputs it depends on InverseSqrt already supplying a correctly rounded seed, as measured on one chip and driver. For example, input `0x017ffffe` has exact result `0x5e800001`; starting from seed `0x5e800000` or `0x5e800002`, exact-FMA host emulation of this sequence returns `0x5e800000`. The reported zero mismatches on the frozen M1/Honeykrisp corpus do not establish seed-independent correctness.

**The corpus is finite. This is not proof for every FP32 input or other hardware/driver versions.** Positive normal inputs use the refinement. Signed zeros and subnormals map to signed infinity, positive infinity to positive zero, and NaNs, negative infinity and negative normals to quiet NaN. NaN payload/sign are not compared. Negative subnormals are characterized separately.

**Do not use `--use-native-builtins=fma` with this function.** With the compiler option proposed in [#102](https://github.com/joshuaswarren/metal2vk/pull/102), use `m2v-compile --native-builtins none`, never `--native-builtins fma`; see its [Math options](m2v-compile.md#math-options). The residual requires the pinned compiler's default software FMA semantics. Native-FMA compilation missed the native control on 1,814 corpus inputs. The scope regression records that combination as unsupported; it does not claim a passing precise operation. It also verifies that unqualified and fast functions keep their existing behavior. No inference-speed claim is made.

## Checks

CPU-only, from the repository root (Python 3 and NumPy):

```sh
python3 m2v_rsqrt_corpus.py check cases/precise_rsqrt
python3 m2v_rsqrt_scoped_status_check.py
```

Build the comparison modules with the pinned toolchain. `B` is its build directory containing `bin/clspv`, `third_party/llvm/bin/{clang-24,opt}` and `third_party/SPIRV-Tools/tools/{spirv-val,spirv-dis}`; `SPIRV_AS` names the assembler. The lowering command preserves failures and uses the reviewed base shim for the old control.

```sh
B=/path/to/clspv/build SPIRV_AS=/path/to/spirv-as bash cases/precise_rsqrt/lower.sh out/rsqrt
```

On a prepared Linux Vulkan host, use the existing `m2v-run` and keep submits bounded:

```sh
python3 m2v-test.py out/rsqrt-results ./m2v-run out/rsqrt \
  precise_rsqrt_fp32_boundaries precise_rsqrt_fp32_old_fail_new_pass \
  precise_rsqrt_fp32_special_values precise_rsqrt_fp32_rounding_boundaries \
  precise_rsqrt_scoped_math_unchanged
```

Missing or changed native captures, changed input corpus, absent modules and failed comparisons cannot pass. The old-fail/new-pass case is bound to the recorded old module; it is a hardware-specific regression rather than a portable expectation that every implementation must get the old approximation wrong.

The native capture path is included in this PR. To generate another control instead of treating it as the frozen verdict, the existing `bench/native-metal/run.sh --rsqrt` captures raw uint32 output for the same runtime-loaded source. Document the new chip, OS, compiler options and hashes separately.
