# metal2vk

Run Metal GPU kernels on Linux: metal2vk translates Apple's Metal compute code to Vulkan with open source tools.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink)](https://github.com/sponsors/joshuaswarren)

Sponsor this work or help stock the lab: [joshuawarren.com/sponsors](https://joshuawarren.com/sponsors)

The fast local AI software for Apple Silicon gets its speed from GPU kernels. That software includes oMLX, mlx-lm, mlx-vlm and the uzu and TensorFold engines. The kernels are written in Metal, Apple's GPU language. Metal only exists on Apple's operating systems. This project reads those kernels as they are. It translates them to Vulkan with clang, clspv and SPIRV-Tools, with no Apple software. They can then run on Apple Silicon under Linux through an open source Vulkan driver.

## What works today

Measured, with the account of how in [docs/what-this-enables.md](docs/what-this-enables.md):

| Measured | Result |
|---|---|
| Kernels from uzu and TensorFold that compile to valid Vulkan programs | 610 of 610 (606 by default, 4 in an opt-in 64-bit mode) |
| uzu kernels that compile | 121 of 121 |
| oMLX, mlx-lm, mlx-vlm kernels sampled: compile | 31 of 32 |
| Same kernels, run on an M1 and checked against a reference | 31 of 31 agree (5 values of about 9,200 differ by 2 to 3 units in the last place in one kernel; about 0.4% of the values in one attention kernel differ by at most 0.02% of their row's largest value) |
| Refused | 1 kernel needs the M5 chip's tensor operations |
| Simple tiled matrix multiply, M1 Max | 4.27 TFLOP/s (hand-tuned Vulkan: 4.07) |
| uzu tiled matrix multiply, M1 Max | about 0.3 times the hand-tuned kernel |
| Translated kernels against MLX's own kernels on the same GPU, M2 Max, two runs (Mesa git 6543eeb7df7, load 1.30 at the start) | softmax f32 takes 0.75 and 0.76 of the MLX time, softmax f16 0.91 and 0.92. The 4x4 register-tiled matrix multiply, 1024³, takes 1.28 and 1.22 times the MLX time (4.77 and 4.76 TFLOP/s against 6.11 and 5.80). The SiLU kernel takes 0.44 and 0.45 of the unfused MLX time in f32 and 2.0 times the fused MLX time in f16. The runs differ by 1.2% or less on softmax and by 5% on the MLX matmul time |
| Compile time per kernel | 1.8 s first time, 0.8 s cached |
| Whole-model speed through this route | not measured yet |

A kernel that fails to translate or fails its check is never run. It falls back to the older route or stops with an error that names the kernel. The reference for a check is the original translator's output where it had one. Otherwise it is a step-by-step Python model of the Metal source.

## Why it matters

Linux gets the Mac's kernels. The speed of these engines is in their custom kernels. Translating the Metal source avoids rewriting each kernel in another language.

The long tail works too. The models that depend on custom kernels are the ones a plain port misses. They are Qwen3.5-class mixture-of-experts and gated-delta-net models, served through oMLX, mlx-lm and mlx-vlm, and the uzu and TensorFold engines.

It is open end to end. The pieces are clang, clspv, SPIRV-Tools and Mesa's Vulkan driver for Apple silicon. Nothing from Apple is needed.

## Quick start

You need Linux, `git`, `python3` with `numpy`, `cmake` and `ninja`, clang and LLVM 23 (with C++ for OpenCL), SPIRV-Tools, and the Vulkan loader and headers. The headers must be 1.3.255 or newer, the first that define `VK_KHR_cooperative_matrix`, which `m2v-run.c` uses. Running (the last step) needs a Vulkan 1.3 GPU. Compiling does not.

**0. Get the code.** Everything below runs from one new directory:

```bash
mkdir m2v && cd m2v
git clone https://github.com/joshuaswarren/metal2vk
git clone https://github.com/google/clspv
git clone https://github.com/trymirai/uzu && git -C uzu checkout c92723bdc04e40f55286de458a457172ce62c5b6
```

**1. Build the patched clspv.** metal2vk uses clspv at a pinned commit plus the patches in `metal2vk/clspv-patches/`. From `m2v/`:

```bash
cd clspv
git checkout f2b01dd6dce7058585eb4bcdb5326997cfeebfd4
python3 utils/fetch_sources.py --shallow
../metal2vk/tools/apply-clspv-patches.sh .
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLSPV_BUILD_TESTS=OFF -B build && ninja -C build clspv
cd ../metal2vk
```

The build compiles LLVM, so it takes a while. You are back in `m2v/metal2vk/` afterwards.

**2. Translate a kernel.** Take uzu's activation and softmax kernels, unmodified. From `m2v/metal2vk/`:

```bash
export UZU_ROOT=$PWD/../uzu CLSPV=$PWD/../clspv/build/bin/clspv CLANG=clang-23 OPT=opt-23   # your clang 23 and opt 23
./compile.sh out activation softmax
```

The folder `out/` now holds `activation.spv` and `softmax.spv`. They are Vulkan 1.3 SPIR-V, checked by `spirv-val`. Next to each is a `.json` that says which buffer is which argument. You also get the LLVM IR and the clspv log.

**3. Run it** on a machine with a Vulkan 1.3 GPU. One example is Apple Silicon under Linux with Mesa's Honeykrisp driver. The script compares the result with a CPU reference. Still in `m2v/metal2vk/`:

```bash
cc -O2 m2v-run.c -lvulkan -o m2v-run
python3 m2v-test.py results ./m2v-run out activation softmax     # writes results/results.json
```

### Translate one Metal file

After the toolchain setup above, compile your own assembled `kernel.metal` from the repository root:

```bash
tools/m2v-compile --msl kernel.metal --out out/single --cache .cache/m2v-compile --json
```

The file must contain one kernel entry. For multiple entries, add `--name` with the entry's Metal host name.
Keep `CLSPV` set to an absolute executable path, as in the setup above.
Compilation and SPIR-V validation need no GPU; running the result does.

Integrators must check the exit status and use the returned `spv` and `json` paths.
A failed invocation does not remove outputs from an earlier success.
See [the CLI reference](docs/m2v-compile.md) for reflection, cache limits and subprocess deadlines.

## How it works

First, clang parses the Metal source as C++ for OpenCL. It uses a small `metal_stdlib` shim (`include/`) that gives Metal's types, address spaces, `simdgroup_matrix` and the other builtins.

Second, LLVM's inliner keeps the IR typed. A few small scripts rewrite shapes that clspv cannot lower.

Third, a patched clspv turns the IR into Vulkan 1.3 compute SPIR-V. Cooperative matrix operations use `VK_KHR_cooperative_matrix`.

Fourth, `spirv-val` checks the module. A pass that had to give up makes the build fail instead of keeping wrong code.

Fifth, the module runs on any Vulkan 1.3 device with 32-wide subgroups, storage buffers and push constants. The target is Honeykrisp, Mesa's driver for Apple silicon.

See [docs/how-it-works.md](docs/how-it-works.md) for the details and the reasons for each choice.

## Limits

Every kernel in the corpus translates: 606 of 610 by default and the last four in an opt-in 64-bit mode. See [docs/coverage.md](docs/coverage.md) for how it got there, by cause. The four are the kimi experts kernels, which read a table of raw GPU addresses. The mode is `M2V_PSB_CASES`. It compiles the named kernels as 64-bit modules with physical storage buffer pointers. Those four are valid SPIR-V only. None has run on a GPU. A host has to pass real device addresses to them, from `vkGetBufferDeviceAddress`.

Translated is not the same as correct. A kernel was checked only if it is in the 31 above. Where Metal leaves the order of floating-point sums or the rounding of `exp` open, results can differ in the last bits.

Matrix multiplies are slower. uzu's tiled matrix multiply runs at about 0.3 times the hand-tuned Vulkan kernel. Matrix multiplies therefore keep using hand-tuned kernels. The translator covers the many smaller ones.

Newest-chip features need care. One sampled kernel uses a Metal feature that only exists on the M5 chip. It is refused by name. Metal 4's tensor operations are emulated in the shim for the matrix layout that MLX, uzu and TensorFold pack for. A kernel written for Apple's other (strict) layout translates but returns permuted results. The shim header says which.

Cooperative matrix needs a driver that offers it. Stock Mesa Honeykrisp does not yet. The [omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx) fork does.

There is no whole-model speed number yet. Kernel speed is measured against MLX's own kernels on the same GPU. It has not been measured against native Metal. That needs the same kernel timed under macOS on the same chip. Tokens per second through this route is not measured.

## Status and roadmap

The translator works and has a growing, gated test. CI runs the whole uzu and TensorFold sweep on every pull request. It fails if a kernel that used to translate stops translating.

Next is whole engines on Linux, starting with oMLX. That route is merged in omarchy-mlx (pull request 70). It is opt-in through an environment variable. The default stays the old translator. There is no tokens-per-second number for it yet.

The `host/` directory holds the host layer (buffers, queues, pipelines). It gives Metal-style host code a Vulkan backend.

## Documentation

Read [What metal2vk enables](docs/what-this-enables.md) for the measured results and what is not measured yet. [How it works](docs/how-it-works.md) covers the pipeline, the shim and the clspv patches. [Kernel measurements](docs/measurements.md) and [coverage by failure class](docs/coverage.md) have the numbers. [m2v-compile](docs/m2v-compile.md) describes the command line front door and its reflection format. The [host layer](host/DESIGN.md) and the [bench harness](docs/bench-status.md) are documented too.

## Contributing

Issues with a failing kernel and its first error message are the most useful thing you can send. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to run the sweep and the rules for shim and clspv changes.

## Support

Every bit of support helps keep metal2vk alive and free. If you are able, [sponsor on GitHub](https://github.com/sponsors/joshuaswarren) or send a Lightning donation to `joshuaswarren@strike.me` to directly fund continued development and new integrations. If financial support is not an option, [star the repo](https://github.com/joshuaswarren/metal2vk), share it, or recommend it to a colleague.

## License

MIT. See [LICENSE](LICENSE).
