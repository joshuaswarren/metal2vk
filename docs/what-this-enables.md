# What metal2vk enables

Many fast AI programs for Apple Silicon are written in Metal, Apple's GPU language, and only run on macOS. metal2vk reads
those Metal kernels and turns them into Vulkan programs with open source tools (clang, clspv and SPIRV-Tools), so they can run
on Apple Silicon under Linux through an open source Vulkan driver, with no Apple software involved. The models that depend on
these kernels are the ones whose speed comes from custom code: Qwen3.5-class mixture-of-experts models and gated-delta-net
models served through oMLX, mlx-lm and mlx-vlm, the Mirai uzu engine, and the TensorFold Zig engine.

What has been measured so far: out of 610 Metal kernels taken from the real uzu and TensorFold sources, 598 compile to valid
Vulkan programs, including all 121 uzu kernels. Of 32 sampled oMLX, mlx-lm and mlx-vlm kernels, 31 compile. These 31 were run
on an M1 and checked against a reference (the original translator's output where it had one, otherwise a step-by-step Python
model of the Metal source), and all 31 agree. Two differences stay inside what the Metal source leaves open. Five values out
of about 9,200 in one mixture-of-experts kernel differ by 2 to 3 units in the last place of a 16-bit float, which is the
order in which the GPU adds partial sums. In one attention kernel, about 0.4 percent of the values differ by at most 0.02
percent of the largest value in their row, because Metal does not fix the exact rounding of its exponential function. The
check still fails when a block of keys is skipped or the mask is shifted by one row. The remaining sampled kernel uses a
Metal feature that only exists on the newest chip (M5) and is refused by name. A kernel that fails to translate or fails its
check is never run: it falls back to the older route or stops with an error that names the kernel.

What has not been measured yet: how fast whole models run through this route. The cutover of oMLX's custom kernels to metal2vk
is being built now, so there is no tokens-per-second number for it. What is measured is kernel speed in isolation: a simple
register-tiled matrix multiply written in Metal runs at 4.27 TFLOP/s on an M1 Max through metal2vk, against 4.07 for the
hand-tuned Vulkan version; uzu's own tiled matrix multiply runs at about 0.3 times the hand-tuned one, so matrix multiplies
keep using the hand-tuned kernels and metal2vk covers the long tail of smaller kernels. Compiling a kernel takes about 1.8
seconds the first time and 0.8 seconds when cached.

| Measured | Result |
|---|---|
| Kernels from uzu and TensorFold that compile to valid Vulkan programs | 598 of 610 |
| uzu kernels that compile | 121 of 121 |
| oMLX, mlx-lm, mlx-vlm kernels sampled: compile | 31 of 32 |
| Same kernels, run on an M1 and checked against a reference | 31 of 31 agree (5 values of about 9,200 differ by 2 to 3 units in the last place in one kernel; about 0.4% of the values in one attention kernel differ by at most 0.02% of their row's largest value) |
| Refused | 1 kernel needs the M5 chip's tensor operations |
| Simple tiled matrix multiply, M1 Max | 4.27 TFLOP/s (hand-tuned Vulkan: 4.07) |
| uzu tiled matrix multiply, M1 Max | about 0.3 times the hand-tuned kernel |
| Compile time per kernel | 1.8 s first time, 0.8 s cached |
| Whole-model speed through this route | not measured yet |
