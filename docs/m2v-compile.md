# m2v-compile

`tools/m2v-compile` is the command line front door over the metal2vk pipeline: one assembled Metal kernel file in,
Vulkan SPIR-V plus a reflection description out, through a content-addressed cache. A host (omarchy-mlx's C++
kernel compiler) runs it the way it runs glslang today: `fork`/`exec`, MSL text on disk, artifacts plus one status
number back out. The compile pipeline is exactly `sweep/m2v_sweep.py`'s (`compile_entry`); the reflection is
exactly `m2v-reflect.py`'s. This tool adds selection, the cache, timeouts and the exit-code contract, nothing
else: a module it produces is byte-identical to what the sweep produces for the same entry.

## Invocation

```
m2v-compile --msl FILE.metal --out DIR [--name KERNEL_NAME] [--cache DIR] [--timeout SEC] [--json]
```

| argument | meaning |
| --- | --- |
| `--msl FILE.metal` | the complete assembled MSL, exactly as mlx generates it: header text, then one or more `[[kernel]]` functions, or a template kernel with `template [[host_name("...")]] [[kernel]] decltype(f<A, B>) f<A, B>;` instantiations |
| `--out DIR` | receives `<name>.spv` and `<name>.json` on success; created if missing; never touched otherwise |
| `--name KERNEL_NAME` | the instantiation (host name) or kernel entry to select. Without it the file must hold exactly one entry |
| `--cache DIR` | cache directory; default `$M2V_CACHE_DIR`, else `$XDG_CACHE_HOME/m2v-compile` |
| `--timeout SEC` | per-stage cap (clang, opt, clspv, spirv-val, reflection), default 120; a stage past the cap is killed and the run exits 4 |
| `--json` | one JSON object on stdout describing the result |

The tool is a Python 3 script with no dependencies beyond the toolchain the sweep already needs (`clang`, `opt`,
the patched `clspv`, `spirv-tools`). Binaries come from the same environment as the sweep: `CLANG`, `OPT`,
`CLSPV`, `SPIRV_VAL`, `M2V_OPT`, `M2V_DEFS`, `M2V_INCLUDE`.

## Outputs

On status ok, DIR holds exactly two files:

- `<name>.spv`: the SPIR-V module (SPIR-V 1.5, valid for Vulkan 1.3, `spirv-val --target-env vulkan1.3` clean).
- `<name>.json`: the reflection. `m2v-reflect.py`'s module view verbatim (capabilities, extensions,
  `uses_cooperative_matrix`, `workgroup_size_spec_constant_ids`, `push_constant_regions`, `kernels`), plus:
  - `name` and `kernel`: the selected kernel; `kernel.args` is the argument list in kernel-argument order, each
    with `ordinal`, `kind` (`storage_buffer`, `uniform_buffer`, `pod_push_constant`, `workgroup_memory`), `set`
    and `binding` for buffers, `offset` and `size` for push-constant POD, `spec_id` and `elem_size` for
    workgroup memory, and the Metal-side declaration as `metal_name`, `metal_type`, `metal_dims` (clspv drops
    unused arguments, so an ordinal may have no Vulkan slot).
  - `kernel.workgroup_size`: `[x, y, z]` when the module fixes it; when the size is spec-constant driven this is
    null and `workgroup_size_spec_constant_ids` holds the three spec ids to set at pipeline creation.
  - `toolchain`: the fingerprint that went into the cache key (see below).

A kernel name longer than 200 bytes (mlx instantiation names can be) truncates the file stem to 140 bytes plus 12
hex characters of the name's sha256; `name` in the JSON and the `--json` line always carry the full name.

## Reflection schema

Top-level fields of `<name>.json`:

| field | type | meaning |
| --- | --- | --- |
| `module` | string | the SPIR-V file this description came from, relative to the output directory |
| `name` | string | the selected kernel's name (also the output file stem) |
| `kernel` | object | the selected kernel; same object as `kernels[i]` |
| `kernels` | array | every kernel in the module; one compile produces exactly one |
| `capabilities` | string array | `OpCapability` names, e.g. `["Shader"]` |
| `extensions` | string array | `OpExtension` strings, e.g. `SPV_KHR_non_semantic_info` |
| `uses_cooperative_matrix` | bool | the `CooperativeMatrixKHR` capability is present |
| `workgroup_size_spec_constant_ids` | int[3] or null | the three spec ids for the x, y, z workgroup size when the size is spec-constant driven |
| `push_constant_regions` | object | clspv's own push-constant blocks, name to `{"offset", "size"}`; the pipeline's push-constant range must cover these plus every `pod_push_constant` argument |
| `toolchain` | object | the cache fingerprint: `clang`, `opt`, `clspv` version strings; `clspv_sha256`, `include_sha256`, `clspv_patches_sha256`, `sources_sha256` (64 hex each) |

`kernel`:

| field | type | meaning |
| --- | --- | --- |
| `name` | string | the kernel name in the module |
| `workgroup_size` | int[3] or null | the fixed `[x, y, z]` when the module declares one; null means the spec-constant path above, which is what MSL input produces (Metal has no `reqd_work_group_size`) |
| `args` | array | the kernel's Vulkan arguments in kernel-argument order, `ordinal` ascending |

Each element of `kernel.args` (fields outside an argument's kind are absent):

| field | type | kinds | meaning |
| --- | --- | --- | --- |
| `ordinal` | int | all | position in the kernel's argument order |
| `kind` | string | all | `storage_buffer`, `uniform_buffer`, `pod_push_constant`, `pod_buffer`, `workgroup_memory` |
| `set`, `binding` | int | `storage_buffer`, `uniform_buffer`, `pod_buffer` | descriptor set and binding to bind the buffer at |
| `offset`, `size` | int | `pod_push_constant` (`pod_buffer` too) | byte range inside the push-constant block (or the POD buffer) |
| `spec_id`, `elem_size` | int | `workgroup_memory` | set the spec constant `spec_id` to the element count the host passes; elements are `elem_size` bytes |
| `metal_name`, `metal_type`, `metal_dims` | string | when the ordinal maps to a declared Metal argument | the Metal-side declaration; clspv drops unused arguments, so an ordinal can lack them |

## Worked example

Input, an mlx-style kernel with storage buffers, a by-value scalar and threadgroup scratch (the 32 sibling
kernels are all storage-buffer-only: TensorFold lowers Metal threadgroup declarations to in-body storage, so no
sibling exercises all three kinds and this minimal kernel stands in; every JSON line below is a real run):

```metal
template <typename T>
[[kernel]] void block_sum(device const T* values [[buffer(0)]],
                          device T* partials [[buffer(1)]],
                          const uint count,
                          threadgroup T* scratch,
                          uint gid [[thread_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint lsize [[threads_per_threadgroup]],
                          uint2 tgid [[threadgroup_position_in_grid]]) { ... }
template [[host_name("block_sum_f32")]] [[kernel]] decltype(block_sum<float>) block_sum<float>;
```

`m2v-compile --msl block_sum.metal --out DIR --json` exits 0 and writes `block_sum_f32.spv` and
`block_sum_f32.json`:

```json
{
 "module": "block_sum_f32.spv",
 "capabilities": ["Shader"],
 "extensions": ["SPV_KHR_non_semantic_info"],
 "uses_cooperative_matrix": false,
 "workgroup_size_spec_constant_ids": [0, 1, 2],
 "push_constant_regions": {
  "PushConstantRegionOffset": {"offset": 0, "size": 12},
  "PushConstantRegionGroupOffset": {"offset": 16, "size": 12}
 },
 "kernels": [ { "...": "the one entry shown under kernel" } ],
 "name": "block_sum_f32",
 "kernel": {
  "name": "block_sum_f32",
  "args": [
   {"ordinal": 0, "kind": "storage_buffer", "set": 0, "binding": 0,
    "metal_name": "values", "metal_type": "device const float*"},
   {"ordinal": 1, "kind": "storage_buffer", "set": 0, "binding": 1,
    "metal_name": "partials", "metal_type": "device float*"},
   {"ordinal": 2, "kind": "pod_push_constant", "offset": 32, "size": 4,
    "metal_name": "count", "metal_type": "const uint"}
  ],
  "workgroup_size": null
 },
 "toolchain": {"clang": "clang version 23.1.1 (...)", "opt": "LLVM (http://llvm.org/):",
  "clspv": "LLVM (http://llvm.org/):", "clspv_sha256": "00bc...65", "include_sha256": "465c...50",
  "clspv_patches_sha256": "9557...5e", "sources_sha256": "36e3...38"}
}
```

(the only elisions are the repeated `kernels` entry and the 64-hex hashes), and how an omarchy-mlx style
dispatcher maps it:

- `values`, `partials`: descriptor set 0, bindings 0 and 1, `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`.
- `count`: push constant, uint at byte 32. The push-constant block must also cover clspv's own regions
  (`PushConstantRegionOffset` at 0..12, `PushConstantRegionGroupOffset` at 16..28), so declare 36 bytes and write
  the region words the m2v host writes (zeros for a dispatch at offset 0, else the element offsets).
- workgroup size: `workgroup_size` is null, so set spec constants with ids 0, 1, 2 to the threadgroup x, y, z via
  `VkSpecializationInfo` at pipeline creation. A module with a fixed size reports it in `workgroup_size` instead.
- the `threadgroup T* scratch` parameter: lowered to a fixed module-scope Workgroup array (4096 elements of the
  element type in this build). Nothing to bind, no spec id, and no `workgroup_memory` argument appears: attr-style
  MSL kernels never produce one. `workgroup_memory` (set spec constant `spec_id` to the element count) comes only
  from the uzu DSL route today.
- grid and threadgroup counts: from the caller, as in Metal dispatch (ceil over the threadgroup dims).

## Status and exit codes

No SPIR-V or JSON is ever written to DIR for a non-ok status, and a refusal or error caches nothing.

| exit | stdout (no `--json`) | stderr | the integrator should |
| --- | --- | --- | --- |
| 0 | nothing | nothing | load `DIR/<name>.spv` and `DIR/<name>.json` |
| 2 | nothing | `error: <reason>` | fix the call; surface the line. Observed reasons, verbatim: `error: no such file: PATH`, `error: --timeout must be positive`, `error: CLANG=TOOL is not runnable; the toolchain is the sweep's (CLANG, OPT, CLSPV, SPIRV_VAL)` (likewise `OPT=`, `CLSPV=`, `SPIRV_VAL=`), `error: file does not hold exactly one entry; pass --name (N entries or instantiations)` (also what a template with no instantiation looks like without `--name`), and under `--name`: `error: no kernel entry named 'NAME'`, `error: name 'NAME' matches N entries or instantiations`, `error: template kernel without host_name instantiation` |
| 3 | `refused: CONSTRUCT in kernel NAME` | nothing | do not retry: the kernel needs a Metal 4 tensor-op lowering that does not exist (CONSTRUCT is `dextents`, `tensor<` or `mpp::`) |
| 4 | nothing | `error: FIRST-DIAGNOSTIC (log: PATH)` | surface the diagnostic and keep PATH: the full command log of the failed compile. A stage past `--timeout` has the shape `error: TOOL timed out after N s in kernel NAME (log: PATH)` |
| 5 | nothing | `error: FIRST-VALIDATOR-LINE (log: PATH)` | same; the module is invalid. The line literally begins `error: error: line ...`: the tool's prefix plus spirv-val's own |

With `--json`, stdout carries exactly one line: `status` (`ok`, `refused`, `error`), `cache` (`hit` or `miss`),
and for ok also `name`, `key` (the cache key, short), `spv`, `json` and `seconds`; for `refused` also `refused`
(the construct) and `kernel`; for errors `exit`, `seconds`, `detail` and, where one exists, `log`. The exit code
and `status` are the machine contract; stderr is for humans.

## Calling it from C++

fork/exec with the arguments in argv, capture both pipes, switch on the exit code; the one `--json` stdout line is
the machine-readable result and stderr is the diagnostic. The tool enforces `--timeout` per stage itself, so the
caller only needs a generous wall-clock cap to guard a wedged child, and killing the child's process group takes
the whole compiler tree down. This exact program compiles and runs (g++ 12, against the tool in this repo):

```cpp
struct M2vRun { int exit_code = -1; std::string json_line, diagnostics; };

static M2vRun m2v_compile(const std::string& tool, const std::string& msl, const std::string& out_dir,
                          const std::string& cache_dir, const std::string& name,
                          int stage_timeout = 120, int wall_cap_s = 600) {
  int out_pipe[2], err_pipe[2];
  if (pipe(out_pipe) || pipe(err_pipe)) return {};
  std::vector<std::string> args = {tool, "--msl", msl, "--out", out_dir, "--cache", cache_dir,
                                   "--timeout", std::to_string(stage_timeout), "--json"};
  if (!name.empty()) { args.push_back("--name"); args.push_back(name); }
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);

  pid_t pid = fork();
  if (pid == 0) {
    dup2(out_pipe[1], STDOUT_FILENO);
    dup2(err_pipe[1], STDERR_FILENO);
    close(out_pipe[0]); close(out_pipe[1]); close(err_pipe[0]); close(err_pipe[1]);
    setpgid(0, 0); // own process group, so the cap below can kill compiler and children together
    execvp(tool.c_str(), argv.data());
    _exit(127);
  }
  close(out_pipe[1]); close(err_pipe[1]);

  M2vRun run;
  char buf[4096];
  ssize_t n;
  while ((n = read(out_pipe[0], buf, sizeof buf)) > 0) run.json_line.append(buf, n);
  while ((n = read(err_pipe[0], buf, sizeof buf)) > 0) run.diagnostics.append(buf, n);
  close(out_pipe[0]); close(err_pipe[0]);

  int status = 0;
  for (int waited = 0;;) {
    pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) break;
    if (++waited > wall_cap_s) { // kill the whole group: the tool is past its own deadline
      kill(-pid, SIGKILL);
      waitpid(pid, &status, 0);
      run.exit_code = 4;
      return run;
    }
    sleep(1);
  }
  run.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 4;
  return run;
}
```

`tool` is the path to the installed `m2v-compile` script (it needs only python3 at runtime); exit 127 from the
child means that path did not execute. On 0, load the two paths from the JSON line. On 3, blacklist the kernel
until a Metal 4 lowering lands. On 4 or 5, log `diagnostics` and the `log` path, and treat the cache key as
burned for this toolchain (the failure is cached nowhere, so a retry after a fix simply misses). The toolchain
environment the child inherits is the sweep's: `CLANG`, `OPT`, `CLSPV`, `SPIRV_VAL` (paths or PATH names),
`M2V_INCLUDE`, `M2V_OPT`, `M2V_DEFS`; give the child an explicit `M2V_CACHE_DIR` (or `--cache`) so the shared
cache lands where the host wants it. `M2V_CLSPV_TIMEOUT` is ignored: `--timeout` governs every stage.

## Cache

The key is `sha256` over: the MSL text, the requested `--name` (empty when absent), the toolchain fingerprint
(clang, opt and clspv versions, the clspv binary hash, sha256 of the `include/` tree, sha256 of `clspv-patches/`,
and one hash over every file of the MSL's own directory and its parent, which is where its project includes
resolve from), the include directory paths, and the compile flags (`M2V_OPT`, `M2V_DEFS`).

Layout under the cache directory:

```
keys/<key>/kernel.spv  kernel.json  meta.json   # one complete, atomically renamed entry per key
                                               # (plus the pipeline's src/, ll/, spv/ intermediates for debugging)
locks/<key>.lock                                   # per-key lock, one compiler at a time
logs/<key>.log                                     # full command log of a failed compile
tmp/<key-prefix>-<pid>/                            # staging, renamed into keys/ on success
```

A hit copies `kernel.spv` and `kernel.json` into DIR with no compiler run. A miss takes the per-key lock (waiting
up to `--timeout`), rechecks (the other process may have finished), compiles into a staging directory and lands
the entry with one `rename`, so two processes compiling the same key never corrupt the cache and a reader never
sees a partial entry. Changed toolchain, headers or source simply produce a new key; old entries stay until you
delete the cache directory. Two ceilings, by design: trees over 100k entries, 4096 files or 32 MiB of content are
hashed by count and newest mtime instead of content, and clocks that move file mtimes backwards invalidate
nothing (an edit that keeps every mtime on an oversized tree can go unnoticed).

## Tests

`python3 tools/tests/run.py` from the repo root. It drives the CLI as a subprocess exactly as a host would, on
generated mlx-style kernels: a plain kernel, a template kernel with two `host_name` instantiations, a Metal 4
`tensor<` refusal and a syntax error. It checks the JSON shape, the exit codes, cache hit after miss, the
per-key lock under two concurrent invocations, that failures leave the output directory empty, and that every
produced module is byte-identical to what `sweep/m2v_sweep.py --dir` compiles for the same entry. CI runs it in
the sweep job. With `M2V_SIBLING_ROOT` pointing at a TensorFold-layout checkout whose `zig/kernels/metal` holds
the 32 sibling kernels, it also runs the sibling parity block: 30 must compile, the `dextents` kernel must exit 3,
the invalid `wide_partial` module must exit 5, and every module must be byte-identical to the sweep's; it also
prints cold and warm medians for the 30.
