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

## Status and exit codes

| exit | status | meaning |
| --- | --- | --- |
| 0 | `ok` | DIR received `<name>.spv` and `<name>.json` |
| 2 | `error` | usage: missing `--msl`, bad `--timeout`, no entry by that name, ambiguous selection, a file with zero or several entries and no `--name`, a template kernel without an instantiation |
| 3 | `refused` | a Metal feature known untranslatable (Metal 4 tensor ops): stdout prints `refused: <construct> in kernel <name>` |
| 4 | `error` | compile error or timeout: the first diagnostic goes to stderr, the full command log to `<cache>/logs/<key>.log` (path in the message) |
| 5 | `error` | the module was produced but `spirv-val` rejects it |

No SPIR-V or JSON is ever written to DIR for a non-ok status, and a refusal or error caches nothing.

With `--json`, stdout carries exactly one line: `status` (`ok`, `refused`, `error`), `cache` (`hit` or `miss`),
and for ok also `name`, `key` (the cache key, short), `spv`, `json` and `seconds`; for errors `exit`, `seconds`,
`detail` and, where one exists, `log`.

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
