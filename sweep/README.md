# Bulk sweep

Runs every uzu `.metal` file and every TensorFold Zig `.metal` file through the pipeline (shim, clang, clspv, spirv-val) and
writes a per-kernel table and a ranked list of the Metal features that still block kernels.

```
export UZU_ROOT=<uzu checkout> TF_ROOT=<TensorFold checkout> CLANG=clang CLSPV=<patched clspv>
python3 sweep/m2v_sweep.py OUT --set all --jobs 4 --tag run1        # compile stages; OUT/run1.json, OUT/{src,ll,spv}
python3 sweep/m2v_sweep.py OUT --dir DIR1 --dir DIR2 --tag run2     # extra flat sets: every *.metal directly in each DIR,
                                                                    # set named after the directory, includes = DIR and its parent
python3 sweep/report.py OUT/run1.json --table sweep/results/table.tsv --features sweep/results/FEATURES.md --compare BASELINE.json
python3 sweep/report.py OUT/run1.json --primary                      # rank by the first error of each entry (what blocks it first)
```

`--dir` needs no uzu or TensorFold checkout (`--set` defaults to none when `--dir` is given). An entry that fails and whose first
diagnostic names a Metal 4 tensor op (`dextents`, `tensor<`, `mpp::`) gets a `refused` row naming the construct instead of a failure:
the mpp shim emulates some of these, so only the failing diagnostic decides and entries that compile are never touched. report.py
counts refused separately from failures.

The run stage needs a GPU host and goes through the GPU queue wrapper (`gpu-job.sh` logs estimated and actual seconds):

```
cc -O2 m2v-run.c -lvulkan -o m2v-run
python3 sweep/m2v_sweep_run.py OUT/run1.json OUT m2v-run OUT/run   # one dispatch per kernel, synthetic data, a few CPU references
```

`results/` holds the table (`table.tsv`, one row per entry), the ranked features (`FEATURES.md`) and the per-entry failure classes
(`FAILURES.md`: the first clspv or spirv-val message, grouped).

The driver's docstring describes how entry points are found (TensorFold's attribute style with `host_name` template instantiations,
uzu's DSL) and how a Metal entry becomes an OpenCL kernel.

Between the front end and clspv the driver runs two IR text passes, each a no-op on modules they do not apply to:
`sweep/bfloat_to_i16.py` retypes the shim's bfloat to i16 and widens the arithmetic, including the `-inf` / `nan` /
bare-integer-constant spellings clang 23 emits; `sweep/flatten_single_member_structs.py` rewrites single-member
vector-wrapper structs (the simdgroup_matrix shape) into the member type, which clspv needs for dynamic-index arrays
of matrices. `sweep/tests/run.py` is the self-check for both (CI runs it with clspv, so a regression in a pass fails
the build).

CI (`.github/workflows/sweep.yml`) runs the compile stages on every PR and fails when an entry listed in `coverage-floor.txt` no longer has valid
SPIR-V (`check_coverage.py`). `docs/coverage.md` is the burn-down by failure class.
