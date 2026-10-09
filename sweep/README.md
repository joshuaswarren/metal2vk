# Bulk sweep

Runs every uzu `.metal` file and every TensorFold Zig `.metal` file through the pipeline (shim, clang, clspv, spirv-val) and
writes a per-kernel table and a ranked list of the Metal features that still block kernels.

```
export UZU_ROOT=<uzu checkout> TF_ROOT=<TensorFold checkout> CLANG=clang CLSPV=<patched clspv>
python3 sweep/m2v_sweep.py OUT --set all --jobs 4 --tag run1        # compile stages; OUT/run1.json, OUT/{src,ll,spv}
python3 sweep/report.py OUT/run1.json --table sweep/results/table.tsv --features sweep/results/FEATURES.md --compare BASELINE.json
python3 sweep/report.py OUT/run1.json --primary                      # rank by the first error of each entry (what blocks it first)
```

The run stage needs a GPU host and goes through the GPU queue wrapper (`gpu-job.sh` logs estimated and actual seconds):

```
cc -O2 m2v-run.c -lvulkan -o m2v-run
python3 sweep/m2v_sweep_run.py OUT/run1.json OUT m2v-run OUT/run   # one dispatch per kernel, synthetic data, a few CPU references
```

`results/` holds the table (`table.tsv`, one row per entry), the ranked features (`FEATURES.md`) and the per-entry failure classes
(`FAILURES.md`: the first clspv or spirv-val message, grouped).

The driver's docstring describes how entry points are found (TensorFold's attribute style with `host_name` template instantiations,
uzu's DSL) and how a Metal entry becomes an OpenCL kernel.
