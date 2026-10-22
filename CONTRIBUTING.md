# Contributing to metal2vk

Contributions are welcome: a Metal kernel that does not translate, a missing piece of the `metal_stdlib` shim, a clspv fix, a
measurement on hardware we do not have. Open an issue with the kernel (or a small excerpt) and the first error message, or open a
pull request directly.

## Set up

Follow the Quick start in the [README](README.md): a patched clspv, clang 23 with C++ for OpenCL, SPIRV-Tools, and the kernel
sources you want to test (`uzu` and `TensorFold` are the two the CI uses).

## CPU-only checks

The coverage gate tests need only Python 3. No compiler, GPU, kernel sources or extra packages are needed:

```bash
python3 -m unittest discover -s sweep/tests -p test_coverage.py -v
```

CI runs these tests in a separate job, without waiting for the compiler build.
They check floor regressions, timeouts, variant identity and floor updates. They do not check generated SPIR-V.
Only clspv timeouts get a warning. An error that merely names `timeout` still fails the gate.

## Compiler checks

`sweep/m2v_sweep.py` runs every uzu and TensorFold kernel through the whole chain and `sweep/check_coverage.py` compares the result
with `sweep/coverage-floor.txt`, the list of kernels that already produce valid SPIR-V:

```bash
export UZU_ROOT=$PWD/uzu TF_ROOT=$PWD/TensorFold CLANG=clang-23 OPT=opt-23 CLSPV=$PWD/clspv/build/bin/clspv
python3 sweep/m2v_sweep.py out --set all --jobs "$(nproc)" --tag local     # a few minutes on a multi-core machine
python3 sweep/check_coverage.py out/local.json                              # fails if a kernel in the floor stopped working
python3 sweep/check_coverage.py out/local.json --update                     # raise the floor in the PR that gains kernels
python3 sweep/report.py out/local.json --table sweep/results/table.tsv --features sweep/results/FEATURES.md --failures sweep/results/FAILURES.md
```

CI runs the same sweep on every pull request, plus a compile check of four reference kernels, and builds the patched clspv (cached).
A change that loses a kernel from the floor does not merge. [docs/coverage.md](docs/coverage.md) is the burn-down by failure class:
add or update the row for what you fixed, with the measured count.

## Rules of the road

- **Shim changes** go in `include/` (`gen-shim-extras.py` regenerates `m2v_vec_ctors.h`). Keep Metal's semantics, not the easiest
  C++: conversions, rounding and overload rules should do what Metal does.
- **clspv changes** are scripts in `clspv-patches/`: idempotent, guarded by anchors that fail loudly when the text moves, and tested
  against clspv `f2b01dd6dce7058585eb4bcdb5326997cfeebfd4`. Add the script to the list in `.github/workflows/sweep.yml`; the order
  matters, and `tools/apply-clspv-patches.sh` reads it from there. A pass that gives up must print `M2V-GIVEUP` so the sweep refuses
  the module instead of keeping wrong code.
- **A valid module is not a correct module.** If you can run it, say what you checked it against. The numbers in the README come from
  [docs/what-this-enables.md](docs/what-this-enables.md) and every claim there names its reference.
- **GPU runs are short.** Keep each submit to a few seconds; the test drivers size their batches that way.
- Commits are signed by the maintainers. Pull requests are merged with a merge commit once CI is green and the change has been
  reviewed.

## License

By contributing you agree that your contribution is released under the [MIT License](LICENSE).
