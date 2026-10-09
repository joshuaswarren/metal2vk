# jw16 (Apple M1 Max, Linux) env for bench/run-all.sh; sourced by run-all.sh when the
# hostname starts with jw16. Host paths live here, not in the scripts.
export UZU_ROOT=$HOME/scratch/uzu-port/uzu
export CLSPV=${CLSPV:-$HOME/scratch/metal2vk/clspv/build/bin/clspv}
export PYTHON=${PYTHON:-$HOME/.venvs/omlx-perf16/bin/python}
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-$HOME/.local/share/coreglass/vulkan-6543eeb7df/honeykrisp_icd.aarch64.json}
