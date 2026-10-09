#!/bin/sh
# Apply the Linux/vulkan-host cfg swap to a detached uzu worktree.
# Usage: apply-uzu.sh /path/to/uzu-worktree
# After this, `cargo check -p uzu-engine --features metal,vulkan-host` proceeds until
# the missing kernel bindings (OUT_DIR/metal.rs include) — see host/STATUS.md.
set -e
UZU="$1"
HOST_DIR="$(cd "$(dirname "$0")" && pwd)"
test -n "$UZU" && test -d "$UZU/crates/uzu-engine" || { echo "usage: $0 <uzu worktree>" >&2; exit 1; }
cd "$UZU"
# The compat crates live next to this script (../compat/... relative to metal2vk root).
M2V="$HOST_DIR/.."
git apply "$HOST_DIR/uzu-linux.patch" && echo "uzu-linux.patch applied to $UZU"
echo "next: cargo check -p uzu-engine --features metal,vulkan-host"
