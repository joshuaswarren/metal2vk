#!/bin/sh
# Apply the Linux/vulkan-host cfg swap to a detached uzu worktree.
# Usage: apply-uzu.sh /path/to/uzu-worktree
#
# The patch carries a __M2V_COMPAT__ placeholder for the compat-crate directory;
# it is substituted here with the absolute path of the compat dir that ships
# next to this script (metal2vk/host/compat). After applying, run:
#   cargo check -p uzu-engine --features metal,vulkan-host
# It proceeds until the macOS-generated kernel bindings are missing
# (include!(OUT_DIR/metal.rs)) — see metal2vk host/STATUS.md.
set -e
UZU="$1"
HOST_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPAT_DIR="$(cd "$HOST_DIR/compat" && pwd)"
test -n "$UZU" && test -d "$UZU/crates/uzu-engine" || { echo "usage: $0 <uzu worktree>" >&2; exit 1; }
test -d "$COMPAT_DIR/mtl-rs" && test -d "$COMPAT_DIR/objc2" || { echo "compat crates missing next to $HOST_DIR" >&2; exit 1; }
cd "$UZU"
git apply "$HOST_DIR/uzu-linux.patch"
sed -i "s|__M2V_COMPAT__|$COMPAT_DIR|g" Cargo.toml
echo "uzu-linux.patch applied; [patch] paths point at $COMPAT_DIR"
