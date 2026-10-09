#!/usr/bin/env python3
"""Retype pointer-typed fields of argument structs to equally wide integers.

The TensorFold kimi/mla and kimi/experts kernels pass a `constant Args&` struct that
carries `device` pointers as plain members (Metal happily stores device addresses in
structs). In Vulkan SPIR-V a pointer inside a UBO/push-constant Block is only expressible
as a physical storage-buffer pointer, which clspv cannot emit on a 32-bit module, so the
producer dies on the struct itself: "POD arguments must map to structures!" (the printed
`ptr addrspace(N)` line before the silent llvm_unreachable) even when the kernel never
reads the pointer field.

When NO instruction in the module loads or stores a pointer value, every pointer field is
dead runtime data and this pass retypes it to the integer type of the same width as the
module datalayout's pointer (e-p:32:32 -> i32). Field offsets and the host-side byte
layout are unchanged, GEPs are by index with opaque pointers, and a struct with an
explicit non-zero pointer initializer or a load/store of a pointer anywhere means the
pointer is live data: the module is left untouched for clspv to reject (those entries are
the real physical-pointer limit).

Runs as a text pass between the clang/opt front end and clspv; a module without a
pointer-typed struct field is a no-op.

Usage: dead_ptr_fields.py FILE.ll
"""
import pathlib
import re
import sys

TYPE_LINE = re.compile(r"^%[\w.$-]+ = type \{")
PTR_LOAD_STORE = re.compile(r"^\s*(?:%[\w.]+ = )?(?:volatile |atomic )?(?:load|store|cmpxchg|atomicrmw) ptr\b", re.M)
DATALAYOUT = re.compile(r"-p(\d*):(\d+)")


def die(msg):
    sys.exit("dead_ptr_fields: " + msg)


def main():
    if len(sys.argv) != 2:
        die("usage: dead_ptr_fields.py FILE.ll")
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()

    if PTR_LOAD_STORE.search(text):
        return  # pointer values are live data; clspv must reject this module

    m = DATALAYOUT.search(text)
    if not m:
        return
    bits = int(m.group(1) or m.group(2))
    if bits not in (32, 64):
        return  # only rewrite what the Vulkan path can spell
    intty = f"i{bits}"

    out = []
    changed = False
    for line in text.splitlines(keepends=True):
        if TYPE_LINE.match(line) and "ptr" in line:
            new = re.sub(r"\bptr addrspace\(\d+\)", intty, line)
            new = re.sub(r"\bptr\b", intty, new)
            if new != line:
                changed = True
            line = new
        elif re.search(r"^\s*@[\w.$]+ = (?:global|constant) ", line) and "ptr" in line:
            # a static initializer for a (possibly just-retyped) struct: keep it well typed
            if not re.fullmatch(r"\s*@[\w.$]+ = (?:global|constant) %[\w.$]+ zeroinitializer.*", line):
                if re.search(r"\bptr\b", line):
                    die(f"pointer initializer outside zeroinitializer form: {line.strip()}")
        out.append(line)

    if changed:
        path.write_text("".join(out))


if __name__ == "__main__":
    main()
