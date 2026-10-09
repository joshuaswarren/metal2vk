#!/usr/bin/env python3
"""Expand llvm.memcpy calls whose two pointer operands live in different address spaces.

clspv's ReplaceLLVMIntrinsicsPass lowers a memcpy to element load/stores only when it can
infer one element type for both sides; a copy between distinct address spaces (private
alloca <- global buffer, the shape SROA leaves behind after inlining a device-side struct
assignment) survives that pass and hits clspv's "Unexpected instruction" abort. The copies
this sweep produces are small fixed-size struct/record copies, so a byte-wise expansion is
exact and cheap: every source byte is loaded first, then stored (loads up front keep the
expansion correct even if the operands overlap; the signature says noalias, but the cost is
trivial at these sizes). Same-address-space memcpys are left for clspv, which lowers them
to OpCopyMemory itself.

Runs as a text pass between the clang/opt front end and clspv; a module without a
mixed-address-space memcpy is a no-op.

Usage: memcpy_mixed_as.py FILE.ll
"""
import pathlib
import re
import sys

CALL = re.compile(
    r"^(\s*)(?:tail )?call void @llvm\.memcpy\.[\w.]+\("
    r"(?P<args>.*)\)"
    r"(?P<md>, ![\w. !<>{}]+)?$"
)
PTR = re.compile(r"^ptr(?: addrspace\((\d+)\))?(?: align (\d+))?(?: (?P<val>%[\w.]+|null))?$")


def die(msg):
    sys.exit("memcpy_mixed_as: " + msg)


def split_args(text):
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_ptr(arg):
    m = PTR.match(arg)
    if not m:
        return None
    return (int(m.group(1) or 0), int(m.group(2) or 0), m.group("val"))


def main():
    if len(sys.argv) != 2:
        die("usage: memcpy_mixed_as.py FILE.ll")
    path = pathlib.Path(sys.argv[1])
    lines = path.read_text().splitlines(keepends=True)

    out = []
    changed = False
    seq = 0
    for line in lines:
        m = CALL.match(line)
        if not m:
            out.append(line)
            continue
        args = split_args(m.group("args"))
        if len(args) != 4:
            out.append(line)
            continue
        dst, src = parse_ptr(args[0]), parse_ptr(args[1])
        size_m = re.fullmatch(r"i32 (\d+)", args[2]) or re.fullmatch(r"i64 (\d+)", args[2])
        vol = args[3].strip() == "true"
        if not dst or not src or not size_m or dst[2] is None or src[2] is None:
            out.append(line)
            continue
        if dst[0] == src[0]:
            out.append(line)  # same address space: clspv lowers this itself
            continue
        size = int(size_m.group(1))
        indent = m.group(1)
        changed = True
        if size == 0:
            continue  # drop the no-op copy
        stail = ", volatile\n" if vol else "\n"
        # loads first: every source byte, in offset order
        names = []
        for i in range(size):
            if i == 0:
                sptr = src[2]
            else:
                seq += 1
                out.append(f"{indent}%mcp{seq}g = getelementptr i8, ptr addrspace({src[0]}) {src[2]}, i64 {i}\n")
                sptr = f"%mcp{seq}g"
            seq += 1
            out.append(f"{indent}%mcp{seq} = load i8, ptr addrspace({src[0]}) {sptr}, align 1{stail}")
            names.append(f"%mcp{seq}")
        # then the stores, in offset order
        for i in range(size):
            if i == 0:
                dptr = dst[2]
            else:
                seq += 1
                out.append(f"{indent}%mcp{seq}g = getelementptr i8, ptr addrspace({dst[0]}) {dst[2]}, i64 {i}\n")
                dptr = f"%mcp{seq}g"
            out.append(f"{indent}store i8 {names[i]}, ptr addrspace({dst[0]}) {dptr}, align 1{', volatile' if vol else ''}\n")

    if changed:
        path.write_text("".join(out))
        print(f"memcpy_mixed_as: expanded mixed-address-space memcpy calls in {path}")
    else:
        print("memcpy_mixed_as: no mixed-address-space memcpy calls")


if __name__ == "__main__":
    main()
