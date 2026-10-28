#!/usr/bin/env python3
"""Promote small private arrays of pointers into scalar allocas with select chains.

The qmv_wide and mla kernels keep a per-dispatch buffer-select table in private memory:
`alloca [2 x ptr addrspace(1)]`, a store of the selected buffer per loop iteration, and a
load feeding a later GEP. SROA declines to promote the array because the index is a loop
bound it cannot prove in range, and clspv's SPIRV producer cannot represent a
pointer-typed storage object: generating the SPIR-V for the load asks for the pointee type
of a pointer, the inferred type is itself a pointer, and the producer hits its
"Unsupported type" path (a bare `ptr addrspace(1)` on stderr followed by a segfault,
because llvm_unreachable is compiled out in Release).

This pass rewrites the pattern into scalar allocas indexed by select chains:

  store ptr addrspace(1) %v, (gep [2 x ptr addrspace(1)] %arr, 0, %i)
    ->  %c0 = icmp eq i32 %i, 0
        %o0 = load ptr addrspace(1), ptr %arr.s0
        %n0 = select i1 %c0, ptr addrspace(1) %v, ptr addrspace(1) %o0
        store ptr addrspace(1) %n0, ptr %arr.s0      (repeat per slot)

  %l = load ptr addrspace(1), (gep [2 x ptr addrspace(1)] %arr, 0, %i)
    ->  %v0 = load ptr addrspace(1), ptr %arr.s0     (per slot)
        %l = select i1 %c0, ptr addrspace(1) %v0, ptr addrspace(1) %v1

Each scalar is initialized to null after allocation, or after each lifetime start when
markers exist. Only inbounds GEPs are rewritten; dereferencing an out-of-range index is
outside the source program's defined behavior. Only arrays whose every GEP feeds a plain
load or store of the array (and nothing else consumes the GEP) are rewritten; unsupported
shapes are left untouched for clspv. Arrays longer than 16 slots are skipped. Value names repeat across functions,
so the analysis runs per function body.

Runs as a text pass between the clang/opt front end and clspv; a module without a
pointer-array alloca is a no-op.

Usage: promote_ptr_arrays.py FILE.ll
"""
import pathlib
import re
import sys

MAX_SLOTS = 16

ALLOCA = re.compile(
    r"^(\s*)(%[\w.]+) = alloca \[(\d+) x ptr addrspace\((\d+)\)\], align (\d+)\s*$"
)
GEP = re.compile(
    r"^(\s*)(%[\w.]+) = getelementptr (?:inbounds )?\[(\d+) x ptr addrspace\((\d+)\)\], "
    r"ptr (%[\w.]+), i32 0, (i32|i64) (%[\w.]+)\s*$"
)
STORE = re.compile(
    r"^\s*store ptr addrspace\((\d+)\) (%[\w.]+|null), ptr (%[\w.]+), align (\d+)\s*$"
)
LOAD = re.compile(
    r"^(\s*)(%[\w.]+) = load ptr addrspace\((\d+)\), ptr (%[\w.]+), align (\d+)\s*$"
)


def slot(arr, j):
    # numeric LLVM names (%7) cannot take suffixes; re-root the name on letters
    return f"%pa{arr[1:]}.s{j}"


def die(msg):
    sys.exit("promote_ptr_arrays: " + msg)


def pointer_index_bits(text):
    """Return an explicit default pointer index width, or None if ambiguous."""
    layouts = re.findall(r'^target datalayout = "([^"]*)"\s*$', text, re.M)
    if len(layouts) != 1:
        return None
    defaults = [item for item in layouts[0].split("-") if item.startswith(("p:", "p0:"))]
    if len(defaults) != 1:
        return None
    match = re.fullmatch(r"p(?:0)?:([0-9]+):[0-9]+(?::[0-9]+)?(?::([0-9]+))?", defaults[0])
    return int(match.group(2) or match.group(1)) if match else None


def rewrite_function(chunk, index_bits=None):
    """chunk: list of lines of one function body (or module header chunk). Returns new lines."""
    arrays = {}  # alloca name -> (slots, addrspace, align)
    for line in chunk:
        m = ALLOCA.match(line)
        if m and 1 <= int(m.group(3)) <= MAX_SLOTS:
            arrays[m.group(2)] = (int(m.group(3)), int(m.group(4)), int(m.group(5)))
    if not arrays or index_bits is None or index_bits < 32:
        return chunk, 0

    # Quoted names and leading-zero numeric locals may alias other names.
    # Generated %pa names share the local namespace with inputs and labels.
    # Decline these functions instead of guessing; supported paired lifetime
    # markers are handled below.
    if any('%"' in line or '@"' in line or
           re.search(r"%pa|%0[0-9]+(?![\w.$-])|^\s*(?:pa[\w.$-]*:|0[0-9]+:|\")", line) for line in chunk):
        return chunk, 0

    geps = {}  # gep name -> (indent, array, idx, idxtypetext, slots, aspace, align)
    for line in chunk:
        m = GEP.match(line)
        if m and m.group(5) in arrays:
            # A plain GEP can wrap its scaled byte offset into a valid slot.
            # Inbounds makes that wrap poison, so defined accesses cannot do so.
            if "getelementptr inbounds " not in line:
                return chunk, 0
            slots, aspace, align = arrays[m.group(5)]
            if int(m.group(3)) == slots and int(m.group(4)) == aspace:
                geps[m.group(2)] = (m.group(1), m.group(5), m.group(7), m.group(6), slots, aspace, align)

    # GEP truncates its index to the data layout's pointer-index width.
    # Only i32 is handled here; a wider operand requires explicit truncation.
    if any(g[3] != "i32" for g in geps.values()):
        return chunk, 0

    # Freeze each index at the original GEP, before repeated comparisons.
    # LLVM undef values can differ on every use; one store must choose one slot.
    frozen = {name: f"%paidx{n}" for n, name in enumerate(geps)}
    lifetimes = {}
    lifetime_re = re.compile(r"^\s*call void @llvm\.lifetime\.(start|end)\.[\w.]+\(ptr (%[\w.]+)\)\s*$")
    for line in chunk:
        m = lifetime_re.match(line)
        if m and m.group(2) in arrays:
            lifetimes.setdefault(m.group(2), set()).add(m.group(1))
    # Only the matched paired intrinsics are transformed. Other references
    # to an array are rejected by the use scan below.
    for arr, kinds in lifetimes.items():
        if kinds != {"start", "end"}:
            arrays.pop(arr, None)

    # Every array and GEP use must be accounted for before rewriting. Match
    # complete LLVM names: %2 is not a use of %20. Multiple plain loads or
    # stores of the same GEP are rewritten at their original program points.
    consumers = set()
    bad = set()
    names = re.compile(r"%[\w.]+")
    for line in chunk:
        # Include comments conservatively: a semicolon can also occur inside a
        # quoted LLVM name, and dropping the rest could hide an escape.
        refs = names.findall(line)
        am = ALLOCA.match(line)
        gm = GEP.match(line)
        sm = STORE.match(line)
        lm = LOAD.match(line)
        life = lifetime_re.match(line)
        for arr in arrays:
            if arr not in refs:
                continue
            allowed = ((am and am.group(2) == arr) or
                       (gm and gm.group(2) in geps and gm.group(5) == arr) or
                       (life and life.group(2) == arr))
            if not allowed or refs.count(arr) != 1:
                bad.add(arr)
        for name, g in geps.items():
            if name not in refs:
                continue
            if gm and gm.group(2) == name and refs.count(name) == 1:
                continue
            plain_store = sm and sm.group(3) == name and int(sm.group(1)) == g[5]
            plain_load = lm and lm.group(4) == name and int(lm.group(3)) == g[5]
            if (plain_store or plain_load) and refs.count(name) == 1:
                consumers.add(name)
            else:
                bad.add(g[1])
    for gname, g in geps.items():
        if g[1] in bad or gname not in consumers:
            arrays.pop(g[1], None)
    keep = {g[1] for g in geps.values() if g[1] in arrays}
    arrays = {k: v for k, v in arrays.items() if k in keep}
    if not arrays:
        return chunk, 0

    seq = 0
    out = []
    promoted = set()
    for line in chunk:
        m = ALLOCA.match(line)
        if m and m.group(2) in arrays:
            out.append(line)
            indent, name = m.group(1), m.group(2)
            slots, aspace, align = arrays[name]
            for j in range(slots):
                out.append(f"{indent}{slot(name, j)} = alloca ptr addrspace({aspace}), align {align}\n")
                if name not in lifetimes:
                    out.append(f"{indent}store ptr addrspace({aspace}) null, ptr {slot(name, j)}, align {align}\n")
            promoted.add(name)
            continue
        life = lifetime_re.match(line)
        if life and life.group(2) in arrays:
            name = life.group(2)
            slots, aspace, align = arrays[name]
            for j in range(slots):
                out.append(line.replace(f"ptr {name})", f"ptr {slot(name, j)})"))
                if life.group(1) == "start":
                    out.append(f"  store ptr addrspace({aspace}) null, ptr {slot(name, j)}, align {align}\n")
            continue
        m = GEP.match(line)
        if m and m.group(2) in geps and geps[m.group(2)][1] in arrays:
            g = geps[m.group(2)]
            out.append(f"{g[0]}{frozen[m.group(2)]} = freeze {g[3]} {g[2]}\n")
            continue
        m = STORE.match(line)
        if m and m.group(3) in geps and geps[m.group(3)][1] in arrays:
            indent, arr, idx, idxty, slots, aspace, align = geps[m.group(3)]
            idx = frozen[m.group(3)]
            for j in range(slots):
                seq += 1
                out.append(f"{indent}%pa{seq}c = icmp eq {idxty} {idx}, {j}\n")
                out.append(f"{indent}%pa{seq}o = load ptr addrspace({aspace}), ptr {slot(arr, j)}, align {align}\n")
                out.append(f"{indent}%pa{seq}n = select i1 %pa{seq}c, ptr addrspace({aspace}) {m.group(2)}, ptr addrspace({aspace}) %pa{seq}o\n")
                out.append(f"{indent}store ptr addrspace({aspace}) %pa{seq}n, ptr {slot(arr, j)}, align {align}\n")
            continue
        m = LOAD.match(line)
        if m and m.group(4) in geps and geps[m.group(4)][1] in arrays:
            indent, arr, idx, idxty, slots, aspace, align = geps[m.group(4)]
            idx = frozen[m.group(4)]
            vals = []
            for j in range(slots):
                seq += 1
                out.append(f"{indent}%pa{seq} = load ptr addrspace({aspace}), ptr {slot(arr, j)}, align {align}\n")
                vals.append(f"%pa{seq}")
            if slots == 1:
                out.append(f"{indent}{m.group(2)} = load ptr addrspace({aspace}), ptr {slot(arr, 0)}, align {align}\n")
            else:
                sel = vals[-1]
                for j in range(slots - 2, -1, -1):
                    seq += 1
                    out.append(f"{indent}%pa{seq}c = icmp eq {idxty} {idx}, {j}\n")
                    out.append(f"{indent}%pa{seq}s = select i1 %pa{seq}c, ptr addrspace({aspace}) {vals[j]}, ptr addrspace({aspace}) {sel}\n")
                    sel = f"%pa{seq}s"
                out.append(f"{indent}{m.group(2)} = select i1 %pa{seq}c, ptr addrspace({aspace}) {vals[0]}, ptr addrspace({aspace}) {sel}\n")
            continue
        out.append(line)
    return out, len(promoted)


def main():
    if len(sys.argv) != 2:
        die("usage: promote_ptr_arrays.py FILE.ll")
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    lines = text.splitlines(keepends=True)

    # split into per-function chunks so value names stay unambiguous
    out, total = [], 0
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("define ") and start != i:
            # header chunk before this define belongs with the define
            pass
    chunk_starts = [i for i, line in enumerate(lines) if line.startswith("define ")]
    if not chunk_starts:
        print("promote_ptr_arrays: no functions")
        return
    bounds = []
    prev = 0
    for cs in chunk_starts:
        if cs > prev:
            bounds.append((prev, cs))  # header chunk
        prev = cs
    bounds.append((chunk_starts[-1], len(lines)))

    for (a, b) in bounds:
        new, n = rewrite_function(lines[a:b], pointer_index_bits(text))
        out.extend(new)
        total += n

    if total:
        path.write_text("".join(out))
        print(f"promote_ptr_arrays: promoted {total} pointer array(s) in {path}")
    else:
        print("promote_ptr_arrays: no promotable pointer arrays")


if __name__ == "__main__":
    main()
