#!/usr/bin/env python3
"""Hoist pointer members of kernel argument structs into kernel buffer arguments.

The kimi/mla and kimi/kda kernels take `constant MlaArgs&` / `constant
KdaState&` / `constant KdaWeights&` structs (Metal: [[buffer(n)]], the host
writes GPU addresses into the struct bytes). On a 32-bit Vulkan module clspv
cannot express a pointer inside a UBO/push-constant Block: its producer dies
with a bare `ptr addrspace(1)` (or, after the struct is copied through memory,
on the byte-reassembled `bitcast <4 x i8> to ptr addrspace(1)`) - even for
members the kernel never reads.

Where a member IS read, the loaded pointer value is a fixed buffer of this
dispatch (the arena base, e.g. `a.cache` or `base.S`): the Vulkan port of that
shape is its own storage-buffer binding. Where it is not read, the pointer
bytes are dead. This pass rewrites both, keeping every field offset and the
host byte layout unchanged:

  %c = load ptr addrspace(1), ptr addrspace(2) %a          (member 0)
    ->  a new trailing kernel argument `ptr addrspace(1) %m2vhN` and every
        use of %c replaced by it

  %g = getelementptr %struct.KdaState, ptr addrspace(2) %a, i32 0, i32 1
  %p = load ptr addrspace(1), ptr addrspace(2) %g
    ->  the same rewrite via the constant member index

  a pointer member whose pointer value is never loaded
    ->  retyped to the datalayout-width integer (i32 for e-p:32:32) in the
        %struct line: same 4 bytes at the same offset

The kernel signature grows by one pointer argument per hoisted member and the
!kernel_arg_* metadata arrays gain the matching entry, so clspv sees plain
buffer arguments where Metal had struct members. Buffers selected at RUNTIME
from an array of structs (the kimi/experts ExpertPtrs table) do not match the
shape - a compile-time argument list cannot follow a runtime index - and are
left untouched: those pointers are genuine runtime data.

Functions that are called somewhere keep their signature (the pass skips
them), and the sweep modules' uncalled __clang_ocl_kern_imp_ copies get the
same rewrite as their kernel, keeping the shared !kernel_arg_* metadata
consistent.

Runs as a text pass between the clang/opt front end and clspv, after
promote_ptr_arrays; a module without pointer loads from a struct argument and
without dead pointer members is a no-op.

Usage: hoist_struct_ptr_args.py FILE.ll
"""
import pathlib
import re
import sys

LOADP = re.compile(
    r"^(\s*)(%[\w.]+) = load (ptr addrspace\(\d+\)), ptr addrspace\((\d+)\) ([\w.%]+), align (\d+)\s*$"
)
GEP_STRUCT = re.compile(
    r"^(\s*)(%[\w.]+) = getelementptr (?:inbounds )?(?:nuw )?(%[\w.]+), "
    r"ptr addrspace\((\d+)\) ([\w.%]+), i32 0, i32 (\d+)\s*$"
)
STRUCT_DEF = re.compile(r"^(%[\w.]+) = type \{(.*)\}\s*$")
SIG = re.compile(r"^define [^@]+ @([\w.$]+)\((.*)\)\s*(#\d+)(.*)\{\s*$")


def split_top(text):
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "<({[":
            depth += 1
        elif ch in ">)}]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts]


def arg_names(argtext):
    return [a.split()[-1] for a in split_top(argtext)] if argtext.strip() else []


def die(msg):
    sys.exit("hoist_struct_ptr_args: " + msg)


def replace_tokens(line, repl):
    if not repl:
        return line
    pattern = re.compile(
        r"(?<![\w.%])(" + "|".join(re.escape(k) for k in repl) + r")(?![\w.])"
    )
    return pattern.sub(lambda m: repl[m.group(1)], line)


def analyse(chunk):
    """-> (sig, {arg: struct type}, {load result: (arg, member, gep gepname)})"""
    sig = SIG.match(chunk[0])
    if not sig:
        return None, {}, {}
    args = arg_names(sig.group(2))
    if not args:
        return sig, {}, {}
    argset = set(args)

    structs = {}      # arg -> struct type name (None when inconsistent)
    gep_member = {}   # gep result name -> (base arg, member index)
    for line in chunk[1:]:
        m = GEP_STRUCT.match(line)
        if not m:
            continue
        base = m.group(5)
        if base in argset:
            prev = structs.get(base, m.group(3))
            structs[base] = prev if prev == m.group(3) else None
        gep_member[m.group(2)] = (base, int(m.group(6)))

    loads = {}  # load result name -> (arg, member index, struct type)
    for line in chunk[1:]:
        m = LOADP.match(line)
        if not m:
            continue
        src = m.group(5)
        if src in argset:
            arg, member = src, 0
        elif src in gep_member:
            arg, member = gep_member[src]
        else:
            continue
        st = structs.get(arg)
        if st:
            loads[m.group(2)] = (arg, member, st)
    return sig, structs, loads


def main():
    if len(sys.argv) != 2:
        die("usage: hoist_struct_ptr_args.py FILE.ll")
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    lines = text.splitlines(keepends=True)

    # module-level struct member lists, and the pointer width from the layout
    structs = {}
    for line in lines:
        sm = STRUCT_DEF.match(line)
        if sm and "ptr addrspace" in sm.group(2):
            structs[sm.group(1)] = split_top(sm.group(2))
    int_ty = "i64" if re.search(r"e-p:64:64", text) else "i32"

    # collect per-function rewrite plans
    plans = []  # (start, end, sig, loads)
    start = None
    for i, line in enumerate(lines + ["}"]):
        if start is None and line.startswith("define "):
            start = i
            continue
        if start is not None and (line.startswith("define ") or i == len(lines)):
            chunk = lines[start:i]
            sig, _structs, loads = analyse(chunk)
            if sig and loads:
                fname = sig.group(1)
                outside = "".join(lines[:start]) + "".join(lines[i:])
                if not re.search(r"@%s\s*\(" % re.escape(fname), outside):
                    plans.append((start, i, sig, loads))
            start = i if i < len(lines) else None
    if not plans:
        pass  # the retype below still applies (dead pointer members)
    arg_structs = {st for (_a, _b, _s, loads) in plans
                   for (_arg, _m, st) in loads.values()}
    # every struct type a function parameter addresses (from any gep on it):
    # the retype also covers arg structs whose pointer members are all dead
    params = set()
    for i, line in enumerate(lines):
        if line.startswith("define "):
            params |= set(arg_names(SIG.match(line).group(2)))
    for line in lines:
        m = GEP_STRUCT.match(line)
        if m and m.group(5) in params:
            arg_structs.add(m.group(3))

    # rewrite bottom-up so earlier chunk bounds stay valid
    new_args_total = 0
    meta_adds = {}  # metadata id -> [entry, ...] (added once per id: the
    # kernel and its imp copy share the same !kernel_arg_* nodes)
    seen_meta = set()
    for (a, b, sig, loads) in sorted(plans, key=lambda p: -p[0]):
        chunk = lines[a:b]
        argtext, attrs = sig.group(2), sig.group(4)
        meta_ids = dict(re.findall(r"!(kernel_arg_\w+) !(\d+)", attrs))
        repl = {}
        for loadname, (arg, member, st) in sorted(
                loads.items(), key=lambda kv: (int(kv[1][1]), kv[0])):
            newname = f"%m2vh{member}_of_{arg.lstrip('%')}"
            if newname in repl.values():
                newname = newname + f"_x{len(repl)}"
            repl[loadname] = newname
        # argument texts and metadata entries; metadata only for the first
        # function that carries each id
        new_meta = {} if set(meta_ids.values()) & seen_meta else meta_adds
        seen_meta |= set(meta_ids.values())
        new_args = []
        for loadname, (arg, member, st) in sorted(
                loads.items(), key=lambda kv: (int(kv[1][1]), kv[0])):
            newname = repl[loadname]
            new_args.append(f"ptr addrspace(1) noundef align 4 {newname}")
            # reflected pointee: the element type of the first gep off the value
            pointee = "i8"
            for line in chunk[1:]:
                gm = re.match(
                    r"^\s*[\w.%]+ = getelementptr (?:inbounds )?(?:nuw )?([\w.<> ]+?), "
                    r"ptr addrspace\(\d+\) " + re.escape(loadname) + r",", line)
                if gm:
                    pointee = gm.group(1).strip()
                    break
            if new_meta is meta_adds and meta_ids:
                entries = {
                    "kernel_arg_addr_space": "i32 1",
                    "kernel_arg_access_qual": '!"none"',
                    "kernel_arg_type": f'!"{pointee}*"',
                    "kernel_arg_base_type": f'!"{pointee}*"',
                    "kernel_arg_type_qual": '!""',
                }
                if "kernel_arg_name" in meta_ids:
                    entries["kernel_arg_name"] = f'!"m2v_h{member}_of_{arg.lstrip("%")}"'
                done = set()  # kernel_arg_type and kernel_arg_base_type can be one uniqued node
                for key, entry in entries.items():
                    if meta_ids[key] not in done:
                        done.add(meta_ids[key])
                        meta_adds.setdefault(meta_ids[key], []).append(entry)

        newargtext = argtext + (", " if argtext.strip() else "") + ", ".join(new_args)
        head, rest = chunk[0].split("(", 1)
        _args, tail = rest.rsplit(")", 1)
        lines[a] = head + "(" + newargtext + ")" + tail

        body = []
        for line in chunk[1:]:
            m = LOADP.match(line)
            if m and m.group(2) in repl:
                continue
            body.append(replace_tokens(line, repl))
        lines[a + 1:b] = body
        new_args_total += len(new_args)

    # retype pointer members of the argument structs. Hoisted members are
    # retyped too: their value now arrives as its own buffer argument and the
    # struct bytes are dead. A member still loaded through a shape the hoist
    # does not handle (for example a runtime-indexed table) keeps its pointer
    # type; if any remaining pointer load cannot be traced away from the arg
    # structs at all, the retype is skipped entirely.
    def_line = {}
    for i, line in enumerate(lines):
        m = re.match(r"^\s*([\w.%]+) = (\w+)[^\n]*$", line)
        if m and m.group(1).startswith("%"):
            def_line[m.group(1)] = (m.group(2), line)
    arg_names_mod = {}
    ascast = {}
    gep_member_mod = {}
    for name, (op, line) in def_line.items():
        m = re.match(r"^\s*[\w.%]+ = addrspacecast ptr addrspace\(\d+\) ([\w.%]+) to", line)
        if m:
            ascast[name] = m.group(1)
            continue
        m = GEP_STRUCT.match(line)
        if m:
            gep_member_mod[name] = (m.group(3), int(m.group(6)))
            continue
        m = re.match(r"^\s*[\w.%]+ = load [\w ]+, ptr addrspace\(\d+\) ([\w.%]+),", line)
        if m:
            arg_names_mod[name] = m.group(1)

    def trace(v, depth=0):
        """Follow a pointer value back to its origin; None = unknown."""
        while True:
            if v in ascast:
                v = ascast[v]
            elif v in gep_member_mod:
                return ("member",) + gep_member_mod[v]
            elif v in arg_names_mod:
                v = arg_names_mod[v]
            elif v in def_line and depth < 6:
                op, line = def_line[v]
                if op == "load":
                    m = re.match(r"^\s*[\w.%]+ = load [\w ]+, ptr addrspace\(\d+\) ([\w.%]+),", line)
                    if not m:
                        return None
                    v = m.group(1)
                    depth += 1
                elif op == "alloca" or v.startswith("@"):
                    return ("private",)
                else:
                    return None
            else:
                return ("private",) if (v.startswith("@") or v in def_line and def_line[v][0] == "alloca") else None

    unsure = False
    still = set()
    for line in lines:
        m = LOADP.match(line)
        if not m:
            continue
        t = trace(m.group(5))
        if t is None:
            unsure = True
        elif t[0] == "member":
            still.add((t[1], t[2]))
    nretyped = 0
    if not unsure:
        for i, line in enumerate(lines):
            sm = STRUCT_DEF.match(line)
            if not sm or sm.group(1) not in arg_structs:
                continue
            members = structs.get(sm.group(1))
            if not members:
                continue
            changed = False
            for mi, mt in enumerate(members):
                if (re.fullmatch(r"ptr addrspace\(\d+\)", mt)
                        and (sm.group(1), mi) not in still):
                    members[mi] = int_ty
                    changed = True
            if changed:
                lines[i] = f"{sm.group(1)} = type {{ {', '.join(members)} }}\n"
                nretyped += 1

    # append the metadata entries
    for i, line in enumerate(lines):
        m = re.match(r"^!(\d+) = !\{(.*)\}\s*$", line)
        if m and m.group(1) in meta_adds:
            entries = ", ".join(meta_adds[m.group(1)])
            lines[i] = f"!{m.group(1)} = !{{{m.group(2)}, {entries}}}\n"

    if new_args_total or nretyped:
        path.write_text("".join(lines))
        print(f"hoist_struct_ptr_args: hoisted {new_args_total} pointer member(s), "
              f"retyped {nretyped} struct(s) in {path}")


if __name__ == "__main__":
    main()
