#!/usr/bin/env python3
"""Rewrite LLVM IR text: named single-member struct types whose only member is a vector become the member type.

clspv cannot lower a dynamic-index GEP into a private alloca array whose element type is an aggregate, no matter
the access shape: an array of struct wrapping <2 x float> with a dynamic index and a vector store dies in
GetIdxsForTyFromOffset ("Unexpected offset for type"), and so does the memberwise scalar form, while the same
stores through a static index, and dynamic indexing of a raw vector array, all pass. The simdgroup_matrix shim
type is exactly such a wrapper (its only member is the two-lane fragment vector), and kernels keep their matrix
arrays in rolled loops, so the aggregate shape itself is the blocker.

A one-member struct and its member have identical size, alignment and ABI, and a named LLVM type may alias any
first-class type, so rewriting

  %"struct.metal::simdgroup_matrix" = type { <2 x i16> }
    ->  %"struct.metal::simdgroup_matrix" = type <2 x i16>

changes no addresses and keeps every GEP, load, store and phi that mentions the name textually valid. Types are
only rewritten when nothing in the module needs the struct shape:

  no GEP whose source type is the struct itself (a member walk),
  no GEP into an array of the struct with a trailing member index,
  no extractvalue / insertvalue on the type,
  no by-value argument, return or struct constant literal.

Anything else is left untouched for clspv.

Runs as a text pass between the front end and clspv; a module without a candidate type is a no-op.

Usage: flatten_single_member_structs.py FILE.ll
"""
import pathlib
import re
import sys

DEF = re.compile(r'^%("([^"]+)"|[\w.$-]+) = type \{ (.+) \}\s*$')
VECTOR = re.compile(r"^<\d+ x [iufhd][\w*]*>$")
GEP = re.compile(r"^((?:tail )?getelementptr(?:\s+\w+)*)\s+(.+)$")


def die(msg):
    sys.exit("flatten_single_member_structs: " + msg)


def spellings(name):
    """The type token as it can appear, quoted and bare, each with a boundary that keeps suffixes out."""
    e = re.escape(name)
    return [f'%"{e}"', f"%{e}"]


def needs_struct_shape(name, body):
    """True when some instruction uses the struct shape of the type and would not survive flattening."""
    toks = "|".join(re.escape(s) for s in spellings(name))
    for ln in body.splitlines():
        code = ln.split(";")[0]
        if not re.search(toks, code):
            continue
        if "extractvalue" in code or "insertvalue" in code:
            return True
        g = GEP.match(code.strip())
        if g:
            src = g.group(2).split(",")[0].strip()
            if re.fullmatch(toks, src):
                return True
            m = re.fullmatch(r"\[\d+ x (?:" + toks + r")\]", src)
            if m:
                # Indices are everything after the pointer operand; a third one walks the (former) struct.
                idxs = g.group(2).split(",")[2:]
                if len(idxs) > 2:
                    return True
        # By-value use: on a call, define or declare the type name directly followed by a value, function or
        # struct literal opening. (A store or phi spelling the type before a value or bracket list survives
        # flattening, because the name keeps aliasing a first-class type.)
        if re.match(r"^\s*(define|declare)\b", code) and re.search(toks + r"\s+([%@][\w.$-]+|\{)", code):
            return True
        if re.search(r"\bcall\b[^\n]*" + toks + r"\s+([%@][\w.$-]+|\{)", code):
            return True
    return False


def main():
    if len(sys.argv) != 2:
        die("usage: flatten_single_member_structs.py FILE.ll")
    p = pathlib.Path(sys.argv[1])
    lines = p.read_text().splitlines(keepends=True)
    candidates = []
    for i, ln in enumerate(lines):
        m = DEF.match(ln.rstrip("\n"))
        if m and VECTOR.fullmatch(m.group(3)):
            name = m.group(2) if m.group(2) is not None else m.group(1)
            candidates.append((i, name, m.group(3)))
    if not candidates:
        return
    body = "".join(ln for ln in lines if not DEF.match(ln.rstrip("\n")))
    for i, name, inner in candidates:
        if not needs_struct_shape(name, body):
            lines[i] = re.sub(r" = type \{ " + re.escape(inner) + r" \}", " = type " + inner, lines[i], count=1)
    p.write_text("".join(lines))


main()
