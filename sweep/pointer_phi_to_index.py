#!/usr/bin/env python3
"""Rewrite LLVM IR text: loops that carry pointers in phis become integer induction plus byte-indexed GEPs.

clspv's SimplifyPointerBitcast cycles on loops whose header phis are pointers advanced by a constant
stride (the gate_up gt::qmv_rows loop: x += 256 i16 elements, w += 128 i8 bytes, s/b += 8 i16 entries).
Its sub-passes fold GEP pairs around the cycle in both directions and never settle; at the 200-iteration
cap the patched pass continues with mid-drift GEPs and the loop-exit users read one block too far
(sweep/tests/spb_tail_drift.ll, PR #47). Upstream clspv does not finish the module at all. The cycle is
the pointer phi, so remove it before clspv sees the module:

  %p = phi ptr addrspace(1) [ %v0, %pre ], [ %adv, %latch ]   (%adv: constant-stride GEP off %p)
    ->
  %idx = phi i32 [ off0, %pre ], [ %next, %latch ]            (byte index from the same root base)
  %p2  = getelementptr inbounds i8, ptr addrspace(1) BASE, i32 %idx
  %next = add i32 %idx, STEP                                  (STEP in bytes)

and every use of %p becomes %p2. %p2 computes the same address %p held on the path that reaches the
header, so loads, stores, inner GEPs and the loop-exit phis that carry %p (or the latch advance) out of
the loop are unchanged - the exit users are exactly what the drift corrupted, and they now sit on
integer arithmetic that SimplifyPointerBitcast never folds.

Rules, applied per pointer phi; a phi that does not fit is left untouched and the module keeps working
exactly as before (a module with no candidate is a byte-identical no-op):

  exactly one incoming is the advance (single-index all-constant GEP whose base, through bitcasts, is
  the phi itself), and exactly one other incoming (the entry value),
  the entry value resolves through bitcasts and single-index GEPs with at most one dynamic index to a
  root base that is a function argument or a global,
  element sizes are known scalars / vectors / arrays of them (no structs), offsets fit in i32/i64.

addrspacecast chains, multi-index GEPs and quoted labels are out of scope. Runs as a text pass before
clspv, next to bfloat_to_i16 and flatten_single_member_structs; skip it with M2V_KEEP_PTR_PHI=1.

Usage: pointer_phi_to_index.py FILE.ll
"""
import pathlib
import re
import sys

SIZES = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "half": 2, "bfloat": 2, "float": 4, "double": 8}
PTR = r"ptr(?:\s+addrspace\(\d+\))?"
PHI = re.compile(rf"^(\s*)(%[\w.$]+)\s+= phi ({PTR}) ((?:\[[^\]]*\](?:,\s*)?)+)\s*$")
INCOMING = re.compile(r"\[\s*(%[\w.$]+|@[\w.$]+|true|false|null)\s*,\s*(%[\w.$-]+)\s*\]")
DEF = re.compile(r"^\s*(%[\w.$]+)\s+= (.+)$")
LABEL = re.compile(r"^([\w.$-]+):\s*(?:;.*)?$")
INT = re.compile(r"-?\d+$")


def die(msg):
    sys.exit("pointer_phi_to_index: " + msg)


def type_size(ty):
    """Byte size of a type token; None when unknown (structs, odd spellings)."""
    ty = ty.strip()
    m = re.fullmatch(r"<(\d+) x (.+)>", ty) or re.fullmatch(r"\[(\d+) x (.+)\]", ty)
    if m:
        inner = type_size(m.group(2))
        return None if inner is None else inner * int(m.group(1))
    return SIZES.get(ty)


def top_split(s):
    """Split on top-level commas; <>, (), [] are nesting."""
    parts, depth, cur = [], 0, []
    for ch in s:
        depth += ch in "<([" 
        depth -= ch in ">)]"
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def parse_gep(rhs):
    """-> (src_ty, base_typed_text, base_name, idx_ty, idx_text) for a plain single-index GEP, else None."""
    m = re.match(r"getelementptr\b(.*)$", rhs)
    if not m:
        return None
    rest = m.group(1)
    while True:  # strip the flag tokens LLVM may print between the opcode and the source type
        stripped = re.sub(r"^\s+(?:inbounds|nuw|nusw|inrange(?:\(\d+(?:,\s*\d+)?\)))\b", "", rest, count=1)
        if stripped == rest:
            break
        rest = stripped
    parts = top_split(rest)
    if len(parts) != 3:
        return None  # wrong operand count or a multi-index GEP (element-size math across members)
    base = parts[1].rsplit(None, 1)
    idx = parts[2].split(None, 1)
    if len(base) != 2 or len(idx) != 2 or idx[0].startswith("<"):
        return None
    return parts[0], parts[1], base[1], idx[0], idx[1]


class Func:
    def __init__(self, lines):
        self.lines = lines
        self.defs = {}
        for i, ln in enumerate(lines):
            m = DEF.match(ln)
            if m:
                self.defs[m.group(1)] = i


def fresh(func, hint):
    """A '%m2v.<hint>.*' name family not colliding with anything in the function."""
    base = "m2v." + re.sub(r"[^\w]", ".", hint)[:36].strip(".")
    while any("%" + base + s in func.defs for s in ("", ".idx", ".next", ".ptr", ".off")):
        base += ".x"
    for s in (".idx", ".next", ".ptr", ".off"):
        func.defs["%" + base + s] = -1
    return base


def walk(func, val, phi_name):
    """Follow one incoming value down through bitcasts and single-index GEPs to its root base.

    -> None when out of scope, else a dict: advance=True for the constant-stride GEP off the phi
    (const_bytes, listed incoming block, def line), or the entry chain (root base, its typed text,
    total constant bytes, the one dynamic index + its byte scale, the line to anchor helpers after,
    the index width seen).
    """
    name, walked, dyn, dyn_scale, const_bytes, idx_ty, anchor = val, False, None, 1, 0, None, None
    last_base = last_base_ty = None
    while True:
        i = func.defs.get(name)
        if i is None:
            # root: function argument (%0 style) or a global; with no GEP walked its spelled type is
            # the phi's own, which plan_phi substitutes
            return {"advance": False, "base": name, "base_ty": None, "const_bytes": const_bytes,
                    "dyn": dyn, "dyn_scale": dyn_scale, "idx_ty": idx_ty, "anchor_i": anchor,
                    "last_base": last_base, "last_base_ty": last_base_ty}
        rhs = DEF.match(func.lines[i]).group(2)
        if rhs.startswith("bitcast "):
            m = re.match(rf"bitcast\s+{PTR}\s+(%[\w.$]+)\s+to\s+{PTR}\s*$", rhs)
            if not m:
                return None
            name = m.group(1)
            continue
        if rhs.startswith("addrspacecast "):
            return None  # byte offsets do not carry across address spaces
        g = parse_gep(rhs)
        if g is None:
            return None  # fed from a select, load, another phi, an alloca...: out of scope
        src_ty, base_ty_text, base, ity, idx = g
        size = type_size(src_ty)
        if size is None or (idx_ty and ity != idx_ty):
            return None
        idx_ty = ity
        if base == phi_name:
            if not INT.match(idx):
                return None
            return {"advance": True, "const_bytes": int(idx) * size, "idx_ty": idx_ty,
                    "def_i": i}
        if base == name:
            return None  # self-referential but not the advance: a cycle
        if INT.match(idx):
            const_bytes += int(idx) * size
        else:
            if dyn is not None or walked:
                return None  # the one dynamic index must be the innermost GEP
            dyn, dyn_scale, anchor = idx, size, i
        walked = True
        name = base
        last_base_ty = base_ty_text
        last_base = base


def plan_phi(func, i, m):
    """-> plan dict for one rewritable pointer phi, else None to leave it untouched."""
    phi_name, phi_ty, incomings = m.group(2), m.group(3), INCOMING.findall(m.group(4))
    if len(incomings) < 2:
        return None
    advance, entry = None, None
    for val, blk in incomings:
        chain = walk(func, val, phi_name)
        if chain is None:
            return None
        if chain.get("advance"):
            if advance is not None:  # two latches: not the shape
                return None
            advance = chain
            advance["blk"] = blk
        else:
            if entry is not None:  # two entry paths into the loop: not the shape
                return None
            entry = chain
            entry["blk"] = blk
    if advance is None or entry is None:
        return None
    # the remembered base text is the complete typed operand of the innermost GEP; a bare argument
    # root gets the phi's own type
    base_text = entry["last_base_ty"] or f"{phi_ty} {entry['base']}"
    step = advance["const_bytes"]
    width = advance["idx_ty"] or entry["idx_ty"] or "i32"
    if entry["idx_ty"] and entry["idx_ty"] != width:
        return None
    if not -2**31 <= step < 2**31:
        return None

    new = fresh(func, phi_name.strip("%"))
    off_val, off_lines = "0", []
    if entry["dyn"] is not None:
        total = entry["const_bytes"]
        off_lines = [f"  %{new}.off = mul {width} {entry['dyn']}, {entry['dyn_scale']}\n"]
        off_val = f"%{new}.off"
        if total:
            off_lines.append(f"  %{new}.off0 = add {width} {off_val}, {total}\n")
            off_val = f"%{new}.off0"
    elif entry["const_bytes"]:
        off_val = str(entry["const_bytes"])

    return {"phi_i": i, "phi_name": phi_name, "new_name": f"%{new}.ptr",
            "idx_line": (f"  %{new}.idx = phi {width} [ {off_val}, {entry['blk']} ], "
                         f"[ %{new}.next, {advance['blk']} ]\n"),
            "ptr_line": f"  %{new}.ptr = getelementptr inbounds i8, {base_text}, {width} %{new}.idx\n",
            "add_line": f"  %{new}.next = add {width} %{new}.idx, {step}\n",
            "add_i": advance["def_i"],
            "off_lines": off_lines, "off_anchor": entry["anchor_i"] if entry["dyn"] is not None else i}


ANY_PHI = re.compile(r"^\s*%[\w.$]+\s+= phi\b")


def ptr_anchor(func, i):
    """The new GEP goes after the phi group: before the first non-phi line of the block."""
    j = i + 1
    while j < len(func.lines) and (not func.lines[j].strip() or ANY_PHI.match(func.lines[j])):
        j += 1
    return j if j < len(func.lines) else i


def rewrite_function(text, n):
    lines = text.splitlines(keepends=True)
    func = Func(lines)
    plans = [p for i, ln in enumerate(lines) if (m := PHI.match(ln)) and (p := plan_phi(func, i, m))]
    # loop-exit phis that forward one value (the gate_up tail shape) are replaced by that value: a
    # single-incoming phi is its incoming, and dropping the last pointer phis leaves clspv nothing
    # pointer-shaped to fold around
    drop_exit = {}
    for i, ln in enumerate(lines):
        if i in drop_exit or not PHI.match(ln):
            continue
        inc = INCOMING.findall(PHI.match(ln).group(4))
        if len(inc) == 1:
            drop_exit[i] = inc[0][0]

    if not plans and not drop_exit:
        return text

    drop, before, after = set(), {}, {}
    for p in plans:
        drop.add(p["phi_i"])
        after.setdefault(p["phi_i"], []).append(p["idx_line"])
        before.setdefault(ptr_anchor(func, p["phi_i"]), []).append(p["ptr_line"])
        before.setdefault(p["add_i"], []).append(p["add_line"])
        if p["off_lines"]:
            after.setdefault(p["off_anchor"], []).extend(p["off_lines"])
    drop.update(drop_exit)

    out = []
    for i, ln in enumerate(lines):
        if i not in drop:
            out.extend(before.get(i, []))
            out.append(ln)
        out.extend(after.get(i, []))

    out_text = "".join(out)
    # dead exit phis first: forward their (chain-resolved) incoming, so every remaining name the
    # plan RAUW rewrites below is the name the module actually still defines
    exit_subs = {PHI.match(func.lines[i]).group(2): val for i, val in drop_exit.items()}
    for name, val in exit_subs.items():
        seen = {name}
        while val in exit_subs and val not in seen:
            seen.add(val)
            val = exit_subs[val]
        out_text = re.sub(r"(?<![\w.$])" + re.escape(name) + r"(?![\w.$])", val, out_text)
    for p in plans:
        out_text = re.sub(r"(?<![\w.$])" + re.escape(p["phi_name"]) + r"(?![\w.$])", p["new_name"], out_text)
    n[0] += len(plans)
    return out_text


def main():
    if len(sys.argv) != 2:
        die("usage: pointer_phi_to_index.py FILE.ll")
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    n = [0]
    # value names repeat per function, so each function is planned and rewritten on its own
    result = "".join(rewrite_function(p, n) if p.startswith("define") else p
                     for p in re.split(r"(?m)^(?=define\s)", text))
    if result != text:
        path.write_text(result)
    print(f"pointer_phi_to_index: rewrote {n[0]} pointer phi(s) in {path}")


main()
