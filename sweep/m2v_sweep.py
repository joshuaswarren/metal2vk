#!/usr/bin/env python3
"""metal2vk bulk sweep: every uzu .metal and every TensorFold Zig .metal through the pipeline, one row per entry point.

Stages per entry (a row stops at the first failing stage):
  parse   clang (C++ for OpenCL, through the metal_stdlib shim) accepts the translation unit and the entry
  ir      clang emits LLVM IR (templates instantiated, inlined, address spaces resolved)
  spv     clspv lowers the IR to Vulkan SPIR-V
  val     spirv-val --target-env vulkan1.3 accepts the module
run and ref (a GPU dispatch with synthetic data, a CPU reference for a few kernels) are m2v_sweep_run.py's, on a GPU host.

Sources are preprocessed first (clang -E, the Metal system headers replaced by empty stand-ins generated into a temporary directory) so project includes,
#if blocks and macros are resolved. Host-injected macros (TensorFold's -D constants) that turn out undeclared get a dummy value and the
retry is recorded in the row ("defs"). uzu is preprocessed with -DDSL_ANALYZE, which turns its DSL macros (KERNEL, VARIANTS, SPECIALIZE,
GROUPS, THREADS, AXIS, OPTIONAL) into clang::annotate attributes that the entry analysis reads.
Entry points come in two spellings:
  - attribute style (TensorFold): a [[kernel]] function whose parameters carry [[buffer(n)]] and thread-position attributes; plain, or a
    template with `template [[host_name("n")]] [[kernel]] decltype(f<A, B>) f<A, B>;` instantiations (one row each). The entry becomes a real
    __kernel function (template arguments substituted, so threadgroup arrays stay in kernel scope, which OpenCL requires).
  - uzu DSL: `template <typename T> VARIANTS(T, float, half) PUBLIC KERNEL(Name)(...)`. A __kernel wrapper calls the template for the first
    and the last variant of every VARIANTS list; threadgroup declarations in the body move into a struct that lives in the wrapper.
Argument rules: pointers into device/constant memory are buffers; `constant T& x` and by-value structs become __constant T* arguments;
SPECIALIZE (function-constant) bools become uint arguments; threadgroup arrays, pointers and references become __local storage; Metal
thread attributes and GROUPS/THREADS/AXIS/ThreadContext become OpenCL work-item queries.
All the other entry points of a file are dropped from the translation unit of one row, so one row is one kernel.

Usage: m2v_sweep.py OUTDIR [--set uzu|tf|all] [--jobs N] [--limit N] [--only SUBSTR] [--no-spv] [--tag NAME]
env:   UZU_ROOT (uzu checkout), TF_ROOT (TensorFold checkout), CLANG (default clang-19), CLSPV (patched clspv; spv/val skipped if absent),
       OPT (the LLVM opt of CLANG's version), M2V_OPT (clang optimisation flags; default is the typed-GEP route of compile.sh),
       M2V_INCLUDE=DIR uses another shim include directory (a baseline run)
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
INC = pathlib.Path(os.environ.get("M2V_INCLUDE", HERE.parent / "include"))  # another shim, to measure a baseline
CLANG = os.environ.get("CLANG", "clang-19")
CLSPV = os.environ.get("CLSPV", str(pathlib.Path.home() / "scratch/metal2vk/clspv/build/bin/clspv"))
SPIRV_VAL = os.environ.get("SPIRV_VAL", "spirv-val")
CLSPV_TIMEOUT = int(os.environ.get("M2V_CLSPV_TIMEOUT", "90"))  # seconds per module; a loaded build host needs more
M2V_DEFS = os.environ.get("M2V_DEFS", "").split()
OPT = os.environ.get("OPT", "opt")  # the LLVM opt of the same version as CLANG
# Front-end optimisation flags. The default is compile.sh's typed-GEP route: clang -O0 without optnone, then LLVM's inliner and SROA (no
# InstCombine, which rewrites typed GEPs into byte offsets that clspv's pointer passes cannot follow). M2V_OPT overrides it, e.g. the old
# route: M2V_OPT="-O2 -fno-slp-vectorize -fno-vectorize -mllvm -inline-threshold=100000".
FE_OPT = shlex.split(os.environ.get("M2V_OPT", "-O0 -Xclang -disable-O0-optnone"))

THREAD_ATTRS = {
    "thread_position_in_grid": "get_global_id",
    "thread_position_in_threadgroup": "get_local_id",
    "threadgroup_position_in_grid": "get_group_id",
    "threads_per_threadgroup": "get_local_size",
    "dispatch_threads_per_threadgroup": "get_local_size",
    "threadgroups_per_grid": "get_num_groups",
    "threads_per_grid": "get_global_size",
    "dispatch_threads_per_grid": "get_global_size",
}
SCALAR_ATTRS = {
    "thread_index_in_threadgroup": "(uint)get_local_linear_id()",
    "simdgroup_index_in_threadgroup": "(uint)get_sub_group_id()",
    "thread_index_in_simdgroup": "(uint)get_sub_group_local_id()",
    "threads_per_simdgroup": "(uint)get_sub_group_size()",
    "simdgroups_per_threadgroup": "(uint)get_num_sub_groups()",
}


def blank_comments(src):
    """Replace comments with spaces of equal length (offsets and line numbers stay valid); string literals are skipped."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def balanced(s, i, open_c="(", close_c=")"):
    """s[i] == open_c; return the index of the matching close."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_c:
            depth += 1
        elif s[j] == close_c:
            depth -= 1
            if depth == 0:
                return j
    return -1


def split_top(s, sep=","):
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def attr_end(p, i):
    """p[i:i+2] == '[['; index of the closing ']]' (string literals skipped)."""
    j = i + 2
    q = None
    while j < len(p):
        c = p[j]
        if q:
            if c == "\\":
                j += 1
            elif c == q:
                q = None
        elif c in "\"'":
            q = c
        elif p.startswith("]]", j):
            return j
        j += 1
    return len(p) - 2


ANNOTATE_TAG = {"dsl.groups": "GROUPS", "dsl.threads": "THREADS", "dsl.axis": "AXIS", "dsl.specialize": "SPECIALIZE",
                "dsl.specialize_if": "SPECIALIZE_IF", "dsl.optional": "OPTIONAL", "dsl.constraint": "CONSTRAINT"}


def take_attrs(p):
    """-> (decl without [[...]] attributes, [attribute strings])."""
    attrs = []
    while True:
        i = p.find("[[")
        if i < 0:
            break
        j = attr_end(p, i)
        attrs.append(p[i + 2:j].strip())
        p = p[:i] + " " + p[j + 2:]
    return p.strip(), attrs


def annotate_tags(attrs):
    """clang::annotate("", "dsl.x", args...) attributes (uzu DSL under DSL_ANALYZE) -> ({tag: [arg text]}, remaining attrs)."""
    tags, rest = {}, []
    for a in attrs:
        m = re.match(r'clang::annotate\(\s*""\s*,\s*"(dsl\.\w+)"\s*(?:,(.*))?\)$', a, re.S)
        if m:
            if m.group(1) in ANNOTATE_TAG:
                tags.setdefault(ANNOTATE_TAG[m.group(1)], []).append(" ".join((m.group(2) or "").replace('"', "").split()))
        else:
            rest.append(a)
    return tags, rest


def strip_annotate(text):
    out, i = [], 0
    while True:
        j = text.find("[[clang::annotate", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        i = attr_end(text, j) + 2


VEC_ALIASES = "".join(f"typedef __generic {t}{n} m2v_V_{t}{n};\n" for t in ("char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "float", "half")
                      for n in (2, 3, 4, 8, 16))
VEC_DECL = re.compile(r"\b(u?char|u?short|u?int|u?long|float|half|bool)([234]|8|16)(\s+)\((\s*[&*])")


def norm_decls(text):
    """`uint4 (&x)[N]` / `uint4 (*p)` declarators collide with the vector constructor macros (uint4(...)): use a typedef alias.
    The alias is __generic-qualified: member arrays of vectors are __generic, and an unqualified or __private alias can
    neither bind a non-const reference to them nor convert const bindings from them."""
    return VEC_DECL.sub(lambda m: f"m2v_V_{m.group(1)}{m.group(2)}{m.group(3)}({m.group(4)}", text)


def split_decl(decl):
    """'const device T* input' -> (type, name, dims)."""
    m = re.match(r"^(.*?)(\w+)\s*((?:\[[^\]]*\]\s*)*)$", decl.strip(), re.S)
    if not m:
        return decl, "", ""
    return m.group(1).strip(), m.group(2), m.group(3).strip()


POD_TYPE = re.compile(r"(?:bool|u?char|u?short|u?int|u?long|half|float|bfloat|size_t|ptrdiff_t|u?int(?:8|16|32|64)_t|"
                      r"(?:u?char|u?short|u?int|u?long|float|half|bfloat)(?:[234]|8|16))")


class Param:
    pass


def classify(raw, tg_counter, grp_counter, thr_counter, ax_counter):
    p = Param()
    decl, attrs = take_attrs(raw)
    atags, attrs = annotate_tags(attrs)
    tags = atags
    p.raw, p.attrs, p.tags = raw, attrs, tags
    p.type, p.name, p.dims = split_decl(decl)
    p.decl = decl
    p.kind = None
    attr_names = [re.match(r"\w+", a).group(0) if re.match(r"\w+", a) else "" for a in attrs]
    p.init = None
    p.alias = None
    p.kernel_decl = None
    p.call = p.name
    p.elems = 0
    t = re.sub(r"\b(const|volatile)\b", "", p.type).strip()
    p.base = t
    if p.type.strip() == "ThreadContext" or p.type.strip().endswith("ThreadContext"):
        p.kind = "builtin"
        p.init = (f"ThreadContext {p.name}; {p.name}.simd_lane_id = get_sub_group_local_id(); "
                  f"{p.name}.simdgroup_index = get_sub_group_id(); {p.name}.simdgroup_size = get_sub_group_size(); "
                  f"{p.name}.simdgroups_per_threadgroup = get_num_sub_groups(); "
                  f"{p.name}.threadgroup_position = uint3(get_group_id(0), get_group_id(1), get_group_id(2)); "
                  f"{p.name}.threadgroup_size = uint3(get_local_size(0), get_local_size(1), get_local_size(2)); "
                  f"{p.name}.threadgroup_count = uint3(get_num_groups(0), get_num_groups(1), get_num_groups(2)); "
                  f"{p.name}.grid_size = uint3(get_global_size(0), get_global_size(1), get_global_size(2));")
        return p
    for a in attr_names:
        if a in THREAD_ATTRS:
            fn = THREAD_ATTRS[a]
            vec = re.match(r"^(u?int|u?short|u?long)(\d)$", t)
            if vec:
                n = int(vec.group(2))
                comps = ", ".join(f"({vec.group(1)}){fn}({k})" for k in range(n))
                p.init = f"{t} {p.name} = {t}({comps});"
            else:
                p.init = f"{t} {p.name} = ({t}){fn}(0);"
            p.kind = "builtin"
            return p
        if a in SCALAR_ATTRS:
            p.init = f"{t} {p.name} = ({t}){SCALAR_ATTRS[a]};"
            p.kind = "builtin"
            return p
    for tag, fn in (("AXIS", "get_global_id"), ("GROUPS", "get_group_id"), ("THREADS", "get_local_id")):
        if tag in tags:
            ctr = {"AXIS": ax_counter, "GROUPS": grp_counter, "THREADS": thr_counter}[tag]
            idx = ctr[0]
            ctr[0] += 1
            p.init = f"{t} {p.name} = ({t}){fn}({min(idx, 2)});"
            p.kind = "builtin"
            return p
    if "threadgroup" in p.type:
        p.kind = "local"
        if p.dims:
            p.init = f"{p.decl};"
        elif "&" in p.type:
            elem = re.sub(r"\bthreadgroup\b|&|\bconst\b", "", p.type).strip()
            p.init = f"threadgroup {elem} {p.name}_m2v_storage; threadgroup {elem}& {p.name} = {p.name}_m2v_storage;"
        else:
            idx = tg_counter[0]
            tg_counter[0] += 1
            elem = re.sub(r"\bthreadgroup\b|\*|\bconst\b", "", p.type).strip() or "float"
            p.init = f"threadgroup {elem} {p.name}_m2v_storage[4096]; threadgroup {elem}* {p.name} = {p.name}_m2v_storage;"
        return p
    if "&" in p.type:
        base = p.type.replace("&", "").strip()
        p.kind = "ref"
        p.kernel_decl = f"{base}* {p.name}_m2v_p"
        p.alias = f"{p.type} {p.name} = *{p.name}_m2v_p;"
        p.call = p.name
        if not re.search(r"\b(device|constant)\b", base):
            p.kernel_decl = f"constant {base}* {p.name}_m2v_p"
        return p
    if "*" in p.type:
        p.kind = "buffer"
        p.kernel_decl = p.decl
        if not re.search(r"\b(device|constant|threadgroup)\b", p.type):
            p.kernel_decl = "device " + p.decl
        return p
    if p.dims:
        p.kind = "buffer"
        base = re.sub(r"\bconst\b", "", p.type).strip()
        p.kernel_decl = (base if re.search(r"\b(device|constant)\b", base) else "constant " + base) + "* " + p.name
        p.call = p.name
        return p
    if re.fullmatch(r"(const\s+)?bool", p.type.strip()):
        p.kind = "pod"
        p.kernel_decl = f"uint {p.name}_m2v_u"
        p.call = p.name
        p.alias = f"const bool {p.name} = ({p.name}_m2v_u != 0u);"
        return p
    p.kind = "pod"
    if not POD_TYPE.fullmatch(re.sub(r"\bconst\b", "", p.type).strip()):
        # a struct (or enum) by value is not a legal kernel argument: pass a pointer to constant memory and copy it in the body
        base = re.sub(r"\bconst\b", "", p.type).strip()
        p.kind = "ref"
        p.kernel_decl = f"constant {base}* {p.name}_m2v_p"
        p.call = p.name
        p.alias = f"{base} {p.name}; __builtin_memcpy(&{p.name}, {p.name}_m2v_p, sizeof({base}));"
        return p
    p.kernel_decl = re.sub(r"\bconst\b", "", p.type).strip() + " " + p.name
    return p


def template_header_before(text, pos):
    """If the text just before pos (whitespace skipped) ends a `template <...>` header, return (start, inner) else None."""
    j = pos - 1
    while j >= 0 and text[j] in " \t\r\n":
        j -= 1
    if j < 0 or text[j] != ">":
        return None
    depth = 0
    k = j
    while k >= 0:
        if text[k] == ">":
            depth += 1
        elif text[k] == "<":
            depth -= 1
            if depth == 0:
                break
        k -= 1
    if k < 0:
        return None
    m = re.search(r"\btemplate\s*$", text[:k])
    if not m:
        return None
    return m.start(), text[k + 1:j]


def tparam_names(inner):
    names = []
    for p in split_top(inner):
        p = re.sub(r"=.*$", "", p.strip())
        m = re.search(r"(\w+)\s*$", p)
        names.append(m.group(1) if m else p)
    return names


HOST_RE = re.compile(r"\btemplate\s*\[\[\s*host_name\s*\(\s*((?:\"[^\"]*\"\s*)+)\)\s*\]\]\s*(?:\[\[\s*kernel\s*\]\]|\bkernel\b)\s*"
                     r"(?:decltype\s*\(\s*(\w+)\s*<|void\s+(\w+)\s*<)")


DSL_KERNEL_RE = re.compile(r'\[\[\s*clang::annotate\(\s*""\s*,\s*"dsl\.kernel"\s*\)\s*\]\]\s*void\s+(\w+)\s*\(')


def attr_run_start(text, pos):
    """Start of the contiguous run of [[...]] attribute groups that ends just before pos."""
    while True:
        j = pos - 1
        while j >= 0 and text[j] in " \t\r\n":
            j -= 1
        if j >= 1 and text[j] == "]" and text[j - 1] == "]":
            k = text.rfind("[[", 0, j)
            if k < 0:
                return pos
            pos = k
        else:
            return pos


def find_entries(text):
    """Entry points of one file (comments blanked): dicts with name, style, params, head, body span, template info, instantiations."""
    entries = []
    pats = [(re.compile(r"(\[\[\s*kernel\b[^\]]*\]\]|\bkernel\b)\s*(?:\[\[[^\]]*\]\]\s*)*void\s+(\w+)\s*\("), "attr", 2),
            (DSL_KERNEL_RE, "dsl", 1)]
    for rx, style, gi in pats:
        for m in rx.finditer(text):
            close = balanced(text, m.end() - 1)
            if close < 0:
                continue
            j = close + 1
            while j < len(text) and (text[j] in " \t\r\n" or text.startswith("const", j)):
                j += 5 if text.startswith("const", j) else 1
            if j >= len(text) or text[j] != "{":
                continue  # a declaration or an explicit instantiation, not a definition
            bend = balanced(text, j, "{", "}")
            run = attr_run_start(text, m.start()) if style == "dsl" else m.start()
            th = template_header_before(text, run)
            variants = {}
            if style == "dsl":
                for am in re.finditer(r'\[\[\s*clang::annotate\(\s*""\s*,\s*"dsl\.variants"\s*,\s*"([^"]*)"\s*,\s*"((?:[^"\\\\]|\\\\.)*)"\s*\)\s*\]\]', text[run:m.start()]):
                    variants[am.group(1)] = split_top(am.group(2))
            entries.append({"style": style, "name": m.group(gi), "head": (m.start(), close + 1), "params": text[m.end():close],
                            "body": (j, bend), "tmpl": th, "inst": [], "close_paren": close, "variants": variants})
    byname = {}
    for e in entries:
        byname.setdefault(e["name"], []).append(e)
    for m in HOST_RE.finditer(text):
        name = m.group(2) or m.group(3)
        lt = m.end() - 1
        gt = balanced(text, lt, "<", ">")
        end = text.find(";", gt)
        for e in byname.get(name, []):
            e["inst"].append({"host": "".join(re.findall(r'"([^"]*)"', m.group(1))), "args": text[lt + 1:gt], "span": (m.start(), end + 1)})
    return entries


def variant_sets(entry, text):
    """-> list of (kernel name, template args text or '', error)."""
    th = entry["tmpl"]
    names = tparam_names(th[1]) if th else []
    if entry["style"] == "attr":
        if not names:
            return [(entry["name"], "", None)]
        if not entry["inst"]:
            return [(entry["name"], "", "template kernel without host_name instantiation")]
        return [(i["host"], i["args"], None) for i in entry["inst"]]
    if not names:
        return [(entry["name"], "", None)]
    var = entry["variants"]
    cons = dsl_constraints(text)
    out = []
    for tag, pick in (("v0", 0), ("v1", -1)):
        vals = []
        for n in names:
            vs = var.get(n)
            if not vs:
                return [(entry["name"], "", f"template parameter {n} has no VARIANTS")]
            vals.append(vs[pick])
        if cons:
            vals = repair_vals(names, var, cons, vals, pick)
        out.append((entry["name"] + "_" + tag, ", ".join(vals), None))
    if out[0][1] == out[1][1]:
        out = out[:1]
    return out


def dsl_constraints(text):
    """uzu CONSTRAINT(...) expressions (comments blanked): [[clang::annotate("", "dsl.constraint", EXPR)]]."""
    return [m2.group(1).replace('\\"', '"') for m2 in
            re.finditer(r'\[\[\s*clang::annotate\(\s*""\s*,\s*"dsl\.constraint"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\)\s*\]\]', text)]


def constraint_ok(expr, vals):
    """Evaluate one CONSTRAINT expression against {param: token}; anything unevaluable counts as satisfied."""
    s = " " + expr + " "
    s = re.sub(r"!(?!=)", " not ", s)
    s = re.sub(r"\b(\w+)::(\w+)", r"'\1::\2'", s)
    s = re.sub(r"\btrue\b", "True", s)
    s = re.sub(r"\bfalse\b", "False", s)
    s = s.replace("&&", " and ").replace("||", " or ")
    for k, v in vals.items():
        s = re.sub(r"\b" + re.escape(k) + r"\b", repr(v), s)

    def unquote(m):
        g = m.group(1)
        return g if g.isdigit() or g in ("True", "False") else "'" + g + "'"

    s = re.sub(r"'(\w+)'", unquote, s)
    try:
        env = {}
        exec("result = " + s, {"__builtins__": {}}, env)
    except Exception:
        return True  # a name outside the VARIANTS lists (a DSL constant): treat the constraint as satisfied
    return bool(env["result"])


def repair_vals(names, var, cons, vals, pick):
    """The first/last VARIANTS picks can violate the file's CONSTRAINT prunings (the real host never instantiates those
    tiles, the wrapper hits the static_asserts). Greedy repair: walk the violated constraints' parameters through their
    candidate values (the pick first, then the list order) until all constraints hold; give up unchanged if none helps."""
    pos = {n: i for i, n in enumerate(names)}
    vals = list(vals)
    pref = {n: [var[n][pick]] + [v for v in var[n] if v != var[n][pick]] for n in names}

    def bad(vs):
        return sum(0 if constraint_ok(c, dict(zip(names, vs))) else 1 for c in cons)

    best = bad(vals)
    for _ in range(4):
        if best == 0:
            break
        improved = False
        for c in cons:
            for n in names:
                if best == 0 or not re.search(r"\b" + re.escape(n) + r"\b", c):
                    continue
                for cand in pref[n]:
                    if cand == vals[pos[n]]:
                        continue
                    trial = list(vals)
                    trial[pos[n]] = cand
                    b = bad(trial)
                    if b < best:
                        vals, best, improved = trial, b, True
                        break
        if not improved:
            break
    return vals


def subst(s, vals):
    for k, v in vals.items():
        s = re.sub(r"\b" + re.escape(k) + r"\b", v, s)
    return s


TG_DECL = re.compile(r"\bthreadgroup\b([^;{}()=]*?);")


def hoist_threadgroup(text, entry):
    """Threadgroup declarations in the entry body -> members of a struct living in the wrapper kernel (a non-kernel function cannot
    declare __local variables). Returns (struct members [(type, name, dims)], edits [(start, end, replacement)])."""
    b0, b1 = entry["body"]
    members, edits, seen = [], [], {}
    for m in TG_DECL.finditer(text, b0, b1):
        decl = " ".join(m.group(1).split())
        parts = split_top(decl)
        t, name, dims = split_decl(parts[0])
        if not name:
            continue
        base = re.sub(r"[\s\*]+$", "", t)
        decls = [(t, name, dims)]
        for extra in parts[1:]:
            mm = re.match(r"^(\w+)\s*((?:\[[^\]]*\]\s*)*)$", extra.strip())
            if mm:
                decls.append((base, mm.group(1), mm.group(2)))
        repl = []
        for (ty, nm, dm) in decls:
            k = seen.get(nm, 0)
            seen[nm] = k + 1
            mem = nm if k == 0 else f"{nm}__{k}"
            members.append((ty, mem, dm))
            repl.append(f"auto& {nm} = m2v_tg.{mem};")
        edits.append((m.start(), m.end(), " ".join(repl)))
    return members, edits


def build_source(text0, blanked, entry, kname, targs, others=()):
    """-> (source text for the clang TU, wrapper source, params) or (None, None, params) when unsupported."""
    th = entry["tmpl"]
    names = tparam_names(th[1]) if th else []
    args = split_top(targs) if targs else []
    vals = dict(zip(names, args))
    if entry["style"] == "attr":
        return build_inplace(text0, blanked, entry, kname, targs, others)
    members, tg_edits = hoist_threadgroup(blanked, entry)
    tgname = "M2V_TG_" + re.sub(r"\W", "_", kname)
    tg_ref = ""
    if members:
        targs_txt = ("<" + ", ".join(names) + ">") if names else ""
        tg_ref = f"__local {tgname}{targs_txt}& m2v_tg"
    tg = [0]
    grp, thr, ax = [0], [0], [0]
    params = [classify(subst(raw, vals), tg, grp, thr, ax) for raw in split_top(entry["params"])]
    for p in params:
        if p.kind == "unsupported":
            return None, None, params
    edits = list(tg_edits)
    for o in others:
        oth = o["tmpl"]
        edits.append((oth[0] if oth else o["head"][0], o["body"][1] + 1, ""))
        for i in o["inst"]:
            edits.append((i["span"][0], i["span"][1], ""))
    if entry["style"] == "attr":
        plist = []
        for raw in split_top(entry["params"]):
            decl, _ = take_attrs(raw)
            plist.append(decl)
        if tg_ref:
            plist.append(tg_ref)
        s, e = entry["head"]
        edits.append((s, e, f"static inline void {entry['name']}__m2v({', '.join(plist)})"))
        for i in entry["inst"]:
            edits.append((i["span"][0], i["span"][1], ""))
    else:
        if tg_ref:
            cp = entry["close_paren"]
            edits.append((cp, cp, ", " + tg_ref))
    if members:
        struct = (f"{th and 'template <' + th[1] + '> ' or ''}struct {tgname} {{\n" +
                  "".join(f"  {ty} {mem}{dm};\n" for ty, mem, dm in members) + "};\n")
        pos = th[0] if th else entry["head"][0]
        if entry["style"] == "dsl":
            pos = th[0] if th else entry["head"][0]
        edits.append((pos, pos, struct))
    out, pos = [], 0
    for s, e, repl in sorted(edits, key=lambda x: (x[0], x[1])):
        out.append(text0[pos:s])
        out.append(repl)
        pos = max(pos, e)
    out.append(text0[pos:])
    src = "".join(out)
    impl = entry["name"] + "__m2v" if entry["style"] == "attr" else entry["name"]
    kargs = [p.kernel_decl for p in params if p.kernel_decl]
    body = [p.alias for p in params if p.alias]
    if members:
        body.append(f"__local {tgname}{('<' + targs + '>') if targs else ''} m2v_tg;")
    body += [p.init for p in params if p.init and p.kind == "local"]
    body += [p.init for p in params if p.init and p.kind == "builtin"]
    call = [p.call for p in params if p.kind in ("buffer", "ref", "pod", "builtin", "local")]
    if members:
        call.append("m2v_tg")
    tpl = f"<{targs}>" if targs else ""
    wrapper = (f"__kernel void {kname}({', '.join(kargs)}) {{\n  " + "\n  ".join(body) + f"\n  {impl}{tpl}({', '.join(call)});\n}}\n")
    return src, wrapper, params


def build_inplace(text0, blanked, entry, kname, targs, others):
    """Attribute-style entry -> a real __kernel function (template arguments substituted textually, so threadgroup declarations
    stay in kernel scope). Returns (source, "", params) or (None, None, params)."""
    th = entry["tmpl"]
    names = tparam_names(th[1]) if th else []
    vals = dict(zip(names, split_top(targs))) if targs else {}
    tg, grp, thr, ax = [0], [0], [0], [0]
    params = [classify(subst(raw, vals), tg, grp, thr, ax) for raw in split_top(entry["params"])]
    for p in params:
        if p.kind == "unsupported":
            return None, None, params
    pre = [p.alias for p in params if p.alias]
    pre += [p.init for p in params if p.init and p.kind == "local"]
    pre += [p.init for p in params if p.init and p.kind == "builtin"]
    kargs = [p.kernel_decl for p in params if p.kernel_decl]
    b0, b1 = entry["body"]
    body = subst(text0[b0 + 1:b1], vals)
    kern = f"__kernel void {kname}({', '.join(kargs)}) {{\n  " + "\n  ".join(pre) + "\n" + body + "}\n"
    own = th[0] if th else attr_run_start(text0, entry["head"][0])
    edits = [(own, b1 + 1, kern)]
    for i in entry["inst"]:
        edits.append((i["span"][0], i["span"][1], ""))
    for o in others:
        oth = o["tmpl"]
        st = oth[0] if oth else attr_run_start(text0, o["head"][0])
        edits.append((st, o["body"][1] + 1, ""))
        for i in o["inst"]:
            edits.append((i["span"][0], i["span"][1], ""))
    out, pos = [], 0
    for st, en, repl in sorted(edits, key=lambda x: (x[0], x[1])):
        if st < pos:
            continue
        out.append(text0[pos:st])
        out.append(repl)
        pos = en
    out.append(text0[pos:])
    return "".join(out), "", params


_STUB_DIR = tempfile.TemporaryDirectory(prefix="m2v-stubs-")  # empty stand-ins for the Metal system headers, removed at exit
STUB = pathlib.Path(_STUB_DIR.name)
for _n in [f.name for f in INC.iterdir() if f.name.startswith(("metal_", "simd"))] + ["metal_stdlib"]:
    (STUB / _n).write_text("")
(STUB / "MetalPerformancePrimitives").mkdir()
(STUB / "MetalPerformancePrimitives" / "MetalPerformancePrimitives.h").write_text("")
_PRE_CACHE = {}


def preprocess(path, include_dirs, defs):
    """clang -E -P with empty stand-ins for the Metal system headers: project includes and preprocessor conditionals are resolved,
    the Metal text (attributes, address spaces, kernel keywords) is left alone for the entry analysis."""
    key = (str(path), tuple(include_dirs), tuple(defs))
    if key in _PRE_CACHE:
        return _PRE_CACHE[key]
    cmd = [CLANG, "--target=spir", "-x", "cl", "-cl-std=clc++2021", "-E", "-P", "-w", "-cl-no-stdinc", "-I" + str(STUB), *defs,
           *[f"-I{x}" for x in include_dirs], str(path)]
    text = None
    for _ in range(8):
        rc, out, _t = sh(cmd)
        if rc == 0:
            text = out
            break
        miss = re.search(r"'((?:metal|simd)[\w/]*)(?:\.h)?' file not found", out)
        if not miss:
            break
        stub = STUB / miss.group(1)
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text("")
    if text is None:
        text = path.read_text(errors="replace")
    _PRE_CACHE[key] = text
    return text


def undeclared_macros(errs):
    """ALL_CAPS names the compiler reports as undeclared: host-injected macros (the Zig host passes them as -D). -> {name: kind}"""
    out = {}
    for ln in errs:
        m = re.search(r"use of undeclared identifier '([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)'", ln)
        if m:
            out[m.group(1)] = "value"
        m = re.search(r"unknown type name '([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)'", ln)
        if m:
            out[m.group(1)] = "type"
    return out


MACRO_DEFAULT = "64"


TF_DEFAULTS = {"TF_UNROLL": "4", "TF_COL": "4", "TF_GROUP": "64", "TF_OUT_T": "bfloat", "TF_D": "2048", "TF_TOPK": "8",
               "TF_SIMD_FRAGS": "4", "TF_FLOOR": "0", "TF_E": "64", "TF_BITS": "4", "TF_MAXR": "4", "TF_LOW": "0",
               "TF_GATHER_SCATTER": "0", "TF_GATHER": "0", "TF_SIMD_LAYOUT": "0", "TF_K": "64", "TF_ITERS": "4", "TF_HC_EPS_INT": "1000"}

# Host-injected symbols: the TensorFold host prepends a per-family preamble before the kernel text; the checkout carries the
# same code inline in sibling kernels (glm/kda_rows.metal, flashnext/*.metal, nax.h). When a parse fails because one of these
# symbols is missing, the group is injected before the kernel text and recorded in the row ("host_syms"). Definitions are
# verbatim host code (mma_16x32 stays out behind M2V_NAX_MPP: it needs the mpp tensor-op emulation).
HOST_SYMBOL_TRIGGERS = {
    "tf_frag": ("frag", "frag_home", "frag_get", "frag_get_in", "frag_put", "frag_put_in"),
    "tfq6_store": ("tfq6::store", "tfq6::k_loop6"),
    "tf_math": ("bsig", "bsilu", "fsig", "fsoftplus", "log1p_", "simd_topk", "simd_topk_all"),
    "kda_quad_dot": ("quad_dot",),
    "kda_sigmoid": ("mlx_sigmoid_precise",),
    "kda_sq_acc": ("sq_acc",),
    "fz_tile": ("fz_tile",),
}
HOST_SYMBOL_TEXT = {
    "kda_quad_dot": """
template <int BITS, int PER>
inline float quad_dot(device const bfloat* x, device const uint8_t* wb, float s, float bb);
template <>
inline float quad_dot<4, 32>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 32;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], e = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(e)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(e) / 4096.0f;
  }
  device const uint16_t* ws = (device const uint16_t*)wb;
  float accum = 0.0f;
  for (int i = 0; i < PER / 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
template <>
inline float quad_dot<4, 16>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 16;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], e = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(e)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(e) / 4096.0f;
  }
  device const uint16_t* ws = (device const uint16_t*)wb;
  float accum = 0.0f;
  for (int i = 0; i < PER / 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
template <>
inline float quad_dot<8, 32>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 32;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  float accum = 0.0f;
  for (int i = 0; i < PER; i++) accum += xt[i] * wb[i];
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
template <>
inline float quad_dot<8, 16>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 16;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  float accum = 0.0f;
  for (int i = 0; i < PER; i++) accum += xt[i] * wb[i];
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
""",
    "kda_sigmoid": """
template <typename U>
inline U mlx_sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}
""",
    "kda_sq_acc": """
#pragma clang fp contract(off)
inline float sq_acc(float acc, float v) {
  return v * v + acc;
}
#pragma clang fp contract(on)
""",
    "tf_math": """
// Elementwise ops as the checkpoint's training framework does them on bf16 tensors: fp32 math, one rounding.
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float bsilu(float x) { return float(bfloat(x / (1.0f + metal::exp(-x)))); }
inline float fsig(float x) { return 1.0f / (1.0f + metal::exp(-x)); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }
// Rank-k expert of a row inside one simdgroup: lane l holds logits l, l + 32, ...; rounds of (largest logit,
// lowest id); returns the id picked in round k and, through ``picked``, the logits of rounds 0..k.
template <int NE>
inline int simd_topk(const device float* logits, int k, uint lane, thread float* picked) {
  float v[NE / 32];
  for (int j = 0; j < NE / 32; j++) v[j] = logits[j * 32 + int(lane)];
  int id = 0;
  for (int round = 0; round <= k; round++) {
    float best = -INFINITY;
    int bid = NE;
    for (int j = 0; j < NE / 32; j++) {
      const int e = j * 32 + int(lane);
      if (v[j] > best || (v[j] == best && e < bid)) { best = v[j]; bid = e; }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const int oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    picked[round] = best;
    id = bid;
    if (int(lane) == bid % 32) v[bid / 32] = -INFINITY;
  }
  return id;
}
// simd_topk's rounds 0 .. TOPK-1 in one pass: ids[k] and logits picked[k] of each round
template <int NE, int TOPK>
inline void simd_topk_all(const device float* logits, uint lane, thread int* ids, thread float* picked) {
  float v[NE / 32];
  for (int j = 0; j < NE / 32; j++) v[j] = logits[j * 32 + int(lane)];
  for (int round = 0; round < TOPK; round++) {
    float best = -INFINITY;
    int bid = NE;
    for (int j = 0; j < NE / 32; j++) {
      const int e = j * 32 + int(lane);
      if (v[j] > best || (v[j] == best && e < bid)) { best = v[j]; bid = e; }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const int oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    picked[round] = best;
    ids[round] = bid;
    if (int(lane) == bid % 32) v[bid / 32] = -INFINITY;
  }
}
""",
    "tf_frag": """
// M5 tensor-unit fragments shared by every NAX kernel (nax.h): layout, loads, stores.
namespace tfp {
using namespace metal;
// Full unroll: register arrays indexed by a loop counter stay in registers only when the loop unrolls.
#define TF_UNROLL _Pragma("clang loop unroll(full)")
// One simdgroup's 16x16 fragment: 8 values a lane.
template <typename T>
using frag = vec<T, 8>;
#ifndef TF_SIMD_FRAGS
// Lane l holds rows home.y and home.y + 8, columns home.x .. home.x + 3 of every fragment (the M5 operand layout).
inline short2 frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}
// The 16x16 block at (r, c) of a row-major matrix with leading dimension ld, in device or threadgroup memory.
// The element type is fixed per overload: metal::vec<T, 8> cannot carry a deducible T (the alias hides it), and the
// ext_vector spelling cannot be formed for bfloat; the bodies are the host's.
template <typename P>
inline void frag_get(thread frag<float>& f, P p, int ld, int r, int c, short2 home) {
  const P q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = float(q[(e >> 2) * 8 * ld + (e & 3)]);
  }
}
template <typename P>
inline void frag_get(thread frag<bfloat16_t>& f, P p, int ld, int r, int c, short2 home) {
  const P q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = bfloat16_t(q[(e >> 2) * 8 * ld + (e & 3)]);
  }
}
// frag_get with zeros outside rows < nr and columns < nc (both relative to p); nothing outside is read.
template <typename S>
inline void frag_get_in(thread frag<float>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    f[e] = (rr < nr && cc < nc) ? float(p[rr * ld + cc]) : float(0);
  }
}
template <typename S>
inline void frag_get_in(thread frag<bfloat16_t>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    f[e] = (rr < nr && cc < nc) ? bfloat16_t(p[rr * ld + cc]) : bfloat16_t(0);
  }
}
template <typename O>
inline void frag_put(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home) {
  device O* q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    q[(e >> 2) * 8 * ld + (e & 3)] = O(f[e]);
  }
}
template <typename O>
inline void frag_put_in(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    if (rr < nr && cc < nc) {
      p[rr * ld + cc] = O(f[e]);
    }
  }
}
// Element e's column past home.x (a plain index: M5 kernels keep their exact text); its row is home.y + (e >> 2) * 8.
#define TF_COL(e) (e & 3)
// A transposed right operand's block (its rows are the op's columns): on M5 it loads like any other.
#define frag_get_t frag_get
#define frag_get_t_in frag_get_in
#else
// TF_SIMD_FRAGS (before M5, 8x8 simdgroup matrices): lane l holds rows home.y, home.y + 8, columns home.x + {0,1,8,9}.
inline short2 frag_home(ushort l) {
  return short2(short(((l & 8) >> 1) | ((l & 1) << 1)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}
#define TF_COL(e) (((e) & 1) + 8 * (((e) >> 1) & 1))
template <typename T, typename P>
inline void frag_get(thread frag<T>& f, P p, int ld, int r, int c, short2 home) {
  const P q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = T(q[(e >> 2) * 8 * ld + TF_COL(e)]);
  }
}
template <typename T, typename S>
inline void frag_get_in(thread frag<T>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + TF_COL(e);
    f[e] = (rr < nr && cc < nc) ? T(p[rr * ld + cc]) : T(0);
  }
}
// The block at (r, c) as a transposed right operand: its rows on the column pattern, its columns on the row pattern.
template <typename T, typename P>
inline void frag_get_t(thread frag<T>& f, P p, int ld, int r, int c, short2 home) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = T(p[(r + home.x + TF_COL(e)) * ld + c + home.y + (e >> 2) * 8]);
  }
}
template <typename T, typename S>
inline void frag_get_t_in(thread frag<T>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.x + TF_COL(e), cc = c + home.y + (e >> 2) * 8;
    f[e] = (rr < nr && cc < nc) ? T(p[rr * ld + cc]) : T(0);
  }
}
template <typename O>
inline void frag_put(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home) {
  device O* q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    q[(e >> 2) * 8 * ld + TF_COL(e)] = O(f[e]);
  }
}
template <typename O>
inline void frag_put_in(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + TF_COL(e);
    if (rr < nr && cc < nc) {
      p[rr * ld + cc] = O(f[e]);
    }
  }
}
#endif
// Declared so the calls parse; the definition needs the mpp tensor-op emulation (define M2V_NAX_MPP and add it). The
// element types are fixed: the shim's vec alias cannot deduce them (16x32x16, C accumulates in float, A/B bf16).
template <bool TA, bool TB>
inline void mma_16x32(thread frag<float>& lo, thread frag<float>& hi, thread const frag<bfloat16_t>& a,
                      thread const frag<bfloat16_t>& b0, thread const frag<bfloat16_t>& b1);
}  // namespace tfp
using namespace tfp;
""",
    "tfq6_store": """
// The tfq6 helpers of the qmm6 family (prefill/qmm6_nax.metal; the b-file kernels share the namespace's helpers).
template <typename T, typename O>
inline void dequant6r(uint2 a, uint2 c, uint2 d, T scale, T bias, threadgroup O* out) {
  const float s = float(scale), b = float(bias);
  const ulong lo = ulong(a.x) | (ulong(a.y) << 32), mid = ulong(c.x) | (ulong(c.y) << 32), hi = ulong(d.x) | (ulong(d.y) << 32);
  TF_UNROLL
  for (int i = 0; i < 32; i++) {
    const int bit = 6 * i;
    uint v;
    if (bit + 6 <= 64) v = uint(lo >> bit) & 63u;
    else if (bit < 64) v = (uint(lo >> bit) | uint(mid << (64 - bit))) & 63u;
    else if (bit + 6 <= 128) v = uint(mid >> (bit - 64)) & 63u;
    else if (bit < 128) v = (uint(mid >> (bit - 64)) | uint(hi << (128 - bit))) & 63u;
    else v = uint(hi >> (bit - 128)) & 63u;
    out[i] = O(static_cast<T>(s * float(v) + b));
  }
}
namespace tfq6 {
// A simdgroup's TM x 2 fragments to y (row stride ld): rows below live, columns below nc.
template <typename T, int TM>
inline void store(thread const frag<float> (&acc)[TM][2], device T* y, int ld, int live, int nc, short2 home) {
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      if (live == 16 * TM && nc >= 32) {
        frag_put(acc[i][j], y, ld, 16 * i, 16 * j, home);
      } else {
        frag_put_in(acc[i][j], y, ld, 16 * i, 16 * j, home, live, nc);
      }
    }
  }
}
// x (TM 16-row fragments, ld K) times a 6-bit [64 rows, K] block, 64 deep a step; thread t dequantizes row t / 2's group t % 2.
template <typename T, int TM>
inline void k_loop6(thread frag<float> (&acc)[TM][2], const device T* x, int K, int ldx, int live, bool inside,
                    const device uint* wq, const device T* scales, const device T* biases, threadgroup T* tile,
                    int tn, uint t, short2 home) {
  constexpr int PAD = 64 + 16 / sizeof(T);
  threadgroup T* mine = tile + (t / 2) * PAD + 32 * (t % 2);
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  uint2 wa = *(const device uint2*)(wq), wb = *(const device uint2*)(wq + 2), wc = *(const device uint2*)(wq + 4);
  T sc = *scales, bi = *biases;
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dequant6r<T>(wa, wb, wc, sc, bi, mine);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k + 64 < K) {
      wq += 12;
      scales += 2;
      biases += 2;
      wa = *(const device uint2*)(wq);
      wb = *(const device uint2*)(wq + 2);
      wc = *(const device uint2*)(wq + 4);
      sc = *scales;
      bi = *biases;
    }
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < 64; kk += 32) {
      if (live > 0) {
        frag<T> a[TM][2], b[2][2];
        TF_UNROLL
        for (short i = 0; i < 2; i++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            frag_get_t(b[j][i], (const threadgroup T*)tile, PAD, tn + 16 * i, kk + 16 * j, home);
          }
        }
        TF_UNROLL
        for (short i = 0; i < TM; i++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            if (inside) {
              frag_get(a[i][j], x, ldx, 16 * i, kk + 16 * j, home);
            } else {
              frag_get_in(a[i][j], x, ldx, 16 * i, kk + 16 * j, home, live, kk + 32);
            }
          }
        }
        TF_UNROLL
        for (short m = 0; m < TM; m++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            mma_16x32<false, true>(acc[m][0], acc[m][1], a[m][j], b[j][0], b[j][1]);
          }
        }
      }
    }
    x += 64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
}  // namespace tfq6
""",
    "fz_tile": """
// The includer defines fz_tile (a threadgroup's output tile); identity is the host default mapping.
inline int fz_tile(int tgx) { return tgx; }
""",
}


def missing_host_symbols(errs):
    """Identifiers the compiler reports as missing that the TensorFold host preambles provide -> group names to inject."""
    seen = set()
    for ln in errs:
        for m in re.finditer(r"(?:use of undeclared identifier|no template named|unknown type name) '([^']+)'", ln):
            seen.add(m.group(1))
        m = re.search(r"no member named '([^']+)' in namespace '([^']+)'", ln)
        if m:
            seen.add(m.group(2) + "::" + m.group(1))
        m = re.search(r"no matching function for call to '([^']+)'", ln)
        if m:
            seen.add(m.group(1))
    # HOST_SYMBOL_TRIGGERS order is the dependency order (tf_frag defines the macros tfq6_store uses).
    return [g for g, ids in HOST_SYMBOL_TRIGGERS.items() if seen.intersection(ids)]


# Macros an injected group defines (TF_UNROLL, TF_COL, frag_get_t, ...): the retry must not also -D them to a dummy value.
HOST_GROUP_MACROS = {g: set(re.findall(r"^\s*#\s*define\s+(\w+)", t, re.M)) for g, t in HOST_SYMBOL_TEXT.items()}


def sh(cmd, timeout=300):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return r.returncode, r.stdout + r.stderr, time.time() - t0
    except subprocess.TimeoutExpired:
        return 124, "timeout", time.time() - t0


def norm_msg(msg):
    msg = re.sub(r"^.*? error: ", "", msg)
    msg = re.sub(r"'[^']*'", "'X'", msg)
    msg = re.sub(r"\d+", "N", msg)
    return msg[:100]


def feature_keys(errlines):
    """Missing-feature keys from clang/clspv error text: kind:identifier."""
    keys = set()
    for ln in errlines:
        m = re.search(r"unknown type name '([^']+)'", ln)
        if m:
            keys.add("type:" + re.sub(r"\d+", "N", m.group(1)))
            continue
        m = re.search(r"use of undeclared identifier '([^']+)'", ln)
        if m:
            keys.add("ident:" + m.group(1))
            continue
        m = re.search(r"no member named '([^']+)' in (?:namespace )?'([^']+)'", ln)
        if m:
            keys.add(f"member:{m.group(2)}::{m.group(1)}")
            continue
        m = re.search(r"no matching (?:function|member function|constructor) for (?:call to|initialization of) '?([^'(]+)", ln)
        if m:
            keys.add("overload:" + m.group(1).strip())
            continue
        m = re.search(r"no template named '([^']+)'", ln)
        if m:
            keys.add("template:" + m.group(1))
            continue
        m = re.search(r"unknown namespace '([^']+)'|no namespace named '([^']+)'", ln)
        if m:
            keys.add("namespace:" + (m.group(1) or m.group(2)))
            continue
        m = re.search(r"file not found", ln)
        if m:
            inc = re.search(r"'([^']+)' file not found", ln)
            keys.add("include:" + (inc.group(1) if inc else "?"))
            continue
        if "error:" in ln:
            keys.add("other:" + norm_msg(ln))
    return keys


def process(job):
    (setname, src_path, rel, eidx, kname, targs, outdir, include_dirs, use_spv, defs, syms) = job
    defs = list(defs)
    syms = list(syms)
    row = None
    for _ in range(4):
        row, undecl, missing = process_once(setname, src_path, rel, eidx, kname, targs, outdir, include_dirs, use_spv, defs, syms)
        new = sorted(n for n in undecl if not any(d.startswith("-D" + n + "=") for d in defs))
        add_syms = [g for g in missing if g not in syms] if setname == "tf" and row["parse"] == "FAIL" else []
        if "tf_frag" in add_syms and "namespace tfp" in src_path.read_text(errors="replace"):
            add_syms.remove("tf_frag")  # the file defines its own tfp (qmm_nax): its frag_* overloads are that slice's gap
        if add_syms:
            skip = set().union(*(HOST_GROUP_MACROS[g] for g in add_syms + syms))
            new = [n for n in new if n not in skip]
        if setname != "tf" or row["parse"] != "FAIL" or (not new and not add_syms):
            break
        defs += [f"-D{n}={TF_DEFAULTS.get(n, 'bfloat' if undecl[n] == 'type' else MACRO_DEFAULT)}" for n in new]
        syms += add_syms
    if defs:
        row["defs"] = [d[2:] for d in defs]
    if syms:
        row["host_syms"] = syms
    return row


def mentions_mpp(path, include_dirs, seen=None):
    """The wrapper must prepend the real MetalPerformancePrimitives header when the source pulls it in, directly or through
    project includes (nax.h, uzu's ops headers) that preprocessing resolves against empty stubs."""
    text = path.read_text(errors="replace")
    if "MetalPerformancePrimitives" in text:
        return True
    seen = seen if seen is not None else {str(path)}
    for m in re.finditer(r'#\s*include\s*"([^"]+)"', text):
        for base in [path.parent, *[pathlib.Path(x) for x in include_dirs]]:
            p = base / m.group(1)
            if p.exists() and str(p) not in seen:
                seen.add(str(p))
                if mentions_mpp(p, include_dirs, seen):
                    return True
    return False


def process_once(setname, src_path, rel, eidx, kname, targs, outdir, include_dirs, use_spv, defs, syms=()):
    row = {"set": setname, "file": rel, "entry": kname, "variant": targs[:60] if targs else "-", "parse": "", "ir": "", "spv": "",
           "val": "", "error": "", "features": [], "kernel": "", "args": []}
    text0 = preprocess(src_path, include_dirs, defs + (["-DDSL_ANALYZE"] if setname == "uzu" else []))
    blanked = blank_comments(text0)
    allent = find_entries(blanked)
    if eidx >= len(allent):
        row["parse"] = "FAIL"
        row["error"] = "entry not found after preprocessing"
        row["features"] = [row["error"]]
        return row, {}, []
    entry = allent[eidx]
    src, wsrc, params = build_source(text0, blanked, entry, kname, targs, [o for i, o in enumerate(allent) if i != eidx])
    if src is None:
        row["parse"] = "FAIL"
        row["error"] = "wrapper: " + next((getattr(p, "note", "") for p in params if p.kind == "unsupported"), "unsupported parameter")
        row["features"] = [row["error"]]
        return row, {}, []
    row["kernel"] = kname
    row["args"] = [{"kind": p.kind, "name": p.name, "type": p.type, "dims": p.dims, "tags": {k: v for k, v in p.tags.items()},
                    "attrs": p.attrs} for p in params]
    tag = (re.sub(r"[^\w]", "_", rel)[-40:] + "__" + re.sub(r"[^\w]", "_", kname)[:40] + "_" +
           hashlib.sha1((kname + targs).encode()).hexdigest()[:8])
    row["tag"] = tag
    d = pathlib.Path(outdir)
    cl = d / "src" / (tag + ".cl")
    cl.parent.mkdir(parents=True, exist_ok=True)
    pre = '#include "metal_stdlib"\n'
    pre += "".join(HOST_SYMBOL_TEXT[g] for g in syms)
    if mentions_mpp(src_path, include_dirs):
        pre += '#include "MetalPerformancePrimitives/MetalPerformancePrimitives.h"\n'
    cl.write_text(pre + VEC_ALIASES + norm_decls(strip_annotate(src)) + '\n#line 1 "m2v-wrapper"\n' + wsrc)
    ll = d / "ll" / (tag + ".ll")
    ll.parent.mkdir(parents=True, exist_ok=True)
    cmd = [CLANG, "--target=spir", "-x", "cl", "-cl-std=clc++2021", "-Xclang", "-finclude-default-header",
           "-cl-ext=-__opencl_c_generic_address_space", *FE_OPT, "-fno-strict-return", "-cl-kernel-arg-info", "-w", "-ferror-limit=100",
           *M2V_DEFS, *defs, f"-I{INC}", *[f"-I{x}" for x in include_dirs], "-S", "-emit-llvm", str(cl), "-o", str(ll)]
    rc, out, _ = sh(cmd)
    errs = [ln for ln in out.splitlines() if " error: " in ln or "fatal error" in ln]
    if rc != 0 or not ll.exists() or ll.stat().st_size == 0:
        row["parse"] = "FAIL"
        row["error"] = norm_msg(errs[0]) if errs else (out.strip().splitlines() or ["?"])[-1][:100]
        row["features"] = sorted(feature_keys(errs))
        row["primary"] = sorted(feature_keys(errs[:1]))[0] if errs else "other:?"
        lines = out.splitlines()
        for i, ln in enumerate(lines):
            if " error: " in ln or "fatal error" in ln:
                row["detail"] = "\n".join(lines[i:i + 3])[:600]
                break
        return row, undeclared_macros(errs), missing_host_symbols(errs)
    row["parse"] = "ok"
    txt = ll.read_text()
    if "-O0" in FE_OPT:
        # clang marks every -O0 function noinline; drop that, then inline and SROA with LLVM's own passes
        ll.write_text(re.sub(r"\bnoinline\b", "", txt))
        rc, out, _ = sh([OPT, "-passes=cgscc(inline),function(sroa,early-cse,simplifycfg)", "-inline-threshold=100000", "-S", str(ll), "-o", str(ll) + ".opt"])
        if rc != 0 or not pathlib.Path(str(ll) + ".opt").exists():
            row["ir"] = "FAIL"
            row["error"] = ([x for x in out.splitlines() if x.strip() and "failed to create target machine" not in x] or [f"opt exit {rc}"])[0][:140]
            row["features"] = ["opt:" + norm_msg(row["error"])]
            row["primary"] = row["features"][0]
            return row, {}, []
        pathlib.Path(str(ll) + ".opt").replace(ll)
        txt = ll.read_text()
    row["ir"] = "ok"
    if not use_spv:
        return row, {}, []
    txt = re.sub(r", !(alias\.scope|noalias) ![0-9]+", "", txt)
    txt = re.sub(r"^\s*(?:tail |musttail |notail )?call void @llvm\.experimental\.noalias\.scope\.decl\(.*\)\s*$", "", txt, flags=re.M)
    ll.write_text(txt)
    # the shim's bfloat is a real __bf16 on clang 20+; clspv rejects that LLVM type, so retype it to i16 first (no-op otherwise)
    if re.search(r"(?<![\w.])bfloat\b", txt):
        rc, out, _ = sh([sys.executable, str(HERE / "bfloat_to_i16.py"), str(ll)])
        if rc != 0:
            row["ir"] = "FAIL"
            row["error"] = (out.strip().splitlines() or ["bfloat_to_i16 failed"])[-1][:140]
            row["features"] = ["bfp:" + norm_msg(row["error"])]
            row["primary"] = row["features"][0]
            return row, {}
        txt = ll.read_text()
    spv = d / "spv" / (tag + ".spv")
    spv.parent.mkdir(parents=True, exist_ok=True)
    rc, out, _ = sh([CLSPV, "-x", "ir", "--cl-std=CLC++2021", "--fp16", "--inline-entry-points", "--spv-version=1.5", str(ll), "-o", str(spv)], CLSPV_TIMEOUT)
    lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("warning: ")]
    if rc != 0 or not spv.exists() or spv.stat().st_size == 0:
        row["spv"] = "FAIL"
        row["error"] = (lines[0] if lines else (f"clspv exit {rc} without a diagnostic (crash)" if rc != 124 else f"timeout after {CLSPV_TIMEOUT} s (clspv does not finish)"))[:140]
        row["features"] = ["clspv:" + norm_msg(row["error"])]
        row["primary"] = row["features"][0]
        row["detail"] = "\n".join(lines[:6])[:600]
        return row, {}, []
    row["spv"] = "ok"
    rc, out, _ = sh([SPIRV_VAL, "--target-env", "vulkan1.3", str(spv)])
    if rc != 0:
        row["val"] = "FAIL"
        row["error"] = norm_msg((out.strip().splitlines() or ["?"])[0])
        row["features"] = ["spirv-val:" + row["error"]]
        row["primary"] = row["features"][0]
        row["detail"] = out.strip()[:400]
        return row, {}, []
    row["val"] = "ok"
    return row, {}, []


def inventory(args):
    jobs, rows = [], []
    sets = []
    if args.set in ("uzu", "all"):
        root = pathlib.Path(os.environ["UZU_ROOT"])
        k = root / "crates/uzu-engine/src/backends/metal/kernel"
        sets.append(("uzu", k, [k, k / "generated", k / "common"]))
    if args.set in ("tf", "all"):
        root = pathlib.Path(os.environ["TF_ROOT"])
        k = root / "zig/kernels/metal"
        sets.append(("tf", k, [k, k.parent, k / "core"]))
    use_spv = not args.no_spv and pathlib.Path(CLSPV).exists()
    for name, base, incs in sets:
        for f in sorted(base.rglob("*.metal")):
            rel = str(f.relative_to(base))
            if args.only and args.only not in rel:
                continue
            fincs = incs + [f.parent, f.parent.parent]
            text0 = preprocess(f, fincs, ["-DDSL_ANALYZE"] if name == "uzu" else [])
            blanked = blank_comments(text0)
            ents = find_entries(blanked)
            defs = []
            if not ents:
                rows.append({"set": name, "file": rel, "entry": "(none)", "variant": "-", "parse": "n/a", "ir": "", "spv": "", "val": "",
                             "error": "no entry point (header/library file)", "features": [], "kernel": "", "args": []})
                continue
            for ei, e in enumerate(ents):
                for kname, targs, err in variant_sets(e, blanked):
                    if err:
                        rows.append({"set": name, "file": rel, "entry": kname, "variant": "-", "parse": "FAIL", "ir": "", "spv": "", "val": "",
                                     "error": "wrapper: " + err, "features": ["wrapper:" + err], "kernel": "", "args": []})
                    else:
                        jobs.append((name, f, rel, ei, kname, targs, args.out, fincs, use_spv, defs, []))
    return jobs, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--set", default="all")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-spv", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--retry-timeouts", metavar="RESULT.json", help="rerun only the entries whose clspv run timed out in RESULT.json and merge")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs, rows = inventory(args)
    kept = []
    if args.retry_timeouts:
        old_rows = json.load(open(args.retry_timeouts))
        redo = {(r["file"], r["entry"]) for r in old_rows if r.get("spv") == "FAIL" and r["error"].startswith("timeout")}
        jobs = [j for j in jobs if (j[2], j[4]) in redo]
        kept = [r for r in old_rows if (r["file"], r["entry"]) not in redo]
        rows = []
    if args.limit:
        jobs = jobs[:args.limit]
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for i, r in enumerate(ex.map(process, jobs)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} entries, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    rows = kept + rows
    name = args.tag or "sweep"
    json.dump(rows, open(os.path.join(args.out, name + ".json"), "w"), indent=1)
    for s in ("uzu", "tf"):
        rs = [r for r in rows if r["set"] == s and r["parse"] != "n/a"]
        if rs:
            n = len(rs)
            ir, sp, va = (sum(r[k] == "ok" for r in rs) for k in ("ir", "spv", "val"))
            print(f"{s}: entries {n} ir_ok {ir} ({100 * ir / n:.1f}%) spv_ok {sp} val_ok {va} ({100 * va / n:.1f}%)")
    print(f"{len(rows)} rows, {time.time() - t0:.0f} s wall")


if __name__ == "__main__":
    main()
