#!/usr/bin/env python3
"""Rewrite LLVM IR text: the bfloat type (the shim's clang 20+ __bf16 spelling) becomes i16.

clspv rejects the LLVM bfloat type. The memory layout is identical (16 bits), so every bfloat
SSA value, memory operand, GEP source type, signature and constant is retyped to i16 unchanged.
Only the conversion edges and the arithmetic ops carry semantics, and they expand to float ops
plus integer bit twiddling:

  fpext bfloat %x to float   ->  zext i16 %x to i32; shl i32 .., 16; bitcast i32 .. to float
                                 (bf16 is a truncated float32, so widening is exact)
  fptrunc float %x to bfloat ->  bitcast float .. to i32; (u + 0x7fff + ((u >> 16) & 1)) >> 16;
                                 trunc to i16 (round to nearest even, byte-identical to the
                                 formula the pre-typedef shim struct used)
  fadd/fsub/fmul/fdiv/frem/fneg bfloat  ->  widen both operands, float op, narrow (a bf16 op is
                                 an fp32 op plus round to nearest even)
  fcmp bfloat                ->  widen both operands, fcmp float
  call/declare of an @llvm.*.bf16 intrinsic  ->  the same intrinsic at .f32 with widened operands
                                 and a narrowed result

Vector forms (<N x bfloat>) expand lane-wise. bfloat constants widen exactly: the decimal
spelling is representable in f32, the 0xHhhhh spelling shifts left by 16. Anything unparseable
fails loudly instead of being silently retyped. Runs as a text pass between the clang/opt front
end and clspv; a module without a bfloat type is a no-op.

Usage: bfloat_to_i16.py FILE.ll
"""
import pathlib
import re
import sys

VT = r"(?:<(\d+) x )?(bfloat|float)(?: x \d+)?>?"  # scalar or vector float type, lane count in group 1


def die(msg):
    sys.exit("bfloat_to_i16: " + msg)


def splat(lanes, text):
    return "<" + ", ".join([text] * lanes) + ">"


def widen_value(dst, tok, lanes, out):
    """Emit code making a float value named dst out of an i16-bits register operand."""
    if not tok.startswith("%"):
        die(f"expected a register operand to widen, got: {tok}")
    v16, v32, vf = (("i16", "i32", "float") if not lanes else
                    (f"<{lanes} x i16>", f"<{lanes} x i32>", f"<{lanes} x float>"))
    sh = splat(lanes, "i32 16") if lanes else "16"
    out.append(f"  %w{dst}a = zext {v16} {tok} to {v32}\n")
    out.append(f"  %w{dst}b = shl {v32} %w{dst}a, {sh}\n")
    out.append(f"  %{dst} = bitcast {v32} %w{dst}b to {vf}\n")


def narrow_value(dst, tok, lanes, out):
    """Emit code making an i16-bits value named dst out of a float operand (register or constant)."""
    v32, v16 = (("i32", "i16") if not lanes else (f"<{lanes} x i32>", f"<{lanes} x i16>"))
    sf = f"<{lanes} x float>" if lanes else "float"
    s16 = splat(lanes, "i32 16") if lanes else "16"
    s1 = splat(lanes, "i32 1") if lanes else "1"
    s32767 = splat(lanes, "i32 32767") if lanes else "32767"
    out.append(f"  %n{dst}u = bitcast {sf} {tok} to {v32}\n")
    out.append(f"  %n{dst}h = lshr {v32} %n{dst}u, {s16}\n")
    out.append(f"  %n{dst}l = and {v32} %n{dst}h, {s1}\n")
    out.append(f"  %n{dst}a = add {v32} %n{dst}u, {s32767}\n")
    out.append(f"  %n{dst}r = add {v32} %n{dst}a, %n{dst}l\n")
    out.append(f"  %n{dst}t = lshr {v32} %n{dst}r, {s16}\n")
    out.append(f"  %{dst} = trunc {v32} %n{dst}t to {v16}\n")


def widen_tok(tok, name, lanes, out, typed=False):
    """Operand token for a float op: constants convert in place, registers get widened. With typed=True the
    operand carries its float type spelling (intrinsic call arguments need it)."""
    if tok.startswith("0xH"):
        hexs = f"0x{int(tok[3:], 16) << 16:08x}"
        const = splat(lanes, f"float {hexs}") if lanes else hexs
        return f"float {const}" if typed else const
    if re.fullmatch(r"[-\d.e+]+", tok):
        const = splat(lanes, f"float {tok}") if lanes else tok
        return f"float {const}" if typed else const
    widen_value(name, tok, lanes, out)
    reg = f"%{name}"
    return f"{f'<{lanes} x float>' if lanes else 'float'} {reg}" if typed else reg


def arg_tokens(rest):
    """Split a call argument list on top-level commas (vector constants contain commas)."""
    args, depth, cur = [], 0, ""
    for ch in rest:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def bf16_bits(dec):
    """The bf16 bit pattern of a decimal float spelling (exact: LLVM prints bf16 values, which widen losslessly)."""
    import struct
    f = struct.unpack(">I", struct.pack(">f", float(dec)))[0]
    if f != f:
        return 0x7FC0
    return (f + 0x7FFF + ((f >> 16) & 1)) >> 16


def main():
    if len(sys.argv) != 2:
        die("usage: bfloat_to_i16.py FILE.ll")
    p = pathlib.Path(sys.argv[1])
    txt = p.read_text()
    # A bare bfloat type token. Names like %struct.bfloat (the pre-typedef shim's struct) are not LLVM
    # bfloat types; clspv already accepts them, so leave such modules untouched.
    bare = re.compile(r"(?<![\w.])bfloat\b")
    if not bare.search(txt):
        return
    out = []
    serial = 0
    rx_binop = re.compile(rf"^\s*%([\w.$.-]+)\s*=\s*(fadd|fsub|fmul|fdiv|frem)\s+{VT}\s+(\S+)\s*,\s*(\S+)\s*$")
    rx_fneg = re.compile(rf"^\s*%([\w.$.-]+)\s*=\s*fneg\s+{VT}\s+(\S+)\s*$")
    rx_fcmp = re.compile(rf"^\s*%([\w.$.-]+)\s*=\s*fcmp\s+(\w+)\s+{VT}\s+(\S+)\s*,\s*(\S+)\s*$")
    rx_fpext = re.compile(rf"^\s*%([\w.$.-]+)\s*=\s*fpext\s+{VT}\s+(\S+)\s+to\s+(?:<\d+ x )?float(?: x \d+)?>?\s*$")
    rx_fptrunc = re.compile(rf"^\s*%([\w.$.-]+)\s*=\s*fptrunc\s+(?:<\d+ x )?float(?: x \d+)?>?\s+(\S+)\s+to\s+{VT}\s*$")
    rx_call = re.compile(rf"^\s*%([\w.$.-]+)\s*=\s*call\s+{VT}\s+(@llvm\.[\w.]+)\((.*)\)\s*$")
    rx_decl = re.compile(rf"^declare\s+{VT}\s+(@llvm\.[\w.]+)\(([^)]*)\)\s*(#.*)?$")
    rx_hex = re.compile(r"\bbfloat 0xH([0-9A-Fa-f]{4})\b")
    float_intrinsics = set(re.findall(r"^declare\s+\S+\s+(@llvm\.[\w.]+)\(", txt, flags=re.M))

    def const_hex(mm):
        return f"i16 {int(mm.group(1), 16)}"

    for ln in txt.splitlines(keepends=True):
        code = ln.split(";")[0]
        if "bfloat" not in code:
            out.append(ln)
            continue
        m = rx_binop.match(code)
        if m:
            serial += 1
            dst, op, lanes, a, b = m.group(1), m.group(2), int(m.group(3) or 0), m.group(5), m.group(6)
            ta = widen_tok(a, f"bin{serial}a", lanes, out)
            tb = widen_tok(b, f"bin{serial}b", lanes, out)
            vf = f"<{lanes} x float>" if lanes else "float"
            out.append(f"  %bin{serial} = {op} {vf} {ta}, {tb}\n")
            narrow_value(dst, f"%bin{serial}", lanes, out)
            continue
        m = rx_fneg.match(code)
        if m:
            serial += 1
            dst, lanes, a = m.group(1), int(m.group(2) or 0), m.group(4)
            ta = widen_tok(a, f"fn{serial}a", lanes, out)
            vf = f"<{lanes} x float>" if lanes else "float"
            out.append(f"  %fn{serial} = fneg {vf} {ta}\n")
            narrow_value(dst, f"%fn{serial}", lanes, out)
            continue
        m = rx_fcmp.match(code)
        if m:
            serial += 1
            dst, pred, lanes, a, b = m.group(1), m.group(2), int(m.group(3) or 0), m.group(5), m.group(6)
            ta = widen_tok(a, f"fc{serial}a", lanes, out)
            tb = widen_tok(b, f"fc{serial}b", lanes, out)
            vf = f"<{lanes} x float>" if lanes else "float"
            out.append(f"  %{dst} = fcmp {pred} {vf} {ta}, {tb}\n")
            continue
        m = rx_fpext.match(code)
        if m:
            widen_value(m.group(1), m.group(4), int(m.group(2) or 0), out)
            continue
        m = rx_fptrunc.match(code)
        if m:
            narrow_value(m.group(1), m.group(2), int(m.group(3) or 0), out)
            continue
        m = re.match(rf"^\s*%([\w.$.-]+)\s*=\s*([su])itofp\s+((?:<\d+ x )?i\d+>?)\s+(\S+)\s+to\s+{VT}\s*$", code)
        if m:
            serial += 1
            dst, sx, ity, tok, lanes = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5) or 0)
            vf = f"<{lanes} x float>" if lanes else "float"
            out.append(f"  %if{serial} = {sx}itofp {ity} {tok} to {vf}\n")
            narrow_value(dst, f"%if{serial}", lanes, out)
            continue
        m = re.match(rf"^\s*%([\w.$.-]+)\s*=\s*(fptosi|fptoui)\s+{VT}\s+(\S+)\s+to\s+((?:<\d+ x )?i\d+>?)\s*$", code)
        if m:
            serial += 1
            dst, op, lanes, tok, ity = m.group(1), m.group(2), int(m.group(3) or 0), m.group(5), m.group(6)
            widen_value(f"fp{serial}", tok, lanes, out)
            sf = f"<{lanes} x float>" if lanes else "float"
            out.append(f"  %{dst} = {op} {sf} %fp{serial} to {ity}\n")
            continue
        m = rx_call.match(code)
        if m and ".bf16" in m.group(4):
            serial += 1
            dst, lanes, callee, argstr = m.group(1), int(m.group(2) or 0), m.group(4), m.group(5)
            newname = callee.replace(".bf16", f".v{lanes}f32" if lanes else ".f32")
            if newname == callee:
                die(f"unsupported intrinsic spelling: {code.strip()}")
            wargs = []
            for i, ar in enumerate(arg_tokens(argstr)):
                am = re.match(rf"^{VT}\s+(\S+)$", ar)
                if am and am.group(2) == "bfloat":
                    wargs.append(widen_tok(am.group(3), f"cl{serial}x{i}", int(am.group(1) or 0), out, typed=True))
                else:
                    wargs.append(re.sub(r"\bbfloat\b", "i16", ar))
            vf = f"<{lanes} x float>" if lanes else "float"
            out.append(f"  %bfcl{serial} = call {vf} {newname}({', '.join(wargs)})\n")
            narrow_value(dst, f"%bfcl{serial}", lanes, out)
            continue
        m = rx_decl.match(code)
        if m and ".bf16" in m.group(3):
            lanes, callee, params = int(m.group(1) or 0), m.group(3), m.group(4)
            vf = f"<{lanes} x float>" if lanes else "float"
            newname = callee.replace(".bf16", f".v{lanes}f32" if lanes else ".f32")
            nparams = len(arg_tokens(params)) if params.strip() else 0
            if newname not in float_intrinsics:
                attrs = m.group(5)
                out.append(f"declare {vf} {newname}({', '.join([vf] * nparams)}){' ' + attrs if attrs else ''}\n")
            continue
        if re.search(r"\b(fadd|fsub|fmul|fdiv|frem|fneg|fcmp)\b", code):
            die(f"unhandled float op on a bfloat: {code.strip()}")
        if "@llvm." in code and ".bf16" in code:
            die(f"unhandled llvm intrinsic at bfloat: {code.strip()}")
        ln = rx_hex.sub(const_hex, ln)
        ln = re.sub(r"\bbfloat\b", "i16", ln)
        # Bare constants inherit the instruction's (former bfloat, now i16) type: phi incomings and select arms
        # spelled as decimal floats or bare 0xHhhhh. Skip literals carrying an explicit float type keyword.
        ln = re.sub(r"(?<![\w.])(?<!float )(?<!double )(?<!half )0xH([0-9A-Fa-f]{4})\b",
                    lambda mm: str(int(mm.group(1), 16)), ln)
        ln = re.sub(r"(?<![\w.])(?<!float )(?<!double )(?<!half )(-?\d+\.\d+e[+-]\d+|-?\d+\.\d+|-?inf|nan)(?![\w.])",
                    lambda mm: str(bf16_bits(mm.group(1))), ln)
        if bare.search(ln.split(";")[0]):
            die(f"unhandled bfloat reference after the rewrite: {ln.strip()}")
        out.append(ln)
    p.write_text("".join(out))


main()
