; Self-check input for sweep/tests/run.py: the bfloat_to_i16 pass must survive the constant spellings the
; clang 23 front end and opt emit around kernels that use numeric_limits. The -inf / +inf / nan operands came
; from omlx_gdn_verify_main_replay (fcmp against numeric_limits<half>::infinity()), the bare integer phi
; incoming is the clang 23 spelling that clspv's LLVM parser rejects before the pass respells it.
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir"

define spir_kernel void @bf_consts(bfloat %x, ptr addrspace(1) %out, i32 %n) {
entry:
  %c1 = fcmp oeq bfloat %x, -inf
  %c2 = fcmp oeq bfloat %x, +inf
  %c3 = fcmp ord bfloat %x, nan
  %c4 = fcmp ogt bfloat %x, 0xH8080
  %s = select i1 %c1, bfloat -inf, bfloat 0.000000e+00
  br label %loop

loop:
  %acc = phi float [ 0, %entry ], [ %next, %loop ]
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %w = fpext bfloat %s to float
  %next = fadd float %acc, %w
  %inext = add i32 %i, 1
  %more = icmp slt i32 %inext, %n
  br i1 %more, label %loop, label %exit

exit:
  %c = and i1 %c4, %c3
  %r = select i1 %c, float %next, float 0x7FF0000000000000
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}
