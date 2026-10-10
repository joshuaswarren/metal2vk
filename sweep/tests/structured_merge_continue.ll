; Reduced from the omlx_verify_attn_wide_partial module (the sib3 verify path):
; an inner loop whose only exit block is also the latch (the SPIR-V continue
; target) of the enclosing loop, with a threadgroup barrier on that block and
; one inside the inner body. A producer that merges the inner loop's exit
; straight into the continue target emits an OpLoopMerge pair spirv-val
; rejects: "Header block 'X' is contained in the loop construct headed by 'Y',
; but its merge block 'Z' is not". The inner loop must merge into a block of
; its own ahead of the old block, which keeps the latch and barrier role.
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir"

declare dso_local spir_func void @_Z7barrierj(i32 noundef)

define dso_local spir_kernel void @structured_merge_continue(
    ptr addrspace(1) noundef %out, i32 noundef %n) #0 {
entry:
  br label %outer.header

outer.header:                                       ; preds = %entry, %outer.latch
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %outer.latch ]
  %cont = icmp ult i32 %i, %n
  br i1 %cont, label %inner.header, label %exit

inner.header:                                       ; preds = %outer.header, %inner.body
  %j = phi i32 [ 0, %outer.header ], [ %j.next, %inner.body ]
  %inner.cont = icmp ult i32 %j, 4
  br i1 %inner.cont, label %inner.body, label %outer.latch

inner.body:                                         ; preds = %inner.header
  call spir_func void @_Z7barrierj(i32 noundef 1)
  %j.next = add i32 %j, 1
  br label %inner.header

; The inner loop's only exit is the enclosing loop's latch, and the latch
; carries the barrier: the exact shape that must not merge into the continue
; target.
outer.latch:                                        ; preds = %inner.header
  call spir_func void @_Z7barrierj(i32 noundef 1)
  %acc.next = add i32 %acc, %j
  %i.next = add i32 %i, 1
  %loop.cont = icmp ult i32 %i.next, %n
  br i1 %loop.cont, label %outer.header, label %exit

exit:                                               ; preds = %outer.header, %outer.latch
  %res = phi i32 [ %acc, %outer.header ], [ %acc.next, %outer.latch ]
  store i32 %res, ptr addrspace(1) %out
  ret void
}

attributes #0 = { convergent mustprogress norecurse nounwind }
