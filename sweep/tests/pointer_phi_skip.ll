; Self-check input for sweep/tests/run.py: the refusal path of pointer_phi_to_index. The loop advances
; its pointer phi by a DYNAMIC stride, so the pass cannot prove a constant byte step and must leave the
; module untouched: the `phi ptr` has to survive this pass byte for byte (clspv handles converging
; pointer phis; only the non-converging strided shape is rewritten).
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir"

define spir_kernel void @skipme(ptr addrspace(1) %x, ptr addrspace(1) %out, i32 %stride) #0 {
entry:
  br label %loop

loop:                                              ; preds = %entry, %loop
  %k = phi i32 [ %k.next, %loop ], [ 0, %entry ]
  %xp = phi ptr addrspace(1) [ %x.next, %loop ], [ %x, %entry ]
  %v = load i16, ptr addrspace(1) %xp, align 2
  %x.next = getelementptr inbounds i16, ptr addrspace(1) %xp, i32 %stride
  %k.next = add nsw i32 %k, 1
  %more = icmp slt i32 %k.next, 4
  br i1 %more, label %loop, label %done

done:                                              ; preds = %loop
  %xf = sitofp i16 %v to float
  store float %xf, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { "cl-std"="CLC++2021" }
