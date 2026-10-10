; Self-check input for sweep/tests/run.py: clspv's SimplifyPointerBitcast cycles on loops whose header
; phis are pointers advanced by a constant stride, and the patched pass then continues with mid-drift
; GEPs, so the loop-exit users read one block too far (the gate_up tail, PR #47). The
; pointer_phi_to_index pass must rewrite every loop-carried pointer phi here into an integer byte
; index plus one byte-indexed GEP before clspv sees the module: no `phi ptr` may survive, the byte
; steps must match the strides (x: 3 x i16 = 6, w: 5 x i8 = 5, s: 7 x i16 = 14, b: 1 x i16 = 2) and
; the constant preheader offsets must land in the index phi (x starts at element 1 = 2 bytes, s at
; element 2 = 4 bytes, w and b at 0). The exit-value phis in tail are the shape the drift corrupted;
; they keep working because the latch advance they carry is now byte arithmetic off the same index.
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir"

define spir_kernel void @pphi(ptr addrspace(1) %x, ptr addrspace(1) %w,
                              ptr addrspace(1) %s, ptr addrspace(1) %b,
                              ptr addrspace(1) %out) #0 {
entry:
  %xs = getelementptr inbounds i16, ptr addrspace(1) %x, i32 1
  %ss = getelementptr inbounds i16, ptr addrspace(1) %s, i32 2
  br label %loop

loop:                                              ; preds = %entry, %loop
  %k   = phi i32 [ %k.next, %loop ], [ 0, %entry ]
  %xp  = phi ptr addrspace(1) [ %x.next, %loop ], [ %xs, %entry ]
  %wp  = phi ptr addrspace(1) [ %w.next, %loop ], [ %w, %entry ]
  %sp  = phi ptr addrspace(1) [ %s.next, %loop ], [ %ss, %entry ]
  %bp  = phi ptr addrspace(1) [ %b.next, %loop ], [ %b, %entry ]
  %acc = phi float [ %acc.next, %loop ], [ 0.000000e+00, %entry ]
  %xv1 = load i16, ptr addrspace(1) %xp, align 2
  %wv1 = load i8, ptr addrspace(1) %wp, align 1
  %sv1 = load i16, ptr addrspace(1) %sp, align 2
  %bv1 = load i16, ptr addrspace(1) %bp, align 2
  %xg = getelementptr inbounds i16, ptr addrspace(1) %xp, i32 %k
  %xv2 = load i16, ptr addrspace(1) %xg, align 2
  %x.next = getelementptr inbounds i16, ptr addrspace(1) %xp, i32 3
  %w.next = getelementptr inbounds i8, ptr addrspace(1) %wp, i32 5
  %s.next = getelementptr inbounds i16, ptr addrspace(1) %sp, i32 7
  %b.next = getelementptr inbounds i16, ptr addrspace(1) %bp, i32 1
  %k.next = add nsw i32 %k, 1
  %xf1 = sitofp i16 %xv1 to float
  %xf2 = sitofp i16 %xv2 to float
  %wf1 = sitofp i8 %wv1 to float
  %sf1 = sitofp i16 %sv1 to float
  %bf1 = sitofp i16 %bv1 to float
  %a1 = fadd float %acc, %xf1
  %a2 = fadd float %a1, %xf2
  %a3 = fadd float %a2, %wf1
  %a4 = fadd float %a3, %sf1
  %acc.next = fadd float %a4, %bf1
  %more = icmp slt i32 %k.next, 4
  br i1 %more, label %loop, label %tail

tail:                                              ; preds = %loop
  %xp.t = phi ptr addrspace(1) [ %x.next, %loop ]
  %wp.t = phi ptr addrspace(1) [ %w.next, %loop ]
  %sp.t = phi ptr addrspace(1) [ %s.next, %loop ]
  %bp.t = phi ptr addrspace(1) [ %b.next, %loop ]
  %acc.t = phi float [ %acc.next, %loop ]
  %xt = load i16, ptr addrspace(1) %xp.t, align 2
  %wt = load i8, ptr addrspace(1) %wp.t, align 1
  %st = load i16, ptr addrspace(1) %sp.t, align 2
  %bt = load i16, ptr addrspace(1) %bp.t, align 2
  %xtf = sitofp i16 %xt to float
  %wtf = sitofp i8 %wt to float
  %stf = sitofp i16 %st to float
  %btf = sitofp i16 %bt to float
  %t1 = fadd float %acc.t, %xtf
  %t2 = fadd float %t1, %wtf
  %t3 = fadd float %t2, %stf
  %t4 = fadd float %t3, %btf
  store float %t4, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { "cl-std"="CLC++2021" }
