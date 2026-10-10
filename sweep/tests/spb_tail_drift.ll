; Self-check for the SimplifyPointerBitcast non-convergence drift that broke
; the gt::qmv_rows tail in omlx_qwen35_moe_gate_up_decode (the shared-gate scalar
; in block 0, block 0 namespace gt::, FAST=0, K=2048). The pattern: a loop
; with three per-iteration pointer advances of different strides (x += 256
; elements of bfloat16, w += 128 bytes of uchar, s += 8 entries of bfloat16)
; whose exit values are then used by a tail clamp that walks the last lane's
; slice. The SimplifyPointerBitcast sub-passes 5 and 7 flip a GEP pair
; between two valid shapes without convergence, so the patched clspv breaks
; out at the 200-iter cap with mid-drift GEPs: the tail then reads one full
; block too far (xp + 256, wp + 128, sp + 8).
;
; The .cl for moe_gate_up produces 1000+ lines of IR; the test feeds
; clspv this minimal form and runs the same pipeline (clspv -x ir -fp16
; -inline-entry-points -spv-version=1.5 --long-vector), then asserts the
; tail x-pointer load uses an offset within VALUES_PER_THREAD of the exit
; phi value, not (exit value + BLOCK_SIZE). The CL-side equivalent of
; this assertion is the gate_up parity check (g7 dumps), which compares
; the GPU output against a per-block-floating-point numpy reference.
;
; The IR must hold the data layout the SPIR-V producer expects:
;   - x: addrspace(1) of 384 bfloat16 elements (BLOCK=256, K=384)
;   - w: addrspace(1) of 96 u8 bytes (lane 0 owns 4 bytes, BLOCK/2 = 128
;     per iteration, total row bytes 192 = 6*32 = 192)
;   - s, b: addrspace(1) of 12 bfloat16 each (lane group size 4, BLOCK/4=64
;     per iteration, total 12 entries = 384/32 = 12)
;   - out: addrspace(1) of 1 float (lane 0 writes the reduced result)
;
; Block count: 1 (only 256<K, so 1 main iteration, then the tail)
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir"

; The clspv SimplifyPointerBitcast non-convergence breaks the
; omlx_qwen35_moe_gate_up_decode SPIR-V: the per-iteration strided
; phis (x += 256 elements, w += 128 bytes, s += 8 entries) of
; gt::qmv_rows' main loop exit on the +1-block too-far state when
; the 200-iter convergence cap fires mid-drift. The tail then reads
; x[2048..], w[1024..], s[64..] - all past the input.
;
; The clspv flag to detect this is the literal "%uint_256" feeding
; the ushort storage chain inside the tail-body block of the kernel
; function. The main loop never produces this pattern, so a single
; presence in the disassembly is a sufficient signal.
;
; A saved g7-style dec.ll is the real reproducer. The minimal form
; below is what the .cl compiles to when the structure is reduced to
; one main iteration + one tail. clspv on this minimal form does NOT
; trigger the non-convergence (the production module does, because of
; its size, the qmv qdot chains, the swiglu T-rounding, and the
; gt/st/rt triplicate). The end-to-end reproducer is the .ll the
; sweep saves at $BLEND_TOOLS's src/<tag>.ll; the run.py below locates
; it and runs the SPIR-V check on that.
;
%0 = type { i32 }

define spir_kernel void @repro(ptr addrspace(1) %x, ptr addrspace(1) %w,
                               ptr addrspace(1) %s, ptr addrspace(1) %b,
                               ptr addrspace(1) %out) #0 {
entry:
  %lid = call i32 @_Z22get_sub_group_local_idv()
  %xp0 = getelementptr inbounds i16, ptr addrspace(1) %x, i32 %lid
  %wp0 = getelementptr inbounds i8,  ptr addrspace(1) %w, i32 %lid
  %sp0 = getelementptr inbounds i16, ptr addrspace(1) %s, i32 %lid
  %bp0 = getelementptr inbounds i16, ptr addrspace(1) %b, i32 %lid
  br label %loop

loop:                                              ; preds = %loop, %entry
  %xp = phi ptr addrspace(1) [ %xp_loop, %loop ], [ %xp0, %entry ]
  %wp = phi ptr addrspace(1) [ %wp_loop, %loop ], [ %wp0, %entry ]
  %sp = phi ptr addrspace(1) [ %sp_loop, %loop ], [ %sp0, %entry ]
  %bp = phi ptr addrspace(1) [ %bp_loop, %loop ], [ %bp0, %entry ]
  %k  = phi i32 [ %k_next, %loop ], [ 0, %entry ]
  %acc = phi float [ %acc_next, %loop ], [ 0.0, %entry ]
  %xv0 = load i16, ptr addrspace(1) %xp, align 2
  %wv0 = load i8,  ptr addrspace(1) %wp, align 1
  %sv0 = load i16, ptr addrspace(1) %sp, align 2
  %bv0 = load i16, ptr addrspace(1) %bp, align 2
  %sum = fadd float 0.0, 0.0
  %acc_next = fadd float %acc, %sum
  %xp_loop = getelementptr inbounds i16, ptr addrspace(1) %xp, i32 256
  %wp_loop = getelementptr inbounds i8,  ptr addrspace(1) %wp, i32 128
  %sp_loop = getelementptr inbounds i16, ptr addrspace(1) %sp, i32 8
  %bp_loop = getelementptr inbounds i16, ptr addrspace(1) %bp, i32 8
  %k_next = add nsw i32 %k, 256
  %more = icmp slt i32 %k, 128
  br i1 %more, label %loop, label %tail

tail:                                              ; preds = %loop
  %xp_t = phi ptr addrspace(1) [ %xp_loop, %loop ]
  %wp_t = phi ptr addrspace(1) [ %wp_loop, %loop ]
  %sp_t = phi ptr addrspace(1) [ %sp_loop, %loop ]
  %bp_t = phi ptr addrspace(1) [ %bp_loop, %loop ]
  %k_t  = phi i32 [ %k_next, %loop ]
  %acc_t = phi float [ %acc_next, %loop ]
  %sub_k = sub nsw i32 384, %k_t
  %sub_lid = sub nsw i32 %sub_k, %lid
  %rem = call i32 @_Z5clampiii(i32 %sub_lid, i32 0, i32 8)
  %has = icmp sgt i32 %rem, 0
  br i1 %has, label %tail_body, label %tail_done

tail_body:                                         ; preds = %tail
  %xv_t = load i16, ptr addrspace(1) %xp_t, align 2
  %wv_t = load i8,  ptr addrspace(1) %wp_t, align 1
  %sv_t = load i16, ptr addrspace(1) %sp_t, align 2
  %bv_t = load i16, ptr addrspace(1) %bp_t, align 2
  %sum_t = fadd float 0.0, 0.0
  %acc_done = fadd float %acc_t, %sum_t
  br label %tail_done

tail_done:                                         ; preds = %tail, %tail_body
  %acc_final = phi float [ %acc_t, %tail ], [ %acc_done, %tail_body ]
  %reduced = call float @_Z20sub_group_reduce_addf(float %acc_final)
  %is_zero = icmp eq i32 %lid, 0
  br i1 %is_zero, label %store, label %exit

store:                                             ; preds = %tail_done
  store float %reduced, ptr addrspace(1) %out, align 4
  br label %exit

exit:                                              ; preds = %tail_done, %store
  ret void
}

declare i32 @_Z22get_sub_group_local_idv()
declare i32 @_Z5clampiii(i32, i32, i32)
declare float @_Z20sub_group_reduce_addf(float)

attributes #0 = { "cl-std"="CLC++2021" convergent }
