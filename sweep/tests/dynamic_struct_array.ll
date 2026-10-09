; Self-check input for sweep/tests/run.py: clspv cannot lower a dynamic-index GEP into a private alloca array
; whose element type is an aggregate, while the same array of the member type lowers fine. The flatten pass must
; rewrite the single-member wrapper (the simdgroup_matrix shape) into the member type; this module then has no
; aggregate left and clspv must produce valid SPIR-V.
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir"

%"struct.metal::simdgroup_matrix" = type { <2 x i16> }

define spir_kernel void @dyn_arr(i32 %n, ptr addrspace(1) %out) {
entry:
  %mats = alloca [8 x %"struct.metal::simdgroup_matrix"], align 4
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %slot = getelementptr inbounds [8 x %"struct.metal::simdgroup_matrix"], ptr %mats, i32 0, i32 %i
  store <2 x i16> zeroinitializer, ptr %slot, align 4
  %ld = load <2 x i16>, ptr %slot, align 4
  %inext = add i32 %i, 1
  %more = icmp slt i32 %inext, %n
  br i1 %more, label %loop, label %exit

exit:
  %first = getelementptr inbounds [8 x %"struct.metal::simdgroup_matrix"], ptr %mats, i32 0, i32 0
  %v = load <2 x i16>, ptr %first, align 4
  store <2 x i16> %v, ptr addrspace(1) %out, align 4
  ret void
}
