; Reduced from cases/aggregate_select.metal (select_aggregate): memcpy_mixed_as.py expands a struct copy out of a
; dynamically indexed threadgroup array into byte loads, `gep i8, (gep [8 x %Conf], 0, %g), k`. Offset k = 0 or 4
; lands on a field; k = 1..3 does not. For those, clspv's constant-offset fold in SimplifyPointerBitcastPass already
; adds the inner offset (%g * 16) and moves to the base; apply-implicit-casts-lossless-merge.py must not merge the
; pair a second time, which read byte (%g << 5) | k instead of (%g << 4) | k.
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir"

%struct.Conf = type { float, i32, float, float }

@groups = internal addrspace(3) global [8 x %struct.Conf] undef, align 4

define dso_local spir_kernel void @lossless_merge_guard(ptr addrspace(1) noundef align 1 %out,
    ptr addrspace(2) noundef align 4 %gp) #0 {
entry:
  %g = load i32, ptr addrspace(2) %gp, align 4
  %e = getelementptr inbounds nuw [8 x %struct.Conf], ptr addrspace(3) @groups, i32 0, i32 %g
  %b0 = load i8, ptr addrspace(3) %e, align 1
  %p1 = getelementptr i8, ptr addrspace(3) %e, i32 1
  %b1 = load i8, ptr addrspace(3) %p1, align 1
  %p2 = getelementptr i8, ptr addrspace(3) %e, i32 2
  %b2 = load i8, ptr addrspace(3) %p2, align 1
  %p3 = getelementptr i8, ptr addrspace(3) %e, i32 3
  %b3 = load i8, ptr addrspace(3) %p3, align 1
  store i8 %b0, ptr addrspace(1) %out, align 1
  %o1 = getelementptr i8, ptr addrspace(1) %out, i32 1
  store i8 %b1, ptr addrspace(1) %o1, align 1
  %o2 = getelementptr i8, ptr addrspace(1) %out, i32 2
  store i8 %b2, ptr addrspace(1) %o2, align 1
  %o3 = getelementptr i8, ptr addrspace(1) %out, i32 3
  store i8 %b3, ptr addrspace(1) %o3, align 1
  ret void
}

attributes #0 = { convergent norecurse nounwind "uniform-work-group-size" }

!opencl.ocl.version = !{!0}
!0 = !{i32 3, i32 0}
