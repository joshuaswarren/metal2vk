target datalayout = "e-p:64:64:64:64"

define i32 @test(i32 %idx) {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 7, ptr %a, align 4
  store i32 13, ptr %b, align 4
  %ap = addrspacecast ptr %a to ptr addrspace(1)
  %bp = addrspacecast ptr %b to ptr addrspace(1)
  %table = alloca [2 x ptr addrspace(1)], align 8
  %s0 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i32 0, i32 %idx
  store ptr addrspace(1) %ap, ptr %s0, align 8
  %p0 = load ptr addrspace(1), ptr %s0, align 8
  store ptr addrspace(1) %bp, ptr %s0, align 8
  %p1 = load ptr addrspace(1), ptr %s0, align 8
  %v0 = load i32, ptr addrspace(1) %p0, align 4
  %v1 = load i32, ptr addrspace(1) %p1, align 4
  %r = sub i32 %v1, %v0
  ret i32 %r
}
define i32 @main() {
  %a = call i32 @test(i32 0)
  %b = call i32 @test(i32 1)
  %c = icmp eq i32 %a, 6
  %d = icmp eq i32 %b, 6
  %ok = and i1 %c, %d
  %exit = select i1 %ok, i32 0, i32 1
  ret i32 %exit
}
