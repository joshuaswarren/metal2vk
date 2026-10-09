// tiny smoke kernels for the m2v-host GPU tests: plain OpenCL C, no shim needed.
// Fixed local size: clspv bakes it as spec constants 0..2 and the reflection
// reports it, so pipelines get the real shape without per-dispatch spec work.
kernel void add(global float* a, global float* b)
    __attribute__((reqd_work_group_size(128, 1, 1))) {
  size_t i = get_global_id(0);
  b[i] = a[i] + b[i];
}

kernel void add_offset(global float* a, global float* b, float k)
    __attribute__((reqd_work_group_size(128, 1, 1))) {
  size_t i = get_global_id(0);
  b[i] = a[i] + k;
}
