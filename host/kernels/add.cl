// tiny smoke kernels for the m2v-host GPU tests: plain OpenCL C, no shim needed.
kernel void add(global float* a, global float* b) {
  size_t i = get_global_id(0);
  b[i] = a[i] + b[i];
}

kernel void add_offset(global float* a, global float* b, float k) {
  size_t i = get_global_id(0);
  b[i] = a[i] + k;
}
