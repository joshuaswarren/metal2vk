// m2v_extra.h: Metal features the bulk sweep (sweep/results/FEATURES.md) found missing. Included at the end of metal_stdlib.
#pragma once

// OpenCL C++ reserves `local` and `global` as address-space keywords, while Metal code uses them as variable names (uzu's
// threadgroup_reduce.h does). The macros apply to every file included after the shim, so C++ for OpenCL code that spells the address
// spaces without the leading underscores does not compile through this shim; Metal never spells them that way.
#define local m2v_local
#define global m2v_global

namespace metal {

// ---- barriers: mem_flags combine with | ----
constexpr mem_flags operator|(mem_flags a, mem_flags b) { return (mem_flags)((int)a | (int)b); }
constexpr mem_flags operator&(mem_flags a, mem_flags b) { return (mem_flags)((int)a & (int)b); }

// ---- thread fences with a scope (Metal 3.2) ----
enum thread_scope { thread_scope_thread = 0, thread_scope_simdgroup = 1, thread_scope_threadgroup = 2, thread_scope_device = 3 };
METAL_FUNC void atomic_thread_fence(mem_flags f, memory_order, thread_scope = thread_scope_device) {
  barrier(((int)f & mem_device) ? CLK_GLOBAL_MEM_FENCE : CLK_LOCAL_MEM_FENCE);
}

// ---- integer bit functions (Metal integer functions with no OpenCL spelling) ----
template <typename T> METAL_FUNC T extract_bits(T v, uint offset, uint bits) {
  if (bits == 0u) return T(0);
  const uint w = sizeof(T) * 8u;
  const T shifted = (T)(v >> offset);
  if (__is_signed(T)) {  // arithmetic shift sign-extends from the top of the field
    const uint s = w - bits;
    return (T)((T)(shifted << s) >> s);
  }
  return bits >= w ? shifted : (T)(shifted & (T)(((T)1 << bits) - (T)1));
}
template <typename T> METAL_FUNC T insert_bits(T base, T insert, uint offset, uint bits) {
  if (bits == 0u) return base;
  const T mask = bits >= sizeof(T) * 8u ? (T)~(T)0 : (T)(((T)1 << bits) - (T)1);
  return (T)((base & (T)~(mask << offset)) | ((insert & mask) << offset));
}
METAL_FUNC uint reverse_bits(uint x) {
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
  x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
  return (x >> 16) | (x << 16);
}

// ---- scalar math ----
template <typename T> METAL_FUNC T sign(T x) { return ::sign(x); }
namespace precise {
METAL_FUNC float divide(float a, float b) { return a / b; }
}
namespace fast {
METAL_FUNC float divide(float a, float b) { return native_divide(a, b); }
}
template <> struct numeric_limits<bool> { static constexpr bool max() { return true; } static constexpr bool min() { return false; } };

// ---- subgroup shuffles up/down: lowered to the plain subgroup shuffle (clspv has no OpGroupNonUniformShuffleDown/Up lowering) ----
// Out-of-range lanes keep their own value (Metal leaves them undefined).
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_down(T x, ushort d) {
  const uint l = get_sub_group_local_id();
  return sub_group_shuffle(x, l + d < (uint)get_sub_group_size() ? l + d : l);
}
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_up(T x, ushort d) {
  const uint l = get_sub_group_local_id();
  return sub_group_shuffle(x, l >= d ? l - d : l);
}
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_down(V x, ushort d) {
  V r;
  for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_down(x[i], d);
  return r;
}
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_up(V x, ushort d) {
  V r;
  for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_up(x[i], d);
  return r;
}

// ---- simd reductions and scans for every arithmetic type, bfloat, and vectors ----
#define M2V_SIMD_RED(NAME, BUILTIN)                                                                                    \
  template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> NAME(T x) {                                      \
    static_assert(sizeof(T) <= 4, "simd reductions and scans on 64-bit types are not supported: the value would be truncated to 32 bits"); \
    using W = conditional_t<__is_integral(T), conditional_t<__is_signed(T), int, uint>, float>;                        \
    return (T)BUILTIN((W)x);                                                                                            \
  }                                                                                                                    \
  METAL_FUNC bfloat NAME(bfloat x) { return bfloat(BUILTIN((float)x)); }                                               \
  template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V) && !is_same_v<V, bfloat>, V> NAME(V x) {            \
    V r;                                                                                                               \
    for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = NAME(x[i]);                                      \
    return r;                                                                                                          \
  }
M2V_SIMD_RED(simd_sum, sub_group_reduce_add)
M2V_SIMD_RED(simd_max, sub_group_reduce_max)
M2V_SIMD_RED(simd_min, sub_group_reduce_min)
M2V_SIMD_RED(simd_prefix_inclusive_sum, sub_group_scan_inclusive_add)
M2V_SIMD_RED(simd_prefix_exclusive_sum, sub_group_scan_exclusive_add)

// ---- quad (4-lane) operations ----
template <typename T> METAL_FUNC T quad_sum(T x) {
  T y = x + simd_shuffle_xor(x, (ushort)1);
  return y + simd_shuffle_xor(y, (ushort)2);
}
template <typename T> METAL_FUNC T quad_max(T x) {
  T y = max(x, simd_shuffle_xor(x, (ushort)1));
  return max(y, simd_shuffle_xor(y, (ushort)2));
}
template <typename T> METAL_FUNC T quad_min(T x) {
  T y = min(x, simd_shuffle_xor(x, (ushort)1));
  return min(y, simd_shuffle_xor(y, (ushort)2));
}

// ---- fast:: and precise:: math on vectors ----
#define M2V_FAST_VEC(NAME, SCALAR)                                                                                     \
  namespace fast {                                                                                                     \
  template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> NAME(V x) {                                     \
    V r;                                                                                                               \
    for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = SCALAR(x[i]);                                    \
    return r;                                                                                                          \
  }                                                                                                                    \
  }
M2V_FAST_VEC(exp, native_exp)
M2V_FAST_VEC(exp2, native_exp2)
M2V_FAST_VEC(log, native_log)
M2V_FAST_VEC(log2, native_log2)
M2V_FAST_VEC(sqrt, native_sqrt)
M2V_FAST_VEC(rsqrt, native_rsqrt)
M2V_FAST_VEC(tanh, ::tanh)
M2V_FAST_VEC(sin, native_sin)
M2V_FAST_VEC(cos, native_cos)

// ---- bfloat vectors (storage + conversion; arithmetic goes through float) ----
// Fields are raw i16 at the IR level (see the vec_sel<bfloat, N> note in metal_stdlib); the element interface keeps
// bfloat semantics with round-to-nearest-even narrowing on every write.
#define M2V_BFVEC_COMMON(NAME, N)                                                                                      \
  static constexpr int m2v_n = N;                                                                                      \
  METAL_FUNC NAME() = default;                                                                                         \
  METAL_FUNC bfloat operator[](int i) const { bfloat r; r.bits = (&x)[i]; return r; }                                  \
  struct ref {                                                                                                         \
    ushort* slot;                                                                                                      \
    METAL_FUNC operator bfloat() const { bfloat r; r.bits = *slot; return r; }                                         \
    METAL_FUNC operator float() const { return as_float((uint)*slot << 16); }                                          \
    METAL_FUNC ref& operator=(bfloat b) { *slot = b.bits; return *this; }                                              \
    METAL_FUNC ref& operator=(float f) { bfloat b(f); *slot = b.bits; return *this; }                                  \
  };                                                                                                                   \
  METAL_FUNC ref operator[](int i) { return ref{&x + i}; }                                                             \
  template <typename E> METAL_FUNC NAME(E v) { for (int i = 0; i < N; i++) (&x)[i] = bfloat((float)v[i]).bits; }
struct bfloat2 { ushort x, y;
  M2V_BFVEC_COMMON(bfloat2, 2)
  METAL_FUNC bfloat2(bfloat a, bfloat b) : x(a.bits), y(b.bits) {} };
struct bfloat3 { ushort x, y, z;
  M2V_BFVEC_COMMON(bfloat3, 3)
  METAL_FUNC bfloat3(bfloat a, bfloat b, bfloat c) : x(a.bits), y(b.bits), z(c.bits) {} };
struct bfloat4 { ushort x, y, z, w;
  M2V_BFVEC_COMMON(bfloat4, 4)
  METAL_FUNC bfloat4(bfloat a, bfloat b, bfloat c, bfloat d) : x(a.bits), y(b.bits), z(c.bits), w(d.bits) {} };

// ---- metal::array ----
template <typename T, size_t N> struct array {
  T _data[N];
  METAL_FUNC constexpr size_t size() const { return N; }
  METAL_FUNC T& operator[](size_t i) { return _data[i]; }
  METAL_FUNC const T& operator[](size_t i) const { return _data[i]; }
  METAL_FUNC T* data() { return _data; }
};

// ---- Apple built-ins that standard-library headers wrap (glm_glue.metal re-declares bfloat math through them) ----
#define __METAL_MAYBE_FAST_MATH__ 0
#define M2V_BUILTIN1(N, F) template <typename T> METAL_FUNC T __metal_##N(T x, int) { return F(x); }
M2V_BUILTIN1(fabs, ::fabs) M2V_BUILTIN1(exp, ::exp) M2V_BUILTIN1(exp2, ::exp2) M2V_BUILTIN1(log, ::log) M2V_BUILTIN1(log2, ::log2)
M2V_BUILTIN1(sqrt, ::sqrt) M2V_BUILTIN1(rsqrt, ::rsqrt) M2V_BUILTIN1(tanh, ::tanh) M2V_BUILTIN1(sin, ::sin) M2V_BUILTIN1(cos, ::cos)

} // namespace metal

// packed_* vector constructors
#define packed_char4(...) m2v::mk<packed_char4>(__VA_ARGS__)
#define packed_uchar4(...) m2v::mk<packed_uchar4>(__VA_ARGS__)
#define packed_short4(...) m2v::mk<packed_short4>(__VA_ARGS__)
#define packed_int4(...) m2v::mk<packed_int4>(__VA_ARGS__)
#define packed_uint4(...) m2v::mk<packed_uint4>(__VA_ARGS__)
#define packed_float2(...) m2v::mk<packed_float2>(__VA_ARGS__)
#define packed_float4(...) m2v::mk<packed_float4>(__VA_ARGS__)
#define packed_half2(...) m2v::mk<packed_half2>(__VA_ARGS__)
#define packed_half4(...) m2v::mk<packed_half4>(__VA_ARGS__)

// float2(bfloat2) and friends
namespace m2v {
template <typename V, typename B> METAL_FUNC metal::enable_if_t<(B::m2v_n > 0), V> mk1(B b) {
  V r;
  for (int i = 0; i < B::m2v_n; i++) r[i] = (elem_t<V>)(float)b[i];
  return r;
}
} // namespace m2v

typedef metal::bfloat bfloat16_t;  // the MLX / Zig-host spelling
typedef half float16_t;

namespace metal {
typedef simdgroup_matrix<half, 8, 8> simdgroup_half8x8;
typedef simdgroup_matrix<float, 8, 8> simdgroup_float8x8;
typedef simdgroup_matrix<bfloat, 8, 8> simdgroup_bfloat8x8;
} // namespace metal
