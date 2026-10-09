#pragma once
// Metal 4 MetalPerformancePrimitives (mpp::tensor_ops) emulated for Vulkan.
//
// Fragment layout (the one MLX's steel NAX kernels, uzu's MxuFragmentOps<true> and TensorFold's
// flashnext/nemotron kernels all pack for; verified against all four call sites): lane l of a
// 32-lane group holds x = (l & 8) + 4 * (l & 1), y = 4 * ((l >> 4) & 1) + ((l >> 1) & 3); element i
// of a lane's slice of a 16-row tile is at row y + 8 * ((i >> 2) & 1), column x + 4 * (i & 3)
// + 16 * (i >> 3). execution_simdgroups<N> gives each consecutive 32-lane group one of N column
// bands of width COLS / N (the band comes from the threadgroup's subgroup index, which matches how
// kernels pair simdgroups). uzu's MxuStrictFragmentOps and TensorFold's TF_SIMD_FRAGS nax branch
// pack for the other (strict) hardware layout; under this emulation their results land permuted -
// the modules are valid SPIR-V, the numerics are not Apple's.
//
// run() computes each lane's destination elements as an fp32 sum (half/bfloat/4-bit inputs widen
// exactly); cooperative-tensor sources are read with subgroup shuffles, so they are supported at
// execution_simdgroup scope only and with 16 rows (all uses). The relaxed-precision flag is
// accepted: the emulation always accumulates in fp32.
#include "metal_stdlib"

namespace metal {

// ---- scope tags (Apple: these live in metal) ----
struct execution_simdgroup {};
template <int N> struct execution_simdgroups {};

}  // namespace metal

namespace mpp {
namespace tensor_ops {

// ---- unsigned 4-bit element format, two per byte (0..15, widened exactly) ----
struct uint4b_format {
  static constexpr unsigned m2v_bits = 4;
  unsigned char m2v_v;
  METAL_FUNC uint4b_format() = default;
  METAL_FUNC constexpr uint4b_format(unsigned v) : m2v_v((unsigned char)(v & 0xFu)) {}
  METAL_FUNC operator float() const { return (float)m2v_v; }
};

// ---- matmul2d_descriptor(M, N, K, transposeA, transposeB, relaxed, mode) ----
// matmul2d takes the descriptor as an integer non-type template parameter (class-type NTTPs need
// C++20; the constexpr conversion is evaluated in the template argument), so it packs to a long long.
struct matmul2d_descriptor {
  enum class mode { multiply, multiply_accumulate };
  int m2v_m, m2v_n, m2v_k;
  bool m2v_ta, m2v_tb, m2v_relaxed;
  mode m2v_mode;
  METAL_FUNC constexpr matmul2d_descriptor(int m, int n, int k, bool ta, bool tb, bool relaxed, mode md)
      : m2v_m(m), m2v_n(n), m2v_k(k), m2v_ta(ta), m2v_tb(tb), m2v_relaxed(relaxed), m2v_mode(md) {}
  METAL_FUNC constexpr operator long long() const {
    return (long long)m2v_m | ((long long)m2v_n << 12) | ((long long)m2v_k << 24) | ((long long)m2v_ta << 36) |
           ((long long)m2v_tb << 37) | ((long long)m2v_relaxed << 38) |
           ((long long)(m2v_mode == mode::multiply_accumulate) << 39);
  }
};

// ---- tensor_inline: the storage tag every tensor use passes ----
struct tensor_inline {};

// ---- dextents<I, R>: rank-R extents, all dynamic, (width, height, ...) ----
template <typename I, int R> struct dextents {
  I m2v_e[R];
  METAL_FUNC dextents() = default;
  template <typename... A> METAL_FUNC constexpr dextents(A... a) : m2v_e{(I)a...} {}
  METAL_FUNC I operator[](int i) const { return m2v_e[i]; }
};

// ---- tensor<T, dextents<I, R>, Tag>: a strided view over device/threadgroup memory.
// element (x, y) sits at element offset x * stride[0] + y * stride[1], default strides
// {extent[0], 1}; uint4b_format packs two elements per byte. ----
template <typename T, typename Ex, typename Tag>
struct tensor {
  using m2v_elem = metal::remove_addrspace_t<T>;
  static constexpr bool m2v_q4 = metal::is_same<m2v_elem, uint4b_format>::value;
  T* m2v_p;
  long long m2v_s0, m2v_s1;

  METAL_FUNC tensor() = default;
  template <typename P> METAL_FUNC tensor(P p, Ex ex) : m2v_p((T*)p), m2v_s0((long long)ex[0]), m2v_s1(1) {}
  template <typename P, typename S>
  METAL_FUNC tensor(P p, Ex ex, S st) : m2v_p((T*)p), m2v_s0((long long)st[0]), m2v_s1((long long)st[1]) {}

  METAL_FUNC tensor slice(long long ox, long long oy) const {
    tensor r;
    if constexpr (m2v_q4) {
      const long long e = ox * m2v_s0 + oy * m2v_s1;
      r.m2v_p = (T*)((unsigned char*)m2v_p + (e >> 1));
    } else {
      r.m2v_p = &m2v_p[ox * m2v_s0 + oy * m2v_s1];
    }
    r.m2v_s0 = m2v_s0;
    r.m2v_s1 = m2v_s1;
    return r;
  }

  METAL_FUNC float m2v_at(long long x, long long y) const {
    if constexpr (m2v_q4) {
      const long long e = x * m2v_s0 + y * m2v_s1;
      return (float)(((unsigned char*)m2v_p)[e >> 1] >> ((e & 1) * 4));
    } else {
      return (float)m2v_p[x * m2v_s0 + y * m2v_s1];
    }
  }
};

// ---- cooperative tensor: one thread's slice of a 16-row tile ----
namespace detail {
METAL_FUNC void m2v_home(int lane, int& x, int& y) {
  x = (lane & 8) + ((lane & 1) << 2);
  y = (((lane >> 4) & 1) << 2) + ((lane >> 1) & 3);
}
METAL_FUNC void m2v_loc(int i, int lane, int band_w, int rows_per_band, int& r, int& c) {
  int x, y;
  m2v_home(lane, x, y);
  if (rows_per_band == 32) {  // one 32-lane group covers a whole 32-row tile
    r = y + (((i >> 2) & 3) << 3);
    c = x + (i & 3) + ((i >> 4) << 4);
  } else if (rows_per_band) {
    r = (lane >> 5) * rows_per_band + y + (((i >> 2) & 1) << 3);
    c = x + (i & 3) + ((i >> 3) << 4);
  } else {
    r = y + (((i >> 2) & 1) << 3);
    c = (lane >> 5) * band_w + x + (i & 3) + ((i >> 3) << 4);
  }
}
METAL_FUNC int m2v_owner(int r, int c, int band_w, int& i) {
  const int band = c / band_w;
  const int cc = c - band * band_w;
  i = (cc & 3) | ((r & 8) ? 4 : 0) | ((cc >> 4) << 3);
  return (band << 5) | (cc & 8) | ((cc >> 2) & 1) | ((r & 3) << 1) | (((r >> 2) & 1) << 4);
}
template <typename S> struct m2v_is_coop {
  template <typename U, typename V = typename U::m2v_coop> static metal::true_type test(int);
  template <typename U> static metal::false_type test(long);
  static constexpr bool value = decltype(test<S>(0))::value;
};
}  // namespace detail

template <typename T, int ROWS, int COLS, int LANES>
struct cooperative_tensor {
  using m2v_coop = int;
  using m2v_elem = metal::remove_addrspace_t<T>;
  static constexpr int m2v_rows = ROWS;
  static constexpr int m2v_cols = COLS;
  static constexpr int m2v_elems = ROWS * COLS / LANES;
  // the 32-lane groups tile rows when the tile is taller than one group's 16 rows, columns otherwise
  static constexpr int m2v_rows_per_band =
      (ROWS >= 16 * (LANES / M2V_SUBGROUP) && ROWS % (LANES / M2V_SUBGROUP) == 0) ? ROWS / (LANES / M2V_SUBGROUP) : 0;
  static_assert(m2v_rows_per_band == 0 || m2v_rows_per_band == 16 || (m2v_rows_per_band == 32 && LANES == M2V_SUBGROUP),
                "matmul2d emulation covers 16-row and 32-row tiles");
  static constexpr int m2v_band_w = m2v_rows_per_band ? COLS : COLS / (LANES / M2V_SUBGROUP);
  m2v_elem m2v_v[m2v_elems];

  // this thread's lane in the op's tile, including the execution_simdgroups<N> band
  METAL_FUNC int m2v_lane() const {
    return (int)get_sub_group_local_id() +
           ((LANES > M2V_SUBGROUP ? (int)get_sub_group_id() & (LANES / M2V_SUBGROUP - 1) : 0) << 5);
  }

  METAL_FUNC m2v_elem& operator[](int i) { return m2v_v[i]; }
  METAL_FUNC const m2v_elem& operator[](int i) const { return m2v_v[i]; }
  METAL_FUNC metal::array<int, 2> get_multidimensional_index(int i) const {
    metal::array<int, 2> ids;
    int r, c;
    detail::m2v_loc(i, m2v_lane(), m2v_band_w, m2v_rows_per_band, r, c);
    ids[0] = c;
    ids[1] = r;
    return ids;
  }
};

// ---- the matmul2d op ----
template <typename S> struct m2v_scope_lanes { static constexpr int v = M2V_SUBGROUP; static constexpr int n = 1; };
template <int N> struct m2v_scope_lanes<metal::execution_simdgroups<N>> {
  static constexpr int v = M2V_SUBGROUP * N;
  static constexpr int n = N;
};

template <long long DESC, typename SCOPE>
struct matmul2d {
  static constexpr int m2v_m = (int)(DESC & 0xFFF), m2v_n = (int)((DESC >> 12) & 0xFFF), m2v_k = (int)((DESC >> 24) & 0xFFF);
  static constexpr bool m2v_ta = (DESC >> 36) & 1, m2v_tb = (DESC >> 37) & 1;
  static constexpr bool m2v_acc = (DESC >> 39) & 1;
  static constexpr int m2v_lanes = m2v_scope_lanes<SCOPE>::v;
  static constexpr int m2v_sn = m2v_scope_lanes<SCOPE>::n;

  template <typename A, typename B, typename C>
  METAL_FUNC cooperative_tensor<metal::remove_addrspace_t<A>, m2v_m, m2v_k, m2v_lanes>
  get_left_input_cooperative_tensor() const {
    return {};
  }
  template <typename A, typename B, typename C, typename L>
  METAL_FUNC cooperative_tensor<metal::remove_addrspace_t<A>, m2v_m, m2v_k, m2v_lanes>
  get_left_input_cooperative_tensor(const L& src) const {
    using out = cooperative_tensor<metal::remove_addrspace_t<A>, m2v_m, m2v_k, m2v_lanes>;
    out r;
    for (int i = 0; i < out::m2v_elems; i++) r[i] = (typename out::m2v_elem)src[i];
    return r;
  }
  template <typename A, typename B, typename C>
  METAL_FUNC cooperative_tensor<metal::remove_addrspace_t<B>, m2v_k, m2v_n, m2v_lanes>
  get_right_input_cooperative_tensor() const {
    return {};
  }
  template <typename A, typename B, typename C>
  METAL_FUNC cooperative_tensor<C, m2v_m, m2v_n, m2v_lanes> get_destination_cooperative_tensor() const {
    return {};
  }

  // D = op(A) * op(B) (+ D): every lane computes its own destination elements in fp32.
  template <typename SA, typename SB, typename SC>
  METAL_FUNC void run(const SA& a, const SB& b, SC& c) const {
    static_assert(SC::m2v_rows_per_band == 0 || SC::m2v_rows_per_band == 16 || (SC::m2v_rows_per_band == 32 && m2v_lanes == M2V_SUBGROUP),
                  "matmul2d emulation covers 16-row and 32-row tiles");
    static_assert(!detail::m2v_is_coop<SA>::value || m2v_sn == 1, "cooperative sources are single-simdgroup");
    static_assert(!detail::m2v_is_coop<SB>::value || m2v_sn == 1, "cooperative sources are single-simdgroup");
    const int lane = c.m2v_lane();
    for (int i = 0; i < SC::m2v_elems; i++) {
      int r, col;
      detail::m2v_loc(i, lane, SC::m2v_band_w, SC::m2v_rows_per_band, r, col);
      float acc = m2v_acc ? (float)c[i] : 0.0f;
      for (int k = 0; k < m2v_k; k++) {
        acc = ::fma(m2v_a(a, r, k), m2v_b(b, k, col), acc);
      }
      c[i] = (typename SC::m2v_elem)acc;
    }
  }

 private:
  // source reads: (r, k) from the left operand, (k, c) from the right one; tensor views read
  // memory (the transpose flags pick the stored orientation), cooperative tensors read the
  // owning lane's element through a subgroup shuffle
  template <typename S>
  METAL_FUNC float m2v_a(const S& a, int r, int k) const {
    if constexpr (detail::m2v_is_coop<S>::value) {
      static_assert(S::m2v_rows == 16, "cooperative left source needs 16 rows");
      const int ar = m2v_ta ? k : r, ac = m2v_ta ? r : k;
      int i, l;
      l = detail::m2v_owner(ar, ac, S::m2v_band_w, i);
      return sub_group_shuffle((float)a.m2v_v[i], (uint)l);
    } else {
      return a.m2v_at(m2v_ta ? r : k, m2v_ta ? k : r);
    }
  }
  template <typename S>
  METAL_FUNC float m2v_b(const S& b, int k, int c) const {
    if constexpr (detail::m2v_is_coop<S>::value) {
      const int br = m2v_tb ? c : k, bc = m2v_tb ? k : c;
      int i, l;
      l = detail::m2v_owner(br, bc, S::m2v_band_w, i);
      return sub_group_shuffle((float)b.m2v_v[i], (uint)l);
    } else {
      return b.m2v_at(m2v_tb ? k : c, m2v_tb ? c : k);
    }
  }
};

}  // namespace tensor_ops
}  // namespace mpp
