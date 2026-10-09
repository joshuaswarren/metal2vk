#pragma once
// Stub: Metal 4 tensor ops (matmul2d) exist only on Apple's M5-class matrix units. uzu's MXU kernel variants are never
// instantiated for Vulkan, but their headers must parse.
#include "metal_stdlib"
namespace metal {
struct execution_simdgroup {};
struct tensor_inline {};
template <typename I, int... E> struct dextents {};
template <typename T, typename Ex, typename Tag = tensor_inline> struct tensor {};
}
namespace mpp {
namespace tensor_ops {
struct matmul2d_descriptor {
  enum class mode { multiply, multiply_accumulate };
  template <typename... A> constexpr matmul2d_descriptor(A...) {}
  constexpr operator int() const { return 0; }
};
template <int D, typename Scope> struct matmul2d {
  template <typename... A> void run(A...) {}
};
}
}
