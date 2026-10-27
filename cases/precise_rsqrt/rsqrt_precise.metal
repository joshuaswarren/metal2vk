// One scalar precise::rsqrt per lane over raw binary32 bits. The same file is the native-Metal comparator source
// (bench/native-metal capture mode) and the metal2vk input (tools/m2v-compile), so both sides run one operation.
// Input and output are uint buffers: nothing is converted to a host float, and the operand is a runtime load.
#include <metal_stdlib>
using namespace metal;

[[kernel]] void rsqrt_precise(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {
  out[i] = as_type<uint>(precise::rsqrt(as_type<float>(in[i])));
}
