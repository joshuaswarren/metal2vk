#!/bin/bash
# Metal kernels include several umbrella headers; they all resolve to the one shim.
set -eu
D=$(cd "$(dirname "$0")" && pwd)/include
for h in metal_simdgroup metal_simdgroup_matrix metal_atomic metal_math metal_compute metal_geometric metal_integer \
         metal_common metal_relational metal_pack metal_texture metal_graphics metal_types metal_array; do
  printf '#pragma once\n#include "metal_stdlib"\n' > "$D/$h"
done
ls "$D"
