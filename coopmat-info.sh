#!/bin/bash
# Print, per Vulkan device, the cooperative matrix feature flags and the supported (M,N,K,A,B,C,Result,scope) shapes.
vulkaninfo 2>/dev/null | awk '
  /^GPU[0-9]+:/ { g=$1; dev="" }
  /deviceName/ && dev=="" { dev=$0; sub(/.*= /,"",dev); print g, "device:", dev }
  /cooperativeMatrix +=/ { print g, $0 }
  /VkCooperativeMatrixPropertiesKHR/ { inprops=1; n=0 }
  inprops && /MSize|NSize|KSize|AType|BType|CType|ResultType|scope|saturating/ { gsub(/^[ \t]+/,""); printf "%s   %s\n", g, $0; n++ }
  inprops && n>40 { inprops=0 }
'
