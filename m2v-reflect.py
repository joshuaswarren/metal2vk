#!/usr/bin/env python3
"""Dump the kernels in a clspv-produced SPIR-V module as JSON, read from the embedded NonSemantic.ClspvReflection records.
Usage: m2v-reflect.py MODULE.spv > MODULE.json      (needs spirv-dis from SPIRV-Tools)
The runner contract (set 0, binding = argument order; POD arguments in push constants after the offsets block; workgroup size
in spec constants 0 to 2) is described per kernel so a host does not need to parse SPIR-V itself."""
import json
import re
import subprocess
import sys

dis = subprocess.run(["spirv-dis", "--no-header", sys.argv[1]], capture_output=True, text=True, check=True).stdout
strings = {}
consts = {}
for ln in dis.splitlines():
    m = re.match(r"\s*(%\S+) = OpString \"(.*)\"", ln)
    if m:
        strings[m.group(1)] = m.group(2)
    m = re.match(r"\s*(%\S+) = OpConstant %\S+ (\d+)", ln)
    if m:
        consts[m.group(1)] = int(m.group(2))


def num(tok):
    if tok in consts:
        return consts[tok]
    m = re.match(r"%uint_(\d+)$", tok)
    return int(m.group(1)) if m else None


kernels = {}
order = []
global_regions = {}
spec_wg = None
for ln in dis.splitlines():
    m = re.match(r"\s*(%\S+) = OpExtInst %void %\S+ (\w+)(.*)", ln)
    if not m:
        continue
    rid, op, rest = m.groups()
    toks = rest.split()
    if op == "Kernel":
        name = strings.get(toks[1], toks[1])
        kernels[rid] = {"name": name, "args": [], "workgroup_size": None}
        order.append(rid)
    elif op in ("ArgumentStorageBuffer", "ArgumentUniform"):
        k = kernels[toks[0]]
        k["args"].append({"ordinal": num(toks[1]), "kind": "storage_buffer" if op == "ArgumentStorageBuffer" else "uniform_buffer",
                          "set": num(toks[2]), "binding": num(toks[3])})
    elif op == "ArgumentPodPushConstant":
        kernels[toks[0]]["args"].append({"ordinal": num(toks[1]), "kind": "pod_push_constant", "offset": num(toks[2]), "size": num(toks[3])})
    elif op in ("ArgumentPodStorageBuffer", "ArgumentPodUniform"):
        kernels[toks[0]]["args"].append({"ordinal": num(toks[1]), "kind": "pod_buffer", "set": num(toks[2]), "binding": num(toks[3]),
                                         "offset": num(toks[4]), "size": num(toks[5])})
    elif op == "ArgumentWorkgroup":
        kernels[toks[0]]["args"].append({"ordinal": num(toks[1]), "kind": "workgroup_memory", "spec_id": num(toks[2]), "elem_size": num(toks[3])})
    elif op == "PropertyRequiredWorkgroupSize":
        kernels[toks[0]]["workgroup_size"] = [num(toks[1]), num(toks[2]), num(toks[3])]
    elif op == "SpecConstantWorkgroupSize":
        spec_wg = [num(toks[0]), num(toks[1]), num(toks[2])]
    elif op.startswith("PushConstant"):
        global_regions[op] = {"offset": num(toks[0]), "size": num(toks[1])}

caps = sorted(set(re.findall(r"OpCapability (\w+)", dis)))
exts = sorted(set(re.findall(r"OpExtension \"([^\"]+)\"", dis)))
out = {"module": sys.argv[1], "capabilities": caps, "extensions": exts, "uses_cooperative_matrix": "CooperativeMatrixKHR" in caps,
       "workgroup_size_spec_constant_ids": spec_wg, "push_constant_regions": global_regions,
       "kernels": [kernels[k] for k in order]}
for k in out["kernels"]:
    k["args"].sort(key=lambda a: a["ordinal"])
json.dump(out, sys.stdout, indent=1)
print()
