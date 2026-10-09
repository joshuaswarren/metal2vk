#!/usr/bin/env python3
"""Pack SPIR-V + m2v-reflect.py JSON into a .m2vlib container.

This is the reference emitter the kernel pipeline adopts: compile.sh (or a
driver) runs `m2v-reflect.py MODULE.spv > MODULE.json`, then

    make-m2vlib.py MODULE.spv MODULE.json OUT.m2vlib [--constants C.json] [--rename K=N]

Container layout (host/DESIGN.md): "M2VL", u32 LE version 1, u32 LE json_len,
UTF-8 JSON, then 4-byte-aligned SPIR-V blobs. The JSON section mirrors
m2v-reflect.py: set 0 binding = argument order, POD arguments in push
constants after the 16-byte offsets block, workgroup size in spec constants
0..2 (or the module's SpecConstantWorkgroupSize ids), OpenCL local pointers as
spec-sized threadgroup arrays. Metal function constants do not exist in clspv
reflection; pass them per kernel in --constants FILE ({"kernel": [{"index",
"spec_id", "type"}...]}).
"""
import argparse
import json
import struct
import sys

MAGIC = b"M2VL"
VERSION = 1


def function_desc(kernel, spv_off, spv_len, constants_by_name, regions):
    buffers, push_words, threadgroup = [], [], []
    pod_end = 0
    for a in kernel["args"]:
        if a["kind"] in ("storage_buffer", "uniform_buffer", "pod_buffer"):
            buffers.append({"arg": a["ordinal"], "binding": a["binding"], "set": a["set"]})
        elif a["kind"] == "pod_push_constant":
            push_words.append({"arg": a["ordinal"], "offset": a["offset"], "size": a["size"]})
            pod_end = max(pod_end, a["offset"] + a["size"])
        elif a["kind"] == "workgroup_memory":
            threadgroup.append({"arg": a["ordinal"], "spec_id": a["spec_id"]})
    # clspv kernels always declare the group/region offsets push block
    # (PushConstantRegion*), even with no POD arguments: the layout must cover
    # it or the driver sees a shader push block the layout does not describe.
    region_end = max((r["offset"] + r["size"] for r in regions.values()), default=0)
    wg = kernel.get("workgroup_size") or [1, 1, 1]
    wg_spec = None if kernel.get("workgroup_size") else None
    return {
        "name": kernel["name"],
        "entry": kernel["name"],
        "spv": {"offset": spv_off, "len": spv_len},
        "workgroup_size": wg,
        "workgroup_size_spec_ids": wg_spec,
        "buffers": sorted(buffers, key=lambda b: b["arg"]),
        "push": {"size": max(pod_end, region_end), "words": push_words},
        "constants": constants_by_name.get(kernel["name"], []),
        "threadgroup": threadgroup,
        "uses_coopmat": False,
    }


def main():
    ap = argparse.ArgumentParser(description="pack SPIR-V + reflection JSON into .m2vlib")
    ap.add_argument("spv")
    ap.add_argument("reflection")
    ap.add_argument("out")
    ap.add_argument("--constants", help='JSON file {"kernel": [{"index","spec_id","type"}...]}')
    ap.add_argument("--uses-coopmat", action="store_true",
                    help="mark the module as needing VK_KHR_cooperative_matrix "
                         "(m2v-reflect.py detects the capability itself when given)")
    args = ap.parse_args()

    with open(args.spv, "rb") as f:
        spv = f.read()
    with open(args.reflection) as f:
        refl = json.load(f)
    constants = {}
    if args.constants:
        with open(args.constants) as f:
            constants = json.load(f)

    if not args.uses_coopmat and "uses_cooperative_matrix" in refl:
        uses_coopmat = bool(refl["uses_cooperative_matrix"])
    else:
        uses_coopmat = args.uses_coopmat

    wg_spec_ids = refl.get("workgroup_size_spec_constant_ids")

    header_len = 12
    json_len = 0
    spv_off = 0
    for _ in range(8):  # json length embeds spv_off; iterate to the fixed point
        spv_off = (header_len + json_len + 3) & ~3
        descs = []
        for k in refl["kernels"]:
            d = function_desc(k, spv_off, len(spv), constants, refl.get("push_constant_regions", {}))
            d["uses_coopmat"] = uses_coopmat
            if wg_spec_ids and not k.get("workgroup_size"):
                d["workgroup_size_spec_ids"] = wg_spec_ids
            descs.append(d)
        blob = json.dumps({"functions": descs}).encode()
        if len(blob) == json_len:
            break
        json_len = len(blob)
    else:
        sys.exit("make-m2vlib: JSON length did not converge")

    out = bytearray()
    out += MAGIC
    out += struct.pack("<II", VERSION, json_len)
    out += blob
    out += b"\0" * (spv_off - len(out))
    out += spv
    with open(args.out, "wb") as f:
        f.write(out)
    print(f"{args.out}: {len(descs)} function(s), json {json_len} B, spv {len(spv)} B at {spv_off}")


if __name__ == "__main__":
    main()
