#!/usr/bin/env python3
"""Count Metal host API usage in crates/uzu-engine/src/backends/metal (mtl-rs `metal::` and objc2 calls).
Usage: host-api-survey.py [UZU_ROOT]"""
import collections
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".") / "crates/uzu-engine/src/backends/metal"
types = collections.Counter()
methods = collections.Counter()
objc = collections.Counter()
files = list(root.rglob("*.rs"))
loc = 0
for f in files:
    t = f.read_text()
    loc += t.count("\n")
    for m in re.finditer(r"\b(MTL[A-Za-z0-9]+|Metal[A-Za-z]+Device|metal::[A-Z][A-Za-z0-9]+)\b", t):
        types[m.group(1)] += 1
    for m in re.finditer(r"\.(new_[a-z_]+|set_[a-z_]+|dispatch_[a-z_]+|encode_[a-z_]+|commit|wait_until_[a-z_]+|add_[a-z_]+|use_[a-z_]+|make_[a-z_]+|compute_command_encoder|blit_command_encoder|copy_from_[a-z_]+|fill_buffer|end_encoding|contents|gpu_start_time|gpu_end_time|status|enqueue|label|argument_buffer[a-z_]*|function_with_name|new_library_with_[a-z_]+|max_threads_per_threadgroup|thread_execution_width|threadgroup_memory_[a-z_]+|resource_options|recommended_max_working_set_size|current_allocated_size|has_unified_memory|supports_[a-z_]+)\(", t):
        methods[m.group(1)] += 1
    for m in re.finditer(r"\b(objc2(?:_[a-z_]+)?::[A-Za-z_:]+|msg_send!?|NSString|NSBundle|autoreleasepool|Retained|ProtocolObject)\b", t):
        objc[m.group(1)] += 1
print(f"files={len(files)} rust_loc={loc}")
print("TYPES", types.most_common(40))
print("METHODS", methods.most_common(60))
print("OBJC2", objc.most_common(20))
