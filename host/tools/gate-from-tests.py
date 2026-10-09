#!/usr/bin/env python3
"""Turn uzu metal unit-test results into the M2V_POLICY_FILE `kernels` section.

Input: JSON lines, one per test process (as written by
`run-uzu-tests-traced.sh`):

    {"test": "attention_basic", "status": "pass", "kernels": ["add"],
     "receipt": "host/receipts/G13C/run-g13c.txt"}

Output (stdout): the `kernels` object read by tools/make-gate.py and
`m2v_host::policy::Policy::from_json`:

    {"add": {"state": "verified", "evidence": "pass: attention_basic; receipt: ..."}}

Semantics (host/m2v-host/src/policy.rs):
- a kernel is `verified` only when EVERY test that dispatched it passed AND each
  such test's receipt file exists AND contains no FAIL line; evidence names the
  passing tests and their receipts;
- a kernel dispatched by ANY failing test is `failed` (it produced wrong output
  in that run; policy.rs routes Failed to fallback/refusal, never translated);
- otherwise (no failing test, but a receipt is missing or has a FAIL line) the
  kernel stays `unverified`.
"""
import json
import sys


def kernels_section(records, exists=lambda path: True, contains_fail=lambda path: False):
    passed = {}
    failed = {}
    broken_receipt = {}
    for rec in records:
        status = rec.get("status")
        test = rec.get("test", "")
        for kernel in rec.get("kernels", []):
            if status == "pass":
                passed.setdefault(kernel, []).append(rec)
            else:
                failed.setdefault(kernel, []).append(rec)
                broken_receipt.setdefault(kernel, []).append(rec)
                continue
            receipt = rec.get("receipt", "")
            if not receipt or not exists(receipt) or contains_fail(receipt):
                broken_receipt.setdefault(kernel, []).append(rec)
    kernels = {}
    names = set(passed) | set(failed) | set(broken_receipt)
    for name in sorted(names):
        if name in failed:
            tests_ = ", ".join(sorted({r.get("test", "") for r in failed[name]}))
            kernels[name] = {"state": "failed", "evidence": f"failed: {tests_}"}
        elif name in broken_receipt:
            why = "receipt missing" if any(
                not r.get("receipt", "") or not exists(r.get("receipt", ""))
                for r in broken_receipt[name]
            ) else "receipt has FAIL line"
            kernels[name] = {"state": "unverified", "evidence": why}
        else:
            recs = passed[name]
            tests_ = ", ".join(sorted({r.get("test", "") for r in recs}))
            receipts = sorted({r.get("receipt", "") for r in recs})
            kernels[name] = {
                "state": "verified",
                "evidence": f"pass: {tests_}; receipts: {', '.join(receipts)}",
            }
    return {"kernels": kernels}


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} RESULTS.jsonl", file=sys.stderr)
        return 2
    records = []
    with open(argv[1]) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    def exists(path):
        try:
            import os
            return os.path.isfile(path)
        except OSError:
            return False

    def contains_fail(path):
        try:
            import os
            with open(path) as f:
                return any(line.startswith("FAIL ") or " FAIL " in line for line in f)
        except OSError:
            return False

    json.dump(kernels_section(records, exists, contains_fail), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
