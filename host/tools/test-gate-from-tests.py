#!/usr/bin/env python3
"""Unit tests for gate-from-tests.py: pass, fail, missing receipt, FAIL line in
receipt, and a kernel shared by a passing and a failing test. No GPU, no network."""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "gate-from-tests.py")


def run(records, files=None):
    with tempfile.TemporaryDirectory() as tmp:
        for path, content in (files or {}).items():
            full = os.path.join(tmp, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(content)
        results = os.path.join(tmp, "results.jsonl")
        with open(results, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        proc = subprocess.run(
            [sys.executable, TOOL, results],
            capture_output=True,
            text=True,
            cwd=tmp,
        )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["kernels"]


def main():
    # 1. pass with an existing, FAIL-free receipt -> verified
    out = run(
        [{"test": "t1", "status": "pass", "kernels": ["add"], "receipt": "r.txt"}],
        {"r.txt": "PASS t1\n"},
    )
    assert out["add"]["state"] == "verified", out
    assert "t1" in out["add"]["evidence"] and "r.txt" in out["add"]["evidence"], out

    # 2. failing test using the kernel -> failed, evidence names the test
    out = run(
        [
            {"test": "t1", "status": "pass", "kernels": ["add"], "receipt": "r.txt"},
            {"test": "t2", "status": "fail", "kernels": ["add"], "receipt": "r.txt"},
        ],
        {"r.txt": "PASS t1\nFAIL t2\n"},
    )
    assert out["add"]["state"] == "failed", out
    assert "t2" in out["add"]["evidence"], out

    # 3. missing receipt -> unverified
    out = run([{"test": "t1", "status": "pass", "kernels": ["add"], "receipt": "missing.txt"}])
    assert out["add"]["state"] == "unverified", out
    assert "receipt missing" in out["add"]["evidence"], out

    # 4. receipt containing a FAIL line -> unverified
    out = run(
        [{"test": "t1", "status": "pass", "kernels": ["add"], "receipt": "r.txt"}],
        {"r.txt": "PASS t1\nFAIL other\n"},
    )
    assert out["add"]["state"] == "unverified", out
    assert "FAIL line" in out["add"]["evidence"], out

    # 5. kernel shared by a pass and a fail -> failed (never verified over a fail)
    out = run(
        [
            {"test": "tp", "status": "pass", "kernels": ["shared", "only_pass"], "receipt": "r.txt"},
            {"test": "tf", "status": "fail", "kernels": ["shared"], "receipt": "r.txt"},
        ],
        {"r.txt": "PASS tp\n"},
    )
    assert out["shared"]["state"] == "failed", out
    assert out["only_pass"]["state"] == "verified", out

    # 6. no FAIL keyword in the receipt even when the test status is pass
    out = run(
        [{"test": "t1", "status": "pass", "kernels": ["add"], "receipt": "r.txt"}],
        {"r.txt": "ok everything fine\n"},
    )
    assert out["add"]["state"] == "verified", out

    print("PASS gate-from-tests: 6/6 cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
