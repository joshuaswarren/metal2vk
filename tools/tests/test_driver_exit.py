"""No GPU: failures in the real test driver must reach the calling process."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

DRIVER = Path(__file__).resolve().parents[2] / "m2v-test.py"

class DriverExitTests(unittest.TestCase):
    def run_case(self, case):
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run([sys.executable, str(DRIVER), td, "/bin/false", td, case],
                                    capture_output=True, text=True, timeout=15)
            receipt = Path(td) / "results.json"
            return result, json.loads(receipt.read_text()) if receipt.exists() else None

    def test_unknown_case_fails_before_dispatch(self):
        result, receipt = self.run_case("not_a_real_case")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("unknown case", result.stderr)
        self.assertIsNone(receipt)

    def test_runner_failure_is_not_a_successful_suite(self):
        result, receipt = self.run_case("activation")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue(receipt)
        self.assertTrue(all(not entry["ok"] for entry in receipt.values()))
        self.assertIn("FAIL", result.stdout)

    def test_missing_precise_module_fails_suite(self):
        result, receipt = self.run_case("precise_rsqrt_fp32_boundaries")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(receipt["precise_rsqrt_fp32_boundaries"]["ok"])

    def test_precise_status_controls_common_exit(self):
        # Stand in only for device execution; use the real case selection, result writer and exit gate.
        script = """import runpy,sys
path,td,status=sys.argv[1:]
case='precise_rsqrt_fp32_boundaries'
sys.argv=[path,td,'/bin/false',td,case]
state=runpy.run_path(path,run_name='driver_for_test')['_run_cases'].__globals__
state[case]=lambda: {'status':status}
state['_run_cases']()
"""
        for status in ("pass", "fail", "incomplete", "unsupported"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as td:
                result = subprocess.run([sys.executable, "-c", script, str(DRIVER), td, status],
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0 if status == "pass" else 1, result.stderr)
                entry = json.loads((Path(td) / "results.json").read_text())["precise_rsqrt_fp32_boundaries"]
                self.assertEqual(entry["ok"], status == "pass")

    def test_failed_special_predicate_is_retained(self):
        script = """import runpy,sys
path,td=sys.argv[1:]
sys.argv=[path,td,'/bin/false',td]
judge=runpy.run_path(path,run_name='driver_for_test')['rsqrt_verdict']
bad=judge([0x7f800000],0,[0x3f800000],[0])
good=judge([0x7f800000],0,[0],[0])
assert bad['verdict']=='fail' and len(bad['special_failures'])==1
assert good['verdict']=='pass' and not good['special_failures']
"""
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run([sys.executable, "-c", script, str(DRIVER), td],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == "__main__":
    unittest.main()
