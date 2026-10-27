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

if __name__ == "__main__":
    unittest.main()
