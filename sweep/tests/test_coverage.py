"""Exercise the coverage gate CLI without a compiler, GPU, or third-party packages."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


GATE = Path(__file__).resolve().parents[1] / "check_coverage.py"


def row(entry, *, variant="-", parse="ok", spv="ok", val="ok", error=""):
    return {"set": "uzu", "file": "kernel.metal", "entry": entry,
            "variant": variant, "parse": parse, "spv": spv, "val": val, "error": error}


class CoverageGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.floor = self.directory / "floor.txt"

    def run_gate(self, rows, floor, *, update=False):
        result = self.directory / "result.json"
        result.write_text(json.dumps(rows))
        self.floor.write_text("".join(entry + "\n" for entry in floor))
        command = [sys.executable, str(GATE), str(result), "--floor", str(self.floor)]
        if update:
            command.append("--update")
        return subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)

    def test_only_floor_entries_must_be_valid(self):
        floor = ["uzu/kernel.metal::kept[-]"]
        result = self.run_gate([row("kept"), row("new", val="FAIL", error="invalid module")], floor)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.floor.read_text().splitlines(), floor)

    def test_missing_floor_entry_fails(self):
        missing = "uzu/kernel.metal::missing[-]"
        result = self.run_gate([row("other")], [missing])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(missing, result.stdout)
        self.assertEqual(self.floor.read_text().splitlines(), [missing])

    def test_invalid_floor_entry_fails(self):
        failed = "uzu/kernel.metal::kept[-]"
        result = self.run_gate([row("kept", val="FAIL", error="invalid module")], [failed])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(failed, result.stdout)

    def test_valid_variant_does_not_cover_a_missing_variant(self):
        missing = "uzu/kernel.metal::kept[half]"
        result = self.run_gate([row("kept", variant="float")], [missing])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(missing, result.stdout)

    def test_clspv_timeout_warns_without_lowering_floor(self):
        floor = ["uzu/kernel.metal::slow[-]"]
        for error in ("timeout", "timeout after 300 s (clspv does not finish)"):
            with self.subTest(error=error):
                result = self.run_gate([row("slow", spv="FAIL", val="", error=error)], floor)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(floor[0], result.stdout)
                self.assertEqual(self.floor.read_text().splitlines(), floor)

    def test_timeout_does_not_hide_a_separate_regression(self):
        floor = ["uzu/kernel.metal::slow[-]", "uzu/kernel.metal::broken[-]"]
        result = self.run_gate([
            row("slow", spv="FAIL", val="", error="timeout after 300 s (clspv does not finish)"),
            row("broken", val="FAIL", error="invalid module"),
        ], floor)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(floor[1], result.stdout)

    def test_timeout_identifier_in_parse_error_is_a_regression(self):
        floor = ["uzu/kernel.metal::broken[-]"]
        result = self.run_gate([
            row("broken", parse="FAIL", spv="", val="", error="use of undeclared identifier 'timeout'"),
        ], floor)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(floor[0], result.stdout)

    def test_clspv_error_starting_with_timeout_is_a_regression(self):
        floor = ["uzu/kernel.metal::broken[-]"]
        result = self.run_gate([
            row("broken", spv="FAIL", val="", error="timeout option is invalid"),
        ], floor)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(floor[0], result.stdout)

    def test_update_replaces_floor_with_sorted_valid_entries(self):
        result = self.run_gate([
            row("zeta"), row("broken", val="FAIL", error="invalid module"), row("alpha"),
        ], ["uzu/kernel.metal::old[-]"], update=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.floor.read_text(), "uzu/kernel.metal::alpha[-]\nuzu/kernel.metal::zeta[-]\n")


if __name__ == "__main__":
    unittest.main()
