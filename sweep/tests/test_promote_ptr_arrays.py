"""Repeated pointer-array accesses must preserve ordering and reject escapes."""
import importlib.util
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("promote", HERE.parent / "promote_ptr_arrays.py")
promote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promote)


class ReusedPointerArrayTest(unittest.TestCase):
    def setUp(self):
        self.source = (HERE / "promote_ptr_arrays_reused.ll").read_text()

    def rewrite(self, source):
        return promote.rewrite_function(source.splitlines(keepends=True), promote.pointer_index_bits(source))

    def test_repeated_load_store_execution(self):
        # The oracle in main checks both dynamic indices and the value across
        # a store between two loads. Execute original and transformed IR.
        lines, count = self.rewrite(self.source)
        self.assertEqual(count, 1)
        self.assertNotIn("ptr %s0", "".join(lines))
        clang = os.environ.get("CLANG", "clang")
        if not shutil.which(clang):
            self.skipTest("clang required to execute IR")
        with tempfile.TemporaryDirectory() as tmp:
            for name, source in (("original", self.source), ("promoted", "".join(lines))):
                ll = pathlib.Path(tmp) / (name + ".ll")
                ll.write_text(source.replace('target datalayout = "e-p:64:64:64:64"\n', "", 1))
                exe = pathlib.Path(tmp) / name
                subprocess.run([clang, "-Wno-override-module", str(ll), "-o", str(exe)], check=True)
                subprocess.run([str(exe)], check=True)

    def test_unknown_uses_decline_promotion(self):
        cases = [
            "  call void @escape(ptr %s0)\n",
            "  call void @escape(ptr %table)\n",
            "  %other = getelementptr i8, ptr %table, i32 0\n",
            "  %unexpected = load ptr addrspace(2), ptr %s0, align 8\n",
            "  %unexpected = load volatile ptr addrspace(1), ptr %s0, align 8\n",
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                source = self.source.replace("  %p0 =", extra + "  %p0 =", 1)
                lines, count = self.rewrite(source)
                self.assertEqual(count, 0)
                self.assertEqual("".join(lines), source)

    def test_quoted_semicolon_callee_cannot_hide_escape(self):
        # These are valid LLVM symbols. The semicolon is not a comment, and
        # both the array and its GEP must remain untouched when passed out.
        opt = os.environ.get("OPT", "opt")
        for escaped in ("%s0", "%table"):
            with self.subTest(escaped=escaped):
                source = self.source.replace("  %p0 =", f'  call void @"touch;slot"(ptr {escaped})\n  %p0 =', 1)
                source += 'declare void @"touch;slot"(ptr)\n'
                lines, count = self.rewrite(source)
                self.assertEqual(count, 0)
                self.assertEqual("".join(lines), source)
                if shutil.which(opt):
                    with tempfile.TemporaryDirectory() as tmp:
                        path = pathlib.Path(tmp) / "escaped.ll"
                        path.write_text(source)
                        subprocess.run([opt, "-passes=verify", "-disable-output", str(path)], check=True)

    def test_quoted_callee_array_mutation_preserves_result(self):
        source = self.source.replace(
            "  store ptr addrspace(1) %bp, ptr %s0, align 8\n",
            '  call void @"set;slot"(ptr %table, i32 %idx, ptr addrspace(1) %bp)\n',
        )
        source += ("""define void @"set;slot"(ptr %array, i32 %idx, ptr addrspace(1) %value) {
  %element = getelementptr [2 x ptr addrspace(1)], ptr %array, i32 0, i32 %idx
  store ptr addrspace(1) %value, ptr %element, align 8
  ret void
}
""")
        lines, count = self.rewrite(source)
        self.assertEqual(count, 0)
        self.assertEqual("".join(lines), source)
        clang = os.environ.get("CLANG", "clang")
        if not shutil.which(clang):
            self.skipTest("clang required to execute IR")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "mutation.ll"
            path.write_text("".join(lines).replace('target datalayout = "e-p:64:64:64:64"\n', "", 1))
            exe = pathlib.Path(tmp) / "mutation"
            subprocess.run([clang, "-Wno-override-module", str(path), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

    def test_quoted_alias_mutation_preserves_result(self):
        source = self.source.replace(
            "  store ptr addrspace(1) %bp, ptr %s0, align 8\n",
            '  call void @setslot(ptr %"table", i32 %idx, ptr addrspace(1) %bp)\n',
        )
        source += """define void @setslot(ptr %array, i32 %idx, ptr addrspace(1) %value) {
  %element = getelementptr [2 x ptr addrspace(1)], ptr %array, i32 0, i32 %idx
  store ptr addrspace(1) %value, ptr %element, align 8
  ret void
}
"""
        lines, count = self.rewrite(source)
        self.assertEqual(count, 0)
        self.assertEqual("".join(lines), source)
        clang = os.environ.get("CLANG", "clang")
        if not shutil.which(clang):
            self.skipTest("clang required to execute IR")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "alias.ll"
            path.write_text("".join(lines).replace('target datalayout = "e-p:64:64:64:64"\n', "", 1))
            exe = pathlib.Path(tmp) / "alias"
            subprocess.run([clang, "-Wno-override-module", str(path), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

    def assert_valid_ir_declined(self, source):
        lines, count = self.rewrite(source)
        self.assertEqual(count, 0)
        self.assertEqual("".join(lines), source)
        opt = os.environ.get("OPT", "opt")
        if shutil.which(opt):
            with tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "escape.ll"
                path.write_text(source)
                subprocess.run([opt, "-passes=verify", "-disable-output", str(path)], check=True)

    def test_quoted_array_and_element_aliases_decline(self):
        for name in ('%"table"', '%"s0"'):
            with self.subTest(name=name):
                source = self.source.replace("  %p0 =", f"  call void @escape(ptr {name})\n  %p0 =", 1)
                self.assert_valid_ir_declined(source + "declare void @escape(ptr)\n")

    def test_generated_temporary_collision_declines(self):
        source = self.source.replace("  %p0 =", "  %pa1c = add i32 1, 2\n  %p0 =", 1)
        self.assert_valid_ir_declined(source)

    def test_generated_slot_collision_declines(self):
        source = self.source.replace("  %p0 =", "  %patable.s0 = alloca i32, align 4\n  %p0 =", 1)
        self.assert_valid_ir_declined(source)

    def test_generated_label_collisions_decline(self):
        for label in ('pa1c', '"pa1c"', '"\\70a1c"'):
            with self.subTest(label=label):
                source = self.source.replace("  ret i32 %r\n}", f"  ret i32 %r\n{label}:\n  ret i32 0\n}}", 1)
                self.assert_valid_ir_declined(source)

    def test_narrow_pointer_index_declines_and_preserves_semantics(self):
        source = self.source.replace('e-p:64:64:64:64', 'e-p:64:64:64:32')
        source = source.replace('getelementptr inbounds [2 x ptr addrspace(1)]',
                                'getelementptr [2 x ptr addrspace(1)]')
        source = source.replace('i32 %idx', 'i64 %idx').replace('@test(i32 0)', '@test(i64 4294967296)')
        source = source.replace('@test(i32 1)', '@test(i64 4294967297)')
        self.assert_valid_ir_declined(source)
        opt = os.environ.get("OPT", "opt")
        if not shutil.which(opt):
            self.skipTest("opt required to check index truncation")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "index.ll"
            path.write_text(source)
            optimized = subprocess.run([opt, "-passes=default<O2>", "-S", str(path), "-o", "-"],
                                       check=True, capture_output=True, text=True).stdout
            self.assertRegex(optimized, r"(?s)define [^\n]*@main\([^)]*\)[^\n]*\{.*?ret i32 0")

    def test_inbounds_i64_index_declines(self):
        source = self.source.replace('i32 %idx', 'i64 %idx')
        source = source.replace('@test(i32 0)', '@test(i64 0)').replace('@test(i32 1)', '@test(i64 1)')
        self.assert_valid_ir_declined(source)

    def test_plain_gep_scaled_offset_wrap_declines(self):
        source = self.source.replace('e-p:64:64:64:64', 'e-p:32:32')
        source = source.replace('getelementptr inbounds [2 x ptr addrspace(1)]',
                                'getelementptr [2 x ptr addrspace(1)]')
        source = source.replace('@test(i32 0)', '@test(i32 1073741824)')
        source = source.replace('@test(i32 1)', '@test(i32 1073741825)')
        self.assert_valid_ir_declined(source)
        opt = os.environ.get("OPT", "opt")
        if not shutil.which(opt):
            self.skipTest("opt required to check wrapped GEP offsets")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "wrap.ll"
            path.write_text(source)
            optimized = subprocess.run([opt, "-passes=default<O2>", "-S", str(path), "-o", "-"],
                                       check=True, capture_output=True, text=True).stdout
            self.assertRegex(optimized, r"(?s)define [^\n]*@main\([^)]*\)[^\n]*\{.*?ret i32 0")

    def test_ambiguous_or_narrow_layout_declines(self):
        for layout in ('', 'e-p:64:64:64:16', 'e-p1:64:64'):
            with self.subTest(layout=layout):
                source = self.source.replace('target datalayout = "e-p:64:64:64:64"',
                                             f'target datalayout = "{layout}"')
                lines, count = self.rewrite(source)
                self.assertEqual(count, 0)
                self.assertEqual("".join(lines), source)

    def test_lifetime_restart_preserves_values(self):
        source = self.source.replace('  %s0 = getelementptr',
                                     '  call void @llvm.lifetime.start.p0(ptr %table)\n  %s0 = getelementptr', 1)
        source = source.replace('  store ptr addrspace(1) %bp, ptr %s0, align 8',
                                '  call void @llvm.lifetime.end.p0(ptr %table)\n'
                                '  call void @llvm.lifetime.start.p0(ptr %table)\n'
                                '  store ptr addrspace(1) %bp, ptr %s0, align 8', 1)
        source = source.replace('  ret i32 %r', '  call void @llvm.lifetime.end.p0(ptr %table)\n  ret i32 %r', 1)
        source += 'declare void @llvm.lifetime.start.p0(ptr)\ndeclare void @llvm.lifetime.end.p0(ptr)\n'
        lines, count = self.rewrite(source)
        self.assertEqual(count, 1)
        transformed = "".join(lines)
        self.assertNotIn('llvm.lifetime.start.p0(ptr %table)', transformed)
        for j in range(2):
            slot = f'%patable.s{j}'
            start = f'call void @llvm.lifetime.start.p0(ptr {slot})'
            init = f'store ptr addrspace(1) null, ptr {slot}, align 8'
            self.assertEqual(transformed.count(start), 2)
            self.assertEqual(transformed.count(init), 2)
            self.assertLess(transformed.index(start), transformed.index(init))
        clang = os.environ.get("CLANG", "clang")
        if not shutil.which(clang):
            self.skipTest("clang required to execute IR")
        with tempfile.TemporaryDirectory() as tmp:
            for name, ir in (("original", source), ("promoted", transformed)):
                path = pathlib.Path(tmp) / (name + ".ll")
                path.write_text(ir.replace('target datalayout = "e-p:64:64:64:64"\n', '', 1))
                exe = pathlib.Path(tmp) / name
                subprocess.run([clang, "-Wno-override-module", str(path), "-o", str(exe)], check=True)
                subprocess.run([str(exe)], check=True)

    def test_undef_index_frozen_once_per_gep(self):
        source = self.source.replace('define i32 @test(i32 %idx)', 'define i32 @test(i32 %u)')
        source = source.replace('entry:\n', 'entry:\n  %idx = and i32 %u, 1\n', 1)
        lines, count = self.rewrite(source)
        self.assertEqual(count, 1)
        transformed = "".join(lines)
        self.assertEqual(transformed.count('%paidx0 = freeze i32 %idx'), 1)
        self.assertNotIn('icmp eq i32 %idx,', transformed)
        self.assertGreaterEqual(transformed.count('icmp eq i32 %paidx0,'), 4)
        opt = os.environ.get("OPT", "opt")
        if shutil.which(opt):
            with tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "freeze.ll"
                path.write_text(transformed)
                subprocess.run([opt, "-passes=verify", "-disable-output", str(path)], check=True)

    def test_leading_zero_numeric_aliases_decline(self):
        source = self.source.replace('%table', '%0')
        source = source.replace('  store ptr addrspace(1) %bp, ptr %s0, align 8\n',
                                '  call void @setslot(ptr %00, i32 %idx, ptr addrspace(1) %bp)\n')
        source += """define void @setslot(ptr %array, i32 %idx, ptr addrspace(1) %value) {
  %element = getelementptr inbounds [2 x ptr addrspace(1)], ptr %array, i32 0, i32 %idx
  store ptr addrspace(1) %value, ptr %element, align 8
  ret void
}
"""
        self.assert_valid_ir_declined(source)
        clang = os.environ.get("CLANG", "clang")
        if not shutil.which(clang):
            self.skipTest("clang required to execute IR")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "numeric.ll"
            path.write_text(source.replace('target datalayout = "e-p:64:64:64:64"\n', '', 1))
            exe = pathlib.Path(tmp) / "numeric"
            subprocess.run([clang, "-Wno-override-module", str(path), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
        # The element GEP can have a numeric alias too.
        source = self.source.replace('%table', '%0').replace('%s0', '%1')
        source = source.replace('  %p0 =', '  call void @escape(ptr %01)\n  %p0 =', 1)
        self.assert_valid_ir_declined(source + 'declare void @escape(ptr)\n')

    def test_zero_slot_array_declines(self):
        source = self.source.replace('[2 x ptr addrspace(1)]', '[0 x ptr addrspace(1)]')
        self.assert_valid_ir_declined(source)

    def test_complete_ssa_names(self):
        source = self.source.replace("  %p0 =", "  %s00 = add i32 1, 2\n  %p0 =", 1)
        _, count = self.rewrite(source)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
