"""
tests/test_harness.py
=====================
Comprehensive test suite for Kartavya's SLM-compatible harness.

Tests cover:
  - str_replace cascade strategies (all 9 opencode strategies)
  - Indent-delta adjustment for SLM outputs
  - Syntax validation and rejection
  - run_python pass/fail banners
  - read_file with line numbers
  - Full code-correction end-to-end with mocked SLM
  - Plan parsing (JSON extraction, single-quoted JSON, trailing commas)

Run:  cd kartavya_fixed && python -m pytest tests/test_harness.py -v
"""

import ast
import os
import sys
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tools import (
    ToolRegistry,
    _apply_replace,
    _check_python_syntax,
    _remove_indent,
    _get_min_indent,
    _delta_indent,
    _levenshtein,
    _simple_replacer,
    _line_trimmed_replacer,
    _indent_flexible_replacer,
    _whitespace_normalized_replacer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_tmp(content: str, suffix: str = ".py") -> str:
    """Write content to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False,
                                    encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


def cleanup(path: str):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helper function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers(unittest.TestCase):

    def test_remove_indent_basic(self):
        text = "    def foo():\n        return 1"
        result = _remove_indent(text)
        self.assertEqual(result, "def foo():\n    return 1")

    def test_remove_indent_mixed(self):
        text = "        a\n        b\n            c"
        result = _remove_indent(text)
        self.assertEqual(result, "a\nb\n    c")

    def test_get_min_indent_zero(self):
        self.assertEqual(_get_min_indent("def f():\n    return 1"), 0)

    def test_get_min_indent_four(self):
        self.assertEqual(_get_min_indent("    return a + b"), 4)

    def test_delta_indent_add(self):
        result = _delta_indent("if x:\n    return x", 4)
        self.assertEqual(result, "    if x:\n        return x")

    def test_delta_indent_remove(self):
        result = _delta_indent("    if x:\n        return x", -4)
        self.assertEqual(result, "if x:\n    return x")

    def test_levenshtein_identical(self):
        self.assertEqual(_levenshtein("abc", "abc"), 0)

    def test_levenshtein_one_edit(self):
        self.assertEqual(_levenshtein("abc", "ab"), 1)

    def test_check_syntax_valid(self):
        ok, err = _check_python_syntax("def f():\n    return 1\n")
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_check_syntax_invalid(self):
        ok, err = _check_python_syntax("def f():\nreturn 1\n")
        self.assertFalse(ok)
        self.assertIn("SyntaxError", err)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Individual replacer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReplacers(unittest.TestCase):

    FILE = textwrap.dedent("""\
        def subtract(a, b):
            \"\"\"Bug: returns sum.\"\"\"
            return a + b  # BUG

        def divide(a, b):
            return a / b  # BUG
    """)

    def test_simple_replacer_exact_line(self):
        """SimpleReplacer must find exact line-boundary matches."""
        hits = list(_simple_replacer(self.FILE, "    return a + b  # BUG"))
        self.assertEqual(len(hits), 1)

    def test_simple_replacer_no_mid_line(self):
        """SimpleReplacer must NOT match mid-line substrings."""
        hits = list(_simple_replacer(self.FILE, "return a + b  # BUG"))
        # This MUST be empty — no line-boundary match for stripped string
        self.assertEqual(len(hits), 0)

    def test_line_trimmed_finds_stripped(self):
        """LineTrimmedReplacer should match when old_str is stripped."""
        hits = list(_line_trimmed_replacer(self.FILE, "return a + b  # BUG"))
        self.assertGreater(len(hits), 0)
        # The yielded block must have the ORIGINAL indentation
        self.assertIn("    return a + b", hits[0])

    def test_indent_flexible_finds_stripped(self):
        """IndentationFlexibleReplacer should match stripped old_str."""
        hits = list(_indent_flexible_replacer(self.FILE, "return a / b  # BUG"))
        self.assertGreater(len(hits), 0)
        self.assertIn("    return a / b", hits[0])

    def test_whitespace_normalized_finds(self):
        hits = list(_whitespace_normalized_replacer(self.FILE, "return a + b  # BUG"))
        self.assertGreater(len(hits), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. _apply_replace integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyReplace(unittest.TestCase):

    BASE = textwrap.dedent("""\
        def subtract(a, b):
            return a + b  # BUG: should be a - b

        def divide(a, b):
            return a / b  # BUG: no zero guard

        def factorial(n):
            if n == 0:
                return 0  # BUG: must be 1
            return n * factorial(n - 1)
    """)

    def test_exact_with_indent(self):
        """Exact match preserving indentation."""
        patched, method = _apply_replace(
            self.BASE,
            "    return a + b  # BUG: should be a - b",
            "    return a - b"
        )
        self.assertIn("    return a - b", patched)
        self.assertNotIn("BUG", patched.split("def divide")[0])

    def test_stripped_single_line(self):
        """SLM gives old_str without leading spaces; delta must be applied."""
        patched, method = _apply_replace(
            self.BASE,
            "return a + b  # BUG: should be a - b",
            "return a - b"
        )
        self.assertIn("    return a - b", patched)
        self.assertNotIn("BUG: should be a - b", patched)

    def test_stripped_multiline(self):
        """SLM gives stripped old_str and relative-indented new_str."""
        patched, method = _apply_replace(
            self.BASE,
            "return a / b  # BUG: no zero guard",
            "if b == 0:\n    return None\nreturn a / b"
        )
        self.assertIn("    if b == 0:", patched)
        self.assertIn("        return None", patched)
        self.assertIn("    return a / b", patched)
        ok, _ = _check_python_syntax(patched)
        self.assertTrue(ok, "Patched code must be syntactically valid")

    def test_nested_stripped(self):
        """SLM strips 8-space indent from nested block."""
        patched, method = _apply_replace(
            self.BASE,
            "return 0  # BUG: must be 1",
            "return 1"
        )
        self.assertIn("        return 1", patched)

    def test_not_found_raises_lookup_error(self):
        with self.assertRaises(LookupError):
            _apply_replace(self.BASE, "this does not exist anywhere", "x")

    def test_identical_raises(self):
        """str_replace tool rejects identical old/new."""
        tr = ToolRegistry()
        tmp = write_tmp(self.BASE)
        try:
            r = tr.execute("str_replace", {
                "path": tmp,
                "old_str": "    return a + b  # BUG: should be a - b",
                "new_str": "    return a + b  # BUG: should be a - b",
            })
            self.assertFalse(r.success)
            self.assertIn("identical", r.error.lower())
        finally:
            cleanup(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# 4. str_replace tool (via ToolRegistry)
# ─────────────────────────────────────────────────────────────────────────────

class TestStrReplaceTool(unittest.TestCase):

    def setUp(self):
        self.tr = ToolRegistry()
        self.base = textwrap.dedent("""\
            def subtract(a, b):
                return a + b  # BUG: should be a - b

            def divide(a, b):
                return a / b  # BUG: no zero guard

            def factorial(n):
                if n == 0:
                    return 0  # BUG: must be 1
                return n * factorial(n - 1)
        """)
        self.tmp = write_tmp(self.base)

    def tearDown(self):
        cleanup(self.tmp)

    def test_patch1_exact(self):
        r = self.tr.execute("str_replace", {
            "path": self.tmp,
            "old_str": "    return a + b  # BUG: should be a - b",
            "new_str": "    return a - b",
        })
        self.assertTrue(r.success, r.error)
        content = Path(self.tmp).read_text()
        self.assertIn("    return a - b", content)

    def test_patch1_stripped(self):
        r = self.tr.execute("str_replace", {
            "path": self.tmp,
            "old_str": "return a + b  # BUG: should be a - b",
            "new_str": "return a - b",
        })
        self.assertTrue(r.success, r.error)
        content = Path(self.tmp).read_text()
        self.assertIn("    return a - b", content)

    def test_patch2_multiline_stripped(self):
        r = self.tr.execute("str_replace", {
            "path": self.tmp,
            "old_str": "return a / b  # BUG: no zero guard",
            "new_str": "if b == 0:\n    return None\nreturn a / b",
        })
        self.assertTrue(r.success, r.error)
        content = Path(self.tmp).read_text()
        self.assertIn("    if b == 0:", content)
        self.assertIn("        return None", content)
        self.assertIn("    return a / b", content)
        ok, _ = _check_python_syntax(content)
        self.assertTrue(ok)

    def test_patch3_nested_stripped(self):
        r = self.tr.execute("str_replace", {
            "path": self.tmp,
            "old_str": "return 0  # BUG: must be 1",
            "new_str": "return 1",
        })
        self.assertTrue(r.success, r.error)
        content = Path(self.tmp).read_text()
        self.assertIn("        return 1", content)

    def test_syntax_validation_rejects_bad_patch(self):
        r = self.tr.execute("str_replace", {
            "path": self.tmp,
            "old_str": "return a + b  # BUG: should be a - b",
            "new_str": "if True:\nreturn a - b",  # broken indent
        })
        self.assertFalse(r.success)
        self.assertIn("SyntaxError", r.error)
        # File must be unchanged
        content = Path(self.tmp).read_text()
        self.assertIn("BUG: should be a - b", content)

    def test_not_found_gives_hint(self):
        r = self.tr.execute("str_replace", {
            "path": self.tmp,
            "old_str": "completely nonexistent text xyz",
            "new_str": "replacement",
        })
        self.assertFalse(r.success)
        self.assertIn("not found", r.error.lower())

    def test_file_not_found(self):
        r = self.tr.execute("str_replace", {
            "path": "/nonexistent/path/file.py",
            "old_str": "x",
            "new_str": "y",
        })
        self.assertFalse(r.success)

    def test_all_three_patches_and_run(self):
        """Full workflow: apply all 3 patches, run, check PASSED."""
        demo = textwrap.dedent("""\
            def subtract(a, b):
                return a + b  # BUG: should be a - b

            def divide(a, b):
                return a / b  # BUG: no zero guard

            def factorial(n):
                if n == 0:
                    return 0  # BUG: must be 1
                return n * factorial(n - 1)

            if __name__ == '__main__':
                assert subtract(5,3)==2, f'sub={subtract(5,3)}'
                r = divide(10,0); assert r is None, f'div0={r}'
                assert factorial(5)==120, f'fac={factorial(5)}'
                assert factorial(0)==1, f'fac0={factorial(0)}'
                print('[ALL PASS]')
        """)
        tmp2 = write_tmp(demo)
        try:
            for old, new in [
                ("return a + b  # BUG: should be a - b",
                 "return a - b"),
                ("return a / b  # BUG: no zero guard",
                 "if b == 0:\n    return None\nreturn a / b"),
                ("return 0  # BUG: must be 1",
                 "return 1"),
            ]:
                r = self.tr.execute("str_replace", {"path": tmp2, "old_str": old, "new_str": new})
                self.assertTrue(r.success, f"Patch failed: {r.error}")
            r = self.tr.execute("run_python", {"path": tmp2})
            self.assertIn("[PASSED]", r.output)
            self.assertIn("[ALL PASS]", r.output)
        finally:
            cleanup(tmp2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. run_python tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPython(unittest.TestCase):

    def setUp(self):
        self.tr = ToolRegistry()

    def test_passing_script(self):
        tmp = write_tmp("print('hello')\n")
        try:
            r = self.tr.execute("run_python", {"path": tmp})
            self.assertIn("[PASSED]", r.output)
        finally:
            cleanup(tmp)

    def test_failing_script(self):
        tmp = write_tmp("raise ValueError('oops')\n")
        try:
            r = self.tr.execute("run_python", {"path": tmp})
            self.assertIn("[FAILED]", r.output)
        finally:
            cleanup(tmp)

    def test_syntax_error_detection(self):
        tmp = write_tmp("def f():\nreturn 1\n")
        try:
            r = self.tr.execute("run_python", {"path": tmp})
            self.assertIn("[SYNTAX ERROR]", r.output)
            self.assertFalse(r.success)
        finally:
            cleanup(tmp)

    def test_assertion_error_fails(self):
        tmp = write_tmp("assert 1 == 2, 'bad'\n")
        try:
            r = self.tr.execute("run_python", {"path": tmp})
            self.assertIn("[FAILED]", r.output)
        finally:
            cleanup(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# 6. read_file tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReadFile(unittest.TestCase):

    def setUp(self):
        self.tr = ToolRegistry()
        self.content = "line1\nline2\nline3\nline4\nline5\n"
        self.tmp = write_tmp(self.content, suffix=".txt")

    def tearDown(self):
        cleanup(self.tmp)

    def test_read_all(self):
        r = self.tr.execute("read_file", {"path": self.tmp})
        self.assertTrue(r.success)
        self.assertIn("1: line1", r.output)
        self.assertIn("5: line5", r.output)

    def test_read_range(self):
        r = self.tr.execute("read_file", {"path": self.tmp,
                                           "start_line": 2, "end_line": 4})
        self.assertTrue(r.success)
        self.assertIn("2: line2", r.output)
        self.assertIn("4: line4", r.output)
        self.assertNotIn("1: line1", r.output)

    def test_not_found(self):
        r = self.tr.execute("read_file", {"path": "/no/such/file.txt"})
        self.assertFalse(r.success)


# ─────────────────────────────────────────────────────────────────────────────
# 7. End-to-end code correction with mocked LLM
# ─────────────────────────────────────────────────────────────────────────────

BUGGY_CODE = textwrap.dedent("""\
    def subtract(a, b):
        return a + b  # BUG: should be a - b

    def divide(a, b):
        return a / b  # BUG: no zero guard

    def factorial(n):
        if n == 0:
            return 0  # BUG: must be 1
        return n * factorial(n - 1)

    if __name__ == '__main__':
        assert subtract(5,3)==2, f'sub={subtract(5,3)}'
        r = divide(10,0); assert r is None, f'div0={r}'
        assert factorial(5)==120, f'fac={factorial(5)}'
        assert factorial(0)==1, f'fac0={factorial(0)}'
        print('[ALL PASS]')
""")


class TestCodeCorrectionMocked(unittest.TestCase):
    """
    Simulate _code_correction_execute without a real LLM.
    The 'LLM' response is mocked to return a JSON patch array.
    """

    def _make_mocked_patches(self, filepath: str, style: str = "exact") -> list:
        """Return mock patches. style='exact' has full indent; 'stripped' has none."""
        if style == "exact":
            return [
                {"path": filepath, "old_str": "    return a + b  # BUG: should be a - b", "new_str": "    return a - b"},
                {"path": filepath, "old_str": "    return a / b  # BUG: no zero guard",
                 "new_str": "    if b == 0:\n        return None\n    return a / b"},
                {"path": filepath, "old_str": "        return 0  # BUG: must be 1", "new_str": "        return 1"},
            ]
        else:  # stripped
            return [
                {"path": filepath, "old_str": "return a + b  # BUG: should be a - b", "new_str": "return a - b"},
                {"path": filepath, "old_str": "return a / b  # BUG: no zero guard",
                 "new_str": "if b == 0:\n    return None\nreturn a / b"},
                {"path": filepath, "old_str": "return 0  # BUG: must be 1", "new_str": "return 1"},
            ]

    def _apply_patches(self, filepath: str, patches: list) -> tuple:
        """Apply patches deterministically, return (applied, last_run_output)."""
        tr = ToolRegistry()
        applied = 0
        last_out = ""
        for patch in patches:
            r = tr.execute("str_replace", {
                "path": patch["path"],
                "old_str": patch["old_str"],
                "new_str": patch["new_str"],
            })
            if r.success:
                applied += 1
            else:
                # Try stripped version
                from core.agent_os import AgentOS
                cleaned_old = AgentOS._strip_line_number_prefixes(patch["old_str"])
                if cleaned_old != patch["old_str"]:
                    r2 = tr.execute("str_replace", {
                        "path": patch["path"],
                        "old_str": cleaned_old,
                        "new_str": AgentOS._strip_line_number_prefixes(patch["new_str"]),
                    })
                    if r2.success:
                        applied += 1
            rr = tr.execute("run_python", {"path": filepath, "timeout": 30})
            last_out = rr.output
            if "[PASSED]" in last_out:
                break
        return applied, last_out

    def test_exact_patches(self):
        tmp = write_tmp(BUGGY_CODE)
        try:
            patches = self._make_mocked_patches(tmp, "exact")
            applied, last_out = self._apply_patches(tmp, patches)
            self.assertEqual(applied, 3)
            self.assertIn("[PASSED]", last_out)
            self.assertIn("[ALL PASS]", last_out)
        finally:
            cleanup(tmp)

    def test_stripped_patches(self):
        """SLM outputs stripped old_str — cascade must still fix all 3 bugs."""
        tmp = write_tmp(BUGGY_CODE)
        try:
            patches = self._make_mocked_patches(tmp, "stripped")
            applied, last_out = self._apply_patches(tmp, patches)
            self.assertEqual(applied, 3, f"Expected 3 patches applied, got {applied}")
            self.assertIn("[PASSED]", last_out)
            self.assertIn("[ALL PASS]", last_out)
        finally:
            cleanup(tmp)

    def test_already_passing_detected(self):
        """If code already passes, code_correction_execute should detect it."""
        passing_code = textwrap.dedent("""\
            def subtract(a, b):
                return a - b
            if __name__ == '__main__':
                assert subtract(5,3)==2
                print('[PASS]')
        """)
        tmp = write_tmp(passing_code)
        try:
            tr = ToolRegistry()
            r = tr.execute("run_python", {"path": tmp})
            self.assertIn("[PASSED]", r.output)
        finally:
            cleanup(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Plan parsing (JSON extraction robustness)
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanParsing(unittest.TestCase):
    """Test the JSON extraction logic used in _plan_all_patches."""

    def _parse(self, raw: str) -> list:
        """Mirror the logic in _plan_all_patches."""
        import re
        # Strip think blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL).strip()
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            m2 = re.search(r"\{.*\}", raw, re.DOTALL)
            if m2:
                try:
                    obj = json.loads(m2.group(0))
                    if "old_str" in obj:
                        return [obj]
                except Exception:
                    pass
            return []
        try:
            patches = json.loads(m.group(0))
            return [p for p in patches if isinstance(p, dict)
                    and p.get("old_str") and p.get("new_str")]
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
            cleaned = cleaned.replace("'", '"')
            try:
                patches = json.loads(re.search(r"\[.*\]", cleaned, re.DOTALL).group(0))
                return [p for p in patches if isinstance(p, dict)
                        and p.get("old_str") and p.get("new_str")]
            except Exception:
                return []

    def test_clean_json_array(self):
        raw = '[{"path": "f.py", "old_str": "a", "new_str": "b"}]'
        result = self._parse(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["old_str"], "a")

    def test_json_with_markdown_fences(self):
        raw = '```json\n[{"path": "f.py", "old_str": "a", "new_str": "b"}]\n```'
        result = self._parse(raw)
        self.assertEqual(len(result), 1)

    def test_json_with_think_blocks(self):
        raw = '<think>Let me analyze the bugs.</think>\n[{"path": "f.py", "old_str": "a", "new_str": "b"}]'
        result = self._parse(raw)
        self.assertEqual(len(result), 1)

    def test_single_object_not_array(self):
        raw = '{"path": "f.py", "old_str": "a", "new_str": "b"}'
        result = self._parse(raw)
        self.assertEqual(len(result), 1)

    def test_trailing_comma_json(self):
        raw = '[{"path": "f.py", "old_str": "a", "new_str": "b",}]'
        result = self._parse(raw)
        self.assertEqual(len(result), 1)

    def test_multiple_patches(self):
        raw = json.dumps([
            {"path": "f.py", "old_str": "a", "new_str": "b"},
            {"path": "f.py", "old_str": "c", "new_str": "d"},
        ])
        result = self._parse(raw)
        self.assertEqual(len(result), 2)

    def test_malformed_dropped(self):
        raw = json.dumps([
            {"path": "f.py", "old_str": "a", "new_str": "b"},
            {"path": "f.py"},   # missing old_str/new_str
        ])
        result = self._parse(raw)
        self.assertEqual(len(result), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Line-number prefix stripping
# ─────────────────────────────────────────────────────────────────────────────

class TestLineNumberStripping(unittest.TestCase):

    def setUp(self):
        from core.agent_os import AgentOS
        self.strip = AgentOS._strip_line_number_prefixes

    def test_strips_numeric_prefix(self):
        s = "21:     return a + b  # BUG"
        self.assertEqual(self.strip(s), "    return a + b  # BUG")

    def test_strips_multiline(self):
        s = "21:     return a + b\n22:     # comment"
        result = self.strip(s)
        self.assertIn("    return a + b", result)
        self.assertIn("    # comment", result)

    def test_no_prefix_unchanged(self):
        s = "    return a + b  # comment"
        self.assertEqual(self.strip(s), s)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
