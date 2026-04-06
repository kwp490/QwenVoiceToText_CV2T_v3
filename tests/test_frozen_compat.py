"""Tests for PyInstaller frozen-build compatibility.

These tests catch issues that only manifest in --noconsole PyInstaller builds:
- Relative imports in __main__.py (no parent package context)
- APIs that assume real file descriptors (faulthandler, fileno)
- Modules that must be importable via absolute paths
- Dynamic imports must be listed in cv2t.spec hiddenimports
"""

import ast
import io
import os
import re
import sys
import unittest
from pathlib import Path

# Root of the cv2t package
_CV2T_PKG = Path(__file__).resolve().parent.parent / "cv2t"
_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestNoRelativeImportsInMain(unittest.TestCase):
    """__main__.py must use absolute imports for PyInstaller compatibility.

    When PyInstaller runs __main__.py as the entry point, there is no parent
    package context, so ``from .config import ...`` raises ImportError.
    """

    def test_no_relative_imports(self):
        source = (_CV2T_PKG / "__main__.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="__main__.py")

        relative_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                relative_imports.append(
                    f"line {node.lineno}: from {'.' * node.level}{node.module or ''} import ..."
                )

        self.assertEqual(
            relative_imports,
            [],
            f"__main__.py must not use relative imports (breaks PyInstaller):\n"
            + "\n".join(relative_imports),
        )


class TestFaulthandlerWithStringIO(unittest.TestCase):
    """faulthandler.enable() must be guarded for --noconsole builds.

    PyInstaller --noconsole sets sys.stderr to None (replaced with StringIO
    in __main__.py). faulthandler.enable() calls stderr.fileno() which raises
    io.UnsupportedOperation on StringIO objects.
    """

    def test_faulthandler_tolerates_stringio_stderr(self):
        import faulthandler

        original_stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            # This is the pattern used in __main__.py — must not raise
            try:
                faulthandler.enable()
            except io.UnsupportedOperation:
                pass  # Expected in --noconsole builds; the guard works
        finally:
            sys.stderr = original_stderr

    def test_main_guards_faulthandler(self):
        """Verify __main__.py wraps faulthandler.enable() in try/except."""
        source = (_CV2T_PKG / "__main__.py").read_text(encoding="utf-8")
        self.assertIn("io.UnsupportedOperation", source,
                       "faulthandler.enable() must be guarded with "
                       "except io.UnsupportedOperation")


class TestStdioSafetyPatches(unittest.TestCase):
    """__main__.py must patch None stdout/stderr for --noconsole builds."""

    def test_stdout_none_guard_exists(self):
        source = (_CV2T_PKG / "__main__.py").read_text(encoding="utf-8")
        self.assertIn("sys.stdout is None", source,
                       "__main__.py must guard against sys.stdout being None")

    def test_stderr_none_guard_exists(self):
        source = (_CV2T_PKG / "__main__.py").read_text(encoding="utf-8")
        self.assertIn("sys.stderr is None", source,
                       "__main__.py must guard against sys.stderr being None")


class TestAllModulesImportable(unittest.TestCase):
    """Every .py file in cv2t/ must be importable via absolute paths.

    This catches missing dependencies, syntax errors, and circular imports
    that would crash the frozen build at startup.
    """

    _SKIP_MODULES = frozenset({
        # These need GPU/hardware or heavy dependencies at import time
        "cv2t.engine.canary",
        # Standalone worker script — has torch/NeMo imports at function level
        "cv2t.engine.canary_worker",
    })

    def test_import_all_modules(self):
        failures = []
        for py_file in sorted(_CV2T_PKG.rglob("*.py")):
            rel = py_file.relative_to(_CV2T_PKG.parent)
            module_name = str(rel.with_suffix("")).replace("\\", ".").replace("/", ".")

            if module_name in self._SKIP_MODULES:
                continue
            # Skip __pycache__
            if "__pycache__" in module_name:
                continue

            try:
                __import__(module_name)
            except Exception as exc:
                failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

        self.assertEqual(
            failures,
            [],
            f"Failed to import the following modules:\n" + "\n".join(failures),
        )


class TestRelativeImportsInSubpackages(unittest.TestCase):
    """Relative imports within sub-packages (e.g., cv2t.engine) are fine
    because PyInstaller preserves the package structure for non-entry-point
    modules. This test ensures those imports actually resolve correctly.
    """

    def test_engine_subpackage_imports(self):
        """The engine __init__ must be importable (it uses relative imports internally)."""
        from cv2t.engine import ENGINES
        self.assertIsInstance(ENGINES, dict)

    def test_engine_whisper_imports(self):
        from cv2t.engine.whisper import WhisperEngine
        self.assertTrue(callable(WhisperEngine))

    def test_engine_base_imports(self):
        from cv2t.engine.base import SpeechEngine
        self.assertTrue(callable(SpeechEngine))


class TestHiddenImportsInSpec(unittest.TestCase):
    """Dynamic imports in __main__.py must be listed in cv2t.spec hiddenimports.

    PyInstaller only detects static top-level imports. Any module imported
    inside a function (deferred/dynamic import) must be explicitly listed in
    the spec's hiddenimports or it will be missing from the frozen binary.
    """

    # Imports within the cv2t package itself — PyInstaller bundles these via
    # the Analysis scripts entry, so they don't need hiddenimports entries.
    _INTERNAL_PREFIXES = ("cv2t.",)

    # Standard library modules — always available, never need hiddenimports.
    _STDLIB = frozenset({
        "argparse", "ctypes", "faulthandler", "io", "json", "logging",
        "logging.handlers", "os", "sys", "re", "pathlib", "tempfile",
        "time", "subprocess", "unittest", "importlib", "threading",
        "collections", "functools", "typing", "traceback", "copy",
        "shutil", "signal", "struct", "abc", "dataclasses", "enum",
    })

    def _parse_hidden_imports(self) -> set[str]:
        """Extract the hiddenimports list from cv2t.spec."""
        spec_path = _REPO_ROOT / "cv2t.spec"
        spec_text = spec_path.read_text(encoding="utf-8")
        # Match the hiddenimports=[...] block
        match = re.search(
            r"hiddenimports\s*=\s*\[(.*?)\]", spec_text, re.DOTALL
        )
        self.assertIsNotNone(match, "Could not find hiddenimports in cv2t.spec")
        entries = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
        return set(entries)

    def _collect_deferred_imports(self, filepath: Path) -> list[tuple[int, str]]:
        """Find all imports inside functions in the given file.

        Returns list of (line_number, top_level_module_name).
        """
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filepath.name)

        deferred: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        deferred.append((child.lineno, alias.name.split(".")[0]))
                elif isinstance(child, ast.ImportFrom):
                    if child.level == 0 and child.module:
                        deferred.append((child.lineno, child.module.split(".")[0]))
        return deferred

    def test_dynamic_imports_in_hiddenimports(self):
        """Every third-party dynamic import in __main__.py must be in hiddenimports."""
        hidden = self._parse_hidden_imports()
        deferred = self._collect_deferred_imports(_CV2T_PKG / "__main__.py")

        missing = []
        for lineno, top_module in deferred:
            if top_module in self._STDLIB:
                continue
            if any(top_module.startswith(p.rstrip(".")) for p in self._INTERNAL_PREFIXES):
                continue
            # Check if the module (or a parent) is in hiddenimports
            if not any(h == top_module or h.startswith(top_module + ".") for h in hidden):
                missing.append(f"line {lineno}: {top_module}")

        self.assertEqual(
            missing,
            [],
            "Dynamic imports in __main__.py not listed in cv2t.spec hiddenimports:\n"
            + "\n".join(missing)
            + "\nAdd them to hiddenimports in cv2t.spec.",
        )

    def test_dynamic_imports_in_main_window(self):
        """Third-party dynamic imports in main_window.py must be in hiddenimports."""
        hidden = self._parse_hidden_imports()
        deferred = self._collect_deferred_imports(_CV2T_PKG / "main_window.py")

        missing = []
        for lineno, top_module in deferred:
            if top_module in self._STDLIB:
                continue
            if any(top_module.startswith(p.rstrip(".")) for p in self._INTERNAL_PREFIXES):
                continue
            if not any(h == top_module or h.startswith(top_module + ".") for h in hidden):
                missing.append(f"line {lineno}: {top_module}")

        self.assertEqual(
            missing,
            [],
            "Dynamic imports in main_window.py not listed in cv2t.spec hiddenimports:\n"
            + "\n".join(missing)
            + "\nAdd them to hiddenimports in cv2t.spec.",
        )


class TestTransitiveDependenciesInSpec(unittest.TestCase):
    """Transitive dependencies used at runtime must be bundled in the spec.

    Some libraries (e.g. faster-whisper) dynamically import optional packages
    internally. These won't be caught by AST-scanning CV2T's own source, but
    they must still be present in the frozen build.
    """

    def _read_spec(self) -> str:
        return (_REPO_ROOT / "cv2t.spec").read_text(encoding="utf-8")

    def _parse_hidden_imports(self) -> set[str]:
        spec_text = self._read_spec()
        match = re.search(
            r"hiddenimports\s*=\s*\[(.*?)\]", spec_text, re.DOTALL
        )
        assert match, "Could not find hiddenimports in cv2t.spec"
        return set(re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)))

    def _parse_excludes(self) -> set[str]:
        spec_text = self._read_spec()
        match = re.search(
            r"excludes\s*=\s*\[(.*?)\]", spec_text, re.DOTALL
        )
        assert match, "Could not find excludes in cv2t.spec"
        return set(re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)))

    def test_onnxruntime_in_hiddenimports(self):
        """onnxruntime must be in hiddenimports — faster-whisper's Silero VAD needs it."""
        hidden = self._parse_hidden_imports()
        self.assertIn("onnxruntime", hidden,
                       "onnxruntime must be in hiddenimports for faster-whisper VAD")

    def test_onnxruntime_not_excluded(self):
        """onnxruntime must NOT be in the excludes list."""
        excludes = self._parse_excludes()
        self.assertNotIn("onnxruntime", excludes,
                          "onnxruntime must not be excluded — faster-whisper VAD requires it")

    def test_onnxruntime_not_stripped(self):
        """onnxruntime must not be matched by any _STRIP_PATTERNS regex."""
        spec_text = self._read_spec()
        # Extract patterns from _STRIP_PATTERNS
        patterns = re.findall(r"_re\.compile\(r'([^']+)'", spec_text)
        for pattern in patterns:
            self.assertIsNone(
                re.search(pattern, "onnxruntime", re.IGNORECASE),
                f"_STRIP_PATTERNS regex r'{pattern}' matches 'onnxruntime' — "
                "this would strip it from the build (needed for VAD)",
            )

    def test_faster_whisper_data_files_collected(self):
        """cv2t.spec must collect faster_whisper data files (Silero VAD ONNX model).

        faster-whisper ships a Silero VAD ONNX model in its assets/ directory.
        PyInstaller only bundles .py files automatically — data files like
        .onnx must be explicitly collected via collect_data_files or datas.
        """
        spec_text = self._read_spec()
        self.assertIn(
            "collect_data_files('faster_whisper')",
            spec_text,
            "cv2t.spec must call collect_data_files('faster_whisper') to bundle "
            "the Silero VAD ONNX model (faster_whisper/assets/silero_vad_v6.onnx)",
        )

    def test_silero_vad_onnx_exists_in_package(self):
        """The Silero VAD ONNX model must exist in the installed faster_whisper package.

        If the file is missing from the installed package, the frozen build
        will also miss it — even with collect_data_files.
        """
        import faster_whisper
        pkg_dir = Path(faster_whisper.__file__).parent
        vad_model = pkg_dir / "assets" / "silero_vad_v6.onnx"
        self.assertTrue(
            vad_model.is_file(),
            f"Silero VAD model not found at {vad_model} — "
            "faster-whisper VAD will fail at runtime",
        )


class TestBundledDataFilesInDist(unittest.TestCase):
    """Data files listed in cv2t.spec datas must land in the build output.

    PyInstaller 6+ places data files in _internal/ (not beside the exe).
    Both the spec and the Python lookup code must agree on this layout.
    """

    _DIST_DIR = _REPO_ROOT / "dist" / "cv2t"
    _INTERNAL_DIR = _DIST_DIR / "_internal"

    @unittest.skipUnless(
        (_REPO_ROOT / "dist" / "cv2t").is_dir(),
        "dist/cv2t/ not found — run pyinstaller cv2t.spec first",
    )
    def test_enable_canary_in_build(self):
        """Enable-Canary.ps1 must exist in the build output.

        The settings dialog searches app_dir (exe directory) first, then
        sys._MEIPASS (_internal/). At least one must contain the file.
        """
        found = (
            (self._DIST_DIR / "Enable-Canary.ps1").is_file()
            or (self._INTERNAL_DIR / "Enable-Canary.ps1").is_file()
        )
        self.assertTrue(
            found,
            "Enable-Canary.ps1 missing from dist/cv2t/ AND "
            "dist/cv2t/_internal/. Check cv2t.spec datas list.",
        )

    @unittest.skipUnless(
        (_REPO_ROOT / "dist" / "cv2t").is_dir(),
        "dist/cv2t/ not found — run pyinstaller cv2t.spec first",
    )
    def test_canary_worker_in_build(self):
        """canary_worker.py must exist in the build output."""
        found = (
            (self._DIST_DIR / "canary_worker.py").is_file()
            or (self._INTERNAL_DIR / "canary_worker.py").is_file()
        )
        self.assertTrue(
            found,
            "canary_worker.py missing from build output. "
            "Check cv2t.spec datas list.",
        )


class TestSettingsDialogScriptDiscovery(unittest.TestCase):
    """_on_install_canary must find Enable-Canary.ps1 in all layouts.

    Layouts:
    - Dev/source:  repo_root/installer/Enable-Canary.ps1
    - Frozen (PyInstaller 6+): _internal/Enable-Canary.ps1  (sys._MEIPASS)
    - Frozen (legacy):  app_dir/Enable-Canary.ps1
    """

    def _find_script(self, *, frozen: bool, app_dir: str,
                     meipass: str | None = None) -> str | None:
        """Replicate the lookup logic from _on_install_canary."""
        if frozen:
            data_dir = meipass if meipass else app_dir
        else:
            data_dir = app_dir

        path = os.path.join(app_dir, "Enable-Canary.ps1")
        if os.path.isfile(path):
            return path
        path = os.path.join(data_dir, "Enable-Canary.ps1")
        if os.path.isfile(path):
            return path
        path = os.path.join(app_dir, "installer", "Enable-Canary.ps1")
        if os.path.isfile(path):
            return path
        return None

    def test_dev_layout(self):
        """In dev mode, the script is found via installer/ subdirectory."""
        result = self._find_script(frozen=False, app_dir=str(_REPO_ROOT))
        self.assertIsNotNone(result, "Script not found in dev layout")
        self.assertTrue(result.endswith("Enable-Canary.ps1"))

    def test_frozen_pyinstaller6_layout(self):
        """Simulates PyInstaller 6+ where data files are in _internal/."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            internal = os.path.join(tmpdir, "_internal")
            os.makedirs(internal)
            script = os.path.join(internal, "Enable-Canary.ps1")
            Path(script).write_text("# stub", encoding="utf-8")

            result = self._find_script(
                frozen=True, app_dir=tmpdir, meipass=internal,
            )
            self.assertIsNotNone(
                result,
                "Script not found with _internal/ layout (PyInstaller 6+)",
            )
            self.assertEqual(result, script)

    def test_frozen_legacy_layout(self):
        """Simulates pre-PyInstaller-6 where data files sit beside the exe."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "Enable-Canary.ps1")
            Path(script).write_text("# stub", encoding="utf-8")

            result = self._find_script(frozen=True, app_dir=tmpdir)
            self.assertIsNotNone(
                result,
                "Script not found with legacy (beside-exe) layout",
            )
            self.assertEqual(result, script)

    def test_missing_everywhere_returns_none(self):
        """When the script doesn't exist anywhere, None is returned."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._find_script(frozen=True, app_dir=tmpdir)
            self.assertIsNone(result)


class TestCanaryLaunchCommandQuoting(unittest.TestCase):
    """The PowerShell command built by _on_install_canary must quote paths.

    Paths like 'C:\\Program Files\\CV2T\\...' contain spaces. The command
    passes through two PowerShell layers (outer → Start-Process → inner),
    so each path token must be individually quoted to survive both stages.
    """

    @staticmethod
    def _build_commands(script_path: str, app_dir: str) -> tuple[str, str]:
        """Replicate the quoting logic from _on_install_canary.

        Returns (inner_cmd, outer_cmd).
        """
        esc_script = script_path.replace("'", "''")
        esc_app = app_dir.replace("'", "''")
        inner_cmd = (
            "Set-ExecutionPolicy Bypass -Scope Process -Force; "
            f"& '{esc_script}' -AppDir '{esc_app}'"
        )
        outer_cmd = (
            "Start-Process powershell.exe "
            "-ArgumentList @("
            "'-NoExit',"
            "'-Command',"
            f"'{inner_cmd.replace(chr(39), chr(39)*2)}'"
            ") -Verb RunAs"
        )
        return inner_cmd, outer_cmd

    def test_paths_with_spaces_are_quoted(self):
        """Paths with spaces must be individually single-quoted."""
        inner, _ = self._build_commands(
            r"C:\Program Files\CV2T\_internal\Enable-Canary.ps1",
            r"C:\Program Files\CV2T",
        )
        self.assertIn(
            "& 'C:\\Program Files\\CV2T\\_internal\\Enable-Canary.ps1'",
            inner,
        )
        self.assertIn("-AppDir 'C:\\Program Files\\CV2T'", inner)

    def test_outer_cmd_doubles_single_quotes(self):
        """The outer command must double single-quotes for Start-Process."""
        _, outer = self._build_commands(
            r"C:\Program Files\CV2T\_internal\Enable-Canary.ps1",
            r"C:\Program Files\CV2T",
        )
        # Inner single quotes become '' in the outer layer
        self.assertIn("''C:\\Program Files\\CV2T\\_internal\\Enable-Canary.ps1''", outer)
        self.assertIn("-AppDir ''C:\\Program Files\\CV2T''", outer)

    def test_paths_without_spaces_still_quoted(self):
        """Even paths without spaces must be quoted for consistency."""
        inner, _ = self._build_commands(
            r"C:\CV2T\_internal\Enable-Canary.ps1",
            r"C:\CV2T",
        )
        self.assertIn("& 'C:\\CV2T\\_internal\\Enable-Canary.ps1'", inner)
        self.assertIn("-AppDir 'C:\\CV2T'", inner)

    def test_source_uses_list_argv(self):
        """subprocess.Popen must use a list, not a shell string."""
        src = (_CV2T_PKG / "settings_dialog.py").read_text(encoding="utf-8")
        # Must NOT have shell=True
        self.assertNotIn(
            "shell=True",
            src,
            "subprocess calls must use list argv, not shell=True",
        )

    def test_source_quotes_esc_script_in_fstring(self):
        """The f-string building inner_cmd must wrap {esc_script} in quotes.

        A previous bug had f'...'' {var}''...' where '' ended the f-string
        literal, causing {var} to be treated as a Python expression outside
        the string rather than an f-string interpolation.
        """
        src = (_CV2T_PKG / "settings_dialog.py").read_text(encoding="utf-8")
        # Must contain the proper pattern: f"& '{esc_script}' -AppDir ..."
        self.assertIn("'{esc_script}'", src)
        self.assertIn("'{esc_app}'", src)


if __name__ == "__main__":
    unittest.main()
