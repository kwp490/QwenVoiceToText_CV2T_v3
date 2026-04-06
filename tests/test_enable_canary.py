"""Tests for Enable-Canary.ps1 and installer script validation."""

import ast
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INSTALLER_DIR = _REPO_ROOT / "installer"


class TestEnableCanaryScript(unittest.TestCase):
    """Enable-Canary.ps1 must exist and be syntactically valid PowerShell."""

    def test_script_exists(self):
        script = _INSTALLER_DIR / "Enable-Canary.ps1"
        self.assertTrue(script.is_file(), f"Enable-Canary.ps1 not found at {script}")

    def test_script_not_empty(self):
        script = _INSTALLER_DIR / "Enable-Canary.ps1"
        self.assertGreater(script.stat().st_size, 100)

    @unittest.skipUnless(sys.platform == "win32", "PowerShell syntax check is Windows-only")
    def test_powershell_syntax_valid(self):
        """Verify Enable-Canary.ps1 parses without syntax errors."""
        script = _INSTALLER_DIR / "Enable-Canary.ps1"
        result = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-Command",
                f"$tokens = $null; $errors = $null; "
                f"[System.Management.Automation.Language.Parser]"
                f"::ParseFile('{script}', [ref]$tokens, [ref]$errors); "
                f"if ($errors) {{ $errors | ForEach-Object {{ $_.Message }}; exit 1 }}"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Enable-Canary.ps1 has syntax errors:\n{result.stdout}\n{result.stderr}",
        )

    def test_script_contains_key_sections(self):
        """Script must contain the essential installation steps."""
        content = (_INSTALLER_DIR / "Enable-Canary.ps1").read_text(encoding="utf-8")
        # Must check for uv
        self.assertIn("uv", content)
        # Must check for Python
        self.assertIn("Python", content)
        # Must create canary-env
        self.assertIn("canary-env", content)
        # Must download model
        self.assertIn("canary-qwen-2.5b", content)
        # Must verify torch
        self.assertIn("torch", content)

    def test_script_creates_pyproject_toml(self):
        """Enable-Canary.ps1 must create a pyproject.toml for the canary-env."""
        content = (_INSTALLER_DIR / "Enable-Canary.ps1").read_text(encoding="utf-8")
        self.assertIn("pyproject.toml", content)
        self.assertIn("nemo_toolkit", content)

    def test_script_accepts_appdir_parameter(self):
        """Enable-Canary.ps1 must accept -AppDir to override directory detection.

        PyInstaller 6+ places bundled data files in _internal/, so the
        script's own path is no longer the app root. The settings dialog
        passes -AppDir to provide the correct install directory.
        """
        content = (_INSTALLER_DIR / "Enable-Canary.ps1").read_text(encoding="utf-8")
        self.assertIn(
            "$AppDir",
            content,
            "Script must use $AppDir variable",
        )
        self.assertRegex(
            content,
            r"param\s*\(",
            "Script must have a param() block to accept -AppDir",
        )
        # Extract the full param() block (may span multiple lines)
        param_match = re.search(r"param\s*\((.*?)\)", content, re.DOTALL)
        self.assertIsNotNone(param_match, "Could not find param() block")
        self.assertIn(
            "AppDir",
            param_match.group(1),
            "param() block must include [string]$AppDir parameter",
        )

    def test_script_handles_internal_directory(self):
        """Script must detect _internal/ parent and adjust AppDir accordingly."""
        content = (_INSTALLER_DIR / "Enable-Canary.ps1").read_text(encoding="utf-8")
        self.assertIn(
            "_internal",
            content,
            "Script must handle being located inside _internal/ directory",
        )

    def test_script_handles_installer_directory(self):
        """Script must detect installer/ parent and adjust AppDir accordingly.

        Regression: running Enable-Canary.ps1 from the installer/ directory
        resolved $AppDir to installer/ itself, placing canary-env and models
        under installer/ instead of the app root.
        """
        content = (_INSTALLER_DIR / "Enable-Canary.ps1").read_text(encoding="utf-8")
        # The path adjustment must handle both _internal and installer
        self.assertIn(
            "installer",
            content,
            "Script must handle being located inside installer/ directory",
        )
        # Verify the pattern matches both directories in the same condition
        self.assertRegex(
            content,
            r"'_internal'.*'installer'|'installer'.*'_internal'",
            "Script must check for both _internal/ and installer/ directories",
        )

    def test_canary_env_deps_match_worker_imports(self):
        """Every package imported by canary_worker.py must appear in the
        canary-env pyproject.toml embedded in Enable-Canary.ps1.

        Regression: onnxruntime was imported by NeMo at runtime but was
        not listed in the canary-env dependencies, causing ImportError.
        """
        import ast
        import re

        # ── Collect top-level package names from canary_worker.py ──
        worker_path = _REPO_ROOT / "cv2t" / "engine" / "canary_worker.py"
        source = worker_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        worker_imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    worker_imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                worker_imports.add(node.module.split(".")[0])

        # Remove stdlib modules (not installable)
        stdlib = {
            "__future__", "gc", "json", "logging", "os", "sys",
            "tempfile", "importlib", "types", "warnings",
        }
        worker_imports -= stdlib

        # ── Extract dependency list from Enable-Canary.ps1 ──
        script = (_INSTALLER_DIR / "Enable-Canary.ps1").read_text(encoding="utf-8")
        # Find the dependencies array in the embedded pyproject.toml
        # Use a bracket-counting approach since the deps contain [] (e.g. nemo_toolkit[asr])
        dep_start = script.find('dependencies = [')
        self.assertNotEqual(dep_start, -1, "Could not find dependencies in script")
        bracket_start = script.index('[', dep_start)
        depth = 0
        bracket_end = bracket_start
        for i, ch in enumerate(script[bracket_start:], start=bracket_start):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    bracket_end = i
                    break
        dep_block = script[bracket_start + 1:bracket_end]

        # Normalize: map pip package names to importable names
        pip_to_import = {
            "nemo_toolkit": "nemo",
            "nemo-toolkit": "nemo",
            "huggingface-hub": "huggingface_hub",
            "huggingface_hub": "huggingface_hub",
            "soundfile": "soundfile",
            "onnxruntime": "onnxruntime",
        }
        declared_imports = set()
        for dep_line in dep_block.split(","):
            dep_line = dep_line.strip().strip('"').strip("'")
            # Extract package name (before any version spec or extras)
            pkg = re.split(r'[>=<~!\[\]]+', dep_line)[0].strip()
            if pkg:
                import_name = pip_to_import.get(pkg, pkg)
                declared_imports.add(import_name)

        # ── Check coverage ──
        # These are packages that canary_worker imports (directly or that
        # NeMo imports at runtime) that must be in the dep list
        required_in_deps = {
            "numpy", "torch", "soundfile", "nemo",
            "huggingface_hub", "onnxruntime",
        }
        missing = required_in_deps - declared_imports
        self.assertEqual(
            missing, set(),
            f"canary-env pyproject.toml in Enable-Canary.ps1 is missing "
            f"dependencies required by canary_worker.py: {missing}",
        )

    def test_canary_env_deps_verification_step_exists(self):
        """Enable-Canary.ps1 must verify all dependencies after installation.

        Regression: the script only verified the NeMo SALM import, missing
        onnxruntime and other transitive deps.
        """
        content = (_INSTALLER_DIR / "Enable-Canary.ps1").read_text(encoding="utf-8")
        # Must check onnxruntime specifically (was the missing dep)
        self.assertIn(
            "onnxruntime",
            content,
            "Enable-Canary.ps1 must verify onnxruntime is importable",
        )
        # Must check multiple deps, not just SALM
        for dep in ("numpy", "soundfile", "torch", "huggingface_hub"):
            self.assertIn(
                dep,
                content,
                f"Enable-Canary.ps1 must verify {dep} is importable",
            )


class TestBuildInstallerScript(unittest.TestCase):
    """Build-Installer.ps1 must exist and be valid."""

    def test_script_exists(self):
        script = _INSTALLER_DIR / "Build-Installer.ps1"
        self.assertTrue(script.is_file())

    @unittest.skipUnless(sys.platform == "win32", "PowerShell syntax check is Windows-only")
    def test_powershell_syntax_valid(self):
        script = _INSTALLER_DIR / "Build-Installer.ps1"
        result = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-Command",
                f"$tokens = $null; $errors = $null; "
                f"[System.Management.Automation.Language.Parser]"
                f"::ParseFile('{script}', [ref]$tokens, [ref]$errors); "
                f"if ($errors) {{ $errors | ForEach-Object {{ $_.Message }}; exit 1 }}"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Build-Installer.ps1 has syntax errors:\n{result.stdout}\n{result.stderr}",
        )


class TestInnoSetupScript(unittest.TestCase):
    """cv2t-setup.iss must be valid and reference Enable-Canary.ps1."""

    def test_script_exists(self):
        self.assertTrue((_INSTALLER_DIR / "cv2t-setup.iss").is_file())

    def test_references_enable_canary(self):
        """Inno Setup script must mention the Canary add-on."""
        content = (_INSTALLER_DIR / "cv2t-setup.iss").read_text(encoding="utf-8")
        self.assertIn("Enable-Canary", content)


class TestSpecFile(unittest.TestCase):
    """cv2t.spec must not include huggingface_hub and must bundle new files."""

    def test_no_huggingface_hub_in_hiddenimports(self):
        spec = (_REPO_ROOT / "cv2t.spec").read_text(encoding="utf-8")
        match = re.search(r"hiddenimports\s*=\s*\[(.*?)\]", spec, re.DOTALL)
        self.assertIsNotNone(match)
        hidden = match.group(1)
        self.assertNotIn("huggingface_hub", hidden,
                         "huggingface_hub must not be in hiddenimports "
                         "(auto-collected as transitive dep of faster_whisper)")

    def test_huggingface_hub_not_in_excludes(self):
        """huggingface_hub must not be excluded — faster_whisper imports it at module level."""
        spec = (_REPO_ROOT / "cv2t.spec").read_text(encoding="utf-8")
        match = re.search(r"excludes\s*=\s*\[(.*?)\]", spec, re.DOTALL)
        self.assertIsNotNone(match, "could not find excludes= block in cv2t.spec")
        excludes = match.group(1)
        self.assertNotIn("huggingface_hub", excludes,
                         "huggingface_hub must not be in excludes — "
                         "faster_whisper unconditionally imports it at module level")

    def test_strip_enabled(self):
        spec = (_REPO_ROOT / "cv2t.spec").read_text(encoding="utf-8")
        # EXE and COLLECT should have strip=True
        self.assertIn("strip=True", spec)

    def test_cudnn_in_strip_patterns(self):
        spec = (_REPO_ROOT / "cv2t.spec").read_text(encoding="utf-8")
        self.assertIn("cudnn", spec.lower())

    def test_canary_worker_bundled(self):
        spec = (_REPO_ROOT / "cv2t.spec").read_text(encoding="utf-8")
        self.assertIn("canary_worker.py", spec)

    def test_enable_canary_bundled(self):
        spec = (_REPO_ROOT / "cv2t.spec").read_text(encoding="utf-8")
        self.assertIn("Enable-Canary.ps1", spec)

    def test_nvidia_cudnn_excluded(self):
        """nvidia.cudnn must NOT be in the collected packages list."""
        spec = (_REPO_ROOT / "cv2t.spec").read_text(encoding="utf-8")
        # Check that nvidia.cudnn is NOT in the for-loop of collected packages
        collected_match = re.search(
            r"for _nvidia_pkg in \((.+?)\)", spec, re.DOTALL
        )
        self.assertIsNotNone(collected_match)
        collected_pkgs = collected_match.group(1)
        self.assertNotIn("nvidia.cudnn", collected_pkgs,
                         "nvidia.cudnn should not be in the collected packages list")


class TestNoHuggingfaceHubInBinary(unittest.TestCase):
    """Verify that huggingface_hub is not imported at module level in binary-critical paths."""

    def test_main_no_toplevel_hf_import(self):
        """__main__.py must not import huggingface_hub at module level."""
        source = (_REPO_ROOT / "cv2t" / "__main__.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        hf_imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "huggingface" in alias.name:
                        hf_imports.append(f"line {node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and "huggingface" in node.module:
                    hf_imports.append(f"line {node.lineno}: from {node.module}")

        self.assertEqual(
            hf_imports, [],
            "huggingface_hub must not be imported at module level in __main__.py:\n"
            + "\n".join(hf_imports),
        )

    def test_whisper_no_toplevel_hf_import(self):
        """whisper.py must not import huggingface_hub at module level."""
        source = (_REPO_ROOT / "cv2t" / "engine" / "whisper.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        hf_imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "huggingface" in alias.name:
                        hf_imports.append(f"line {node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and "huggingface" in node.module:
                    hf_imports.append(f"line {node.lineno}: from {node.module}")

        self.assertEqual(
            hf_imports, [],
            "huggingface_hub must not be imported at module level in whisper.py:\n"
            + "\n".join(hf_imports),
        )


if __name__ == "__main__":
    unittest.main()
