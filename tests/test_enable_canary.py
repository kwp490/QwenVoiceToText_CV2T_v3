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
                         "huggingface_hub must not be in hiddenimports (replaced by urllib)")

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
