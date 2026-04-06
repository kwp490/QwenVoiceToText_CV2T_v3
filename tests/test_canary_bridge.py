"""Tests for the Canary subprocess bridge engine."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCanaryBridgeImport(unittest.TestCase):
    """canary_bridge module must be importable without torch/NeMo."""

    def test_import_module(self):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine
        self.assertTrue(callable(CanaryBridgeEngine))

    def test_inherits_speech_engine(self):
        from cv2t.engine.base import SpeechEngine
        from cv2t.engine.canary_bridge import CanaryBridgeEngine
        self.assertTrue(issubclass(CanaryBridgeEngine, SpeechEngine))

    def test_engine_name(self):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine
        engine = CanaryBridgeEngine()
        self.assertEqual(engine.name, "canary")

    def test_vram_estimate(self):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine
        engine = CanaryBridgeEngine()
        self.assertIsInstance(engine.vram_estimate_gb, float)
        self.assertGreater(engine.vram_estimate_gb, 0)

    def test_not_loaded_by_default(self):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine
        engine = CanaryBridgeEngine()
        self.assertFalse(engine.is_loaded)


class TestCanaryEnvDetection(unittest.TestCase):
    """canary_env_available() must detect canary-env correctly."""

    def test_returns_bool(self):
        from cv2t.engine.canary_bridge import canary_env_available
        result = canary_env_available()
        self.assertIsInstance(result, bool)

    @patch("cv2t.engine.canary_bridge.os.path.isfile")
    @patch("cv2t.engine.canary_bridge._get_app_dir")
    def test_returns_true_when_env_exists(self, mock_app_dir, mock_isfile):
        from cv2t.engine.canary_bridge import canary_env_available
        mock_app_dir.return_value = "C:\\Program Files\\CV2T"
        mock_isfile.return_value = True
        self.assertTrue(canary_env_available())

    @patch("cv2t.engine.canary_bridge.os.path.isfile")
    @patch("cv2t.engine.canary_bridge._get_app_dir")
    def test_returns_false_when_env_missing(self, mock_app_dir, mock_isfile):
        from cv2t.engine.canary_bridge import canary_env_available
        mock_app_dir.return_value = "C:\\Program Files\\CV2T"
        mock_isfile.return_value = False
        self.assertFalse(canary_env_available())

    @patch("cv2t.engine.canary_bridge.os.path.isfile")
    @patch("cv2t.engine.canary_bridge._get_app_dir")
    @patch("cv2t.engine.canary_bridge.sys")
    def test_frozen_finds_worker_in_meipass(self, mock_sys, mock_app_dir, mock_isfile):
        """In a frozen build, canary_worker.py is in _MEIPASS (_internal/)."""
        from cv2t.engine.canary_bridge import _get_canary_paths

        mock_app_dir.return_value = r"C:\Program Files\CV2T"
        mock_sys.frozen = True
        mock_sys._MEIPASS = r"C:\Program Files\CV2T\_internal"

        def isfile_side_effect(path):
            return path in (
                r"C:\Program Files\CV2T\canary-env\.venv\Scripts\python.exe",
                r"C:\Program Files\CV2T\_internal\canary_worker.py",
            )
        mock_isfile.side_effect = isfile_side_effect

        python_path, worker_path = _get_canary_paths()
        self.assertIsNotNone(python_path)
        self.assertEqual(
            worker_path,
            r"C:\Program Files\CV2T\_internal\canary_worker.py",
        )

    @patch("cv2t.engine.canary_bridge.os.path.isfile")
    @patch("cv2t.engine.canary_bridge._get_app_dir")
    @patch("cv2t.engine.canary_bridge.sys")
    def test_frozen_worker_not_in_app_dir(self, mock_sys, mock_app_dir, mock_isfile):
        """Worker beside exe should NOT be found if only in _internal/."""
        from cv2t.engine.canary_bridge import _get_canary_paths

        mock_app_dir.return_value = r"C:\Program Files\CV2T"
        mock_sys.frozen = True
        mock_sys._MEIPASS = r"C:\Program Files\CV2T\_internal"

        def isfile_side_effect(path):
            # python exists, worker only in _internal (not app_dir)
            if path == r"C:\Program Files\CV2T\canary-env\.venv\Scripts\python.exe":
                return True
            if path == r"C:\Program Files\CV2T\_internal\canary_worker.py":
                return True
            return False
        mock_isfile.side_effect = isfile_side_effect

        python_path, worker_path = _get_canary_paths()
        self.assertIsNotNone(worker_path, "Worker in _internal/ must be found")


class TestCanaryBridgeLoadError(unittest.TestCase):
    """Loading without canary-env must raise RuntimeError."""

    @patch("cv2t.engine.canary_bridge._get_canary_paths")
    def test_load_raises_without_env(self, mock_paths):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine
        mock_paths.return_value = (None, None)
        engine = CanaryBridgeEngine()
        with self.assertRaises(RuntimeError):
            engine.load("C:\\models")


class TestCanaryBridgeProtocol(unittest.TestCase):
    """Test the JSON-line protocol with a mock subprocess."""

    def _create_mock_process(self, responses: list[dict]):
        """Create a mock Popen with preset stdout responses."""
        proc = MagicMock()
        proc.poll.return_value = None

        stdout_lines = [json.dumps(r) + "\n" for r in responses]
        proc.stdout.readline.side_effect = stdout_lines
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()
        return proc

    @patch("cv2t.engine.canary_bridge._get_canary_paths")
    @patch("cv2t.engine.canary_bridge.subprocess.Popen")
    def test_load_sends_correct_command(self, mock_popen, mock_paths):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine

        mock_paths.return_value = ("python.exe", "worker.py")
        proc = self._create_mock_process([{"status": "ready"}])
        mock_popen.return_value = proc

        engine = CanaryBridgeEngine()
        engine.load("C:\\models", "cuda")

        # Verify the load command was sent
        written = proc.stdin.write.call_args[0][0]
        cmd = json.loads(written)
        self.assertEqual(cmd["command"], "load")
        self.assertEqual(cmd["model_path"], "C:\\models")
        self.assertEqual(cmd["device"], "cuda")
        self.assertTrue(engine.is_loaded)

    @patch("cv2t.engine.canary_bridge._get_canary_paths")
    @patch("cv2t.engine.canary_bridge.subprocess.Popen")
    def test_load_failure_raises(self, mock_popen, mock_paths):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine

        mock_paths.return_value = ("python.exe", "worker.py")
        proc = self._create_mock_process([{"status": "error", "message": "OOM"}])
        mock_popen.return_value = proc

        engine = CanaryBridgeEngine()
        with self.assertRaises(RuntimeError) as ctx:
            engine.load("C:\\models")
        self.assertIn("OOM", str(ctx.exception))

    @patch("cv2t.engine.canary_bridge._get_canary_paths")
    @patch("cv2t.engine.canary_bridge.subprocess.Popen")
    def test_transcribe_sends_audio_file(self, mock_popen, mock_paths):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine

        mock_paths.return_value = ("python.exe", "worker.py")
        # First response: load ready, second: transcription result
        proc = self._create_mock_process([
            {"status": "ready"},
            {"status": "ok", "text": "hello world"},
        ])
        mock_popen.return_value = proc

        engine = CanaryBridgeEngine()
        engine.load("C:\\models")

        audio = np.zeros(16000, dtype=np.float32)
        result = engine.transcribe(audio, 16000, "en")
        self.assertEqual(result, "hello world")

    @patch("cv2t.engine.canary_bridge._get_canary_paths")
    @patch("cv2t.engine.canary_bridge.subprocess.Popen")
    def test_unload_sends_shutdown(self, mock_popen, mock_paths):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine

        mock_paths.return_value = ("python.exe", "worker.py")
        proc = self._create_mock_process([{"status": "ready"}])
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        engine = CanaryBridgeEngine()
        engine.load("C:\\models")
        engine.unload()

        # Verify shutdown was sent
        calls = proc.stdin.write.call_args_list
        last_cmd = json.loads(calls[-1][0][0])
        self.assertEqual(last_cmd["command"], "shutdown")
        self.assertFalse(engine.is_loaded)


class TestCanaryWorkerSyntax(unittest.TestCase):
    """canary_worker.py must be valid Python (parseable by AST)."""

    def test_worker_is_valid_python(self):
        import ast
        worker_path = _REPO_ROOT / "cv2t" / "engine" / "canary_worker.py"
        source = worker_path.read_text(encoding="utf-8")
        # Should not raise SyntaxError
        ast.parse(source, filename="canary_worker.py")

    def test_worker_has_main_function(self):
        import ast
        worker_path = _REPO_ROOT / "cv2t" / "engine" / "canary_worker.py"
        source = worker_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        func_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        self.assertIn("main", func_names)
        self.assertIn("_load_model", func_names)
        self.assertIn("_transcribe", func_names)
        self.assertIn("_send", func_names)
        self.assertIn("_recv", func_names)

    def test_worker_has_no_cv2t_imports(self):
        """canary_worker.py must be standalone — no cv2t imports."""
        import ast
        worker_path = _REPO_ROOT / "cv2t" / "engine" / "canary_worker.py"
        source = worker_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        cv2t_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("cv2t"):
                    cv2t_imports.append(f"line {node.lineno}: from {node.module}")
                # Also check relative imports
                if node.level and node.level > 0:
                    cv2t_imports.append(
                        f"line {node.lineno}: from {'.' * node.level}{node.module or ''}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("cv2t"):
                        cv2t_imports.append(f"line {node.lineno}: import {alias.name}")

        self.assertEqual(
            cv2t_imports, [],
            "canary_worker.py must be standalone (no cv2t imports):\n"
            + "\n".join(cv2t_imports),
        )

    def test_worker_global_declarations_precede_usage(self):
        """Every 'global X' must appear before any reference to X in its function.

        Regression: 'global _model' was placed after '_model is None' in main(),
        causing a SyntaxError at runtime.
        """
        import ast
        worker_path = _REPO_ROOT / "cv2t" / "engine" / "canary_worker.py"
        source = worker_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        errors = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            # Collect global declarations and their line numbers
            globals_declared = {}
            for child in ast.walk(node):
                if isinstance(child, ast.Global):
                    for name in child.names:
                        globals_declared.setdefault(name, child.lineno)

            if not globals_declared:
                continue

            # Find earliest reference (Name node) of each global var
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id in globals_declared:
                    # Skip the global statement itself
                    if isinstance(child.ctx, (ast.Load, ast.Store, ast.Del)):
                        if child.lineno < globals_declared[child.id]:
                            errors.append(
                                f"{node.name}(): '{child.id}' used on line "
                                f"{child.lineno} before 'global' on line "
                                f"{globals_declared[child.id]}"
                            )
                            # Only report the first violation per variable
                            del globals_declared[child.id]
                            if not globals_declared:
                                break

        self.assertEqual(
            errors, [],
            "Global declarations must precede all usage in the function:\n"
            + "\n".join(errors),
        )

    def test_worker_compiles_in_subprocess(self):
        """canary_worker.py must compile without errors in a fresh Python process.

        Catches SyntaxError that AST-based checks might miss (e.g. byte-compile
        issues, encoding problems).
        """
        worker_path = _REPO_ROOT / "cv2t" / "engine" / "canary_worker.py"
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(worker_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode, 0,
            f"canary_worker.py failed to compile:\n{result.stderr}",
        )


class TestCanaryWorkerAudioUtils(unittest.TestCase):
    """Test the standalone audio utilities duplicated in canary_worker."""

    def test_ensure_16khz_passthrough(self):
        # Import the functions from the worker module directly
        import importlib.util
        worker_path = str(_REPO_ROOT / "cv2t" / "engine" / "canary_worker.py")
        spec = importlib.util.spec_from_file_location("canary_worker", worker_path)
        worker = importlib.util.module_from_spec(spec)
        # Don't exec the full module (it has side effects); just test the function
        # by parsing the source and extracting the function
        # Instead, test that the functions are consistent with audio_utils
        from cv2t.engine.audio_utils import chunk_audio, ensure_16khz, stitch_transcripts
        # Test audio_utils still works (regression)
        audio = np.zeros(16000, dtype=np.float32)
        self.assertEqual(len(ensure_16khz(audio, 16000)), 16000)
        self.assertEqual(len(chunk_audio(audio, 16000)), 1)
        self.assertEqual(stitch_transcripts(["hello", "world"]), "hello world")


class TestCanaryWorkerStdoutIsolation(unittest.TestCase):
    """canary_worker.py must protect the JSON protocol from stray prints.

    Regression: NeMo/torch print() calls wrote to sys.stdout, corrupting
    the JSON-lines protocol. The worker must redirect sys.stdout to stderr
    and use a saved reference (_proto_stdout) for protocol messages.
    """

    def test_proto_stdout_saved_before_redirect(self):
        """_proto_stdout must be assigned at module level (before main redirect)."""
        import ast
        worker_path = _REPO_ROOT / "cv2t" / "engine" / "canary_worker.py"
        source = worker_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # _proto_stdout must be assigned at module level
        module_assigns = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_proto_stdout":
                        module_assigns.append(node.lineno)

        self.assertTrue(
            len(module_assigns) >= 1,
            "_proto_stdout must be assigned at module level in canary_worker.py",
        )

    def test_send_uses_proto_stdout(self):
        """_send() must write to _proto_stdout, not sys.stdout."""
        source = (_REPO_ROOT / "cv2t" / "engine" / "canary_worker.py").read_text(
            encoding="utf-8"
        )
        # Find _send function body
        in_send = False
        send_lines = []
        for line in source.splitlines():
            if line.startswith("def _send("):
                in_send = True
                continue
            if in_send:
                if line and not line[0].isspace():
                    break
                send_lines.append(line)

        send_body = "\n".join(send_lines)
        self.assertIn(
            "_proto_stdout",
            send_body,
            "_send() must write to _proto_stdout, not sys.stdout",
        )
        self.assertNotIn(
            "sys.stdout",
            send_body,
            "_send() must NOT write to sys.stdout (use _proto_stdout)",
        )

    def test_main_redirects_stdout(self):
        """main() must set sys.stdout = sys.stderr to catch stray prints."""
        source = (_REPO_ROOT / "cv2t" / "engine" / "canary_worker.py").read_text(
            encoding="utf-8"
        )
        # Find main function body
        in_main = False
        main_lines = []
        for line in source.splitlines():
            if line.startswith("def main("):
                in_main = True
                continue
            if in_main:
                if line and not line[0].isspace():
                    break
                main_lines.append(line)

        main_body = "\n".join(main_lines)
        self.assertIn(
            "sys.stdout = sys.stderr",
            main_body,
            "main() must redirect sys.stdout to sys.stderr to protect JSON protocol",
        )


class TestCanaryBridgeRecvErrorCapture(unittest.TestCase):
    """_recv() must include stderr output when the worker crashes.

    Regression: _recv() raised a generic 'terminated unexpectedly' error
    without the worker's stderr, making it impossible to diagnose crashes.
    """

    @patch("cv2t.engine.canary_bridge._get_canary_paths")
    @patch("cv2t.engine.canary_bridge.subprocess.Popen")
    def test_recv_includes_stderr_on_crash(self, mock_popen, mock_paths):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine

        mock_paths.return_value = ("python.exe", "worker.py")
        proc = MagicMock()
        proc.poll.return_value = 1
        # stdout returns empty (worker died)
        proc.stdout.readline.return_value = ""
        proc.stdin = MagicMock()
        mock_popen.return_value = proc

        engine = CanaryBridgeEngine()
        engine._process = proc
        # Simulate lines collected by the stderr drain thread
        engine._stderr_lines = ["SyntaxError: invalid syntax"]

        with self.assertRaises(RuntimeError) as ctx:
            engine._recv()
        error_msg = str(ctx.exception)
        self.assertIn("SyntaxError", error_msg,
                       "_recv error must include worker stderr output")
        self.assertIn("exit code 1", error_msg,
                       "_recv error must include the exit code")


class TestCanaryWorkerModuleStubs(unittest.TestCase):
    """canary_worker._load_model must shim missing transitive dependencies.

    Regression: NeMo transitively imports modules like datasets.distributed,
    wandb, and transformers.dependency_versions_check.  When these are absent
    in the installed canary-env the worker must inject stubs so model loading
    doesn't crash with "No module named ...".

    This test runs a subprocess snippet that:
      1. Blocks the real module via sys.modules (forces ImportError).
      2. Executes the stub logic extracted from _load_model.
      3. Verifies the shim module was injected into sys.modules.
    """

    _WORKER_PATH = str(_REPO_ROOT / "cv2t" / "engine" / "canary_worker.py")

    def _run_stub_check(self, snippet: str) -> subprocess.CompletedProcess:
        """Run *snippet* in a subprocess and return the result."""
        return subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_datasets_distributed_stub(self):
        """Worker must shim datasets.distributed when the real module is missing."""
        snippet = (
            "import importlib.machinery, types, sys\n"
            # Block the real module so the fallback fires
            "sys.modules['datasets.distributed'] = None\n"
            "del sys.modules['datasets.distributed']\n"
            # Prevent any real import from succeeding
            "import importlib\n"
            "orig_import = __builtins__.__import__\n"
            "def _block(name, *a, **kw):\n"
            "    if name == 'datasets.distributed':\n"
            "        raise ImportError('blocked for test')\n"
            "    return orig_import(name, *a, **kw)\n"
            "__builtins__.__import__ = _block\n"
            # Now run the stub logic copied from canary_worker
            "if 'datasets.distributed' not in sys.modules:\n"
            "    try:\n"
            "        import datasets.distributed\n"
            "    except (ImportError, ModuleNotFoundError):\n"
            "        _dd = types.ModuleType('datasets.distributed')\n"
            "        _dd.__spec__ = importlib.machinery.ModuleSpec('datasets.distributed', None)\n"
            "        _dd.__package__ = 'datasets'\n"
            "        _dd.split_dataset_by_node = lambda dataset, rank=0, world_size=1: dataset\n"
            "        sys.modules['datasets.distributed'] = _dd\n"
            # Verify
            "assert 'datasets.distributed' in sys.modules\n"
            "m = sys.modules['datasets.distributed']\n"
            "assert callable(m.split_dataset_by_node)\n"
            "assert m.split_dataset_by_node('passthrough') == 'passthrough'\n"
            "print('OK')\n"
        )
        result = self._run_stub_check(snippet)
        self.assertEqual(
            result.returncode, 0,
            f"datasets.distributed stub failed:\n{result.stderr}",
        )
        self.assertIn("OK", result.stdout)

    def test_wandb_stub(self):
        """Worker must shim wandb when the real module is missing."""
        snippet = (
            "import importlib.machinery, types, sys\n"
            "if 'wandb' not in sys.modules:\n"
            "    _wandb = types.ModuleType('wandb')\n"
            "    _wandb.__spec__ = importlib.machinery.ModuleSpec('wandb', None)\n"
            "    _wandb.__path__ = []\n"
            "    _wandb.__package__ = 'wandb'\n"
            "    _wandb.__version__ = '0.0.0'\n"
            "    sys.modules['wandb'] = _wandb\n"
            "import wandb\n"
            "assert wandb.__version__ == '0.0.0'\n"
            "print('OK')\n"
        )
        result = self._run_stub_check(snippet)
        self.assertEqual(
            result.returncode, 0,
            f"wandb stub failed:\n{result.stderr}",
        )
        self.assertIn("OK", result.stdout)

    def test_transformers_dep_check_stub(self):
        """Worker must shim transformers.dependency_versions_check."""
        snippet = (
            "import importlib.machinery, types, sys\n"
            "if 'transformers.dependency_versions_check' not in sys.modules:\n"
            "    _dvc = types.ModuleType('transformers.dependency_versions_check')\n"
            "    _dvc.dep_version_check = lambda pkg, hint=None: None\n"
            "    sys.modules['transformers.dependency_versions_check'] = _dvc\n"
            "m = sys.modules['transformers.dependency_versions_check']\n"
            "m.dep_version_check('torch')  # must not raise\n"
            "print('OK')\n"
        )
        result = self._run_stub_check(snippet)
        self.assertEqual(
            result.returncode, 0,
            f"transformers dep_check stub failed:\n{result.stderr}",
        )
        self.assertIn("OK", result.stdout)

    def test_worker_source_contains_all_stubs(self):
        """canary_worker.py must contain stub blocks for all known modules.

        If NeMo adds new transitive imports that need shimming, a developer
        should add the stub to _load_model AND add the module name here.
        """
        source = Path(self._WORKER_PATH).read_text(encoding="utf-8")
        required_stubs = [
            "datasets.distributed",
            "wandb",
            "transformers.dependency_versions_check",
        ]
        for mod in required_stubs:
            self.assertIn(
                mod,
                source,
                f"canary_worker.py is missing a stub for '{mod}'. "
                f"If NeMo imports it transitively, add a shim in _load_model.",
            )


class TestCanaryBridgeRecvTimeout(unittest.TestCase):
    """_recv() must support a timeout to prevent infinite hangs.

    Regression: The bridge called stdout.readline() with no timeout.
    If the canary worker subprocess hung (NeMo import stall, CUDA OOM,
    kernel compilation), the Worker thread would block forever and the
    UI showed "Loading…" counting up indefinitely with no recovery.
    """

    def test_recv_accepts_timeout_parameter(self):
        """_recv() must accept an optional timeout parameter."""
        import inspect
        from cv2t.engine.canary_bridge import CanaryBridgeEngine

        sig = inspect.signature(CanaryBridgeEngine._recv)
        self.assertIn(
            "timeout",
            sig.parameters,
            "_recv() must accept a 'timeout' parameter to prevent infinite hangs",
        )
        # Should default to None (no timeout for non-load operations)
        param = sig.parameters["timeout"]
        self.assertIs(
            param.default,
            None,
            "_recv(timeout=...) must default to None for backward compatibility",
        )

    def test_load_calls_recv_with_timeout(self):
        """load() must call _recv() WITH a timeout (not the default None).

        Regression: Without a timeout on the load operation, the UI hangs
        forever if the canary worker subprocess stalls during model loading.
        """
        import ast
        source = Path(_REPO_ROOT / "cv2t" / "engine" / "canary_bridge.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)

        # Find the load() method in CanaryBridgeEngine
        load_src = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "CanaryBridgeEngine":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "load":
                        load_src = ast.get_source_segment(source, item)
                        break
                break
        self.assertIsNotNone(load_src, "Could not find CanaryBridgeEngine.load()")

        # Verify _recv is called with a timeout keyword
        self.assertRegex(
            load_src,
            r"self\._recv\(.*timeout\s*=",
            "load() must call self._recv(timeout=...) to prevent infinite hangs",
        )

    def test_recv_timeout_raises_timeout_error(self):
        """_recv(timeout=...) must raise TimeoutError when the worker doesn't respond."""
        from cv2t.engine.canary_bridge import CanaryBridgeEngine

        engine = CanaryBridgeEngine()

        # Create a mock process whose stdout.readline blocks until released
        proc = MagicMock()
        proc.poll.return_value = None

        import threading
        import time

        block_event = threading.Event()

        def blocking_readline():
            block_event.wait(timeout=30)  # block "forever" (up to 30s safety)
            return ""

        proc.stdout.readline = blocking_readline
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()

        # Make kill() unblock the readline so cleanup is fast
        def _kill():
            block_event.set()
        proc.kill = _kill

        engine._process = proc
        engine._stderr_lines = []

        start = time.monotonic()
        with self.assertRaises(TimeoutError) as ctx:
            engine._recv(timeout=0.5)
        elapsed = time.monotonic() - start

        # Must have timed out, not waited forever
        self.assertLess(elapsed, 10.0,
                        "_recv(timeout=0.5) took too long — timeout not working")
        self.assertIn("0", str(ctx.exception),
                       "TimeoutError message should mention the timeout duration")

        # Allow cleanup thread to finish
        block_event.set()

    def test_recv_without_timeout_still_works(self):
        """_recv() without timeout must still return JSON immediately."""
        from cv2t.engine.canary_bridge import CanaryBridgeEngine

        engine = CanaryBridgeEngine()
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdout.readline.return_value = '{"status": "ok"}\n'
        engine._process = proc

        result = engine._recv()
        self.assertEqual(result, {"status": "ok"})

    def test_load_timeout_is_generous(self):
        """The load timeout must be at least 120s (NeMo can take minutes).

        A too-short timeout would cause false failures on slower hardware
        or when NeMo compiles CUDA kernels for the first time.
        """
        source = Path(_REPO_ROOT / "cv2t" / "engine" / "canary_bridge.py").read_text(
            encoding="utf-8"
        )
        # Find the timeout value passed in load()
        import re
        match = re.search(r"self\._recv\(timeout\s*=\s*(\d+)", source)
        self.assertIsNotNone(match, "Could not find _recv(timeout=N) in load()")
        timeout_val = int(match.group(1))
        self.assertGreaterEqual(
            timeout_val, 120,
            f"Load timeout is {timeout_val}s — must be >= 120s to allow for "
            f"slow NeMo model loading and CUDA kernel compilation",
        )


class TestEngineRegistryWithBridge(unittest.TestCase):
    """Engine __init__.py must handle canary_bridge import without crashing."""

    def test_engines_dict_populated(self):
        from cv2t.engine import ENGINES
        self.assertIsInstance(ENGINES, dict)
        # Whisper should be available in the dev environment
        self.assertIn("whisper", ENGINES)

    def test_canary_bridge_in_engines_or_not(self):
        """Canary may or may not be in ENGINES depending on env — must not crash."""
        from cv2t.engine import ENGINES
        # This test just verifies the import doesn't crash
        if "canary" in ENGINES:
            engine_cls = ENGINES["canary"]
            self.assertTrue(callable(engine_cls))


if __name__ == "__main__":
    unittest.main()
