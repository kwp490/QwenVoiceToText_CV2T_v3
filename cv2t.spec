# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for CV2T — Whisper-only binary (no torch/NeMo).

Build: pyinstaller cv2t.spec
Output: dist/cv2t/cv2t.exe (onedir)
"""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

block_cipher = None

# Collect PortAudio DLL from sounddevice
binaries = collect_dynamic_libs('sounddevice')

# Collect faster-whisper data files (Silero VAD ONNX model in assets/)
datas = []
try:
    datas += collect_data_files('faster_whisper')
except Exception:
    pass

# Collect CUDA DLLs from ctranslate2 (used by faster-whisper)
# Filter out cuDNN DLLs — CTranslate2 only needs cuBLAS for Whisper inference.
try:
    _ct2_bins = collect_dynamic_libs('ctranslate2')
    binaries += [(src, dst) for src, dst in _ct2_bins
                 if 'cudnn' not in src.lower()]
except Exception:
    pass

# Collect PyAV (FFmpeg) DLLs — required by faster-whisper for audio decoding
try:
    binaries += collect_dynamic_libs('av')
except Exception:
    pass

# Collect onnxruntime DLLs — required by faster-whisper's Silero VAD filter
try:
    binaries += collect_dynamic_libs('onnxruntime')
except Exception:
    pass

# Collect cuBLAS DLLs from nvidia pip packages (cublas64_12.dll etc.)
# NOTE: nvidia.cudnn is deliberately excluded — CTranslate2 uses cuBLAS only
# for Whisper inference and does not call any cuDNN APIs. Excluding cuDNN
# saves ~900 MB of DLLs from the build.
for _nvidia_pkg in ('nvidia.cublas', 'nvidia.cuda_runtime'):
    try:
        _pkg_bins = collect_dynamic_libs(_nvidia_pkg)
        # Double-check: strip any cuDNN DLLs that sneak in via transitive deps
        binaries += [(src, dst) for src, dst in _pkg_bins
                     if 'cudnn' not in src.lower()]
    except Exception:
        pass

a = Analysis(
    ['cv2t/__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas + [
        ('cv2t/assets', 'cv2t/assets'),
        ('cv2t/engine/canary_worker.py', '.'),
        ('installer/Enable-Canary.ps1', '.'),
    ],
    hiddenimports=[
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'sounddevice',
        'soundfile',
        '_soundfile_data',
        'numpy',
        'keyboard',
        'pynvml',
        'faster_whisper',
        'ctranslate2',
        'av',
        'tokenizers',
        'onnxruntime',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Engines / ML frameworks not used in Whisper-only binary
        'torch',
        'torchaudio',
        'torchvision',
        'nemo',
        'nemo_toolkit',
        # GUI / image libraries not used
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── Strip unnecessary binaries ───────────────────────────────────────────────
# CV2T only uses QtWidgets/QtCore/QtGui. Remove Qt Quick/Qml/Pdf, the OpenGL
# software renderer and PyAV video codec DLLs.
import re as _re

_STRIP_PATTERNS = [
    _re.compile(r'Qt6Quick', _re.I),
    _re.compile(r'Qt6Qml', _re.I),
    _re.compile(r'Qt6Pdf', _re.I),
    _re.compile(r'opengl32sw', _re.I),
    _re.compile(r'cudnn', _re.I),             # cuDNN — not needed for Whisper
    _re.compile(r'huggingface_hub', _re.I),   # replaced by stdlib urllib
    _re.compile(r'safetensors', _re.I),       # huggingface_hub transitive dep
]

def _should_keep(entry):
    name = entry[0] if isinstance(entry, tuple) else str(entry)
    return not any(p.search(name) for p in _STRIP_PATTERNS)

a.binaries = [b for b in a.binaries if _should_keep(b)]
a.datas = [d for d in a.datas if _should_keep(d)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='cv2t',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='cv2t',
)
