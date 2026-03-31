# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for CV2T — Whisper-only binary (no torch/NeMo).

Build: pyinstaller cv2t.spec
Output: dist/cv2t/cv2t.exe (onedir)
"""

from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

# Collect PortAudio DLL from sounddevice
binaries = collect_dynamic_libs('sounddevice')

# Collect CUDA DLLs from ctranslate2 (used by faster-whisper)
# Filter out cuDNN DLLs — CTranslate2 only needs cuBLAS for Whisper inference.
try:
    _ct2_bins = collect_dynamic_libs('ctranslate2')
    binaries += [(src, dst) for src, dst in _ct2_bins if 'cudnn' not in src.lower()]
except Exception:
    pass

# Collect cuBLAS DLLs from nvidia pip packages (cublas64_12.dll etc.)
# NOTE: nvidia.cudnn is deliberately excluded — CTranslate2 uses cuBLAS only
# for Whisper inference and does not call any cuDNN APIs. Excluding cuDNN
# saves ~900 MB of DLLs from the build.
for _nvidia_pkg in ('nvidia.cublas', 'nvidia.cuda_runtime'):
    try:
        binaries += collect_dynamic_libs(_nvidia_pkg)
    except Exception:
        pass

a = Analysis(
    ['cv2t/__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=[
        ('cv2t/assets', 'cv2t/assets'),
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
        'onnxruntime',
        # PyAV (video codec library) — pulled transitively, not used by cv2t
        'av',
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
# software renderer, PyAV video codec DLLs, and onnxruntime.
import re as _re

_STRIP_PATTERNS = [
    _re.compile(r'Qt6Quick', _re.I),
    _re.compile(r'Qt6Qml', _re.I),
    _re.compile(r'Qt6Pdf', _re.I),
    _re.compile(r'opengl32sw', _re.I),
    _re.compile(r'av[\\/]', _re.I),         # PyAV Python package
    _re.compile(r'av\.libs[\\/]', _re.I),    # PyAV native libs
    _re.compile(r'onnxruntime', _re.I),
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
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='cv2t',
)
