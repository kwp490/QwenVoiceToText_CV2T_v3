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

# Collect CUDA/cuDNN DLLs from ctranslate2 (used by faster-whisper)
try:
    binaries += collect_dynamic_libs('ctranslate2')
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
        'torch',
        'torchaudio',
        'torchvision',
        'nemo',
        'nemo_toolkit',
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
