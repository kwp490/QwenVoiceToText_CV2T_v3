"""
Windows clipboard operations and auto-paste via ctypes.

Uses raw Win32 API to avoid subprocess issues
in PyInstaller --noconsole builds.
"""

from __future__ import annotations

import ctypes
import logging
import time

log = logging.getLogger(__name__)

# ── Win32 API setup (done once at import) ─────────────────────────────────────

_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002

_kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
_user32 = ctypes.windll.user32  # type: ignore[attr-defined]

_kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
_kernel32.GlobalAlloc.restype = ctypes.c_void_p
_kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
_kernel32.GlobalLock.restype = ctypes.c_void_p
_kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
_kernel32.GlobalUnlock.restype = ctypes.c_int
_user32.OpenClipboard.argtypes = [ctypes.c_void_p]
_user32.OpenClipboard.restype = ctypes.c_int
_user32.EmptyClipboard.restype = ctypes.c_int
_user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
_user32.SetClipboardData.restype = ctypes.c_void_p
_user32.CloseClipboard.restype = ctypes.c_int


# ── Clipboard ─────────────────────────────────────────────────────────────────

def set_clipboard_text(text: str) -> bool:
    """Copy *text* to the Windows clipboard.  Returns ``True`` on success."""
    encoded = text.encode("utf-16-le") + b"\x00\x00"
    buf_size = len(encoded)

    if not _user32.OpenClipboard(None):
        log.error("OpenClipboard failed")
        return False
    try:
        _user32.EmptyClipboard()
        h_mem = _kernel32.GlobalAlloc(_GMEM_MOVEABLE, buf_size)
        if not h_mem:
            log.error("GlobalAlloc failed")
            return False
        p_mem = _kernel32.GlobalLock(h_mem)
        if not p_mem:
            log.error("GlobalLock failed")
            return False
        ctypes.memmove(p_mem, encoded, buf_size)
        _kernel32.GlobalUnlock(h_mem)
        _user32.SetClipboardData(_CF_UNICODETEXT, h_mem)
        log.debug("Clipboard set: %s", text[:80])
        return True
    finally:
        _user32.CloseClipboard()


# ── Auto-paste ────────────────────────────────────────────────────────────────

def simulate_paste(wait_for_modifiers: bool = True) -> None:
    """Send Ctrl+V to the active window.

    When *wait_for_modifiers* is ``True`` the function spins until
    Ctrl and Alt are released — this prevents Ctrl+Alt+V being sent
    instead of Ctrl+V when triggered via a global hotkey.
    """
    import keyboard  # imported here so clipboard module can be used without keyboard

    if wait_for_modifiers:
        for _ in range(200):  # 10 s max wait
            if not keyboard.is_pressed("ctrl") and not keyboard.is_pressed("alt"):
                break
            time.sleep(0.05)
        time.sleep(0.05)
    keyboard.send("ctrl+v")
    log.debug("Ctrl+V sent")
