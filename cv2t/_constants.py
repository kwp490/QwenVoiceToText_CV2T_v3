"""Shared UI constants — colours, timings, Win32 message IDs."""

# ── Status colours ───────────────────────────────────────────────────────────
COLOR_SUCCESS = "#2e7d32"
COLOR_VALIDATED = "#1b5e20"
COLOR_WARNING = "#f57f17"
COLOR_ERROR = "#c62828"
COLOR_NEUTRAL = "#757575"
COLOR_INFO = "#1565c0"
COLOR_IDLE = "#424242"

# ── Timer intervals (milliseconds) ──────────────────────────────────────────
METRICS_POLL_MS = 5000
LOADING_TICK_MS = 1000
STATE_RESET_IDLE_MS = 1500
STATE_RESET_ERROR_MS = 2000
SYSTEM_RESUME_DELAY_MS = 2000
SYSTEM_RESUME_DEBOUNCE_S = 10

# ── Windows power-management messages ────────────────────────────────────────
WM_POWERBROADCAST = 0x0218
PBT_APMRESUMEAUTOMATIC = 0x0012
PBT_APMRESUMESUSPEND = 0x0007
