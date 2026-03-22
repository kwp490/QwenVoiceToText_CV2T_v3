"""
GPU and system resource monitoring via nvidia-ml-py and Win32 APIs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class GpuMetrics:
    """NVIDIA GPU metrics from NVML."""
    name: str = ""
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    vram_percent: float = 0.0
    temperature_c: int = 0


@dataclass
class SystemMetrics:
    """Combined system and GPU metrics."""
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    gpu: GpuMetrics = field(default_factory=GpuMetrics)


_host_ram_gb_cache: float = 0.0


def _get_host_ram() -> tuple[float, float, float]:
    """Return (used_gb, total_gb, percent) via Win32 GlobalMemoryStatusEx."""
    global _host_ram_gb_cache
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total = stat.ullTotalPhys / (1024 ** 3)
        avail = stat.ullAvailPhys / (1024 ** 3)
        used = total - avail
        pct = (used / total * 100) if total > 0 else 0
        if total > 0:
            _host_ram_gb_cache = total
        return used, total, pct
    except Exception as exc:
        log.debug("Win32 GlobalMemoryStatusEx failed: %s", exc)
        return 0.0, 0.0, 0.0


def _get_gpu_metrics() -> GpuMetrics:
    """Query NVIDIA GPU via NVML (nvidia-ml-py)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()

        total_gb = mem.total / (1024 ** 3)
        used_gb = mem.used / (1024 ** 3)
        pct = (used_gb / total_gb * 100) if total_gb > 0 else 0

        return GpuMetrics(
            name=name,
            vram_used_gb=used_gb,
            vram_total_gb=total_gb,
            vram_percent=pct,
            temperature_c=temp,
        )
    except Exception as exc:
        log.debug("NVML query failed: %s", exc)
        return GpuMetrics()


def get_system_metrics() -> SystemMetrics:
    """Collect RAM and GPU metrics."""
    ram_used, ram_total, ram_pct = _get_host_ram()
    gpu = _get_gpu_metrics()
    return SystemMetrics(
        ram_used_gb=ram_used,
        ram_total_gb=ram_total,
        ram_percent=ram_pct,
        gpu=gpu,
    )
