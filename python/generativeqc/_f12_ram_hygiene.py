"""Host memory management and phase diagnostics for focal evaluations."""

from __future__ import annotations

import ctypes
import gc
import math
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_increment import FocalF12Increment, FocalF12Sources
    from ._f12_focal_one_electron import FocalOneElectron
    from ._f12_focal_reference import FocalReference
    from ._f12_focal_tplus_sources import SpinOccupied
    from .experimental.f12_focal import FocalRequest

# Stream dense [Q, ext, ext] AO panels whenever a single panel reaches 512 MiB
# or the 5-operator F12 panel set reaches 1.5 GiB (or Cartesian count > 128M).
SINGLE_PANEL_STREAM_BYTES = 512 << 20
F12_TOTAL_PANEL_STREAM_BYTES = 1536 << 20
AO_CART_STREAM_THRESHOLD = 128_000_000

STRICT_GPU_ONLY_ENV = "GENERATIVEQC_F12_STRICT_GPU_ONLY"

_LIBC_MALLOC_TRIM: Callable[[int], int] | None | bool = False
_ACTIVE_RAM_TRACES: ContextVar[tuple[list[dict[str, float | str]], ...]] = ContextVar(
    "_ACTIVE_RAM_TRACES", default=()
)
_ACTIVE_HYBRID_TRACES: ContextVar[tuple[list[dict[str, object]], ...]] = ContextVar(
    "_ACTIVE_HYBRID_TRACES", default=()
)


def _resolve_malloc_trim() -> Callable[[int], int] | None:
    global _LIBC_MALLOC_TRIM
    if _LIBC_MALLOC_TRIM is not False:
        return _LIBC_MALLOC_TRIM  # type: ignore[return-value]
    fn: Callable[[int], int] | None = None
    try:
        libc = ctypes.CDLL("libc.so.6")
        raw = getattr(libc, "malloc_trim", None)
        if raw is not None:
            raw.argtypes = [ctypes.c_size_t]
            raw.restype = ctypes.c_int
            fn = raw
    except OSError:
        fn = None
    _LIBC_MALLOC_TRIM = fn
    return fn


def host_memory_mib() -> dict[str, float]:
    """Read Linux process VmRSS and VmHWM in MiB from /proc/self/status."""
    rss_mib = 0.0
    hwm_mib = 0.0
    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                rss_mib = float(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                hwm_mib = float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return {"vm_rss_mib": rss_mib, "vm_hwm_mib": hwm_mib}


def release_host_memory() -> bool:
    """Collect Python cycles and trim retained glibc heap arenas back to the OS."""
    gc.collect()
    trim = _resolve_malloc_trim()
    if trim is None:
        return False
    try:
        return bool(trim(0))
    except Exception:
        return False


def should_stream_panel(
    *,
    allow_streaming: bool,
    ordinary: bool,
    coulomb_only: bool,
    n_ext: int,
    n_aux: int,
    ao_cart_total: int,
    env_stream: str | None = None,
) -> bool:
    """Decide whether a 3-center F12/JK fitting panel must stream on demand."""
    if not allow_streaming or ordinary:
        return False
    flag = os.environ.get("GENERATIVEQC_F12_STREAM_PANELS") if env_stream is None else env_stream
    if flag == "1":
        return True
    if flag == "0":
        return False
    single_panel_bytes = int(n_aux) * int(n_ext) * int(n_ext) * 8
    est_panel_bytes = (1 if coulomb_only else 5) * single_panel_bytes
    return (
        int(ao_cart_total) > AO_CART_STREAM_THRESHOLD
        or single_panel_bytes >= SINGLE_PANEL_STREAM_BYTES
        or est_panel_bytes >= F12_TOTAL_PANEL_STREAM_BYTES
    )


def strict_gpu_only_enabled() -> bool:
    """Return True when GENERATIVEQC_F12_STRICT_GPU_ONLY=1 forbids host-assisted GPU stages."""
    return os.environ.get(STRICT_GPU_ONLY_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@contextmanager
def capture_ram_trace() -> Iterator[list[dict[str, float | str]]]:
    """Capture per-phase host VmRSS/VmHWM checkpoints emitted by F12 routines.

    Uses a ContextVar so concurrent threads/tasks and interleaved exits never
    corrupt another trace's bucket.
    """
    records: list[dict[str, float | str]] = []
    _ACTIVE_RAM_TRACES.set((*_ACTIVE_RAM_TRACES.get(), records))
    try:
        yield records
    finally:
        cur = _ACTIVE_RAM_TRACES.get()
        _ACTIVE_RAM_TRACES.set(tuple(bucket for bucket in cur if bucket is not records))


@contextmanager
def capture_hybrid_phases() -> Iterator[list[dict[str, object]]]:
    """Capture structured hybrid/host-assisted phase disclosures emitted by F12 routines."""
    records: list[dict[str, object]] = []
    _ACTIVE_HYBRID_TRACES.set((*_ACTIVE_HYBRID_TRACES.get(), records))
    try:
        yield records
    finally:
        cur = _ACTIVE_HYBRID_TRACES.get()
        _ACTIVE_HYBRID_TRACES.set(tuple(bucket for bucket in cur if bucket is not records))


def disclose_host_assisted_stage(
    stage: str,
    *,
    requested_backend: str,
    detail: str,
    scientific_contraction: bool = True,
) -> None:
    """Record a host-assisted scientific stage and enforce GENERATIVEQC_F12_STRICT_GPU_ONLY=1."""
    entry: dict[str, object] = {
        "stage": stage,
        "requested_backend": requested_backend,
        "execution_mode": "hybrid_host_assisted"
        if requested_backend == "cuda"
        else "host",
        "scientific_contraction": bool(scientific_contraction),
        "detail": detail,
    }
    for bucket in _ACTIVE_HYBRID_TRACES.get():
        bucket.append(entry)
    if (
        requested_backend == "cuda"
        and scientific_contraction
        and strict_gpu_only_enabled()
    ):
        raise NotImplementedError(
            f"{STRICT_GPU_ONLY_ENV}=1 forbids host-contracted scientific stage "
            f"{stage!r} on backend='cuda': {detail}"
        )


def record_ram_phase(phase: str) -> None:
    """Record a host VmRSS/VmHWM checkpoint when tracing is enabled."""
    active = _ACTIVE_RAM_TRACES.get()
    print_trace = os.environ.get("GENERATIVEQC_F12_RAM_TRACE") == "1"
    if not active and not print_trace:
        return
    mem = host_memory_mib()
    entry: dict[str, float | str] = {
        "phase": phase,
        "vm_rss_mib": mem["vm_rss_mib"],
        "vm_hwm_mib": mem["vm_hwm_mib"],
    }
    for bucket in active:
        bucket.append(entry)
    if print_trace:
        print(
            f"[f12-ram] {phase}: VmRSS={mem['vm_rss_mib']:.1f} MiB "
            f"({mem['vm_rss_mib'] / 1024.0:.2f} GiB), "
            f"VmHWM={mem['vm_hwm_mib']:.1f} MiB ({mem['vm_hwm_mib'] / 1024.0:.2f} GiB)",
            flush=True,
        )
