"""Lazy CUDA driver identity queries for runtime target selection.

Keep device discovery out of compiler/source generation and avoid requiring
CuPy or the native chemistry library for standalone TensorIR execution.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class CudaDeviceIdentity:
    """Driver-reported product name and compute capability (``sm_XX``) of a device."""

    name: str
    architecture: str


def _driver_device(device: int) -> tuple[Callable[..., None], ctypes.c_int]:
    if type(device) is not int or not 0 <= device <= 2**31 - 1:
        raise ValueError("CUDA device must be a nonnegative int32 ordinal")
    name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    try:
        driver = ctypes.CDLL(name)
    except OSError as error:
        raise RuntimeError(
            "CUDA target detection requires the NVIDIA driver"
        ) from error

    def call(symbol: str, types: list[object], *arguments: object) -> None:
        function = getattr(driver, symbol)
        function.argtypes = types
        function.restype = ctypes.c_int
        status = function(*arguments)
        if status:
            raise RuntimeError(
                f"CUDA target detection failed for device {device}: {symbol} (status {status})"
            )

    call("cuInit", [ctypes.c_uint], 0)
    handle = ctypes.c_int()
    call(
        "cuDeviceGet",
        [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        ctypes.byref(handle),
        device,
    )
    return call, handle


def _compute_capability(call: Callable[..., None], handle: ctypes.c_int) -> str:
    pointer = ctypes.POINTER(ctypes.c_int)
    major, minor = ctypes.c_int(), ctypes.c_int()
    call(
        "cuDeviceComputeCapability",
        [pointer, pointer, ctypes.c_int],
        ctypes.byref(major),
        ctypes.byref(minor),
        handle,
    )
    if major.value < 1 or not 0 <= minor.value <= 9:
        raise RuntimeError("CUDA driver returned an invalid compute capability")
    return f"sm_{major.value}{minor.value}"


def device_architecture(device: int = 0) -> str:
    """Return the compute capability of a CUDA-visible device as ``sm_XX``.

    The driver interprets the ordinal after CUDA_VISIBLE_DEVICES filtering.
    This query does not create/switch a context or modify device visibility.
    Missing drivers/devices fail explicitly instead of choosing a guessed GPU.
    Call only inside the caller's GPU allocation.
    """
    return _compute_capability(*_driver_device(device))


def device_identity(device: int = 0) -> CudaDeviceIdentity:
    """Return the product name and compute capability of a CUDA-visible device.

    The ordinal, context and failure behavior match `device_architecture`.
    """
    call, handle = _driver_device(device)
    architecture = _compute_capability(call, handle)
    product = ctypes.create_string_buffer(256)
    call(
        "cuDeviceGetName",
        [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        product,
        256,
        handle,
    )
    label = product.value.decode("utf-8", errors="replace").strip()
    if not label:
        raise RuntimeError("CUDA driver returned an empty device name")
    return CudaDeviceIdentity(label, architecture)
