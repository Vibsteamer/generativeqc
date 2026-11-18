"""Thin CUDA contraction of fixed AO weights with nuclear integral derivatives."""

from __future__ import annotations

import ctypes as ct
from typing import Any, NamedTuple

import numpy as np

from . import _native
from ._api_types import Atom
from ._context_owner import ContextOwner
from .elements import checked_integer


class _Resources(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("device_bytes", ct.c_uint64),
        ("host_numeric_bytes", ct.c_uint64),
        ("host_to_device_bytes", ct.c_uint64),
        ("device_to_host_bytes", ct.c_uint64),
        ("synchronous_uploads", ct.c_uint64),
        ("stream_synchronizations", ct.c_uint64),
    ]


class IntegralGradient(NamedTuple):
    gradient: np.ndarray
    resources: dict[str, int]


def cuda_one_electron_gradient(
    calculator: Any,
    atoms: Any,
    *,
    overlap: np.ndarray | None = None,
    kinetic: np.ndarray | None = None,
    attraction: np.ndarray | None = None,
    hcore: np.ndarray | None = None,
    native_to_source: np.ndarray | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    maximum_bytes: int = 128 << 20,
    schedule: int = 1,
) -> IntegralGradient:
    """Return +dE/dR for fixed full AO weights, with no SCF or CPU fallback.

    Weights use the full Frobenius convention, including both off-diagonal
    entries. None means a zero channel. By default the weights follow the
    calculator's public AO order. If native_to_source is given, its element i
    is the source index of public AO i, with identical phases and normalization.
    For an adjoint of the core Hamiltonian, hcore validates and stages one
    matrix for kinetic energy and the calculator's attraction channel (which
    includes its ECP term when present). It is mutually exclusive with kinetic
    and attraction.
    The returned gradient excludes nuclear repulsion, orbital response and any
    derivative of the supplied weights. A molecular adjoint consumer assembles
    those contributions separately. Every center is mapped to a physical atom
    by the explicit basis shells; duplicated auxiliary centers need not create
    additional atoms. Numeric staging is bounded by maximum_bytes.

    A private context owns this synchronous transaction and is destroyed even
    after native failures. No singlepoint cache or electronic state is changed.
    """
    if calculator._backend != _native.BACKEND_CUDA:
        raise ValueError("one-electron gradient contraction requires a CUDA calculator")
    if hcore is not None and (kinetic is not None or attraction is not None):
        raise ValueError("hcore and separate kinetic or attraction weights overlap")
    size_limit = (1 << (8 * ct.sizeof(ct.c_size_t))) - 1
    maximum_bytes = checked_integer(
        maximum_bytes, "maximum bytes", low=1, high=size_limit
    )
    schedule = checked_integer(schedule, "gradient schedule", low=0, high=3)
    geometry = tuple(Atom.from_value(a) for a in atoms)
    if not geometry:
        raise ValueError("one-electron gradient requires atoms")
    shells = calculator._shells_for_atoms(
        geometry, operator="one_electron_gradient", derivative_order=1
    )
    spherical = calculator._basis_representation == _native.BASIS_SPHERICAL
    n = sum(
        2 * s.angular_momentum + 1
        if spherical
        else (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2
        for s in shells
    )
    checked_integer(n * n, "AO matrix count", low=1, high=size_limit)
    channels = (overlap, hcore) if hcore is not None else (overlap, kinetic, attraction)
    channel_count = sum(w is not None for w in channels)
    if channel_count * n * n * 8 > maximum_bytes:
        raise MemoryError("integral weights exceed the numeric staging budget")
    ix = None
    if native_to_source is not None:
        p = np.asarray(native_to_source)
        if (
            p.shape != (n,)
            or p.dtype.kind not in "iu"
            or not np.array_equal(np.sort(p), np.arange(n))
        ):
            raise ValueError("native_to_source must be a bijective integer AO map")
        ix = np.ix_(p, p)
    weights = []
    for value in channels:
        if value is None:
            weights.append(None)
            continue
        matrix = np.asarray(value)
        if matrix.dtype != np.float64 or matrix.shape != (n, n):
            raise ValueError("integral weights require a square FP64 public-AO matrix")
        if matrix.nbytes > maximum_bytes:
            raise MemoryError("integral weights exceed the numeric staging budget")
        if not np.isfinite(matrix).all():
            raise ValueError("integral weights must be finite")
        weights.append(np.ascontiguousarray(matrix if ix is None else matrix[ix]))
    if hcore is not None:
        # The native bridge recognizes identical pointers and uploads H once.
        weights.append(weights[1])
    library = calculator._library
    pointer = ct.POINTER(ct.c_double)
    routine = library.generativeqc_system_one_electron_gradient_cuda
    routine.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        pointer,
        pointer,
        pointer,
        ct.c_size_t,
        ct.c_uint,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
        ct.POINTER(_Resources),
    ]
    routine.restype = ct.c_int
    output = np.empty((len(geometry), 3), dtype=np.float64)
    resources = _Resources(ct.sizeof(_Resources), _native.ABI_VERSION)
    owner = ContextOwner(library)
    system = ct.c_void_p()
    try:
        with owner.lock:
            context = owner.get(calculator._context_descriptor())
            system = calculator._create_native_system(
                context,
                geometry,
                charge,
                multiplicity,
                operator="one_electron_gradient",
                derivative_order=1,
            )
            status = routine(
                context,
                system,
                *(None if w is None else w.ctypes.data_as(pointer) for w in weights),
                n * n,
                schedule,
                maximum_bytes,
                output.ctypes.data_as(pointer),
                output.size,
                ct.byref(resources),
            )
            _native.check(library, status, context=context)
    finally:
        try:
            if system:
                library.generativeqc_system_destroy(system)
        finally:
            owner.clear()
    return IntegralGradient(
        output,
        {key: int(getattr(resources, key)) for key, _ in _Resources._fields_[2:]},
    )
