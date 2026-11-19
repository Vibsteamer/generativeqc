"""Coupled unrestricted orbital response for real, canonical DF-UHF F12 forces."""

import ctypes
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from . import _native
from .elements import checked_integer


class UnrestrictedOrbitalForces(NamedTuple):
    rotations: np.ndarray
    orbital_gradient: np.ndarray
    forces: np.ndarray


def _block(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != shape
        or array.dtype != np.float64
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must contain finite FP64 values of shape {shape}")
    return array.ravel(order="C")


def unrestricted_orbital_forces(
    space: tuple[int, int, int, int],
    factors: Mapping[str, np.ndarray],
    electronic_driver: np.ndarray,
    nuclear_rhs: np.ndarray,
    reference_forces: np.ndarray,
    fixed_gradients: Sequence[np.ndarray] = (),
    *,
    device_id: int = 0,
    maximum_bytes: int = 1536 << 20,
    library: ctypes.CDLL | None = None,
) -> UnrestrictedOrbitalForces:
    """Solve Hᵀz=g and return F_ref − Σ fixed gradients − b_Rᵀz on CUDA.

    Space contains alpha/beta occupied counts, OBS orbitals and the shared DF
    rank. Supply vo, oo, vv and gap for each occupied spin under names such as
    vo_alpha. Use the same Coulomb whitening for both spins. Flatten the
    electronic driver in alpha-then-beta virtual-major, occupied-minor order.
    Nuclear right-hand sides have shape (atoms, 3, rotations). They solve
    H x_R=b_R. The fixed-frame energy gradients must exclude this rotation
    response. Reference forces and each fixed gradient have shape (atoms, 3).
    The returned orbital gradient uses the positive energy derivative sign.

    maximum_bytes bounds explicit CUDA arrays including pivot and LU workspaces.
    """
    oa, ob, n, q = (
        checked_integer(value, name)
        for value, name in zip(
            space,
            ("alpha occupation", "beta occupation", "OBS orbitals", "DF rank"),
            strict=True,
        )
    )
    if not oa + ob or n <= max(oa, ob) or q < 1:
        raise ValueError(
            "Expected occupied electrons, OBS virtuals and a nonempty fitting basis"
        )
    device = checked_integer(device_id, "CUDA device")
    budget = checked_integer(
        maximum_bytes, "workspace bytes", low=1, high=np.iinfo(np.uintp).max
    )
    rank = checked_integer(oa * (n - oa) + ob * (n - ob), "orbital rotations", low=1)
    ref = np.asarray(reference_forces)
    if ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError("Reference forces must have three Cartesian axes")
    directions = checked_integer(ref.size, "Cartesian directions", low=1)
    terms = checked_integer(len(fixed_gradients), "fixed gradient terms")
    if 8 * rank * rank > budget:
        raise ValueError(
            "The dense orbital Hessian exceeds the device workspace budget"
        )
    packed = []
    for spin, o in (("alpha", oa), ("beta", ob)):
        if not o:
            continue
        v = n - o
        for key, shape in (
            ("vo", (q, v, o)),
            ("oo", (q, o, o)),
            ("vv", (q, v, v)),
            ("gap", (v, o)),
        ):
            name = f"{key}_{spin}"
            packed.append(_block(factors[name], shape, name))
    packed.append(_block(electronic_driver, (rank,), "electronic driver"))
    packed.append(_block(nuclear_rhs, (*ref.shape, rank), "nuclear RHS"))
    packed.append(_block(ref, ref.shape, "reference forces"))
    packed.extend(_block(g, ref.shape, "fixed gradient") for g in fixed_gradients)
    if (sum(a.size for a in packed) + rank * rank + directions) * 8 + (
        rank + 1
    ) * 4 > budget:
        raise ValueError(
            "Molecular adjoint inputs and Hessian exceed the device workspace budget"
        )
    data = np.concatenate(packed)
    if library is None:
        library = _native.load_library(device="cuda", device_id=device)
    routine = library.generativeqc_uhf_orbital_adjoint_cuda_v1
    pointer = ctypes.POINTER(ctypes.c_double)
    routine.argtypes = [ctypes.c_int] * 7 + [
        pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        pointer,
        pointer,
        pointer,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    routine.restype = ctypes.c_int
    z, gradient, forces = np.empty(rank), np.empty(ref.shape), np.empty(ref.shape)
    error = ctypes.create_string_buffer(512)
    status = routine(
        device,
        oa,
        ob,
        n,
        q,
        directions,
        terms,
        data.ctypes.data_as(pointer),
        data.size,
        budget,
        z.ctypes.data_as(pointer),
        gradient.ctypes.data_as(pointer),
        forces.ctypes.data_as(pointer),
        error,
        len(error),
    )
    if status:
        raise RuntimeError(error.value.decode(errors="replace"))
    return UnrestrictedOrbitalForces(z, gradient, forces)
