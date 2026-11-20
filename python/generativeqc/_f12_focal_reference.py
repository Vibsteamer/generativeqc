"""Converged physical DF-HF references for the experimental focal-point API."""

from __future__ import annotations

import ctypes as ct
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import _native
from .elements import checked_integer

if TYPE_CHECKING:
    from .experimental.f12_focal import Backend, FocalRequest


@dataclass(frozen=True)
class FocalReference:
    """One cold DF-HF solution in the orbital basis declared by the request.

    Spin-major coefficients[:, AO, MO], Fock and density have one RHF or two UHF
    channels. The RHF density includes both electrons per occupied orbital.
    Energies are in Eh and forces are negative derivatives in Eh/Bohr.
    """

    energy: float
    forces: np.ndarray | None
    overlap: np.ndarray
    hcore: np.ndarray
    coefficients: np.ndarray
    orbital_energies: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    occupied: tuple[int, ...]
    iterations: int
    executed_backend: Backend
    surface_id: str
    basis_id: str
    orbital_id: str
    jk_fit_id: str

    def _pair_spin_indices(self, request: FocalRequest) -> tuple[int, int]:
        """Map physical alpha/beta lines to the canonical orbital variables."""
        if (
            request.root_id == "rhf"
            and request.multiplicity == 1
            and len(self.occupied) == 1
        ):
            return (0, 0)
        if request.root_id == "uhf" and len(self.occupied) == 2:
            return (0, 1)
        raise ValueError("Pair request and canonical reference spin channels disagree")


def df_reference(
    request: FocalRequest, *, forces: bool = True, budget_mib: int = 512
) -> FocalReference:
    """Solve canonical DF-RHF/DF-UHF using the declared OBS and JKFit.

    Use root_id="rhf" or "uhf" for the respective cold solver. Both J and K
    use the same Coulomb metric with relative cutoff 1e-10. A scalar ECP on
    the OBS uses native CUDA FP64 fitted UHF hydrides and singlet RHF HBr or HI. The
    JKFit supplies only Coulomb fitting functions. With forces=True
    the same solve also evaluates the reference's analytic nuclear forces.
    The CUDA path uses the native CUDA DF operator and derivative providers
    with host SCF control. The budget limits detached output and native reference
    capacity separately and sets the CUDA DF provider workspace allowance.
    CPU DF integral scratch is additional. Results use public AO order.
    """
    if request.backend not in ("cpu_reference", "cuda"):
        raise ValueError("DF focal reference requires an explicit CPU or CUDA backend")
    if request.root_id not in ("rhf", "uhf") or (
        request.root_id == "rhf" and request.multiplicity != 1
    ):
        raise ValueError(
            "DF focal reference requires a singlet rhf or a canonical uhf root"
        )
    if type(forces) is not bool:
        raise TypeError("forces must be Boolean")
    budget = (
        checked_integer(
            budget_mib, "reference budget in MiB", low=1, high=(2**63 - 1) >> 20
        )
        << 20
    )
    orb, jk = request.basis.orbital, request.basis.jk_fit
    for record in (orb, jk):
        if record.representation != "spherical":
            raise ValueError("DF focal reference requires real-spherical OBS and JKFit")
        if any(
            record.by_element[a.atomic_number].nuclear_charge != a.atomic_number
            for a in request.atoms
        ):
            raise NotImplementedError(
                "DF focal reference requires physical nuclear charges"
            )
    if any(jk.by_element[a.atomic_number].ecp_core_electrons for a in request.atoms):
        raise ValueError("the focal JKFit describes Coulomb fitting functions only")
    if any(orb.by_element[a.atomic_number].ecp_core_electrons for a in request.atoms):
        from .ecp import fitted_hf_hydride_domain

        if request.backend != "cuda" or not fitted_hf_hydride_domain(
            request.root_id, orb, request.atoms, request.charge, request.multiplicity
        ):
            raise NotImplementedError(
                "focal scalar ECP requires the native CUDA fitted HF hydride domain"
            )

    from .calculator import Calculator

    cuda = request.backend == "cuda"
    calc = Calculator(
        method=request.root_id,
        basis=orb,
        auxiliary_basis=jk,
        density_fitting=request.backend,
        device="cuda" if cuda else "cpu",
        initial_guess=None,
        precision="fp64",
    )
    spin = 2 if request.root_id == "uhf" else 1
    natom = len(request.atoms)
    n = sum(
        (2 * shell.angular_momentum + 1) * len(shell.coefficients)
        for a in request.atoms
        for shell in orb.by_element[a.atomic_number].shells
    )
    nf = 3 * natom if forces else 0
    count = 3 + spin + (2 + 3 * spin) * n**2 + spin * n + nf
    if 8 * count > budget:
        raise MemoryError("DF focal reference output exceeds its numeric budget")
    with ExitStack() as owners:
        owners.callback(calc.clear_cache)
        lib = calc._library
        native = getattr(lib, "generativeqc_focal_df_reference_v1", None)
        if native is None:
            raise RuntimeError("the native library lacks the focal DF reference export")
        native.argtypes = [ct.c_void_p] * 3 + [
            ct.c_int,
            ct.c_int,
            ct.c_size_t,
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ct.c_size_t,
        ]
        native.restype = ct.c_int
        calc._preflight_hf_basis(request.atoms, compute_forces=forces)
        ctx = ct.c_void_p()
        _native.check(
            lib,
            lib.generativeqc_context_create(
                ct.byref(calc._context_descriptor()), ct.byref(ctx)
            ),
        )
        owners.callback(lib.generativeqc_context_destroy, ctx)
        geometry = (ctx, request.atoms, request.charge, request.multiplicity)
        system = calc._create_native_system(*geometry)
        owners.callback(lib.generativeqc_system_destroy, system)
        auxiliary = calc._create_native_system(*geometry, jk)
        owners.callback(lib.generativeqc_system_destroy, auxiliary)
        raw = np.empty(count)
        _native.check(
            lib,
            native(ctx, system, auxiliary, spin, forces, budget, raw, count),
            context=ctx,
        )
    if not np.isfinite(raw).all() or raw[2] != (
        _native.BACKEND_CUDA if cuda else _native.BACKEND_CPU_REFERENCE
    ):
        raise RuntimeError(
            "native DF reference returned nonfinite data or used another backend"
        )
    raw.setflags(write=False)
    energy, iterations = raw[:2]
    occupied = tuple(int(v) for v in raw[3 : 3 + spin])
    off = 3 + spin
    overlap, hcore = raw[off : off + 2 * n**2].reshape(2, n, n)
    off += 2 * n**2
    extent = spin * (3 * n**2 + n)
    channels = raw[off : off + extent].reshape(spin, 3 * n**2 + n)
    coeff, density, fock = (
        channels[:, : 3 * n**2].reshape(spin, 3, n, n).transpose(1, 0, 2, 3)
    )
    return FocalReference(
        float(energy),
        raw[-nf:].reshape(natom, 3) if forces else None,
        overlap,
        hcore,
        coeff,
        channels[:, 3 * n**2 :],
        density,
        fock,
        occupied,
        int(iterations),
        request.backend,
        request.surface_id,
        request.basis.basis_id,
        orb.identity,
        jk.identity,
    )
