"""Converged physical DF-HF references for the experimental focal-point API."""

from __future__ import annotations

import ctypes as ct
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from weakref import finalize

import numpy as np

from . import _native
from .elements import checked_integer

if TYPE_CHECKING:
    from .calculator import Calculator
    from .experimental.f12_focal import Backend, FocalRequest


class _NativeFocalReference:
    """Own an immutable native determinant independently of its original context."""

    def __init__(self, library: ct.CDLL, handle: ct.c_void_p) -> None:
        self.library = library
        self.handle = handle
        release = library.generativeqc_focal_reference_destroy_v1
        release.argtypes = [ct.c_void_p]
        release.restype = None
        finalize(self, release, handle)


@dataclass(frozen=True)
class FocalReference:
    """One converged DF-HF solution in the orbital basis declared by the request.

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
    occupied_overlaps: tuple[float, ...] | None = None
    _native_owner: _NativeFocalReference | None = field(
        default=None, repr=False, compare=False
    )

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


@dataclass(frozen=True)
class FocalReferenceSeed:
    """Transport a converged determinant between geometries of one ordered molecule.

    The projected occupied orbitals initialize an independent target SCF. The
    smallest singular value of the source/target occupied overlap in every
    populated spin channel must meet minimum_overlap after convergence.
    """

    request: FocalRequest
    reference: FocalReference
    minimum_overlap: float = 0.99

    def __post_init__(self) -> None:
        req, ref = self.request, self.reference
        if (
            ref.surface_id != req.surface_id
            or ref.basis_id != req.basis.basis_id
            or ref.orbital_id != req.basis.orbital.identity
            or ref.jk_fit_id != req.basis.jk_fit.identity
        ):
            raise ValueError("Seed determinant and reference request identities differ")
        if isinstance(self.minimum_overlap, bool) or not 0 < self.minimum_overlap <= 1:
            raise ValueError("minimum_overlap must be in (0, 1]")

    def _project(
        self, request: FocalRequest, calculator: Calculator, budget: int
    ) -> tuple[np.ndarray, np.ndarray]:
        from .overlap import cross_overlap
        from .projection import project_occupied

        source, ref = self.request, self.reference
        if (
            source.root_id,
            source.charge,
            source.multiplicity,
            tuple(a.atomic_number for a in source.atoms),
        ) != (
            request.root_id,
            request.charge,
            request.multiplicity,
            tuple(a.atomic_number for a in request.atoms),
        ):
            raise ValueError(
                "Seed and target must have the same ordered nuclei, charge and spin reference"
            )
        if (ref.orbital_id, ref.jk_fit_id) != (
            request.basis.orbital.identity,
            request.basis.jk_fit.identity,
        ):
            raise ValueError("Seed and target require identical OBS and JKFit")
        options = {
            "charge": request.charge,
            "multiplicity": request.multiplicity,
            "maximum_bytes": budget,
        }
        cross = cross_overlap(
            calculator, calculator, request.atoms, source_atoms=source.atoms, **options
        )
        target = cross_overlap(calculator, calculator, request.atoms, **options)
        occupancy = 2 if request.root_id == "rhf" else 1
        spin_density = []
        for coeff, nocc in zip(ref.coefficients, ref.occupied, strict=True):
            if nocc:
                projected = project_occupied(
                    ref.overlap, target, cross, coeff[:, :nocc], occupation=occupancy
                )
                spin_density.append(projected.density)
            else:
                spin_density.append(np.zeros_like(target))
        return np.ascontiguousarray(spin_density).reshape(-1), cross

    def _checked_overlaps(
        self, cross: np.ndarray, coefficients: np.ndarray, occupied: tuple[int, ...]
    ) -> tuple[float, ...]:
        if occupied != self.reference.occupied:
            raise ValueError("Target SCF changed the seed spin occupations")
        overlaps = []
        for source, target, nocc in zip(
            self.reference.coefficients, coefficients, occupied, strict=True
        ):
            m = target[:, :nocc].T @ cross @ source[:, :nocc]
            overlaps.append(
                float(np.linalg.svd(m, compute_uv=False)[-1]) if nocc else 1.0
            )
        if not np.isfinite(overlaps).all() or min(overlaps) < self.minimum_overlap:
            raise ValueError(
                f"Target SCF changed occupied subspace: overlaps {overlaps}"
            )
        return tuple(overlaps)


def df_reference(
    request: FocalRequest,
    *,
    forces: bool = True,
    budget_mib: int = 512,
    seed: FocalReferenceSeed | None = None,
) -> FocalReference:
    """Solve canonical DF-RHF/DF-UHF using the declared OBS and JKFit.

    Use root_id="rhf" or "uhf" and optionally transport an occupied seed.
    Seed projection and the cross-geometry overlap check use host setup. Both J and K
    use the same Coulomb metric with relative cutoff 1e-10. A scalar ECP on
    the OBS uses native CUDA FP64 fitted HF on admitted hydrides, C/H monohalides,
    and Br/I-water systems.
    The JKFit supplies only Coulomb fitting functions. With forces=True
    the same solve also evaluates the reference's analytic nuclear forces.
    The CUDA path uses the native CUDA DF operator and derivative providers
    with host SCF control. Its fitted g-shell value route also covers singlet
    hydroxide and the water–hydroxide complex. The budget limits detached output
    and native reference capacity separately and sets the CUDA DF provider
    workspace allowance.
    CPU DF integral scratch is additional. Results use public AO order.
    """
    if seed is not None and not isinstance(seed, FocalReferenceSeed):
        raise TypeError("seed must be a FocalReferenceSeed")
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
        from .ecp import fitted_hf_ecp_domain

        if request.backend != "cuda" or not fitted_hf_ecp_domain(
            request.root_id, orb, request.atoms, request.charge, request.multiplicity
        ):
            raise NotImplementedError(
                "focal scalar ECP requires the native CUDA fitted HF ECP domain"
            )

    from .calculator import Calculator

    cuda = request.backend == "cuda"
    calculator = Calculator
    nuclei = tuple(atom.atomic_number for atom in request.atoms)
    unique_nuclei = sorted(set(nuclei))
    all_electron_obs = all(
        not orb.by_element[z].ecp_core_electrons and not orb.by_element[z].ecp_data
        for z in unique_nuclei
    )
    rhf_values = request.root_id == "rhf" and all_electron_obs
    oh_values = (
        request.root_id == "uhf"
        and request.charge == 0
        and request.multiplicity == 2
        and nuclei == (8, 1)
        and all_electron_obs
    )
    if (
        cuda
        and not forces
        and (rhf_values or oh_values)
        and max(shell.angular_momentum for z in unique_nuclei for shell in orb.by_element[z].shells) == 4
    ):
        class FittedValueCalculator(Calculator):
            def _full_df_basis_consumer(self, operator, derivative_order=0, *, atoms=None):
                if atoms is not None and derivative_order == 0 and operator in {
                    "overlap", "kinetic", "nuclear_attraction", "df_three_center",
                }:
                    return "full_df"
                return super()._full_df_basis_consumer(operator, derivative_order, atoms=atoms)

        calculator = FittedValueCalculator
    calc = calculator(
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
        name = (
            "generativeqc_focal_df_reference_v1"
            if seed is None
            else "generativeqc_focal_df_reference_seeded_v1"
        )
        owned = getattr(lib, "generativeqc_focal_df_reference_owned_v1", None)
        native = owned if owned is not None else getattr(lib, name, None)
        if native is None:
            raise RuntimeError(f"the native library lacks {name}")
        array = np.ctypeslib.ndpointer(np.float64, ndim=1, flags="C_CONTIGUOUS")
        signature = [ct.c_void_p] * 3 + [ct.c_int, ct.c_int, ct.c_size_t]
        if owned is not None:
            signature.extend([ct.c_void_p, ct.c_size_t])
        elif seed is not None:
            signature.extend([array, ct.c_size_t])
        signature.extend([array, ct.c_size_t])
        if owned is not None:
            signature.append(ct.POINTER(ct.c_void_p))
        native.argtypes = signature
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
        handle = ct.c_void_p()
        native_owner = None
        arguments = (None, 0) if owned is not None else ()
        if seed is not None:
            density, cross = seed._project(request, calc, budget)
            arguments = (
                density.ctypes.data if owned is not None else density,
                density.size,
            )
        raw = np.empty(count)
        _native.check(
            lib,
            native(
                ctx,
                system,
                auxiliary,
                spin,
                forces,
                budget,
                *arguments,
                raw,
                count,
                *((ct.byref(handle),) if owned is not None else ()),
            ),
            context=ctx,
        )
        if owned is not None:
            if not handle.value:
                raise RuntimeError("The focal DF solve returned no native reference")
            native_owner = _NativeFocalReference(lib, handle)
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
        seed._checked_overlaps(cross, coeff, occupied) if seed is not None else None,
        native_owner,
    )
