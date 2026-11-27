"""Shared-reference additive MP2-F12 core-valence (Delta-CV) evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import partial
from math import fsum, isfinite
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_molecular import MolecularFocalProvider
    from ._f12_focal_reference import FocalReference, FocalReferenceSeed
    from .experimental.f12_focal import FocalRequest


@dataclass(frozen=True)
class FocalDeltaCVResult:
    """Shared-reference MP2-F12 additive core-valence increment in Eh and Eh/bohr.

    Both 'cv-active' and 'frozen-core' terms are evaluated on the exact same
    core-valence basis, DF-HF determinant, one-electron/ECP operators,
    three-center/metric panels, and fixed-rank CABS projector so that HF+CABS
    cancels identically and only correlation differences enter Delta-CV.
    When cv_frozen == fc_frozen, the Delta-CV difference vanishes identically
    (delta_* == 0.0 and delta_forces == 0) and uncomputed absolute cv_*/fc_*
    energy and force fields are recorded as None rather than placeholder zeros.
    """

    reference: FocalReference
    cabs_rank: int
    cv_frozen: tuple[int, int]
    fc_frozen: tuple[int, int]
    cv_ordinary_mp2_Eh: float | None
    fc_ordinary_mp2_Eh: float | None
    cv_f12_increment_Eh: float | None
    fc_f12_increment_Eh: float | None
    cv_mp2_f12_Eh: float | None
    fc_mp2_f12_Eh: float | None
    delta_ordinary_mp2_Eh: float
    delta_f12_increment_Eh: float
    delta_mp2_f12_Eh: float
    cv_forces: np.ndarray | None = None
    fc_forces: np.ndarray | None = None
    delta_forces: np.ndarray | None = None
    fc_mode: str = "frozen-core"
    reference_provenance: str = "native_df_reference"
    frozen_core_definition_version: int | None = None

    @property
    def cv_frozen_occupied(self) -> tuple[int, int]:
        return self.cv_frozen

    @property
    def fc_frozen_occupied(self) -> tuple[int, int]:
        return self.fc_frozen


from ._f12_frozen_core import (
    DEFAULT_PRODUCTION_CORRELATION_SPACE,
    FROZEN_CORE_DEFINITION_VERSION,
)


def default_valence_correlation_space_id(request: FocalRequest) -> str:
    """Resolve default valence correlation space when request_fc is omitted.

    Molecules containing small-core Br(ECP10MDF) or I(ECP28MDF) use the production
    Hill-Peterson / TA13 FC9 valence definition (DEFAULT_PRODUCTION_CORRELATION_SPACE);
    all other molecules use 'frozen-core' (identical to 'frozen-core-spd' for Z <= 30).
    """
    by_el = request.basis.orbital.by_element
    for atom in request.atoms:
        z = atom.atomic_number
        eb = by_el.get(z)
        ecp_core = eb.ecp_core_electrons if eb is not None else 0
        if (z == 35 and ecp_core == 10) or (z == 53 and ecp_core == 28):
            return DEFAULT_PRODUCTION_CORRELATION_SPACE
    return "frozen-core"


def _is_valid_delta_cv_fc_space(correlation_space_id: str) -> bool:
    return correlation_space_id in (
        "frozen-core",
        DEFAULT_PRODUCTION_CORRELATION_SPACE,
    ) or correlation_space_id.startswith("frozen:")


def _is_eligible_for_focal_source_df_hf_fallback(
    request: FocalRequest, exc: NotImplementedError
) -> bool:
    """Admit the independently covered host HBr and high-angular SCF capacity fallbacks."""
    from . import _native
    from .ecp import fitted_hf_ecp_domain

    orb = request.basis.orbital
    if orb.representation != "spherical" or request.basis.jk_fit.representation != "spherical":
        return False
    if any(
        record.by_element[a.atomic_number].nuclear_charge != a.atomic_number
        for record in (orb, request.basis.jk_fit)
        for a in request.atoms
    ) or any(request.basis.jk_fit.by_element[a.atomic_number].ecp_core_electrons for a in request.atoms):
        return False
    if request.root_id not in ("rhf", "uhf") or (
        request.root_id == "rhf" and request.multiplicity != 1
    ):
        return False
    max_orb_l = max(
        shell.angular_momentum
        for a in request.atoms
        for shell in orb.by_element[a.atomic_number].shells
    )
    detail = str(exc).removeprefix(f"GENERATIVEQC error {_native.STATUS_NOT_IMPLEMENTED}: ")
    if detail == "ECP density-fitting execution is not yet validated":
        # Independent PySCF E/FD and restricted/unrestricted controls for the
        # focal host J/K implementation cover these small-core HBr states.
        return (
            request.backend == "cpu_reference"
            and sorted(a.atomic_number for a in request.atoms) == [1, 35]
            and orb.by_element[1].ecp_core_electrons == 0
            and orb.by_element[35].ecp_core_electrons == 10
            and (request.charge, request.multiplicity) in ((0, 1), (1, 2))
            and max_orb_l <= 2
        )
    # The C ABI returns a status and a detail string. Accept its orbital-shell
    # capacity diagnostic or the equivalent Python orbital preflight.
    angular_limits = {
        "scalar ECP execution currently supports orbital s/p/d/f shells",
        "ECP execution supports orbital s/p/d/f shells",
        "unrestricted CUDA DF execution supports orbital shells only through f (l <= 3)",
        "this fitted CUDA DF-UHF spin domain supports orbital shells through g",
        "CUDA orbital metadata supports l<=4; execution requires a qualified operator route",
    }
    if detail not in angular_limits and not all(
        re.fullmatch(
            r"orbital atom \d+ Z=\d+ shell \d+ contraction \d+ l=[456]: "
            r"(?:cpu|cuda)/(?:overlap|kinetic|nuclear_attraction|df_three_center|eri) "
            r"derivative 0 supports only l<=\d+",
            reason,
        )
        for reason in detail.split("; ")
    ):
        return False
    has_ecp = any(
        orb.by_element[a.atomic_number].ecp_core_electrons for a in request.atoms
    )
    if has_ecp and not fitted_hf_ecp_domain(
        request.root_id, orb, request.atoms, request.charge, request.multiplicity
    ):
        return False
    return 4 <= max_orb_l <= 6


def _paired_cv_fc_requests(
    req_cv: FocalRequest,
    req_fc: FocalRequest | None,
) -> tuple[FocalRequest, FocalRequest]:
    """Validate or construct the matching (cv-active, valence-fc) pair requests."""
    from .experimental.f12_focal import FocalSeries, plan_singlepoint

    if req_cv.correlation_space_id != "cv-active":
        raise ValueError(
            f"Delta-CV primary request must use correlation_space_id='cv-active', "
            f"got {req_cv.correlation_space_id!r}"
        )
    if req_fc is None:
        series = FocalSeries(req_cv.basis)
        plan = plan_singlepoint(
            req_cv.atoms,
            series,
            series,
            None,
            charge=req_cv.charge,
            multiplicity=req_cv.multiplicity,
            root_id=req_cv.root_id,
            correlation_space_id=default_valence_correlation_space_id(req_cv),
            backend=req_cv.backend,
            slater_normalization=req_cv.slater_normalization,
        )
        req_fc = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    if not _is_valid_delta_cv_fc_space(req_fc.correlation_space_id):
        raise ValueError(
            "Delta-CV valence request must use correlation_space_id in "
            "('frozen-core', 'frozen-core-spd', 'frozen:<n>'), "
            f"got {req_fc.correlation_space_id!r}"
        )
    if (
        req_cv.atoms != req_fc.atoms
        or req_cv.charge != req_fc.charge
        or req_cv.multiplicity != req_fc.multiplicity
        or req_cv.root_id != req_fc.root_id
        or req_cv.basis.basis_id != req_fc.basis.basis_id
        or req_cv.role != "mp2_f12_correlation"
        or req_fc.role != "mp2_f12_correlation"
        or req_cv.backend != req_fc.backend
        or req_cv.slater_normalization != req_fc.slater_normalization
    ):
        raise ValueError(
            "Delta-CV cv-active and frozen-core requests must share identical "
            "geometry, charge, multiplicity, root, CV basis, role, backend and Slater normalization"
        )
    return req_cv, req_fc


def molecular_delta_cv_energy(
    provider: MolecularFocalProvider,
    request_cv: FocalRequest,
    request_fc: FocalRequest | None = None,
    *,
    expected_cabs_rank: int | None = None,
) -> FocalDeltaCVResult:
    """Evaluate forward-only Delta-CV = E_MP2-F12(cv-active) - E_MP2-F12(frozen-core).

    Shares a single DF-HF solve, 1e/ECP + 3c/2c integral build, CABS projector,
    MO-transformed pair source, Fock matrix, and single-pass pair summation.
    """
    from generativeqc_compiler.tensor import execute

    from ._f12_focal_energy import (
        _check_request,
        _fock_uses_canonical_spectrum,
        _ordinary_energies,
    )
    from ._f12_focal_frame import canonical_pair_frame
    from ._f12_focal_reference import df_reference
    from ._f12_focal_reference import partition_focal_reference
    from ._f12_spin_ao import _orbital_sources
    from ._f12_spin_fock import spin_fock
    from ._f12_spin_pairs import molecular_pair_energies

    req_cv, req_fc = _paired_cv_fc_requests(request_cv, request_fc)
    budget = _check_request(provider, req_cv)
    cuda = req_cv.backend == "cuda"
    runner = provider.run or partial(execute, max_bytes=budget << 20)

    def run(graph: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        result = runner(graph, inputs)
        if result.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError("Delta-CV energy graph executed on another backend")
        return result

    if hasattr(runner, "batch"):
        from contextlib import contextmanager

        @contextmanager
        def run_batch(graph: Program):
            with runner.batch(graph) as exec_one:
                def wrapped(inputs: Mapping[str, np.ndarray]) -> Execution:
                    result = exec_one(inputs)
                    if result.backend != (
                        "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
                    ):
                        raise RuntimeError(
                            "Delta-CV energy graph executed on another backend"
                        )
                    return result

                yield wrapped

        run.batch = run_batch  # type: ignore[attr-defined]

    ref = df_reference(
        req_cv, forces=False, budget_mib=budget, **provider._seed_options(req_cv)
    )
    ref = partition_focal_reference(
        req_cv, ref, budget, additional_space=req_fc.correlation_space_id,
    )
    assert ref.frozen_frame is not None
    cv_frozen = ref.frozen_frame.counts(req_cv.correlation_space_id)
    fc_frozen = ref.frozen_frame.counts(req_fc.correlation_space_id)
    indices = ref._pair_spin_indices(req_cv)
    occs = tuple(ref.occupied[k] for k in indices)

    if cv_frozen == fc_frozen:
        from ._f12_focal_projector import checked_reference_cabs_rank

        return FocalDeltaCVResult(
            reference=ref,
            cabs_rank=checked_reference_cabs_rank(provider, req_cv, ref, expected_cabs_rank),
            cv_frozen=cv_frozen,
            fc_frozen=fc_frozen,
            fc_mode=req_fc.correlation_space_id,
            frozen_core_definition_version=FROZEN_CORE_DEFINITION_VERSION,
            cv_ordinary_mp2_Eh=None,
            fc_ordinary_mp2_Eh=None,
            cv_f12_increment_Eh=None,
            fc_f12_increment_Eh=None,
            cv_mp2_f12_Eh=None,
            fc_mp2_f12_Eh=None,
            delta_ordinary_mp2_Eh=0.0,
            delta_f12_increment_Eh=0.0,
            delta_mp2_f12_Eh=0.0,
        )

    one, (jk, mp2, f12) = provider._cached_pair_sources(
        req_cv, derivatives=False, budget=budget
    )
    space, ao, jkfit, projector = canonical_pair_frame(
        req_cv,
        ref,
        one,
        jk.joint(req_cv, explicit=False),
        f12.joint(req_cv, explicit=True),
        expected_cabs_rank=expected_cabs_rank,
        relative_cutoff=provider.relative_cutoff,
        library=provider.response_library,
    )
    oa, ob, obs, ext, _ = space
    distinct_jk = jk is not f12
    data, _ = _orbital_sources(
        space,
        ao,
        run,
        None,
        projected=None,
        role="pairs" if distinct_jk else "shared",
        occupied_coulomb=distinct_jk,
    )
    fock_data = data
    if distinct_jk:
        shared = ("overlap_ao", "hcore_ao", "coefficients_alpha", "coefficients_beta")
        fock_ao = {**{k: ao[k] for k in shared}, **jkfit}
        fock_data, _ = _orbital_sources(
            (oa, ob, obs, ext, len(jk.whitener)),
            fock_ao,
            run,
            None,
            projected=None,
            role="fock",
        )

    def sources(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            k.removeprefix("source/"): v
            for k, v in raw.items()
            if k.startswith("source/")
        }

    fock = spin_fock((oa, ob, ext, len(jk.whitener)), sources(fock_data), run)
    fc_by_spin = {"alpha": fc_frozen[0], "beta": fc_frozen[1]}
    shared_pairs = mp2 is f12 and _fock_uses_canonical_spectrum(fock, ref, indices, obs)

    cv_mp2_terms: list[float] = []
    fc_mp2_terms: list[float] = []
    cv_f12_terms: list[float] = []
    fc_f12_terms: list[float] = []

    for item in molecular_pair_energies(
        space,
        data,
        sources(data),
        fock,
        run,
        frozen_occupied=cv_frozen,
        include_ordinary=shared_pairs,
        shell_library=provider.shell_library,
        budget_mib=budget,
        backend=req_cv.backend,
    ):
        sl, sr = item.spins
        ii, jj = item.occupied_indices
        in_fc = ii >= fc_by_spin[sl] and jj >= fc_by_spin[sr]
        cv_f12_terms.append(item.f12)
        if in_fc:
            fc_f12_terms.append(item.f12)
        if item.mp2 is not None:
            cv_mp2_terms.append(item.mp2)
            if in_fc:
                fc_mp2_terms.append(item.mp2)

    if shared_pairs:
        cv_ord = fsum(cv_mp2_terms)
        fc_ord = fsum(fc_mp2_terms)
    else:
        obs_src = mp2.obs(req_cv, obs)
        cv_ord = fsum(
            _ordinary_energies(
                ref,
                indices,
                obs_src,
                run,
                None,
                frozen_occupied=cv_frozen,
                shell_library=provider.shell_library,
                response_library=provider.response_library,
                budget_mib=budget,
                cuda=cuda,
            )
        )
        fc_ord = fsum(
            _ordinary_energies(
                ref,
                indices,
                obs_src,
                run,
                None,
                frozen_occupied=fc_frozen,
                shell_library=provider.shell_library,
                response_library=provider.response_library,
                budget_mib=budget,
                cuda=cuda,
            )
        )

    cv_f12 = fsum(cv_f12_terms)
    fc_f12 = fsum(fc_f12_terms)
    cv_tot = cv_ord + cv_f12
    fc_tot = fc_ord + fc_f12
    d_ord = cv_ord - fc_ord
    d_f12 = cv_f12 - fc_f12
    d_tot = cv_tot - fc_tot

    if not all(isfinite(v) for v in (cv_tot, fc_tot, d_tot)):
        raise ArithmeticError("nonfinite molecular Delta-CV energy")

    return FocalDeltaCVResult(
        reference=ref,
        cabs_rank=projector.rank,
        cv_frozen=cv_frozen,
        fc_frozen=fc_frozen,
        fc_mode=req_fc.correlation_space_id,
        frozen_core_definition_version=FROZEN_CORE_DEFINITION_VERSION,
        cv_ordinary_mp2_Eh=float(cv_ord),
        fc_ordinary_mp2_Eh=float(fc_ord),
        cv_f12_increment_Eh=float(cv_f12),
        fc_f12_increment_Eh=float(fc_f12),
        cv_mp2_f12_Eh=float(cv_tot),
        fc_mp2_f12_Eh=float(fc_tot),
        delta_ordinary_mp2_Eh=float(d_ord),
        delta_f12_increment_Eh=float(d_f12),
        delta_mp2_f12_Eh=float(d_tot),
    )


def molecular_delta_cv_energy_and_forces(
    provider: MolecularFocalProvider,
    request_cv: FocalRequest,
    request_fc: FocalRequest | None = None,
    *,
    expected_cabs_rank: int | None = None,
) -> FocalDeltaCVResult:
    """Evaluate Delta-CV energy and analytic 3N forces sharing one DF-HF + 1e + 3c panel."""
    from ._f12_focal_energy import _check_request
    from ._f12_focal_reference import df_reference
    from ._f12_focal_reference import partition_focal_reference

    req_cv, req_fc = _paired_cv_fc_requests(request_cv, request_fc)
    budget = _check_request(provider, req_cv)

    ref_cv = df_reference(
        req_cv, forces=False, budget_mib=budget, **provider._seed_options(req_cv)
    )
    ref_cv = partition_focal_reference(
        req_cv, ref_cv, budget, additional_space=req_fc.correlation_space_id,
    )
    assert ref_cv.frozen_frame is not None
    cv_frozen = ref_cv.frozen_frame.counts(req_cv.correlation_space_id)
    fc_frozen = ref_cv.frozen_frame.counts(req_fc.correlation_space_id)

    if cv_frozen == fc_frozen:
        from ._f12_focal_projector import checked_reference_cabs_rank

        zeros = np.zeros((len(req_cv.atoms), 3), dtype=np.float64)
        return FocalDeltaCVResult(
            reference=ref_cv,
            cabs_rank=checked_reference_cabs_rank(provider, req_cv, ref_cv, expected_cabs_rank),
            cv_frozen=cv_frozen,
            fc_frozen=fc_frozen,
            fc_mode=req_fc.correlation_space_id,
            frozen_core_definition_version=FROZEN_CORE_DEFINITION_VERSION,
            cv_ordinary_mp2_Eh=None,
            fc_ordinary_mp2_Eh=None,
            cv_f12_increment_Eh=None,
            fc_f12_increment_Eh=None,
            cv_mp2_f12_Eh=None,
            fc_mp2_f12_Eh=None,
            delta_ordinary_mp2_Eh=0.0,
            delta_f12_increment_Eh=0.0,
            delta_mp2_f12_Eh=0.0,
            cv_forces=None,
            fc_forces=None,
            delta_forces=zeros,
        )

    one, panels = provider._cached_pair_sources(req_cv, derivatives=True, budget=budget)
    spin_cv, inc_cv, rank, forces_cv = provider._pair_components(
        req_cv, ref_cv, one, panels, expected_cabs_rank=expected_cabs_rank
    )
    import gc

    gc.collect()

    ref_fc = replace(ref_cv, surface_id=req_fc.surface_id)
    one_fc = replace(one, surface_id=req_fc.surface_id)
    spin_fc, inc_fc, rank_fc, forces_fc = provider._pair_components(
        req_fc, ref_fc, one_fc, panels, expected_cabs_rank=rank
    )
    if rank_fc != rank:
        raise RuntimeError(
            f"Delta-CV CABS rank mismatch between cv-active ({rank}) and frozen-core ({rank_fc})"
        )

    cv_ord = fsum(spin_cv)
    fc_ord = fsum(spin_fc)
    cv_tot = cv_ord + inc_cv
    fc_tot = fc_ord + inc_fc
    delta_forces = forces_cv - forces_fc

    if not all(isfinite(v) for v in (cv_tot, fc_tot, cv_tot - fc_tot)) or not np.isfinite(delta_forces).all():
        raise ArithmeticError("nonfinite molecular Delta-CV energy or forces")

    return FocalDeltaCVResult(
        reference=ref_cv,
        cabs_rank=rank,
        cv_frozen=cv_frozen,
        fc_frozen=fc_frozen,
        fc_mode=req_fc.correlation_space_id,
        frozen_core_definition_version=FROZEN_CORE_DEFINITION_VERSION,
        cv_ordinary_mp2_Eh=float(cv_ord),
        fc_ordinary_mp2_Eh=float(fc_ord),
        cv_f12_increment_Eh=float(inc_cv),
        fc_f12_increment_Eh=float(inc_fc),
        cv_mp2_f12_Eh=float(cv_tot),
        fc_mp2_f12_Eh=float(fc_tot),
        delta_ordinary_mp2_Eh=float(cv_ord - fc_ord),
        delta_f12_increment_Eh=float(inc_cv - inc_fc),
        delta_mp2_f12_Eh=float(cv_tot - fc_tot),
        cv_forces=forces_cv,
        fc_forces=forces_fc,
        delta_forces=delta_forces,
    )


def conventional_mp2_request(
    atoms,
    basis,
    *,
    root_id: str,
    correlation_space_id: str,
    charge: int = 0,
    multiplicity: int = 1,
    backend: str = "cuda",
    protocol_id: str | None = None,
) -> FocalRequest:
    """Build a pure conventional DF-MP2 FocalRequest (zero CABS, zero F12, beta=None)."""
    import json

    from ._api_types import Atom
    from .elements import checked_integer, electron_state
    from .experimental.f12_focal import FocalBasis, FocalRequest
    from .profiles import canonical_hash

    geometry = tuple(Atom.from_value(atom) for atom in atoms)
    if not geometry:
        raise ValueError("at least one atom is required")
    charge = checked_integer(charge, "ionic charge", low=-(2**31))
    multiplicity = checked_integer(multiplicity, "multiplicity", low=1)
    if root_id not in ("rhf", "uhf") or (root_id == "rhf" and multiplicity != 1):
        raise ValueError("conventional DF-MP2 requires singlet rhf or uhf root_id")
    if not isinstance(correlation_space_id, str) or not correlation_space_id.strip():
        raise ValueError("conventional DF-MP2 requires a nonempty correlation_space_id")
    if backend not in ("cuda", "cpu_reference"):
        raise ValueError("focal backend must be cuda or cpu_reference")
    if not isinstance(basis, FocalBasis):
        raise TypeError("conventional_mp2_request requires a FocalBasis")
    nuclei = {atom.atomic_number for atom in geometry}
    for name, record in (
        ("orbital", basis.orbital),
        ("JKFit", basis.jk_fit),
        ("MP2Fit", basis.mp2_fit),
    ):
        if record is None:
            raise ValueError(f"conventional DF-MP2 requires an explicit {name} basis")
        if nuclei - record.by_element.keys():
            raise ValueError(f"{name} basis {record.name!r} lacks a nucleus in the geometry")

    orbital = basis.orbital.by_element
    state = electron_state(
        geometry, charge=charge, multiplicity=multiplicity, element_metadata=orbital
    )
    hamiltonian = tuple(
        (z, core, canonical_hash(json.loads(data)) if data is not None else None)
        for z, core, data in zip(
            state.nuclear_charges,
            state.ecp_core_electrons,
            (orbital[atom.atomic_number].ecp_data for atom in geometry),
            strict=True,
        )
    )
    surface_id = canonical_hash(
        {
            "geometry_bohr": [
                (
                    atom.atomic_number,
                    tuple((v if v else 0.0).hex() for v in atom.position),
                )
                for atom in geometry
            ],
            "charge": charge,
            "multiplicity": multiplicity,
            "hamiltonian": hamiltonian,
            "electronic_root": root_id,
            "correlation_space": correlation_space_id,
            "frozen_core_definition_version": FROZEN_CORE_DEFINITION_VERSION,
            **({"protocol_id": protocol_id} if protocol_id is not None else {}),
        }
    )
    return FocalRequest(
        geometry,
        charge,
        multiplicity,
        root_id,
        correlation_space_id,
        surface_id,
        basis,
        "mp2_f12_correlation",
        backend,  # type: ignore[arg-type]
    )


def _paired_cv_fc_mp2_requests(
    req_cv: FocalRequest,
    req_fc: FocalRequest | None,
) -> tuple[FocalRequest, FocalRequest]:
    """Validate or construct a paired (cv-active, frozen-core) conventional DF-MP2 request pair."""
    if req_cv.correlation_space_id != "cv-active":
        raise ValueError(
            f"Delta-CV primary request must use correlation_space_id='cv-active', "
            f"got {req_cv.correlation_space_id!r}"
        )
    if req_fc is None:
        req_fc = conventional_mp2_request(
            req_cv.atoms,
            req_cv.basis,
            root_id=req_cv.root_id,
            correlation_space_id=default_valence_correlation_space_id(req_cv),
            charge=req_cv.charge,
            multiplicity=req_cv.multiplicity,
            backend=req_cv.backend,
        )
    if not _is_valid_delta_cv_fc_space(req_fc.correlation_space_id):
        raise ValueError(
            "Delta-CV valence request must use correlation_space_id in "
            "('frozen-core', 'frozen-core-spd', 'frozen:<n>'), "
            f"got {req_fc.correlation_space_id!r}"
        )
    if (
        req_cv.atoms != req_fc.atoms
        or req_cv.charge != req_fc.charge
        or req_cv.multiplicity != req_fc.multiplicity
        or req_cv.root_id != req_fc.root_id
        or req_cv.basis.basis_id != req_fc.basis.basis_id
        or req_cv.role != "mp2_f12_correlation"
        or req_fc.role != "mp2_f12_correlation"
        or req_cv.backend != req_fc.backend
    ):
        raise ValueError(
            "Delta-CV cv-active and frozen-core requests must share identical "
            "geometry, charge, multiplicity, root, CV basis, role, and backend"
        )
    return req_cv, req_fc


def _check_conventional_mp2_request(
    provider: MolecularFocalProvider, request: FocalRequest
) -> int:
    from .elements import checked_integer

    basis = request.basis
    if (
        request.role != "mp2_f12_correlation"
        or request.root_id not in ("rhf", "uhf")
        or (request.root_id == "rhf" and request.multiplicity != 1)
        or basis.mp2_fit is None
    ):
        raise ValueError(
            "conventional DF-MP2 Delta-CV requires RHF/UHF with orbital, JKFit, and MP2Fit bases"
        )
    cuda = request.backend == "cuda"
    if request.backend not in ("cuda", "cpu_reference") or (
        cuda
        and (
            provider.run is None
            or provider.metric_library is None
            or provider.response_library is None
        )
    ):
        raise ValueError(
            "conventional DF-MP2 Delta-CV requires its physical backend, CUDA metric library, and graph runner"
        )
    return checked_integer(
        provider.budget_mib,
        "molecular focal budget in MiB",
        low=1,
        high=(2**63 - 1) >> 20,
    )


def _checked_runner(provider: MolecularFocalProvider, cuda: bool, budget: int):
    from contextlib import contextmanager

    from generativeqc_compiler.tensor import execute

    runner = provider.run or partial(execute, max_bytes=budget << 20)
    expected_backend = "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"

    def run(graph: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        result = runner(graph, inputs)
        if result.backend != expected_backend:
            raise RuntimeError("Delta-CV MP2 graph executed on another backend")
        return result

    if hasattr(runner, "batch"):
        @contextmanager
        def run_batch(graph: Program):
            with runner.batch(graph) as exec_one:
                def wrapped(inputs: Mapping[str, np.ndarray]) -> Execution:
                    result = exec_one(inputs)
                    if result.backend != expected_backend:
                        raise RuntimeError("Delta-CV MP2 graph executed on another backend")
                    return result

                yield wrapped

        run.batch = run_batch  # type: ignore[attr-defined]
    return run


def _solve_df_reference_from_focal_sources(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    one,
    jk,
    *,
    run,
    seed: FocalReferenceSeed | None = None,
    max_cycles: int = 160,
) -> FocalReference:
    """Solve canonical DF-RHF/DF-UHF natively from focal 1e/ECP and JKFit Coulomb panels.

    Used as a bounded fallback when frozen libgenerativeqc.so rejects orbital g-shells on
    multi-center or atomic small-core ECP systems even though the focal 1e/ECP and shell
    libraries natively evaluate through i-shells (l <= 6).
    """
    from contextlib import nullcontext

    from generativeqc_compiler.tensor import (
        Index,
        IndexSpace,
        Program,
        TensorSpec,
        einsum,
        input_tensor,
    )

    from ._f12_cv_dfmp2_native import bind_delta_cv_native
    from ._f12_focal_reference import FocalReference
    from ._f12_ram_hygiene import disclose_host_assisted_stage
    from .elements import electron_state

    disclose_host_assisted_stage(
        "delta_cv_fallback_df_hf_host_linear_algebra",
        requested_backend=request.backend,
        detail="host-assisted Lowdin/DIIS/eigh SCF diagonalization in focal-source fallback",
    )

    S = np.ascontiguousarray(one.overlap, dtype=np.float64)
    H = np.ascontiguousarray(one.hcore, dtype=np.float64)
    ao_coulomb = jk.data["ao_coulomb"]
    W = np.ascontiguousarray(jk.whitener, dtype=np.float64)
    n = len(S)
    q = len(W)
    orb = request.basis.orbital
    state = electron_state(
        request.atoms,
        charge=request.charge,
        multiplicity=request.multiplicity,
        element_metadata=orb.by_element,
    )
    nelec = int(state.electron_count)
    n_unpaired = request.multiplicity - 1
    oa = (nelec + n_unpaired) // 2
    ob = (nelec - n_unpaired) // 2
    restricted = request.root_id == "rhf"

    # Nuclear repulsion energy using effective ionic charges Z_eff = Z - ecp_core
    z_eff = [
        float(z - c)
        for z, c in zip(state.nuclear_charges, state.ecp_core_electrons, strict=True)
    ]
    xyz = np.asarray([a.position for a in request.atoms], dtype=np.float64)
    e_nuc = 0.0
    for i_at in range(len(z_eff)):
        for j_at in range(i_at + 1, len(z_eff)):
            r_ij = float(np.linalg.norm(xyz[i_at] - xyz[j_at]))
            e_nuc += z_eff[i_at] * z_eff[j_at] / r_ij

    # Symmetric Lowdin orthogonalizer X = S^{-1/2}
    s_vals, s_vecs = np.linalg.eigh(0.5 * (S + S.T))
    if np.min(s_vals) <= 1e-11:
        raise ValueError(f"Near-singular overlap matrix in focal DF-HF fallback (min eig={np.min(s_vals):.3e})")
    X = np.ascontiguousarray((s_vecs * (s_vals**-0.5)) @ s_vecs.T)

    # Precompute whitened B_jk[Q, p, q] = sum_R W[Q, R] * ao_coulomb[R, p, q]
    B_jk = np.ascontiguousarray(
        (W @ ao_coulomb.reshape(q, -1)).reshape(q, n, n), dtype=np.float64
    )

    q_chunk = max(1, min(q, 50_000_000 // max(1, n * n)))
    aot = bind_delta_cv_native(
        provider.shell_library,
        host_reference=(request.backend == "cpu_reference"),
        budget_bytes=provider.budget_mib << 20,
    )

    if aot is not None and aot.host_reference != (request.backend == "cpu_reference"):
        raise RuntimeError("Delta-CV fallback J/K binding has a different execution backend")

    def _diag(fock_ao: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f_orth = X.T @ (0.5 * (fock_ao + fock_ao.T)) @ X
        f_orth = 0.5 * (f_orth + f_orth.T)
        eps, c_orth = np.linalg.eigh(f_orth)
        return eps, np.ascontiguousarray(X @ c_orth)

    # Initial guess from seed if provided (with full state/basis/overlap verification),
    # otherwise GWH (Generalized Wolfsberg-Helmholtz).
    seed_cross: np.ndarray | None = None
    expected_occ: tuple[int, ...] = (oa,) if restricted else (oa, ob)
    if seed is not None:
        source_req, source_ref = seed.request, seed.reference
        if (
            source_req.root_id,
            source_req.charge,
            source_req.multiplicity,
            tuple(a.atomic_number for a in source_req.atoms),
        ) != (
            request.root_id,
            request.charge,
            request.multiplicity,
            tuple(a.atomic_number for a in request.atoms),
        ):
            raise ValueError(
                "Seed and target must have the same ordered nuclei, charge and spin reference"
            )
        if (source_ref.orbital_id, source_ref.jk_fit_id) != (
            request.basis.orbital.identity,
            request.basis.jk_fit.identity,
        ):
            raise ValueError("Seed and target require identical OBS and JKFit")
        if source_ref.occupied != expected_occ:
            raise ValueError("Seed and target spin occupations differ")
        if source_ref.coefficients.shape != ((1, n, n) if restricted else (2, n, n)):
            raise ValueError("Seed coefficient dimensions differ from target basis")
        if source_req.atoms == request.atoms:
            if not np.allclose(source_ref.overlap, S, atol=2e-8, rtol=0):
                raise ValueError(
                    "Seed and target overlap matrices disagree at identical geometry"
                )
            seed_cross = S
        else:
            from .calculator import Calculator
            from .overlap import cross_overlap

            try:
                calc = Calculator(
                    method=request.root_id,
                    basis=orb,
                    auxiliary_basis=request.basis.jk_fit,
                    density_fitting=request.backend,
                    device="cuda" if request.backend == "cuda" else "cpu",
                    initial_guess=None,
                    precision="fp64",
                )
                try:
                    seed_cross = cross_overlap(
                        calc,
                        calc,
                        request.atoms,
                        source_atoms=source_req.atoms,
                        charge=request.charge,
                        multiplicity=request.multiplicity,
                        maximum_bytes=provider.budget_mib << 20,
                    )
                finally:
                    calc.clear_cache()
            except Exception as exc:
                raise ValueError(
                    "Focal-source DF-HF fallback cannot verify displaced-geometry seed overlap"
                ) from exc
        from .projection import project_occupied

        proj_frames = []
        for coeff_s, nocc_s in zip(source_ref.coefficients, source_ref.occupied, strict=True):
            if nocc_s:
                projected = project_occupied(
                    source_ref.overlap,
                    S,
                    seed_cross,
                    coeff_s[:, :nocc_s],
                    occupation=2 if restricted else 1,
                )
                # The SCF cycle consumes occupied columns only. They are
                # already orthonormal in the target AO overlap metric.
                proj_frames.append(projected.coefficients)
            else:
                proj_frames.append(np.empty((n, 0), dtype=np.float64))
        ca = proj_frames[0]
        cb = ca if restricted else proj_frames[1]
    else:
        h_diag = np.diag(H)
        f_gwh = 0.5 * 1.75 * S * (h_diag[:, None] + h_diag[None, :])
        np.fill_diagonal(f_gwh, h_diag)
        _, ca = _diag(f_gwh)
        if restricted:
            cb = ca
        else:
            # Tiny deterministic diagonal break on GWH breaks degenerate atomic p-shells cleanly
            f_break = f_gwh.copy()
            diag_idx = np.arange(n)
            f_break[diag_idx, diag_idx] += 1e-4 * np.cos(np.arange(1, n + 1, dtype=np.float64))
            _, cb = _diag(f_break)

    padded_chunks: list[np.ndarray] | None = None
    if aot is not None:
        ctx_a = nullcontext(None)
        ctx_b = nullcontext(None)
    else:
        chunks = list(range(0, q, q_chunk))
        a_idx = Index("a", IndexSpace("jk_chunk", "auxiliary", q_chunk))
        obs_ao = IndexSpace("obs_ao", "ao", n)
        p_idx = Index("p", obs_ao)
        q_idx = Index("q", obs_ao)

        def _build_fock_programs(occ_count: int):
            i_idx = Index("i", IndexSpace(f"occ_{occ_count}", "orbital", occ_count))
            b_in = input_tensor("B", TensorSpec((a_idx, p_idx, q_idx), role="input"))
            d_in = input_tensor("D", TensorSpec((p_idx, q_idx), role="input"))
            c_in = input_tensor("Co", TensorSpec((q_idx, i_idx), role="input"))
            rho = einsum("apq,pq->a", b_in, d_in)
            j_out = einsum("a,apq->pq", rho, b_in)
            x_mo = einsum("apq,qi->api", b_in, c_in)
            k_out = einsum("api,aqi->pq", x_mo, x_mo)
            return Program({"J": j_out, "K": k_out})

        def _padded_chunk(q0: int) -> np.ndarray:
            q1 = min(q, q0 + q_chunk)
            if q1 - q0 == q_chunk:
                return np.ascontiguousarray(B_jk[q0:q1])
            pad = np.zeros((q_chunk, n, n), dtype=np.float64)
            pad[: q1 - q0] = B_jk[q0:q1]
            return pad

        padded_chunks = [_padded_chunk(q0) for q0 in chunks]

        # Build reusable GPU/CPU Fock batch executors
        if restricted:
            prog_a = _build_fock_programs(oa)
            ctx_a = run.batch(prog_a) if hasattr(run, "batch") else nullcontext(lambda inp: run(prog_a, inp))
            ctx_b = nullcontext(None)
        else:
            prog_a = _build_fock_programs(oa)
            ctx_a = run.batch(prog_a) if hasattr(run, "batch") else nullcontext(lambda inp: run(prog_a, inp))
            if ob > 0:
                prog_b = _build_fock_programs(ob)
                ctx_b = run.batch(prog_b) if hasattr(run, "batch") else nullcontext(lambda inp: run(prog_b, inp))
            else:
                ctx_b = nullcontext(None)

    diis_focks: list[tuple[np.ndarray, ...]] = []
    diis_errs: list[np.ndarray] = []
    max_diis = 8

    e_prev = float("inf")
    converged = False
    iters = 0
    fa = fb = H.copy()
    da = db = np.zeros((n, n), dtype=np.float64)
    ea = eb = np.zeros(n, dtype=np.float64)

    with ctx_a as exec_a, ctx_b as exec_b:
        for it in range(1, max_cycles + 1):
            iters = it
            coa = np.ascontiguousarray(ca[:, :oa])
            da = coa @ coa.T
            if restricted:
                d_tot = 2.0 * da
                if aot is not None:
                    j_mat, ka_mat, _ = aot.run_df_jk_fock(
                        B_jk, d_tot, coa, None, restricted=True, q_chunk=q_chunk
                    )
                else:
                    assert padded_chunks is not None and exec_a is not None
                    j_mat = np.zeros((n, n), dtype=np.float64)
                    ka_mat = np.zeros((n, n), dtype=np.float64)
                    for b_ch in padded_chunks:
                        out = exec_a({"B": b_ch, "D": d_tot, "Co": coa}).outputs
                        j_mat += out["J"]
                        ka_mat += out["K"]
                fa = 0.5 * ((H + j_mat - ka_mat) + (H + j_mat - ka_mat).T)
                fb = fa
                db = da
                e_elec = float(np.sum(da * (H + fa)))
                err_a = X.T @ (fa @ da @ S - S @ da @ fa) @ X
                err_vec = err_a.ravel()
                grad_max = float(np.max(np.abs(err_a)))
            else:
                cob = np.ascontiguousarray(cb[:, :ob]) if ob > 0 else np.zeros((n, 0), dtype=np.float64)
                db = cob @ cob.T if ob > 0 else np.zeros((n, n), dtype=np.float64)
                d_tot = da + db
                if aot is not None:
                    j_mat, ka_mat, kb_out = aot.run_df_jk_fock(
                        B_jk, d_tot, coa, cob, restricted=False, q_chunk=q_chunk
                    )
                    assert kb_out is not None
                    kb_mat = kb_out
                else:
                    assert padded_chunks is not None and exec_a is not None
                    j_mat = np.zeros((n, n), dtype=np.float64)
                    ka_mat = np.zeros((n, n), dtype=np.float64)
                    kb_mat = np.zeros((n, n), dtype=np.float64)
                    for b_ch in padded_chunks:
                        out_a = exec_a({"B": b_ch, "D": d_tot, "Co": coa}).outputs
                        j_mat += out_a["J"]
                        ka_mat += out_a["K"]
                        if ob > 0 and exec_b is not None:
                            out_b = exec_b({"B": b_ch, "D": d_tot, "Co": cob}).outputs
                            kb_mat += out_b["K"]
                fa = 0.5 * ((H + j_mat - ka_mat) + (H + j_mat - ka_mat).T)
                fb = 0.5 * ((H + j_mat - kb_mat) + (H + j_mat - kb_mat).T)
                e_elec = 0.5 * float(np.sum(da * (H + fa)) + np.sum(db * (H + fb)))
                err_a = X.T @ (fa @ da @ S - S @ da @ fa) @ X
                err_b = X.T @ (fb @ db @ S - S @ db @ fb) @ X
                err_vec = np.concatenate([err_a.ravel(), err_b.ravel()])
                grad_max = float(max(np.max(np.abs(err_a)), np.max(np.abs(err_b))))

            e_tot = e_nuc + e_elec
            if it > 1 and abs(e_tot - e_prev) < 1e-12 and grad_max < 1e-10:
                ea, ca = _diag(fa)
                eb, cb = (ea, ca) if restricted else _diag(fb)
                converged = True
                break
            e_prev = e_tot

            # Pulay DIIS extrapolation
            diis_focks.append((fa.copy(),) if restricted else (fa.copy(), fb.copy()))
            diis_errs.append(err_vec.copy())
            if len(diis_focks) > max_diis:
                diis_focks.pop(0)
                diis_errs.pop(0)

            m = len(diis_focks)
            if m >= 2:
                b_mat = np.full((m + 1, m + 1), -1.0, dtype=np.float64)
                b_mat[m, m] = 0.0
                for ii in range(m):
                    for jj in range(ii, m):
                        val = float(np.dot(diis_errs[ii], diis_errs[jj]))
                        b_mat[ii, jj] = b_mat[jj, ii] = val
                rhs = np.zeros(m + 1, dtype=np.float64)
                rhs[m] = -1.0
                try:
                    w_diis = np.linalg.solve(b_mat, rhs)[:m]
                    if np.isfinite(w_diis).all() and np.max(np.abs(w_diis)) < 1e4:
                        fa_ext = sum(w_diis[k] * diis_focks[k][0] for k in range(m))
                        fb_ext = fa_ext if restricted else sum(w_diis[k] * diis_focks[k][1] for k in range(m))
                    else:
                        fa_ext, fb_ext = fa, fb
                except np.linalg.LinAlgError:
                    fa_ext, fb_ext = fa, fb
            else:
                fa_ext, fb_ext = fa, fb

            # Apply gentle Fock damping on early unseeded iterations so Aufbau ordering settles
            # before DIIS locks onto the ground-state determinant.
            if seed is None and it <= 2:
                ea, ca = _diag(0.5 * (fa + diis_focks[0][0]) if it == 2 else fa)
                eb, cb = (ea, ca) if restricted else _diag(0.5 * (fb + diis_focks[0][1]) if it == 2 else fb)
            else:
                ea, ca = _diag(fa_ext)
                eb, cb = (ea, ca) if restricted else _diag(fb_ext)

    if not converged:
        raise RuntimeError(
            f"Native focal DF-HF fallback did not converge in {max_cycles} iterations "
            f"(grad_max={grad_max:.3e}, e_tot={e_tot:.10f})"
        )

    # Final exact Fock build & diagonalization check on converged orbitals
    coa = np.ascontiguousarray(ca[:, :oa])
    da = coa @ coa.T
    if restricted:
        coeff = ca[None, :, :]
        eps_out = ea[None, :]
        dens = (2.0 * da)[None, :, :]
        fock_out = fa[None, :, :]
        occ_tuple: tuple[int, ...] = (oa,)
        canon_err = float(np.max(np.abs(ca.T @ fa @ ca - np.diag(ea))))
    else:
        cob = np.ascontiguousarray(cb[:, :ob]) if ob > 0 else np.zeros((n, 0), dtype=np.float64)
        db = cob @ cob.T if ob > 0 else np.zeros((n, n), dtype=np.float64)
        coeff = np.stack([ca, cb], axis=0)
        eps_out = np.stack([ea, eb], axis=0)
        dens = np.stack([da, db], axis=0)
        fock_out = np.stack([fa, fb], axis=0)
        occ_tuple = (oa, ob)
        canon_err = float(
            max(
                np.max(np.abs(ca.T @ fa @ ca - np.diag(ea))),
                np.max(np.abs(cb.T @ fb @ cb - np.diag(eb))),
            )
        )
    if not np.isfinite(canon_err) or canon_err > 1e-8:
        raise RuntimeError(f"Native focal DF-HF canonical residual {canon_err:.3e} exceeds 1e-8")
    del B_jk, padded_chunks

    checked_overlaps = (
        seed._checked_overlaps(seed_cross, coeff, occ_tuple)
        if seed is not None and seed_cross is not None
        else None
    )
    return FocalReference(
        float(e_tot),
        None,
        S,
        H,
        coeff,
        eps_out,
        dens,
        fock_out,
        occ_tuple,
        int(iters),
        request.backend,
        request.surface_id,
        request.basis.basis_id,
        request.basis.orbital.identity,
        request.basis.jk_fit.identity,
        occupied_overlaps=checked_overlaps,
        execution_mode="hybrid_host_assisted" if request.backend == "cuda" else "cpu_reference",
    )


def _mp2_factors_q_chunked(
    ao_coulomb: np.ndarray,
    whitener: np.ndarray,
    coefficients: np.ndarray,
    occupied: int,
    frozen: int,
    run,
    *,
    max_chunk_elements: int = 50_000_000,
    aot=None,
):
    """Build whitened MP2Fit B[Q, o_act, v] in Q-chunks and return (B, reverse_fn)."""
    q, n, _ = ao_coulomb.shape
    o_act = occupied - frozen
    n_vir = n - occupied
    co = np.ascontiguousarray(coefficients[:, frozen:occupied], dtype=np.float64)
    cv = np.ascontiguousarray(coefficients[:, occupied:], dtype=np.float64)
    q_chunk = max(1, min(q, max_chunk_elements // max(1, n * n)))

    if aot is not None:
        B = aot.run_mp2_factors_fwd(ao_coulomb, whitener, co, cv, q_chunk=q_chunk)

        def reverse_aot(bar_B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            bar_co, bar_cv, bar_ao, metric_bar = aot.run_mp2_factors_rev(
                ao_coulomb, whitener, co, cv, B, bar_B, q_chunk=q_chunk
            )
            bar_c = np.zeros_like(coefficients)
            bar_c[:, frozen:occupied] = bar_co
            bar_c[:, occupied:] = bar_cv
            return bar_c, bar_ao, metric_bar

        return B, reverse_aot

    from contextlib import nullcontext

    from generativeqc_compiler.tensor import (
        Index,
        IndexSpace,
        Program,
        TensorSpec,
        add,
        einsum,
        input_tensor,
        transpose,
        transpose_program,
    )

    a_ch = Index("A", IndexSpace("mp2_qchunk", "auxiliary", q_chunk))
    ao_sp = IndexSpace("ao_obs", "ao", n)
    u = Index("u", ao_sp)
    v = Index("v", ao_sp)
    i_idx = Index("i", IndexSpace("occ_act", "occupied", o_act))
    vir_idx = Index("a", IndexSpace("vir", "virtual", n_vir))

    raw_in = input_tensor("ao_chunk", TensorSpec((a_ch, u, v), role="input", differentiable=True))
    co_in = input_tensor("co", TensorSpec((u, i_idx), role="input", differentiable=True))
    cv_in = input_tensor("cv", TensorSpec((v, vir_idx), role="input", differentiable=True))
    mo_raw_prog = Program(
        {"mo_raw": einsum("Aiv,va->Aia", einsum("ui,Auv->Aiv", co_in, raw_in), cv_in)}
    )

    mo_raw = np.empty((q, o_act, n_vir), dtype=np.float64)
    batch_ctx = run.batch(mo_raw_prog) if hasattr(run, "batch") else nullcontext(lambda inp: run(mo_raw_prog, inp))
    with batch_ctx as exec_one:
        for q0 in range(0, q, q_chunk):
            q1 = min(q, q0 + q_chunk)
            if q1 - q0 == q_chunk:
                chunk_arr = np.ascontiguousarray(ao_coulomb[q0:q1])
            else:
                chunk_arr = np.zeros((q_chunk, n, n), dtype=np.float64)
                chunk_arr[: q1 - q0] = ao_coulomb[q0:q1]
            out = exec_one({"ao_chunk": chunk_arr, "co": co, "cv": cv}).outputs["mo_raw"]
            mo_raw[q0:q1] = out[: q1 - q0]

    # Whiten B = W @ mo_raw
    aux_sp = IndexSpace("mp2_full", "auxiliary", q)
    a_full = Index("A", aux_sp)
    b_full = Index("B", aux_sp)
    w_in = input_tensor("W", TensorSpec((a_full, b_full), role="input"))
    m_in = input_tensor("mo_raw", TensorSpec((b_full, i_idx, vir_idx), role="input"))
    whiten_prog = Program({"B": einsum("AB,Bia->Aia", w_in, m_in)})
    B = run(whiten_prog, {"W": whitener, "mo_raw": mo_raw}).outputs["B"]
    del mo_raw

    def reverse(bar_B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seed_in = input_tensor("bar_B", TensorSpec((a_full, i_idx, vir_idx), role="input"))
        fac_in = input_tensor("B", TensorSpec((a_full, i_idx, vir_idx), role="input"))
        weighted = einsum("AB,Aia->Bia", w_in, seed_in)
        fitted = einsum("AB,Aia->Bia", w_in, fac_in)
        nonsym = einsum("Aia,Bia->AB", weighted, fitted)
        metric_bar_expr = add(nonsym, transpose(nonsym, (1, 0)), coefficients=("-1/4", "-1/4"))
        met_prog = Program({"weighted": weighted, "metric_bar": metric_bar_expr})
        met_out = run(met_prog, {"W": whitener, "bar_B": bar_B, "B": B}).outputs
        bar_mo_raw = met_out["weighted"]
        metric_bar = met_out["metric_bar"]

        mo_raw_rev = transpose_program(mo_raw_prog, ("mo_raw",)).program
        bar_ao = np.empty_like(ao_coulomb)
        bar_co = np.zeros_like(co)
        bar_cv = np.zeros_like(cv)
        rev_ctx = run.batch(mo_raw_rev) if hasattr(run, "batch") else nullcontext(lambda inp: run(mo_raw_rev, inp))
        with rev_ctx as exec_rev:
            for q0 in range(0, q, q_chunk):
                q1 = min(q, q0 + q_chunk)
                actual = q1 - q0
                if actual == q_chunk:
                    chunk_arr = np.ascontiguousarray(ao_coulomb[q0:q1])
                    bar_chunk = np.ascontiguousarray(bar_mo_raw[q0:q1])
                else:
                    chunk_arr = np.zeros((q_chunk, n, n), dtype=np.float64)
                    chunk_arr[:actual] = ao_coulomb[q0:q1]
                    bar_chunk = np.zeros((q_chunk, o_act, n_vir), dtype=np.float64)
                    bar_chunk[:actual] = bar_mo_raw[q0:q1]
                out = exec_rev(
                    {"ao_chunk": chunk_arr, "co": co, "cv": cv, "bar_mo_raw": bar_chunk}
                ).outputs
                bar_ao[q0:q1] = out["bar_ao_chunk"][:actual]
                bar_co += out["bar_co"]
                bar_cv += out["bar_cv"]
        bar_c = np.zeros_like(coefficients)
        bar_c[:, frozen:occupied] = bar_co
        bar_c[:, occupied:] = bar_cv
        return bar_c, bar_ao, metric_bar

    return B, reverse


def _build_tiled_mp2_pair_program(
    *,
    same: bool,
    q: int,
    o_l: int,
    v_l: int,
    j_tile: int,
    v_r: int,
    need_cotangents: bool,
    compute_right_fo: bool = False,
) -> Program:
    """Build one j-tiled DF-MP2 pair energy (and optional analytic cotangent) TensorIR graph."""
    from generativeqc_compiler.tensor import (
        Index,
        IndexSpace,
        Program,
        TensorSpec,
        add,
        broadcast,
        divide,
        einsum,
        input_tensor,
        multiply,
        reduce_sum,
        transpose,
    )

    q_i = Index("Q", IndexSpace("mp2fit", "auxiliary", q))
    occ_l_sp = IndexSpace("occ_l", "occupied", o_l)
    occ_rj_sp = IndexSpace("occ_rj", "occupied", j_tile)
    vir_l_sp = IndexSpace("vir_l", "virtual", v_l)
    vir_r_sp = vir_l_sp if same else IndexSpace("vir_r", "virtual", v_r)
    i_i = Index("i", occ_l_sp)
    j_i = Index("j", occ_rj_sp)
    a_i = Index("a", vir_l_sp)
    b_i = Index("b", vir_r_sp)

    B_l = input_tensor("B_l", TensorSpec((q_i, i_i, a_i), role="input"))
    B_rj = input_tensor("B_rj", TensorSpec((q_i, j_i, b_i), role="input"))
    eo_l = input_tensor("eo_l", TensorSpec((i_i,), role="input"))
    eo_rj = input_tensor("eo_rj", TensorSpec((j_i,), role="input"))
    ev_l = input_tensor("ev_l", TensorSpec((a_i,), role="input"))
    ev_r = input_tensor("ev_r", TensorSpec((b_i,), role="input"))

    direct = einsum("Qia,Qjb->ijab", B_l, B_rj)
    denom = add(
        *(broadcast(e, direct.spec.indices, (ax,)) for ax, e in enumerate((eo_l, eo_rj, ev_l, ev_r))),
        coefficients=(1, 1, -1, -1),
    )
    num = add(direct, transpose(direct, (0, 1, 3, 2)), coefficients=(1, -1)) if same else direct
    amp = divide(num, denom)
    w_str = "1/4" if same else 1
    pair_e = add(reduce_sum(multiply(amp, num), (2, 3)), coefficients=(w_str,))
    if not need_cotangents:
        return Program({"pair_e": pair_e})

    mask_ij = input_tensor("mask_ij", TensorSpec((i_i, j_i), role="input"))
    t_masked = multiply(amp, broadcast(mask_ij, amp.spec.indices, (0, 1)))
    f_w = 1 if same else 2
    bar_B_l = add(einsum("ijab,Qjb->Qia", t_masked, B_rj), coefficients=(f_w,))
    bar_B_rj = add(einsum("ijab,Qia->Qjb", t_masked, B_l), coefficients=(f_w,))
    v_w = "1/4" if same else 1
    bar_Fv_l = add(einsum("ijab,ijcb->ac", t_masked, t_masked), coefficients=(v_w,))
    bar_Fv_r = add(einsum("ijab,ijac->bc", t_masked, t_masked), coefficients=(v_w,))

    # Full-tile Fo contraction; caller zeroes [shift_l:, shift_l:] for valence j-tiles!
    # Mask padded tail along j using j_valid (shape (j_tile,)).
    j_valid = input_tensor("j_valid", TensorSpec((j_i,), role="input"))
    t_valid = multiply(amp, broadcast(j_valid, amp.spec.indices, (1,)))
    o_w = "-1/4" if same else -1
    bar_Fo_l = add(einsum("ijab,kjab->ik", t_valid, t_valid), coefficients=(o_w,))

    outputs = {
        "pair_e": pair_e,
        "bar_B_l": bar_B_l,
        "bar_B_rj": bar_B_rj,
        "bar_Fv_l": bar_Fv_l,
        "bar_Fv_r": bar_Fv_r,
        "bar_Fo_l": bar_Fo_l,
    }
    if compute_right_fo:
        outputs["bar_Fo_rj"] = add(einsum("ijab,ilab->jl", t_valid, t_valid), coefficients=(o_w,))
    return Program(outputs)


def _tiled_mp2_delta_cv_pairs(
    ref: FocalReference,
    spin_indices: tuple[int, int],
    cv_frozen: tuple[int, int],
    fc_frozen: tuple[int, int],
    B_by_spin: dict[str, np.ndarray],
    run,
    *,
    need_cotangents: bool,
    max_tile_elements: int = 40_000_000,
    aot=None,
    force_uhf_path: bool = False,
):
    """Evaluate exact cv-active and frozen-core DF-MP2 energies (and optional Delta-CV cotangents)."""
    from contextlib import nullcontext

    n = len(ref.hcore)
    spins = ("alpha", "beta")
    occs = tuple(ref.occupied[k] for k in spin_indices)
    cv_act = tuple(o - f for o, f in zip(occs, cv_frozen, strict=True))
    shifts = tuple(fc - cv for fc, cv in zip(fc_frozen, cv_frozen, strict=True))
    restricted_ref = len(ref.occupied) == 1
    restricted = restricted_ref and (not force_uhf_path)

    eos = {
        s: np.ascontiguousarray(ref.orbital_energies[spin_indices[k]][cv_frozen[k] : occs[k]])
        for k, s in enumerate(spins)
    }
    evs = {
        s: np.ascontiguousarray(ref.orbital_energies[spin_indices[k]][occs[k] :])
        for k, s in enumerate(spins)
    }
    bar_B_delta = {s: np.zeros_like(B_by_spin[s]) for s in B_by_spin} if need_cotangents else {}
    fock_seeds_delta = {f"fock_{s}": np.zeros((n, n), dtype=np.float64) for s in spins} if need_cotangents else {}

    if aot is not None:
        va, vb = n - occs[0], n - occs[1]
        max_ol = max(cv_act[0], cv_act[1], 1)
        max_vv = max(va * va, va * vb, vb * vb, 1)
        tile_limit = max(1, max_tile_elements // max(1, max_ol * max_vv))
        cv_e, fc_e, bar_Ba, bar_Bb, bar_Fo_a, bar_Fo_b, bar_Fv_a, bar_Fv_b = aot.run_mp2_pairs(
            restricted=restricted,
            need_cotangents=need_cotangents,
            B_a=B_by_spin.get("alpha"),
            B_b=None if restricted else B_by_spin.get("beta"),
            eo_a=eos["alpha"],
            eo_b=None if restricted else eos["beta"],
            ev_a=evs["alpha"],
            ev_b=None if restricted else evs["beta"],
            sh_a=shifts[0],
            sh_b=shifts[1],
            tile_limit=tile_limit,
        )
        if need_cotangents:
            fa, fb = cv_frozen[0], cv_frozen[1]
            oa_tot, ob_tot = occs[0], occs[1]
            if restricted:
                if bar_Ba is not None and "alpha" in bar_B_delta:
                    bar_B_delta["alpha"][:] = bar_Ba
                if bar_Fo_a is not None:
                    fock_seeds_delta["fock_alpha"][fa:oa_tot, fa:oa_tot] = bar_Fo_a
                    fock_seeds_delta["fock_beta"][fa:oa_tot, fa:oa_tot] = bar_Fo_a
                if bar_Fv_a is not None:
                    fock_seeds_delta["fock_alpha"][oa_tot:, oa_tot:] = bar_Fv_a
                    fock_seeds_delta["fock_beta"][oa_tot:, oa_tot:] = bar_Fv_a
            else:
                if bar_Ba is not None and "alpha" in bar_B_delta:
                    bar_B_delta["alpha"][:] = bar_Ba
                if bar_Bb is not None and "beta" in bar_B_delta:
                    bar_B_delta["beta"][:] = bar_Bb
                if restricted_ref and "alpha" in bar_B_delta and "beta" in bar_B_delta:
                    bar_B_delta["alpha"] = bar_B_delta["alpha"] + bar_B_delta["beta"]
                if bar_Fo_a is not None:
                    fock_seeds_delta["fock_alpha"][fa:oa_tot, fa:oa_tot] = bar_Fo_a
                if bar_Fo_b is not None:
                    fock_seeds_delta["fock_beta"][fb:ob_tot, fb:ob_tot] = bar_Fo_b
                if bar_Fv_a is not None:
                    fock_seeds_delta["fock_alpha"][oa_tot:, oa_tot:] = bar_Fv_a
                if bar_Fv_b is not None:
                    fock_seeds_delta["fock_beta"][ob_tot:, ob_tot:] = bar_Fv_b
        return cv_e, fc_e, bar_B_delta, fock_seeds_delta

    cv_spin_e = [0.0, 0.0, 0.0]
    fc_spin_e = [0.0, 0.0, 0.0]

    channel_list = ((0, 0), (0, 1)) if restricted else ((0, 0), (0, 1), (1, 1))

    for ch_idx, (il, ir) in enumerate(channel_list):
        sl, sr = spins[il], spins[ir]
        ol, orr = cv_act[il], cv_act[ir]
        vl, vr = n - occs[il], n - occs[ir]
        sh_l, sh_r = shifts[il], shifts[ir]
        same = il == ir
        if ol == 0 or orr == 0 or (same and ol < 2):
            continue
        Bl = B_by_spin[sl]
        Br = B_by_spin[sr]
        q = Bl.shape[0]

        # Split j-range into core [0, sh_r) and valence [sh_r, orr) tiles so every j in a tile
        # shares identical core/valence status for exact Delta-CV Fo cancellation.
        j_tile = max(1, min(orr, max_tile_elements // max(1, ol * vl * vr)))
        single_tile_ab = need_cotangents and (not same) and (not restricted) and (j_tile == orr) and (sh_l == 0 or False)
        # Keep right Fo exact even for UHF alpha_beta by doing a clean symmetric pass when needed
        prog = _build_tiled_mp2_pair_program(
            same=same,
            q=q,
            o_l=ol,
            v_l=vl,
            j_tile=j_tile,
            v_r=vr,
            need_cotangents=need_cotangents,
            compute_right_fo=False,
        )

        # Build tile intervals (start, stop, is_core_j)
        tile_specs: list[tuple[int, int, bool]] = []
        for j0 in range(0, sh_r, j_tile):
            tile_specs.append((j0, min(sh_r, j0 + j_tile), True))
        for j0 in range(sh_r, orr, j_tile):
            tile_specs.append((j0, min(orr, j0 + j_tile), False))

        e_cv_acc = 0.0
        e_fc_acc = 0.0
        batch_ctx = run.batch(prog) if hasattr(run, "batch") else nullcontext(lambda inp: run(prog, inp))
        with batch_ctx as exec_one:
            for j0, j1, is_core_j in tile_specs:
                actual = j1 - j0
                if actual == j_tile:
                    Br_tile = np.ascontiguousarray(Br[:, j0:j1, :])
                    eo_rj = np.ascontiguousarray(eos[sr][j0:j1])
                else:
                    Br_tile = np.zeros((q, j_tile, vr), dtype=np.float64)
                    Br_tile[:, :actual, :] = Br[:, j0:j1, :]
                    eo_rj = np.zeros(j_tile, dtype=np.float64)
                    eo_rj[:actual] = eos[sr][j0:j1]

                inp: dict[str, np.ndarray] = {
                    "B_l": Bl,
                    "B_rj": Br_tile,
                    "eo_l": eos[sl],
                    "eo_rj": eo_rj,
                    "ev_l": evs[sl],
                    "ev_r": evs[sr],
                }
                if need_cotangents:
                    mask_ij = np.zeros((ol, j_tile), dtype=np.float64)
                    if is_core_j:
                        mask_ij[:, :actual] = 1.0
                    else:
                        mask_ij[:sh_l, :actual] = 1.0
                    j_valid = np.zeros(j_tile, dtype=np.float64)
                    j_valid[:actual] = 1.0
                    inp["mask_ij"] = mask_ij
                    inp["j_valid"] = j_valid

                out = exec_one(inp).outputs
                pe = out["pair_e"][:, :actual]
                e_cv_acc += float(np.sum(pe))
                if not is_core_j:
                    e_fc_acc += float(np.sum(pe[sh_l:, :]))

                if need_cotangents:
                    fl = cv_frozen[il]
                    fr = cv_frozen[ir]
                    ol_tot = occs[il]
                    or_tot = occs[ir]
                    fo_l = out["bar_Fo_l"].copy()
                    if not is_core_j:
                        fo_l[sh_l:, sh_l:] = 0.0

                    if restricted:
                        # For RHF: alpha_alpha and beta_beta are identical (total 2x on B_alpha);
                        # alpha_beta is symmetric under alpha<->beta exchange.
                        if same:
                            bar_B_delta["alpha"] += 2.0 * out["bar_B_l"]
                            bar_B_delta["alpha"][:, j0:j1, :] += 2.0 * out["bar_B_rj"][:, :actual, :]
                            fock_seeds_delta["fock_alpha"][fl:ol_tot, fl:ol_tot] += 2.0 * fo_l
                            fock_seeds_delta["fock_alpha"][ol_tot:, ol_tot:] += out["bar_Fv_l"] + out["bar_Fv_r"]
                            fock_seeds_delta["fock_beta"][fl:ol_tot, fl:ol_tot] += 2.0 * fo_l
                            fock_seeds_delta["fock_beta"][ol_tot:, ol_tot:] += out["bar_Fv_l"] + out["bar_Fv_r"]
                        else:
                            bar_B_delta["alpha"] += out["bar_B_l"]
                            bar_B_delta["alpha"][:, j0:j1, :] += out["bar_B_rj"][:, :actual, :]
                            fock_seeds_delta["fock_alpha"][fl:ol_tot, fl:ol_tot] += fo_l
                            fock_seeds_delta["fock_beta"][fr:or_tot, fr:or_tot] += fo_l
                            fock_seeds_delta["fock_alpha"][ol_tot:, ol_tot:] += out["bar_Fv_l"]
                            fock_seeds_delta["fock_beta"][or_tot:, or_tot:] += out["bar_Fv_r"]
                    else:
                        bar_B_delta[sl] += out["bar_B_l"]
                        bar_B_delta[sr][:, j0:j1, :] += out["bar_B_rj"][:, :actual, :]
                        fock_seeds_delta[f"fock_{sl}"][fl:ol_tot, fl:ol_tot] += (2.0 if same else 1.0) * fo_l
                        fock_seeds_delta[f"fock_{sl}"][ol_tot:, ol_tot:] += out["bar_Fv_l"]
                        fock_seeds_delta[f"fock_{sr}"][or_tot:, or_tot:] += out["bar_Fv_r"]

        cv_spin_e[ch_idx] = e_cv_acc
        fc_spin_e[ch_idx] = e_fc_acc
        if restricted and same:
            # Mirror alpha_alpha into beta_beta energy slot
            cv_spin_e[2] = e_cv_acc
            fc_spin_e[2] = e_fc_acc

        # For UHF opposite-spin (alpha_beta) E+F, also compute exact bar_Fo_beta via symmetric i-tiled pass
        if need_cotangents and (not restricted) and (not same):
            i_tile = max(1, min(ol, max_tile_elements // max(1, orr * vr * vl)))
            prog_ba = _build_tiled_mp2_pair_program(
                same=False,
                q=q,
                o_l=orr,
                v_l=vr,
                j_tile=i_tile,
                v_r=vl,
                need_cotangents=True,
                compute_right_fo=False,
            )
            itile_specs: list[tuple[int, int, bool]] = []
            for i0 in range(0, sh_l, i_tile):
                itile_specs.append((i0, min(sh_l, i0 + i_tile), True))
            for i0 in range(sh_l, ol, i_tile):
                itile_specs.append((i0, min(ol, i0 + i_tile), False))
            ba_ctx = run.batch(prog_ba) if hasattr(run, "batch") else nullcontext(lambda inp: run(prog_ba, inp))
            with ba_ctx as exec_ba:
                for i0, i1, is_core_i in itile_specs:
                    actual = i1 - i0
                    if actual == i_tile:
                        Bl_tile = np.ascontiguousarray(Bl[:, i0:i1, :])
                        eo_li = np.ascontiguousarray(eos[sl][i0:i1])
                    else:
                        Bl_tile = np.zeros((q, i_tile, vl), dtype=np.float64)
                        Bl_tile[:, :actual, :] = Bl[:, i0:i1, :]
                        eo_li = np.zeros(i_tile, dtype=np.float64)
                        eo_li[:actual] = eos[sl][i0:i1]
                    mask_ji = np.zeros((orr, i_tile), dtype=np.float64)
                    if is_core_i:
                        mask_ji[:, :actual] = 1.0
                    else:
                        mask_ji[:sh_r, :actual] = 1.0
                    i_valid = np.zeros(i_tile, dtype=np.float64)
                    i_valid[:actual] = 1.0
                    out_ba = exec_ba(
                        {
                            "B_l": Br,
                            "B_rj": Bl_tile,
                            "eo_l": eos[sr],
                            "eo_rj": eo_li,
                            "ev_l": evs[sr],
                            "ev_r": evs[sl],
                            "mask_ij": mask_ji,
                            "j_valid": i_valid,
                        }
                    ).outputs
                    fo_r = out_ba["bar_Fo_l"].copy()
                    if not is_core_i:
                        fo_r[sh_r:, sh_r:] = 0.0
                    fr = cv_frozen[ir]
                    or_tot = occs[ir]
                    fock_seeds_delta[f"fock_{sr}"][fr:or_tot, fr:or_tot] += fo_r

    return tuple(cv_spin_e), tuple(fc_spin_e), bar_B_delta, fock_seeds_delta


def molecular_delta_cv_mp2_energy(
    provider: MolecularFocalProvider,
    request_cv: FocalRequest,
    request_fc: FocalRequest | None = None,
) -> FocalDeltaCVResult:
    """Evaluate pure conventional DF-MP2 Delta-CV = E_MP2(cv-active) - E_MP2(frozen-core).

    Zero CABS, zero F12, reusing exact cv-active vs frozen-core orbital counts and a single
    shared DF-HF determinant + single OBS-only MP2Fit Coulomb panel.
    """
    from ._f12_cv_dfmp2_native import bind_delta_cv_native
    from ._f12_focal_one_electron import one_electron_sources
    from ._f12_focal_reference import df_reference
    from ._f12_focal_reference import partition_focal_reference

    req_cv, req_fc = _paired_cv_fc_mp2_requests(request_cv, request_fc)
    budget = _check_conventional_mp2_request(provider, req_cv)
    cuda = req_cv.backend == "cuda"
    run = _checked_runner(provider, cuda, budget)
    aot = bind_delta_cv_native(
        provider.shell_library, host_reference=not cuda, budget_bytes=budget << 20
    )

    seed_options = provider._seed_options(req_cv)
    ref_provenance = "native_df_reference"
    try:
        ref = df_reference(
            req_cv, forces=False, budget_mib=budget, **seed_options
        )
    except NotImplementedError as exc:
        if not _is_eligible_for_focal_source_df_hf_fallback(req_cv, exc):
            raise
        one_fb = one_electron_sources(
            req_cv,
            library=provider.one_electron_library or provider.shell_library,
            derivatives=False,
            orbital_only=True,
            budget_mib=budget,
        )
        jk_fb = provider._panel(req_cv, "jk_fit", orbital_only=True)
        ref = _solve_df_reference_from_focal_sources(
            provider, req_cv, one_fb, jk_fb, run=run, **seed_options
        )
        ref_provenance = "focal_source_df_hf_fallback"

    ref = partition_focal_reference(
        req_cv, ref, budget, additional_space=req_fc.correlation_space_id,
    )
    assert ref.frozen_frame is not None
    cv_frozen = ref.frozen_frame.counts(req_cv.correlation_space_id)
    fc_frozen = ref.frozen_frame.counts(req_fc.correlation_space_id)
    indices = ref._pair_spin_indices(req_cv)
    occs = tuple(ref.occupied[k] for k in indices)

    if cv_frozen == fc_frozen:
        return FocalDeltaCVResult(
            reference=ref,
            cabs_rank=0,
            cv_frozen=cv_frozen,
            fc_frozen=fc_frozen,
            fc_mode=req_fc.correlation_space_id,
            frozen_core_definition_version=FROZEN_CORE_DEFINITION_VERSION,
            cv_ordinary_mp2_Eh=None,
            fc_ordinary_mp2_Eh=None,
            cv_f12_increment_Eh=None,
            fc_f12_increment_Eh=None,
            cv_mp2_f12_Eh=None,
            fc_mp2_f12_Eh=None,
            delta_ordinary_mp2_Eh=0.0,
            delta_f12_increment_Eh=0.0,
            delta_mp2_f12_Eh=0.0,
            reference_provenance=ref_provenance,
        )

    mp2_panel = provider._panel(req_cv, "mp2_fit", orbital_only=True)
    obs = len(ref.hcore)
    mp2_src = mp2_panel.obs(req_cv, obs)

    restricted = len(ref.occupied) == 1
    B_by_spin: dict[str, np.ndarray] = {}
    spins_to_build = (("alpha", 0),) if restricted else (("alpha", 0), ("beta", 1))
    for spin_name, s_idx in spins_to_build:
        if occs[s_idx] - cv_frozen[s_idx] > 0:
            B_s, _ = _mp2_factors_q_chunked(
                mp2_src.three_center,
                mp2_src.whitener,
                ref.coefficients[indices[s_idx]],
                occs[s_idx],
                cv_frozen[s_idx],
                run,
                aot=aot,
            )
            B_by_spin[spin_name] = B_s
    if restricted and "alpha" in B_by_spin:
        B_by_spin["beta"] = B_by_spin["alpha"]

    cv_spins, fc_spins, _, _ = _tiled_mp2_delta_cv_pairs(
        ref, indices, cv_frozen, fc_frozen, B_by_spin, run, need_cotangents=False, aot=aot
    )
    cv_ord = fsum(cv_spins)
    fc_ord = fsum(fc_spins)
    d_ord = cv_ord - fc_ord
    if not all(isfinite(v) for v in (cv_ord, fc_ord, d_ord)):
        raise ArithmeticError("nonfinite molecular DF-MP2 Delta-CV energy")

    return FocalDeltaCVResult(
        reference=ref,
        cabs_rank=0,
        cv_frozen=cv_frozen,
        fc_frozen=fc_frozen,
        fc_mode=req_fc.correlation_space_id,
        frozen_core_definition_version=FROZEN_CORE_DEFINITION_VERSION,
        cv_ordinary_mp2_Eh=float(cv_ord),
        fc_ordinary_mp2_Eh=float(fc_ord),
        cv_f12_increment_Eh=0.0,
        fc_f12_increment_Eh=0.0,
        cv_mp2_f12_Eh=float(cv_ord),
        fc_mp2_f12_Eh=float(fc_ord),
        delta_ordinary_mp2_Eh=float(d_ord),
        delta_f12_increment_Eh=0.0,
        delta_mp2_f12_Eh=float(d_ord),
        reference_provenance=ref_provenance,
    )


def molecular_delta_cv_mp2_energy_and_forces(
    provider: MolecularFocalProvider,
    request_cv: FocalRequest,
    request_fc: FocalRequest | None = None,
) -> FocalDeltaCVResult:
    """Evaluate pure conventional DF-MP2 Delta-CV energy and exact analytic 3N forces.

    Zero CABS, zero F12. Exploits exact linearity of the DF-HF orbital response, canonical
    frame response, and nuclear derivative sinks to solve a single matrix-free Z-vector
    equation and contract nuclear integral derivatives once for Delta = cv-active - frozen-core.
    """
    from ._canonical_frame import canonical_frame_seed
    from ._f12_cv_dfmp2_native import bind_delta_cv_native
    from ._f12_focal_increment import _canonical_occupied_virtual_blocks
    from ._f12_focal_one_electron import one_electron_sources
    from ._f12_focal_reference import df_reference
    from ._f12_focal_reference import partition_focal_reference

    req_cv, req_fc = _paired_cv_fc_mp2_requests(request_cv, request_fc)
    budget = _check_conventional_mp2_request(provider, req_cv)
    cuda = req_cv.backend == "cuda"
    run = _checked_runner(provider, cuda, budget)
    aot = bind_delta_cv_native(
        provider.shell_library, host_reference=not cuda, budget_bytes=budget << 20
    )

    seed_options = provider._seed_options(req_cv)
    ref_provenance = "native_df_reference"
    try:
        ref = df_reference(
            req_cv, forces=False, budget_mib=budget, **seed_options
        )
    except NotImplementedError as exc:
        if not _is_eligible_for_focal_source_df_hf_fallback(req_cv, exc):
            raise
        ref = None

    one = one_electron_sources(
        req_cv,
        library=provider.one_electron_library or provider.shell_library,
        derivatives=True,
        orbital_only=True,
        budget_mib=budget,
    )
    jk = provider._panel(req_cv, "jk_fit", orbital_only=True)
    if ref is None:
        ref = _solve_df_reference_from_focal_sources(
            provider, req_cv, one, jk, run=run, **seed_options
        )
        ref_provenance = "focal_source_df_hf_fallback"

    ref = partition_focal_reference(
        req_cv, ref, budget, additional_space=req_fc.correlation_space_id,
    )
    assert ref.frozen_frame is not None
    cv_frozen = ref.frozen_frame.counts(req_cv.correlation_space_id)
    fc_frozen = ref.frozen_frame.counts(req_fc.correlation_space_id)
    indices = ref._pair_spin_indices(req_cv)
    occs = tuple(ref.occupied[k] for k in indices)

    if cv_frozen == fc_frozen:
        zeros = np.zeros((len(req_cv.atoms), 3), dtype=np.float64)
        return FocalDeltaCVResult(
            reference=ref,
            cabs_rank=0,
            cv_frozen=cv_frozen,
            fc_frozen=fc_frozen,
            fc_mode=req_fc.correlation_space_id,
            frozen_core_definition_version=FROZEN_CORE_DEFINITION_VERSION,
            cv_ordinary_mp2_Eh=None,
            fc_ordinary_mp2_Eh=None,
            cv_f12_increment_Eh=None,
            fc_f12_increment_Eh=None,
            cv_mp2_f12_Eh=None,
            fc_mp2_f12_Eh=None,
            delta_ordinary_mp2_Eh=0.0,
            delta_f12_increment_Eh=0.0,
            delta_mp2_f12_Eh=0.0,
            cv_forces=None,
            fc_forces=None,
            delta_forces=zeros,
            reference_provenance=ref_provenance,
        )

    mp2 = (
        jk
        if jk.fit_id == req_cv.basis.mp2_fit.identity
        else provider._panel(req_cv, "mp2_fit", orbital_only=True)
    )
    n = one.obs_ao_count
    jk_src = jk.obs(req_cv, n)
    mp2_src = mp2.obs(req_cv, n)

    restricted = len(ref.occupied) == 1
    spins = ("alpha", "beta")
    B_by_spin: dict[str, np.ndarray] = {}
    rev_by_spin: dict[str, object] = {}
    spins_to_build = (("alpha", 0),) if restricted else (("alpha", 0), ("beta", 1))
    for spin_name, s_idx in spins_to_build:
        if occs[s_idx] - cv_frozen[s_idx] > 0:
            B_s, rev_s = _mp2_factors_q_chunked(
                mp2_src.three_center,
                mp2_src.whitener,
                ref.coefficients[indices[s_idx]],
                occs[s_idx],
                cv_frozen[s_idx],
                run,
                aot=aot,
            )
            B_by_spin[spin_name] = B_s
            rev_by_spin[spin_name] = rev_s
    if restricted and "alpha" in B_by_spin:
        B_by_spin["beta"] = B_by_spin["alpha"]

    cv_spins, fc_spins, bar_B_delta, fock_seeds_delta = _tiled_mp2_delta_cv_pairs(
        ref, indices, cv_frozen, fc_frozen, B_by_spin, run, need_cotangents=True, aot=aot
    )

    # Reverse Q-chunked MP2Fit factor build -> direct coefficient, 3c, and metric cotangents
    cbar = np.zeros_like(ref.coefficients)
    mp2_raw_bar = np.zeros_like(mp2_src.three_center)
    mp2_metric_bar = np.zeros_like(mp2_src.metric)
    for spin_name, s_idx in spins_to_build:
        if spin_name not in rev_by_spin:
            continue
        rev_fn = rev_by_spin[spin_name]
        bc, bao, bmet = rev_fn(bar_B_delta[spin_name])  # type: ignore[operator]
        cbar[indices[s_idx]] += bc
        mp2_raw_bar += bao
        mp2_metric_bar += bmet
    del B_by_spin, rev_by_spin, bar_B_delta

    if aot is not None:
        ca = np.ascontiguousarray(ref.coefficients[0], dtype=np.float64)
        cb = None if restricted else np.ascontiguousarray(ref.coefficients[indices[1]], dtype=np.float64)
        oa, ob = occs
        df_a, df_b, fock_a, fock_b = aot.run_jk_mo_fock_fwd(
            jk_src.three_center,
            ref.hcore,
            jk_src.whitener,
            ca,
            cb,
            oa,
            ob,
            restricted=restricted,
        )
        fock_by_spin = {"alpha": fock_a, "beta": fock_a if restricted else fock_b}
        for k, spin in enumerate(spins):
            if not np.allclose(
                fock_by_spin[spin],
                np.diag(ref.orbital_energies[indices[k]]),
                rtol=0,
                atol=2e-8,
            ):
                raise ValueError("JKFit does not reproduce the canonical physical HF Fock")

        def aot_fock_pullback(seeds_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            sa = seeds_map["fock_alpha"]
            sb = (
                None
                if (restricted and np.array_equal(sa, seeds_map["fock_beta"]))
                else seeds_map["fock_beta"]
            )
            bca, bcb, bh, bao, bmet = aot.run_jk_fock_pullback(
                jk_src.three_center,
                ref.hcore,
                jk_src.whitener,
                ca,
                cb,
                df_a,
                df_b,
                sa,
                sb,
                oa,
                ob,
                restricted=restricted,
            )
            return {
                "coefficients_alpha": bca,
                "coefficients_beta": bcb,
                "hcore_ao": bh,
                "ao_coulomb": bao,
                "metric/coulomb": bmet,
            }

        direct_bars = aot_fock_pullback(fock_seeds_delta)
        for k, spin in enumerate(spins):
            idx = indices[k]
            direct_bars["coefficients_" + spin] += cbar[idx] * (0.5 if restricted else 1.0)

        canonical_seeds: dict[str, np.ndarray] = {}
        for k, spin in enumerate(spins):
            idx = indices[k]
            rotation = ref.coefficients[idx].T @ direct_bars["coefficients_" + spin]
            eps, occupied = ref.orbital_energies[idx], occs[k]
            canonical_seeds["fock_" + spin] = (
                canonical_frame_seed(
                    rotation,
                    eps,
                    occupied,
                    maximum_bytes=budget << 20,
                    library=provider.canonical_library,
                )[0]
                if cuda
                else _canonical_occupied_virtual_blocks(rotation, eps, occupied)
            )
        canonical_bars = aot_fock_pullback(canonical_seeds)
        for key in direct_bars:
            direct_bars[key] += canonical_bars[key]

        total_cbar = np.zeros_like(ref.coefficients)
        for k, spin in enumerate(spins):
            total_cbar[indices[k]] += direct_bars["coefficients_" + spin]

        overlap_bar = np.zeros((n, n), dtype=np.float64)
        drivers: list[np.ndarray] = []
        gaps: dict[str, np.ndarray] = {}
        for k, (spin, o) in enumerate(zip(spins, occs, strict=True)):
            idx = indices[k]
            coeff = ref.coefficients[idx]
            eps = ref.orbital_energies[idx]
            bar = total_cbar[idx] * (0.5 if restricted else 1.0)
            rotation = coeff.T @ bar
            torque = rotation - rotation.T
            if any(
                b.size and np.max(np.abs(b)) > 1e-8
                for b in (torque[:o, :o], torque[o:, o:])
            ):
                raise ValueError(
                    "electronic observable fails occupied or virtual orbital gauge invariance"
                )
            mask = np.full((n, n), -0.5, dtype=np.float64)
            mask[:o, o:] = -1.0
            mask[o:, :o] = 0.0
            scaled = mask * rotation
            frame = 0.5 * (scaled + scaled.T)
            weight = coeff @ frame @ coeff.T
            overlap_bar += 0.5 * (weight + weight.T)
            if o:
                drivers.append(torque[o:, :o].ravel())
                gaps[spin] = np.ascontiguousarray(
                    eps[o:, None] - eps[None, :o], dtype=np.float64
                )

        from inspect import signature

        from scipy.sparse.linalg import LinearOperator, minres

        tol_key = "rtol" if "rtol" in signature(minres).parameters else "tol"
        if restricted:
            va = n - oa
            ra = va * oa
            g_rhf = drivers[0]
            if not np.allclose(g_rhf, drivers[1], atol=1e-10, rtol=1e-10):
                raise ValueError("RHF orbital response driver is not spin-symmetric")
            inv_diag = 1.0 / np.maximum(np.abs(gaps["alpha"].ravel()), 1e-2)
            precond = LinearOperator(
                (ra, ra), matvec=lambda x: x * inv_diag, dtype=np.float64
            )

            def rhf_matvec(vector: np.ndarray) -> np.ndarray:
                out_a, _ = aot.run_cphf_matvec(
                    df_a,
                    None,
                    gaps["alpha"],
                    None,
                    vector.reshape(va, oa),
                    None,
                    oa,
                    ob,
                    restricted=True,
                )
                return out_a.ravel()

            z_a, status = minres(
                LinearOperator((ra, ra), matvec=rhf_matvec, dtype=np.float64),
                g_rhf,
                M=precond,
                **{tol_key: 1e-12},
                maxiter=max(128, 4 * ra),
            )
            if status:
                raise RuntimeError(
                    f"Matrix-free RHF DF-HF orbital response did not converge ({status})"
                )
            res_norm = np.linalg.norm(rhf_matvec(z_a) - g_rhf)
            if res_norm > 1e-9 * max(1.0, np.linalg.norm(g_rhf)):
                raise ValueError(
                    f"Matrix-free RHF DF-HF orbital response residual {res_norm:.3e} exceeds tolerance"
                )
            z = np.concatenate((z_a, z_a))
        else:
            driver = np.concatenate(drivers)
            r = driver.size
            gap_vec = np.concatenate(
                [gaps[s].ravel() for s, o in zip(spins, occs, strict=True) if o]
            )
            inv_diag = 1.0 / np.maximum(np.abs(gap_vec), 1e-2)
            precond = LinearOperator(
                (r, r), matvec=lambda x: x * inv_diag, dtype=np.float64
            )
            va, vb = n - oa, n - ob
            ra = va * oa
            empty_a = np.empty((va, 0), dtype=np.float64)
            empty_b = np.empty((vb, 0), dtype=np.float64)

            def uhf_matvec(vector: np.ndarray) -> np.ndarray:
                ta = vector[:ra].reshape(va, oa) if oa else empty_a
                tb = vector[ra:].reshape(vb, ob) if ob else empty_b
                out_a, out_b = aot.run_cphf_matvec(
                    df_a,
                    df_b,
                    gaps.get("alpha", empty_a),
                    gaps.get("beta", empty_b),
                    ta,
                    tb,
                    oa,
                    ob,
                    restricted=False,
                )
                parts = []
                if oa:
                    parts.append(out_a.ravel())
                if ob:
                    assert out_b is not None
                    parts.append(out_b.ravel())
                return np.concatenate(parts)

            z, status = minres(
                LinearOperator((r, r), matvec=uhf_matvec, dtype=np.float64),
                driver,
                M=precond,
                **{tol_key: 1e-12},
                maxiter=max(128, 4 * r),
            )
            if status:
                raise RuntimeError(
                    f"Matrix-free UHF DF-HF orbital response did not converge ({status})"
                )
            res_norm = np.linalg.norm(uhf_matvec(z) - driver)
            if res_norm > 1e-9 * max(1.0, np.linalg.norm(driver)):
                raise ValueError(
                    f"Matrix-free UHF DF-HF orbital response residual {res_norm:.3e} exceeds tolerance"
                )

        p_a = np.zeros((n, n), dtype=np.float64)
        sm_a = np.zeros((n, n), dtype=np.float64)
        offset = 0
        if oa:
            size_a = (n - oa) * oa
            z_a_mat = z[offset : offset + size_a].reshape(n - oa, oa)
            offset += size_a
            p_a[oa:, :oa] = -z_a_mat
            sm_a[oa:, :oa] = z_a_mat * ref.orbital_energies[0][:oa][None, :]
        p_b = None
        sm_b = None
        if not restricted:
            p_b = np.zeros((n, n), dtype=np.float64)
            sm_b = np.zeros((n, n), dtype=np.float64)
            if ob:
                size_b = (n - ob) * ob
                z_b_mat = z[offset : offset + size_b].reshape(n - ob, ob)
                offset += size_b
                p_b[ob:, :ob] = -z_b_mat
                sm_b[ob:, :ob] = z_b_mat * ref.orbital_energies[indices[1]][:ob][None, :]

        bar_ds, bar_dh, bar_dr, bar_dj = aot.run_nuclear_source_weights(
            jk_src.whitener,
            ca,
            cb,
            df_a,
            df_b,
            p_a,
            p_b,
            sm_a,
            sm_b,
            oa,
            ob,
            restricted=restricted,
        )
        del df_a, df_b
        w = {
            "overlap": overlap_bar + bar_ds,
            "hcore": direct_bars["hcore_ao"] + bar_dh,
            "three_center": direct_bars["ao_coulomb"] + bar_dr,
            "metric": direct_bars["metric/coulomb"] + bar_dj,
        }
    else:
        from generativeqc_compiler.method.canonical_frame_response import (
            orbital_rotation_cotangent,
        )

        from ._f12_focal_singles_response import relaxed_reference_weights
        from ._f12_spin_ao import _orbital_sources
        from ._f12_spin_fock import spin_fock, spin_fock_pullback

        # Direct Fock pullback + canonical occ-occ/vir-vir frame response (single pass on Delta!)
        space = (*occs, n, n, len(jk_src.metric))
        ao = {
            "coefficients_alpha": ref.coefficients[0],
            "coefficients_beta": ref.coefficients[indices[1]],
            "hcore_ao": ref.hcore,
            "ao_coulomb": jk_src.three_center,
            "metric/coulomb": jk_src.metric,
            "metric_whitener": jk_src.whitener,
        }
        projected, _ = _orbital_sources(
            space, ao, run, None, projected=None, role="ordinary_fock"
        )
        values = {
            k.removeprefix("source/"): v
            for k, v in projected.items()
            if k.startswith("source/")
        }
        fspace = (*occs, n, len(jk_src.metric))
        focks = spin_fock(fspace, values, run)
        for k, spin in enumerate(spins):
            if not np.allclose(
                focks["fock_" + spin],
                np.diag(ref.orbital_energies[indices[k]]),
                rtol=0,
                atol=2e-8,
            ):
                raise ValueError("JKFit does not reproduce the canonical physical HF Fock")

        def fock_pullback(seeds_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            fseed = spin_fock_pullback(fspace, values, seeds_map, run)
            _, fbars = _orbital_sources(
                space,
                ao,
                run,
                {"source/" + k: v for k, v in fseed.items()},
                projected=projected,
                role="ordinary_fock",
            )
            assert fbars is not None
            return fbars

        direct_bars = fock_pullback(fock_seeds_delta)
        for k, spin in enumerate(spins):
            idx = indices[k]
            direct_bars["coefficients_" + spin] += cbar[idx] * (0.5 if restricted else 1.0)

        # Canonical occ-occ/vir-vir response is ALWAYS required for Delta-CV (fc_frozen > cv_frozen >= 0)
        canonical_seeds = {}
        for k, spin in enumerate(spins):
            idx = indices[k]
            rotation = run(
                orbital_rotation_cotangent(n, n),
                {
                    "coefficients": ref.coefficients[idx],
                    "frame_bar": direct_bars["coefficients_" + spin],
                },
            ).outputs["rotation_gradient"]
            eps, occupied = ref.orbital_energies[idx], occs[k]
            canonical_seeds["fock_" + spin] = (
                canonical_frame_seed(
                    rotation,
                    eps,
                    occupied,
                    maximum_bytes=budget << 20,
                    library=provider.canonical_library,
                )[0]
                if cuda
                else _canonical_occupied_virtual_blocks(rotation, eps, occupied)
            )
        canonical_bars = fock_pullback(canonical_seeds)
        for key in direct_bars:
            direct_bars[key] += canonical_bars[key]

        total_cbar = np.zeros_like(ref.coefficients)
        for k, spin in enumerate(spins):
            total_cbar[indices[k]] += direct_bars["coefficients_" + spin]
        direct = {
            "hcore": direct_bars["hcore_ao"],
            "three_center": direct_bars["ao_coulomb"],
            "metric": direct_bars["metric/coulomb"],
            "obs_coefficients": total_cbar,
        }
        w = relaxed_reference_weights(
            ref,
            values,
            jk_src.whitener,
            direct,
            np.zeros((n, n), dtype=np.float64),
            run,
            provider.response_library,
            budget,
            matrix_free=True,
        )

    jk_weights = {"metric": w["metric"], "ao_coulomb": w["three_center"]}
    mp_weights = {"metric": mp2_metric_bar, "ao_coulomb": mp2_raw_bar}
    if mp2 is jk:
        jk_weights["metric"] = jk_weights["metric"] + mp_weights["metric"]
        jk_weights["ao_coulomb"] = jk_weights["ao_coulomb"] + mp_weights["ao_coulomb"]
        grad = provider._gradient(req_cv, jk, jk_weights, coulomb_only=True)
    else:
        grad = provider._gradient(req_cv, jk, jk_weights, coulomb_only=True)
        grad = grad + provider._gradient(req_cv, mp2, mp_weights, coulomb_only=True)
    if aot is not None:
        assert one.d_overlap is not None and one.d_hcore is not None
        grad = grad + (
            one.d_overlap.reshape(-1, n * n) @ w["overlap"].ravel()
            + one.d_hcore.reshape(-1, n * n) @ w["hcore"].ravel()
        ).reshape(len(req_cv.atoms), 3)
    else:
        grad = grad + one.nuclear_gradient(
            w["overlap"], w["hcore"], run=run, shell_library=provider.shell_library
        )
    delta_forces = -grad

    cv_ord = fsum(cv_spins)
    fc_ord = fsum(fc_spins)
    d_ord = cv_ord - fc_ord
    if not all(isfinite(v) for v in (cv_ord, fc_ord, d_ord)) or not np.isfinite(delta_forces).all():
        raise ArithmeticError("nonfinite molecular DF-MP2 Delta-CV energy or forces")

    return FocalDeltaCVResult(
        reference=ref,
        cabs_rank=0,
        cv_frozen=cv_frozen,
        fc_frozen=fc_frozen,
        fc_mode=req_fc.correlation_space_id,
        frozen_core_definition_version=FROZEN_CORE_DEFINITION_VERSION,
        cv_ordinary_mp2_Eh=float(cv_ord),
        fc_ordinary_mp2_Eh=float(fc_ord),
        cv_f12_increment_Eh=0.0,
        fc_f12_increment_Eh=0.0,
        cv_mp2_f12_Eh=float(cv_ord),
        fc_mp2_f12_Eh=float(fc_ord),
        delta_ordinary_mp2_Eh=float(d_ord),
        delta_f12_increment_Eh=0.0,
        delta_mp2_f12_Eh=float(d_ord),
        cv_forces=None,
        fc_forces=None,
        delta_forces=delta_forces,
        reference_provenance=ref_provenance,
    )
