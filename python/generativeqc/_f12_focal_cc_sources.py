"""Spin-resolved fixed-cusp CC equation inputs at one molecular geometry."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from generativeqc_compiler.common.arrays import immutable

from ._f12_cc_fock_pullback import extended_fock_response
from ._f12_cc_spin_sources import unrestricted_cc_f12_fixed_source_feeds
from ._f12_focal_frame import canonical_pair_frame
from ._f12_spin_ao import _orbital_sources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_molecular import MolecularFocalProvider
    from ._f12_focal_reference import FocalReference
    from ._uccsd_rhf_virtual_pairs import RHFDFLadderOptions
    from ._uccsd_uhf_virtual_ladder import UHFDFLadderOptions
    from .experimental.f12_focal import FocalRequest


@dataclass(frozen=True)
class FocalUCCFixedSources:
    """W_vo, W_vv and C in the returned reference's canonical OBS spin frame.

    feeds contains the six spin orderings per tensor and fixed_geminal_energy
    (V+B-X in Eh). The reference holds the same OBS coefficient matrices, metric,
    occupancy, orbital energies, surface, basis and backend used by the tensors.
    """

    reference: FocalReference
    cabs_rank: int
    f12_fit_id: str
    feeds: dict[str, np.ndarray]
    frozen_occupied: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class FocalUCCEquationSources:
    """Ordinary and fixed-geminal CC inputs for the same canonical DF-HF determinant."""

    fixed: FocalUCCFixedSources
    mp2_fit_id: str
    ordinary: dict[str, np.ndarray]
    rhf_df_ladder: RHFDFLadderOptions | None = None
    virtual_factors: dict[str, np.ndarray] | None = None
    uhf_df_ladder: UHFDFLadderOptions | None = None


@overload
def molecular_ucc_sources(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    *,
    include_ordinary: Literal[False],
    reference_forces: bool = False,
    reference: FocalReference | None = None,
    expected_cabs_rank: int | None = None,
) -> FocalUCCFixedSources: ...


@overload
def molecular_ucc_sources(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    *,
    include_ordinary: Literal[True],
    reference_forces: bool = False,
    reference: FocalReference | None = None,
    expected_cabs_rank: int | None = None,
) -> FocalUCCEquationSources: ...


def molecular_ucc_sources(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    *,
    include_ordinary: bool,
    reference_forces: bool = False,
    reference: FocalReference | None = None,
    expected_cabs_rank: int | None = None,
) -> FocalUCCFixedSources | FocalUCCEquationSources:
    """Calculate fixed-geminal UCC equation inputs on an all-active singlet RHF or UHF root."""
    from generativeqc_compiler.tensor import execute

    from ._f12_focal_one_electron import one_electron_sources
    from ._f12_focal_reference import df_reference
    from ._f12_frozen_core import resolve_frozen_occupied

    if (
        request.role != "mp2_f12_correlation"
        or request.root_id not in ("rhf", "uhf")
        or (request.root_id == "rhf" and request.multiplicity != 1)
    ):
        raise ValueError(
            "fixed-cusp UCC sources require a UHF or singlet RHF molecular pair request"
        )
    if (
        request.basis.f12_fit is None
        or request.basis.beta is None
        or request.basis.cabs is None
    ):
        raise ValueError(
            "fixed-cusp UCC sources require F12Fit, OptRI and a Gaussian exponent"
        )
    if request.backend not in ("cuda", "cpu_reference"):
        raise ValueError("fixed-cusp UCC sources require an explicit device backend")
    if include_ordinary and request.basis.mp2_fit is None:
        raise ValueError(
            "ordinary UCC Hamiltonian inputs require an explicit correlation fitting basis"
        )
    cuda = request.backend == "cuda"
    if cuda and (
        provider.run is None
        or provider.metric_library is None
        or provider.response_library is None
    ):
        raise ValueError(
            "CUDA fixed-cusp UCC sources require the device graph, metric and projector providers"
        )
    executor = provider.run or partial(execute, max_bytes=provider.budget_mib << 20)

    def checked(graph: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        out = executor(graph, inputs)
        if out.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError("UCC fixed-source graph executed on a different backend")
        return out

    ref = reference
    if ref is None:
        ref = df_reference(
            request,
            forces=reference_forces,
            budget_mib=provider.budget_mib,
            **provider._seed_options(request),
        )
    elif (
        ref.surface_id != request.surface_id
        or ref.basis_id != request.basis.basis_id
        or ref.executed_backend != request.backend
        or ref.orbital_id != request.basis.orbital.identity
        or ref.jk_fit_id != request.basis.jk_fit.identity
        or (reference_forces and ref.forces is None)
    ):
        raise ValueError("CC equation sources require the matching canonical reference")
    one = one_electron_sources(
        request,
        library=provider.one_electron_library or provider.shell_library,
        derivatives=False,
        budget_mib=provider.budget_mib,
    )
    corr = provider._panel(request, "f12_fit", allow_streaming=True)
    jk = (
        corr
        if (corr.fit_id == request.basis.jk_fit.identity and not corr.streaming)
        else provider._panel(request, "jk_fit", allow_streaming=False)
    )
    space, ao, panel, proj = canonical_pair_frame(
        request,
        ref,
        one,
        jk.joint(request, explicit=False),
        corr.joint(request, explicit=True),
        expected_cabs_rank=expected_cabs_rank,
        relative_cutoff=provider.relative_cutoff,
        library=provider.response_library,
    )
    data, _ = _orbital_sources(space, ao, checked, None, projected=None, role="pairs")
    oa, ob, obs, ext, _ = space
    fock = extended_fock_response(
        space, ao, panel, checked, cuda=cuda, library=provider.response_library
    )[0]
    spin_indices = ref._pair_spin_indices(request)
    for spin, k in zip(("alpha", "beta"), spin_indices, strict=True):
        if not np.allclose(
            fock["fock_" + spin][:obs, :obs],
            np.diag(ref.orbital_energies[k]),
            atol=2e-8,
            rtol=0,
        ):
            raise ValueError(
                "UCC extended Fock disagrees with the canonical DF reference"
            )
    frozen = resolve_frozen_occupied(request, (oa, ob))
    feeds = unrestricted_cc_f12_fixed_source_feeds(
        space, data, fock, corr.whitener, checked, frozen_occupied=frozen
    )
    fixed = FocalUCCFixedSources(ref, proj.rank, corr.fit_id, feeds, frozen)
    if not include_ordinary:
        return fixed

    from ._uccsd_df_hamiltonian import unrestricted_cc_df_hamiltonian

    assert request.basis.mp2_fit is not None
    fit_id = request.basis.mp2_fit.identity
    if fit_id == corr.fit_id:
        raw, whitener = data, corr.whitener
    else:
        # A separate correlation fit needs only the ordinary OBS×OBS Coulomb panel.
        mp = jk if fit_id == jk.fit_id else provider._panel(request, "mp2_fit")
        whitener = mp.whitener
        obs_ao = {
            "ao_coulomb": np.ascontiguousarray(mp.data["ao_coulomb"][:, :obs, :obs]),
            "metric_whitener": whitener,
            **{
                "coefficients_" + spin: ref.coefficients[k]
                for spin, k in zip(("alpha", "beta"), spin_indices, strict=True)
            },
        }
        raw, _ = _orbital_sources(
            (oa, ob, obs, obs, len(whitener)),
            obs_ao,
            checked,
            None,
            projected=None,
            role="ordinary_jk",
        )
    ladder = provider.rhf_df_ladder if request.root_id == "rhf" else None
    uhf_ladder = provider.uhf_df_ladder if request.root_id == "uhf" else None
    factors: dict[str, np.ndarray] | None = (
        {} if ladder is not None or uhf_ladder is not None else None
    )
    ordinary = unrestricted_cc_df_hamiltonian(
        (oa, ob, obs, ext, len(whitener)),
        raw,
        fock,
        whitener,
        checked,
        frozen_occupied=frozen,
        max_output_bytes=(provider.budget_mib << 20)
        - sum(v.nbytes for v in feeds.values()),
        retain_rhf_virtual_factors=factors if ladder is not None else None,
        retain_uhf_virtual_factors=factors if uhf_ladder is not None else None,
    )
    if factors is not None:
        # The solver borrows these permanently read-only buffers instead of
        # retaining a second copy of all ordinary and fixed MO sources.
        for mapping in (ordinary, feeds):
            for name, value in mapping.items():
                mapping[name] = immutable(value)
    return FocalUCCEquationSources(fixed, fit_id, ordinary, ladder, factors, uhf_ladder)
