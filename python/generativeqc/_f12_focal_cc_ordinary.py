"""Molecular force carried by the ordinary Hamiltonian inputs of stationary UCCSD-F12b."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_cc_sources import FocalUCCEquationSources
    from ._f12_focal_molecular import MolecularFocalProvider
    from .experimental.f12_focal import FocalRequest


def ordinary_ucc_source_forces(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    sources: FocalUCCEquationSources,
    weights: Mapping[str, np.ndarray],
    *,
    virtual_factor_weights: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Pull ordinary stationary MO and optional virtual factor weights to the nuclei."""
    from generativeqc_compiler.method.canonical_frame_response import (
        orbital_rotation_cotangent,
    )
    from generativeqc_compiler.tensor import execute

    from ._canonical_frame import canonical_frame_seed
    from ._f12_focal_increment import _canonical_occupied_virtual_blocks
    from ._f12_focal_one_electron import one_electron_sources
    from ._f12_focal_singles_response import relaxed_reference_weights
    from ._f12_spin_ao import _orbital_sources
    from ._f12_spin_fock import spin_fock, spin_fock_pullback
    from ._uccsd_df_eri_response import unrestricted_cc_eri_pullback
    from ._uccsd_df_hamiltonian import _spin_fock_weight

    ref = sources.fixed.reference
    basis = request.basis
    if (
        request.role != "mp2_f12_correlation"
        or request.correlation_space_id != "all-active"
        or basis.mp2_fit is None
        or basis.f12_fit is None
    ):
        raise ValueError(
            "Ordinary CC source forces require an all-active molecular pair request"
        )
    if (
        request.surface_id,
        basis.basis_id,
        request.backend,
        basis.jk_fit.identity,
        basis.mp2_fit.identity,
        basis.f12_fit.identity,
    ) != (
        ref.surface_id,
        ref.basis_id,
        ref.executed_backend,
        ref.jk_fit_id,
        sources.mp2_fit_id,
        sources.fixed.f12_fit_id,
    ):
        raise ValueError(
            "Molecular UCC sources, stationary root and force request must match"
        )
    expected = sources.ordinary.keys()
    if virtual_factor_weights is not None:
        expected = {name for name in expected if not name.startswith("gvvvv_")}
    if weights.keys() != expected:
        raise ValueError("Ordinary UCC force requires weights of every ordinary source")

    cuda = request.backend == "cuda"
    if cuda and (
        provider.run is None
        or provider.metric_library is None
        or provider.canonical_library is None
        or provider.response_library is None
    ):
        raise ValueError(
            "CUDA UCC source forces require the native metric and orbital response libraries"
        )
    runner = (
        provider.run if cuda else partial(execute, max_bytes=provider.budget_mib << 20)
    )
    assert runner is not None

    def run(program: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        result = runner(program, inputs)
        if result.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError(
                "Molecular UCC ordinary response executed on a different backend"
            )
        return result

    one = one_electron_sources(
        request,
        library=provider.one_electron_library or provider.shell_library,
        orbital_only=True,
        derivatives=True,
        budget_mib=provider.budget_mib,
    )
    if not (
        np.allclose(one.overlap, ref.overlap, atol=2e-8, rtol=0)
        and np.allclose(one.hcore, ref.hcore, atol=2e-8, rtol=0)
    ):
        raise ValueError(
            "Ordinary UCC one-electron integrals differ from the stationary root"
        )
    jk = provider._panel(request, "jk_fit", orbital_only=True)
    mp = (
        jk
        if basis.jk_fit.identity == basis.mp2_fit.identity
        else provider._panel(request, "mp2_fit")
    )
    spins = ("alpha", "beta")
    spin_indices = ref._pair_spin_indices(request)
    occupied = tuple(ref.occupied[i] for i in spin_indices)
    energies = tuple(ref.orbital_energies[i] for i in spin_indices)
    c = {s: ref.coefficients[i] for s, i in zip(spins, spin_indices, strict=True)}
    eri = unrestricted_cc_eri_pullback(
        occupied,
        {k: v for k, v in weights.items() if k.startswith("g")},
        c,
        mp.data["ao_coulomb"],
        mp.data["metric"],
        mp.whitener,
        run,
        virtual_factor_weights=virtual_factor_weights,
    )

    n = one.obs_ao_count
    ao = {"coefficients_" + s: coeff for s, coeff in c.items()}
    ao.update(
        hcore_ao=ref.hcore,
        ao_coulomb=jk.data["ao_coulomb"],
        metric_whitener=jk.whitener,
    )
    ao["metric/coulomb"] = jk.data["metric"]
    space = (*occupied, n, n, len(jk.whitener))
    fspace = (*occupied, n, len(jk.whitener))
    projected, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="ordinary_fock"
    )
    values = {
        k.removeprefix("source/"): v
        for k, v in projected.items()
        if k.startswith("source/")
    }
    focks = spin_fock(fspace, values, run)
    for s, eps in zip(spins, energies, strict=True):
        if not np.allclose(focks["fock_" + s], np.diag(eps), atol=2e-8, rtol=0):
            raise ValueError(
                "Ordinary JKFit Fock differs from the stationary canonical frame"
            )

    def fock_pullback(seeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        seed = spin_fock_pullback(fspace, values, seeds, run)
        _, bars = _orbital_sources(
            space,
            ao,
            run,
            {"source/" + k: v for k, v in seed.items()},
            projected=projected,
            role="ordinary_fock",
        )
        assert bars is not None
        return bars

    direct = fock_pullback(
        {
            "fock_" + s: _spin_fock_weight(weights, s, n, o, 0)
            for s, o in zip(spins, occupied, strict=True)
        }
    )
    # The ordinary part alone can carry a same-spin, same-occupation torque.
    # Enforce the canonical oo/vv constraint before the coupled ov relaxation.
    seeds = {}
    rotation = orbital_rotation_cotangent(n, n)
    for spin, eps, o in zip(spins, energies, occupied, strict=True):
        key = "coefficients_" + spin
        direct[key] += eri[key]
        rot = run(
            rotation, {"coefficients": c[spin], "frame_bar": direct[key]}
        ).outputs["rotation_gradient"]
        seeds["fock_" + spin] = (
            canonical_frame_seed(
                rot,
                eps,
                o,
                library=provider.canonical_library,
                maximum_bytes=provider.budget_mib << 20,
            )[0]
            if cuda
            else _canonical_occupied_virtual_blocks(rot, eps, o)
        )
    canonical = fock_pullback(seeds)
    for key in direct:
        direct[key] += canonical[key]
    coefficient_bar = np.zeros_like(ref.coefficients)
    for spin, i in zip(spins, spin_indices, strict=True):
        coefficient_bar[i] += direct["coefficients_" + spin]
    relaxed = relaxed_reference_weights(
        ref,
        values,
        jk.whitener,
        {
            "hcore": direct["hcore_ao"],
            "three_center": direct["ao_coulomb"],
            "metric": direct["metric/coulomb"],
            "obs_coefficients": coefficient_bar,
        },
        np.zeros_like(ref.overlap),
        run,
        provider.response_library,
        provider.budget_mib,
    )

    grad = one.nuclear_gradient(relaxed["overlap"], relaxed["hcore"], run=run)
    jk_bars = {"ao_coulomb": relaxed["three_center"], "metric": relaxed["metric"]}
    mp_bars = {"ao_coulomb": eri["ao_coulomb"], "metric": eri["metric/coulomb"]}
    if mp is jk:
        for key in jk_bars:
            jk_bars[key] += mp_bars[key]
    else:
        grad += provider._gradient(request, mp, mp_bars, coulomb_only=True)
    grad += provider._gradient(request, jk, jk_bars, coulomb_only=True)
    return -grad
