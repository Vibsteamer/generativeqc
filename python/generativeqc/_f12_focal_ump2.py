"""Ordinary all-active DF-MP2 on one canonical focal RHF or UHF determinant."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable, Mapping

    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_reference import FocalReference
    from .experimental.f12_focal import FocalRequest


@dataclass(frozen=True)
class FocalCoulombSource:
    """OBS-only native real-spherical Coulomb integrals for a single fitting role."""

    surface_id: str
    orbital_id: str
    fit_id: str
    executed_backend: str
    three_center: np.ndarray
    metric: np.ndarray
    whitener: np.ndarray


@dataclass(frozen=True)
class FocalUMP2Weights:
    """Spin energies in aa, ab, bb order and relaxed derivatives of their sum."""

    spin_energies: tuple[float, float, float]
    weights: dict[str, np.ndarray]


def reference_df_ump2(
    request: FocalRequest,
    reference: FocalReference,
    jk: FocalCoulombSource,
    mp2: FocalCoulombSource,
    *,
    run: Callable | None = None,
    library: ctypes.CDLL | None = None,
    budget_mib: int = 512,
) -> FocalUMP2Weights:
    """Calculate ordinary MP2 and relax its integral weights using the same canonical root.

    Independently declared JKFit and MP2Fit may have different dimensions.
    Return derivatives with respect to overlap, core, JKFit and MP2Fit raw
    three-center integrals and metrics. Nuclear forces are minus their
    contractions with the corresponding physical integral derivatives.
    """
    from generativeqc_compiler.mp2.ordinary_df_factors import ordinary_df_factors
    from generativeqc_compiler.mp2.unrestricted_df_response import df_ump2_pair_response
    from generativeqc_compiler.tensor import Program, execute, transpose_program

    from ._f12_focal_singles_response import relaxed_reference_weights
    from ._f12_spin_ao import _orbital_sources
    from ._f12_spin_fock import spin_fock, spin_fock_pullback
    from .elements import checked_integer

    cuda = request.backend == "cuda"
    if request.backend not in ("cuda", "cpu_reference") or (cuda and run is None):
        raise ValueError(
            "ordinary MP2 requires its explicit backend and CUDA graph runner"
        )
    if (
        request.role != "mp2_f12_correlation"
        or request.root_id not in ("rhf", "uhf")
        or request.correlation_space_id != "all-active"
    ):
        raise ValueError(
            "ordinary focal DF-UMP2 requires an all-active RHF or UHF pair request"
        )
    if (
        reference.executed_backend,
        reference.surface_id,
        reference.basis_id,
        reference.orbital_id,
        reference.jk_fit_id,
    ) != (
        request.backend,
        request.surface_id,
        request.basis.basis_id,
        request.basis.orbital.identity,
        request.basis.jk_fit.identity,
    ):
        raise ValueError(
            "HF reference and pair request describe different surfaces or bases"
        )
    n = len(reference.hcore)
    spin_indices = reference._pair_spin_indices(request)
    occs = tuple(reference.occupied[k] for k in spin_indices)
    if len(occs) != 2 or any(not 0 <= o < n for o in occs) or not any(occs):
        raise ValueError("invalid HF occupations")
    for source, fit in ((jk, request.basis.jk_fit), (mp2, request.basis.mp2_fit)):
        if fit is None or (
            source.executed_backend,
            source.surface_id,
            source.orbital_id,
            source.fit_id,
        ) != (request.backend, request.surface_id, reference.orbital_id, fit.identity):
            raise ValueError(
                "Coulomb source must match its declared surface, fit and backend"
            )
        q = len(source.metric)
        for field, shape in (
            (source.metric, (q, q)),
            (source.three_center, (q, n, n)),
            (source.whitener, (q, q)),
        ):
            if (
                not q
                or field.dtype != np.float64
                or field.shape != shape
                or not np.isfinite(field).all()
            ):
                raise ValueError("invalid FP64 molecular Coulomb source")
        if not np.allclose(
            source.whitener @ source.metric @ source.whitener.T,
            np.eye(q),
            atol=2e-7,
            rtol=0,
        ):
            raise ValueError("Coulomb whitener must have full rank for the same metric")
    budget = checked_integer(
        budget_mib, "ordinary MP2 budget in MiB", low=1, high=(2**63 - 1) >> 20
    )
    runner = run if run is not None else partial(execute, max_bytes=budget << 20)

    def checked(program: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        result = runner(program, inputs)
        if result.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError("ordinary MP2 graph backend differs from request")
        return result

    spins = ("alpha", "beta")
    q = len(mp2.metric)
    controls, seeds, graphs = {}, {}, {}
    for k, (spin, o) in enumerate(zip(spins, occs, strict=True)):
        if not o:
            continue
        graph = ordinary_df_factors(n, o, q)
        inputs = {
            "coefficients": reference.coefficients[spin_indices[k]],
            "ao_coulomb": mp2.three_center,
            "whitener": mp2.whitener,
        }
        forward = Program({"B": graph.outputs["B"]})
        graphs[spin] = (forward, graph, inputs)
        b = checked(forward, inputs).outputs["B"]
        eps = reference.orbital_energies[spin_indices[k]]
        controls[spin] = {"B_" + spin: b, "eo_" + spin: eps[:o], "ev_" + spin: eps[o:]}
        seeds[spin] = np.zeros_like(b)
    fock_seeds = {"fock_" + spin: np.zeros((n, n)) for spin in spins}
    energies = []
    for il, ir in ((0, 0), (0, 1), (1, 1)):
        left, right = spins[il], spins[ir]
        ol, orr = occs[il], occs[ir]
        if not ol or not orr or (il == ir and ol == 1):
            energies.append(0.0)
            continue
        graph = df_ump2_pair_response(
            left + "_" + right, q, ol, n - ol, orr, n - orr, 1, source_weights=True
        )
        result = checked(
            graph, {**controls[left], **controls[right], "bar_force": np.array([-1.0])}
        ).outputs
        energies.append(float(result["energy"]))
        for k in sorted({il, ir}):
            spin, o = spins[k], occs[k]
            seeds[spin] += result["bar_B_" + spin]
            fock_seeds["fock_" + spin][:o, :o] += result["bar_Fo_" + spin]
            fock_seeds["fock_" + spin][o:, o:] += result["bar_Fv_" + spin]

    mp2_raw_bar = np.zeros_like(mp2.three_center)
    mp2_metric_bar = np.zeros_like(mp2.metric)
    cbar = np.zeros_like(reference.coefficients)
    for k, spin in enumerate(spins):
        if spin not in graphs:
            continue
        fwd, graph, inputs = graphs[spin]
        reverse = transpose_program(fwd, ("B",)).program
        result = checked(reverse, {**inputs, "bar_B": seeds[spin]}).outputs
        cbar[spin_indices[k]] += result["bar_coefficients"]
        mp2_raw_bar += result["bar_ao_coulomb"]
        mg = Program({"metric_bar": graph.outputs["metric_bar"]})
        mp2_metric_bar += checked(mg, {**inputs, "factor_bar": seeds[spin]}).outputs[
            "metric_bar"
        ]

    space = (*occs, n, n, len(jk.metric))
    ao = {
        "coefficients_alpha": reference.coefficients[0],
        "coefficients_beta": reference.coefficients[spin_indices[1]],
        "hcore_ao": reference.hcore,
        "ao_coulomb": jk.three_center,
        "metric/coulomb": jk.metric,
        "metric_whitener": jk.whitener,
    }
    projected, _ = _orbital_sources(
        space, ao, checked, None, projected=None, role="ordinary_fock"
    )
    values = {
        k.removeprefix("source/"): v
        for k, v in projected.items()
        if k.startswith("source/")
    }
    fspace = (*occs, n, len(jk.metric))
    focks = spin_fock(fspace, values, checked)
    for k, spin in enumerate(spins):
        if not np.allclose(
            focks["fock_" + spin],
            np.diag(reference.orbital_energies[spin_indices[k]]),
            rtol=0,
            atol=2e-8,
        ):
            raise ValueError("JKFit does not reproduce the canonical physical HF Fock")
    fseed = spin_fock_pullback(fspace, values, fock_seeds, checked)
    _, bars = _orbital_sources(
        space,
        ao,
        checked,
        {"source/" + k: v for k, v in fseed.items()},
        projected=projected,
        role="ordinary_fock",
    )
    assert bars is not None
    for k, spin in enumerate(spins):
        cbar[spin_indices[k]] += bars["coefficients_" + spin]
    direct = {
        "hcore": bars["hcore_ao"],
        "three_center": bars["ao_coulomb"],
        "metric": bars["metric/coulomb"],
        "obs_coefficients": cbar,
    }
    weights = relaxed_reference_weights(
        reference,
        values,
        jk.whitener,
        direct,
        np.zeros((n, n)),
        checked,
        library,
        budget,
    )
    weights["mp2_three_center"] = mp2_raw_bar
    weights["mp2_metric"] = mp2_metric_bar
    return FocalUMP2Weights(tuple(energies), weights)
