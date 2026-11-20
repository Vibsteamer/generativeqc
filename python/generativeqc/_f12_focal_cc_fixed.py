"""Relaxed molecular forces of the fixed-cusp UCC W, C and V+B−X inputs."""

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


def fixed_ucc_source_forces(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    sources: FocalUCCEquationSources,
    fixed_weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return minus the nuclear derivative action of the stationary fixed-source weights.

    Pair integrals follow F12Fit. The canonical and extended Fock responses
    follow JKFit. The shared OBS/CABS and its projector move with the nuclei.
    """
    from generativeqc_compiler.cc.uccsd_equations import build_f12_program
    from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
    from generativeqc_compiler.tensor import execute

    from ._f12_cc_c_nuclear import unrestricted_cc_c_ao_weights
    from ._f12_cc_scalar_nuclear import unrestricted_cc_scalar_ao_weights
    from ._f12_cc_w_nuclear import unrestricted_cc_w_ao_weights
    from ._f12_focal_frame import canonical_pair_frame
    from ._f12_focal_increment import relaxed_f12_source_weights
    from ._f12_focal_one_electron import one_electron_sources

    ref = sources.fixed.reference
    basis = request.basis
    if (
        (request.role, request.root_id, request.correlation_space_id)
        != ("mp2_f12_correlation", "uhf", "all-active")
        or basis.f12_fit is None
        or basis.mp2_fit is None
    ):
        raise ValueError(
            "Fixed UCC forces require all-active UHF molecular equation sources"
        )
    if (
        request.surface_id,
        basis.basis_id,
        request.backend,
        basis.jk_fit.identity,
        basis.f12_fit.identity,
        basis.mp2_fit.identity,
    ) != (
        ref.surface_id,
        ref.basis_id,
        ref.executed_backend,
        ref.jk_fit_id,
        sources.fixed.f12_fit_id,
        sources.mp2_fit_id,
    ):
        raise ValueError(
            "Fixed UCC weights and molecular request must share the UHF surface and all fitting roles"
        )
    oa, ob = ref.occupied
    n = ref.coefficients.shape[-1]
    cc = build_f12_program(UCCSDLayout(oa, ob, n - oa, n - ob))
    required = {
        node.attrs["name"]
        for node in cc.live_nodes
        if node.op == "input" and node.attrs["name"] in sources.fixed.feeds
    }
    if fixed_weights.keys() != required or any(
        w.dtype != np.float64
        or w.shape != sources.fixed.feeds[k].shape
        or not np.isfinite(w).all()
        for k, w in fixed_weights.items()
    ):
        raise ValueError(
            "Fixed UCC force requires one finite FP64 cotangent for each active spin source"
        )
    cuda = request.backend == "cuda"
    if cuda and any(
        obj is None
        for obj in (
            provider.run,
            provider.metric_library,
            provider.canonical_library,
            provider.response_library,
        )
    ):
        raise ValueError(
            "CUDA fixed UCC forces require the native shell, metric, canonical and coupled response providers"
        )
    runner = (
        provider.run if cuda else partial(execute, max_bytes=provider.budget_mib << 20)
    )
    assert runner is not None

    def checked(graph: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        result = runner(graph, inputs)
        if result.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError(
                "Fixed UCC force tensor program executed on a different backend"
            )
        return result

    one = one_electron_sources(
        request,
        library=provider.one_electron_library or provider.shell_library,
        derivatives=True,
        budget_mib=provider.budget_mib,
    )
    corr = provider._panel(request, "f12_fit")
    jk = (
        corr
        if corr.fit_id == basis.jk_fit.identity
        else provider._panel(request, "jk_fit")
    )
    jk_sources = jk.joint(request, explicit=False)
    space, ao, panel, proj = canonical_pair_frame(
        request,
        ref,
        one,
        jk_sources,
        corr.joint(request, explicit=True),
        expected_cabs_rank=sources.fixed.cabs_rank,
        relative_cutoff=provider.relative_cutoff,
        library=provider.response_library,
    )
    scalar = unrestricted_cc_scalar_ao_weights(
        space, ao, fixed_weights["fixed_geminal_energy"], checked, jkfit=panel
    )
    energy = scalar.linear_v + scalar.b_energy - scalar.x_fock_energy
    if abs(energy - float(sources.fixed.feeds["fixed_geminal_energy"])) > 2e-9:
        raise ValueError("The molecular V+B-X energy differs from the fixed UCC state")
    bars = scalar.weights
    for prefix, pullback, kwargs in (
        ("w_", unrestricted_cc_w_ao_weights, {}),
        ("c_f12_", unrestricted_cc_c_ao_weights, {"jkfit": panel}),
    ):
        selected = {k: v for k, v in fixed_weights.items() if k.startswith(prefix)}
        if selected:
            extra = pullback(space, ao, selected, checked, **kwargs)
            for k, v in extra.items():
                bars[k] += v
            del extra
    wo, wj, wf = relaxed_f12_source_weights(
        request,
        ref,
        one,
        jk_sources,
        panel,
        proj,
        bars,
        checked,
        budget=provider.budget_mib,
        relative_cutoff=provider.relative_cutoff,
        library=provider.response_library,
        canonical_library=provider.canonical_library,
    )
    if jk is corr:
        wf["metric"] += wj["metric"]
        wf["ao_coulomb"] += wj["ao_coulomb"]
    grad = provider._gradient(request, corr, wf, coulomb_only=False)
    if jk is not corr:
        grad += provider._gradient(request, jk, wj, coulomb_only=True)
    grad += one.nuclear_gradient(wo["overlap"], wo["hcore"], run=provider.run)
    return -grad
