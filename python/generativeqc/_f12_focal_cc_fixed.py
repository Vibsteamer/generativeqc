"""Relaxed molecular forces of the fixed-cusp UCC W, C and V+B−X inputs."""

from __future__ import annotations

from contextlib import contextmanager
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

    from ._f12_cc_c_nuclear import _accumulate_c_mo_pullback
    from ._f12_cc_fixed_sources_native import coulomb_fit_forward, coulomb_fit_reverse
    from ._f12_cc_fock_pullback import extended_fock_response
    from ._f12_cc_scalar_nuclear import _accumulate_scalar_mo_pullback
    from ._f12_cc_w_nuclear import _accumulate_w_mo_pullback
    from ._f12_focal_frame import canonical_pair_frame
    from ._f12_focal_increment import relaxed_f12_source_weights
    from ._f12_focal_integral_sink import contract_single_channel
    from ._f12_focal_molecular import _AO_NAME_TO_OP, _METRIC_KEY_TO_OP
    from ._f12_frozen_core import resolve_frozen_occupied
    from ._f12_spin_ao import _orbital_sources

    ref = sources.fixed.reference
    basis = request.basis
    if (
        request.role != "mp2_f12_correlation"
        or basis.f12_fit is None
        or basis.mp2_fit is None
    ):
        raise ValueError("Fixed CC forces require molecular equation sources")
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
            "Fixed UCC weights and molecular request must share the reference surface and all fitting roles"
        )
    oa, ob = (ref.occupied[i] for i in ref._pair_spin_indices(request))
    frozen = resolve_frozen_occupied(request, (oa, ob))
    if frozen != sources.fixed.frozen_occupied:
        raise ValueError("Fixed UCC force frozen-core mismatch with stationary sources")
    fa, fb = frozen
    n = ref.coefficients.shape[-1]
    cc = build_f12_program(UCCSDLayout(oa - fa, ob - fb, n - oa, n - ob))
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

    if hasattr(runner, "batch"):

        @contextmanager
        def checked_batch(graph: Program):
            with runner.batch(graph) as exec_one:

                def wrapped(inputs: Mapping[str, np.ndarray]) -> Execution:
                    result = exec_one(inputs)
                    if result.backend != (
                        "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
                    ):
                        raise RuntimeError(
                            "Fixed UCC force tensor program executed on a different backend"
                        )
                    return result

                yield wrapped

        checked.batch = checked_batch  # type: ignore[attr-defined]

    one, (jk, _mp2, corr) = provider._cached_pair_sources(
        request, derivatives=True, budget=provider.budget_mib
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
    fock, fock_pullback = extended_fock_response(
        space, ao, panel, checked, cuda=cuda, library=provider.response_library
    )

    # Fused single-pass AO->MO projection for scalar (V+B-X), W, and C pullbacks.
    # Coulomb is projected only on [aux, obs, ext], avoiding full [aux, ext, ext].
    oa_s, ob_s, obs, ext, aux = space
    occ_by_spin = {"alpha": oa_s, "beta": ob_s}
    data, _ = _orbital_sources(
        space,
        ao,
        checked,
        None,
        projected=None,
        role="pairs",
        obs_coulomb=True,
        skip_coulomb_fit_solve=True,
    )

    host_ref = request.backend == "cpu_reference"
    whitener = ao["metric_whitener"]
    same_spin = oa_s == ob_s and np.array_equal(
        ao["coefficients_alpha"], ao["coefficients_beta"]
    )
    fit_obs: dict[str, np.ndarray] = {}
    for spin in ("alpha", "beta"):
        if same_spin and spin == "beta" and "alpha" in fit_obs:
            fit_obs["beta"] = fit_obs["alpha"]
        else:
            fit_obs[spin] = coulomb_fit_forward(
                whitener,
                data["source/raw_coulomb_" + spin],
                native_library=provider.shell_library,
                host_reference=host_ref,
                run=checked,
            )
        n_occ = occ_by_spin[spin]
        if n_occ:
            data["source/fit_" + spin] = fit_obs[spin][:, :n_occ, :]

    dbar = {k: np.zeros_like(v) for k, v in data.items() if k != "space"}
    fit_obs_bar = {s: np.zeros_like(v) for s, v in fit_obs.items()}
    fbar = {k: np.zeros_like(v) for k, v in fock.items()}

    # 1. Scalar V + B - X MO pullback (native GPU cuBLAS when available)
    e_v, e_b, e_x = _accumulate_scalar_mo_pullback(
        space,
        data,
        fock,
        fixed_weights["fixed_geminal_energy"],
        checked,
        dbar,
        fbar,
        frozen_occupied=frozen,
        native_library=provider.shell_library,
        host_reference=host_ref,
        budget_bytes=provider.budget_mib << 20,
    )
    energy = e_v + e_b - e_x
    if abs(energy - float(sources.fixed.feeds["fixed_geminal_energy"])) > 2e-9:
        raise ValueError("The molecular V+B-X energy differs from the fixed UCC state")

    # 2. W_vo / W_vv MO pullback
    w_selected = {k: v for k, v in fixed_weights.items() if k.startswith("w_")}
    if w_selected:
        _accumulate_w_mo_pullback(
            space,
            data,
            fit_obs,
            w_selected,
            checked,
            dbar,
            fit_obs_bar,
            frozen_occupied=frozen,
            native_library=provider.shell_library,
            host_reference=host_ref,
            budget_bytes=provider.budget_mib << 20,
        )

    # 3. C_vv MO pullback
    c_selected = {k: v for k, v in fixed_weights.items() if k.startswith("c_f12_")}
    if c_selected:
        _accumulate_c_mo_pullback(
            space,
            data,
            fock,
            c_selected,
            checked,
            dbar,
            fbar,
            frozen_occupied=frozen,
            native_library=provider.shell_library,
            host_reference=host_ref,
            budget_bytes=provider.budget_mib << 20,
        )

    # Fold occupied fit cotangents from scalar+C into full obs fit cotangents from W,
    # then apply Coulomb metric solve adjoint ONCE per distinct spin.
    for spin in ("alpha", "beta"):
        n_occ = occ_by_spin[spin]
        if n_occ:
            fit_obs_bar[spin][:, :n_occ, :] += dbar["source/fit_" + spin]

    metric_coul_bar = np.zeros_like(ao["metric/coulomb"])
    can_share_fit_rev = same_spin and np.array_equal(
        fit_obs_bar["alpha"], fit_obs_bar["beta"]
    )
    if can_share_fit_rev:
        tc_bar, m_bar = coulomb_fit_reverse(
            whitener,
            data["source/raw_coulomb_alpha"],
            fit_obs_bar["alpha"],
            fit_precomputed=fit_obs["alpha"],
            native_library=provider.shell_library,
            host_reference=host_ref,
            run=checked,
        )
        dbar["source/raw_coulomb_alpha"] += tc_bar
        dbar["source/raw_coulomb_beta"] += tc_bar
        metric_coul_bar += 2.0 * m_bar
        del tc_bar, m_bar
    else:
        for spin in ("alpha", "beta"):
            tc_bar, m_bar = coulomb_fit_reverse(
                whitener,
                data["source/raw_coulomb_" + spin],
                fit_obs_bar[spin],
                fit_precomputed=fit_obs[spin],
                native_library=provider.shell_library,
                host_reference=host_ref,
                run=checked,
            )
            dbar["source/raw_coulomb_" + spin] += tc_bar
            metric_coul_bar += m_bar
            del tc_bar, m_bar
    del fit_obs, fit_obs_bar

    # Fused single-pass MO->AO pullback (streaming each AO cotangent directly to GPU sink when large).
    streamed_grad = np.zeros((len(request.atoms), 3), dtype=np.float64)

    def _f12_ao_sink(ao_name: str, bar_ao_mat: np.ndarray) -> None:
        streamed_grad[:] += contract_single_channel(
            corr.layout,
            "ao",
            _AO_NAME_TO_OP[ao_name],
            bar_ao_mat,
            provider.shell_library,
            host_reference=host_ref,
            budget_mib=provider.budget_mib,
            slater_normalization=provider.slater_normalization,
        )

    _, bars = _orbital_sources(
        space,
        ao,
        checked,
        dbar,
        projected=data,
        role="pairs",
        ao_bar_sink=_f12_ao_sink if corr.streaming else None,
        obs_coulomb=True,
        skip_coulomb_fit_solve=True,
    )
    del data, dbar
    assert bars is not None
    bars["metric/coulomb"] += metric_coul_bar
    fock_pullback(fbar, bars)
    del fbar

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
    del bars

    if corr.streaming:
        grad = streamed_grad
        for mkey, op in _METRIC_KEY_TO_OP.items():
            grad += contract_single_channel(
                corr.layout,
                "metric",
                op,
                wf[mkey],
                provider.shell_library,
                host_reference=host_ref,
                budget_mib=provider.budget_mib,
                slater_normalization=provider.slater_normalization,
            )
        grad += provider._gradient(request, jk, wj, coulomb_only=True)
    else:
        if jk is corr:
            wf["metric"] += wj["metric"]
            wf["ao_coulomb"] += wj["ao_coulomb"]
        grad = provider._gradient(request, corr, wf, coulomb_only=False)
        if jk is not corr:
            grad += provider._gradient(request, jk, wj, coulomb_only=True)
    grad += one.nuclear_gradient(
        wo["overlap"], wo["hcore"], run=provider.run, shell_library=provider.shell_library
    )
    return -grad
