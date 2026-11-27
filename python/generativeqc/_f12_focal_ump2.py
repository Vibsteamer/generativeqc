"""Ordinary all-active DF-MP2 on one canonical focal RHF or UHF determinant."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_reference import FocalReference
    from ._f12_focal_tplus_sources import SpinOccupied
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


@dataclass(frozen=True)
class FocalUMP2Unrelaxed:
    """Unrelaxed ordinary DF-MP2 spin energies and fixed-frame cotangent seeds."""

    spin_energies: tuple[float, float, float]
    fock_seeds: dict[str, np.ndarray]
    obs_coefficients: np.ndarray
    mp2_three_center: np.ndarray
    mp2_metric: np.ndarray


def ordinary_df_ump2_unrelaxed(
    request: FocalRequest,
    reference: FocalReference,
    mp2: FocalCoulombSource,
    *,
    run: Callable | None = None,
    budget_mib: int = 512,
    occupied_weights: SpinOccupied | None = None,
    shell_library: Path | str | ctypes.CDLL | None = None,
) -> FocalUMP2Unrelaxed:
    """Evaluate ordinary DF-MP2 energies and unrelaxed MO/Fock/MP2Fit seeds."""
    from ._f12_focal_reference import partition_focal_reference
    from ._f12_focal_tplus_sources import (
        checked_occupied_weights,
        occupied_pair_energy_seed,
    )
    from ._f12_frozen_core import active_occupied_counts, resolve_frozen_occupied
    from ._f12_mp2_pair_native import (
        aot_mp2f12_disabled,
        bind_ordinary_df_mp2,
        run_ordinary_df_mp2_native,
    )
    from .elements import checked_integer

    cuda = request.backend == "cuda"
    if request.backend not in ("cuda", "cpu_reference") or (cuda and run is None):
        raise ValueError(
            "ordinary MP2 requires its explicit backend and CUDA graph runner"
        )
    if (
        request.role != "mp2_f12_correlation"
        or request.root_id not in ("rhf", "uhf")
    ):
        raise ValueError(
            "ordinary focal DF-UMP2 requires an RHF or UHF pair request"
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
    reference = partition_focal_reference(request, reference, budget_mib)
    n = len(reference.hcore)
    spin_indices = reference._pair_spin_indices(request)
    occs = tuple(reference.occupied[k] for k in spin_indices)
    if len(occs) != 2 or any(not 0 <= o < n for o in occs) or not any(occs):
        raise ValueError("invalid HF occupations")
    frozen = resolve_frozen_occupied(request, occs)
    active_occs = active_occupied_counts(occs, frozen)
    if occupied_weights is not None:
        checked_occupied_weights(occupied_weights, active_occs)
    fit = request.basis.mp2_fit
    if fit is None or (
        mp2.executed_backend,
        mp2.surface_id,
        mp2.orbital_id,
        mp2.fit_id,
    ) != (request.backend, request.surface_id, reference.orbital_id, fit.identity):
        raise ValueError(
            "Coulomb source must match its declared surface, fit and backend"
        )
    q = len(mp2.metric)
    for field, shape in (
        (mp2.metric, (q, q)),
        (mp2.three_center, (q, n, n)),
        (mp2.whitener, (q, q)),
    ):
        if (
            not q
            or field.dtype != np.float64
            or field.shape != shape
            or not np.isfinite(field).all()
        ):
            raise ValueError("invalid FP64 molecular Coulomb source")
    if not np.allclose(
        mp2.whitener @ mp2.metric @ mp2.whitener.T,
        np.eye(q),
        atol=2e-7,
        rtol=0,
    ):
        raise ValueError("Coulomb whitener must have full rank for the same metric")
    budget = checked_integer(
        budget_mib, "ordinary MP2 budget in MiB", low=1, high=(2**63 - 1) >> 20
    )

    spins = ("alpha", "beta")
    restricted = len(reference.occupied) == 1
    same_spin_mp2 = (
        occs[0] == occs[1]
        and frozen[0] == frozen[1]
        and (
            restricted
            or (
                np.array_equal(
                    reference.coefficients[spin_indices[0]],
                    reference.coefficients[spin_indices[1]],
                )
                and np.array_equal(
                    reference.orbital_energies[spin_indices[0]],
                    reference.orbital_energies[spin_indices[1]],
                )
            )
        )
        and (
            occupied_weights is None
            or np.array_equal(occupied_weights[0], occupied_weights[1])
        )
    )

    native_fn = bind_ordinary_df_mp2(
        host_reference=not cuda,
        shell_library=shell_library,
    )
    if shell_library is not None and native_fn is None and not aot_mp2f12_disabled():
        suffix = "_host_reference" if not cuda else "_cuda"
        raise RuntimeError(
            f"shell_library {shell_library} does not export gqc_f12_ordinary_df_mp2{suffix}"
        )
    if native_fn is not None:
        (
            spin_energies,
            fock_a,
            fock_b,
            bar_coeff_a,
            bar_coeff_b,
            bar_ao_coul,
            bar_metric,
        ) = run_ordinary_df_mp2_native(
            native_fn,
            same_spin=same_spin_mp2,
            compute_response=True,
            oa=occs[0],
            fa=frozen[0],
            ob=occs[1],
            fb=frozen[1],
            budget_bytes=budget << 20,
            coeff_a=reference.coefficients[spin_indices[0]],
            coeff_b=reference.coefficients[spin_indices[1]],
            eps_a=reference.orbital_energies[spin_indices[0]],
            eps_b=reference.orbital_energies[spin_indices[1]],
            ao_coul=mp2.three_center,
            whitener=mp2.whitener,
            occ_w_a=None if occupied_weights is None else occupied_weights[0],
            occ_w_b=None if occupied_weights is None else occupied_weights[1],
        )
        assert fock_a is not None and fock_b is not None
        assert bar_coeff_a is not None and bar_coeff_b is not None
        assert bar_ao_coul is not None and bar_metric is not None
        cbar = np.zeros_like(reference.coefficients)
        cbar[spin_indices[0]] += bar_coeff_a
        cbar[spin_indices[1]] += bar_coeff_b
        return FocalUMP2Unrelaxed(
            spin_energies,
            {"fock_alpha": fock_a, "fock_beta": fock_b},
            cbar,
            bar_ao_coul,
            bar_metric,
        )

    from generativeqc_compiler.mp2.ordinary_df_factors import ordinary_df_factors
    from generativeqc_compiler.mp2.unrestricted_df_response import df_ump2_pair_response
    from generativeqc_compiler.tensor import Program, execute, transpose_program

    runner = run if run is not None else partial(execute, max_bytes=budget << 20)

    def checked(program: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        result = runner(program, inputs)
        if result.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError("ordinary MP2 graph backend differs from request")
        return result

    if hasattr(runner, "batch"):
        from contextlib import contextmanager

        @contextmanager
        def checked_batch(program: Program):
            with runner.batch(program) as exec_one:

                def wrapped(inputs: Mapping[str, np.ndarray]) -> Execution:
                    result = exec_one(inputs)
                    if result.backend != (
                        "cuda-fp64-ordinary-stream"
                        if cuda
                        else "numpy-cpu-interpreter"
                    ):
                        raise RuntimeError(
                            "ordinary MP2 graph backend differs from request"
                        )
                    return result

                yield wrapped

        checked.batch = checked_batch  # type: ignore[attr-defined]

    controls, seeds, graphs = {}, {}, {}
    for k, (spin, o, f) in enumerate(zip(spins, occs, frozen, strict=True)):
        o_act = o - f
        if not o_act:
            continue
        if same_spin_mp2 and spin == "beta" and "alpha" in controls:
            graphs["beta"] = graphs["alpha"]
            controls["beta"] = {
                "B_beta": controls["alpha"]["B_alpha"],
                "eo_beta": controls["alpha"]["eo_alpha"],
                "ev_beta": controls["alpha"]["ev_alpha"],
            }
            seeds["beta"] = np.zeros_like(controls["alpha"]["B_alpha"])
            continue
        graph = ordinary_df_factors(n, o, q, frozen=f)
        inputs = {
            "coefficients": reference.coefficients[spin_indices[k]],
            "ao_coulomb": mp2.three_center,
            "whitener": mp2.whitener,
        }
        forward = Program({"B": graph.outputs["B"]})
        graphs[spin] = (forward, graph, inputs)
        b = checked(forward, inputs).outputs["B"]
        eps = reference.orbital_energies[spin_indices[k]]
        controls[spin] = {"B_" + spin: b, "eo_" + spin: eps[f:o], "ev_" + spin: eps[o:]}
        seeds[spin] = np.zeros_like(b)
    fock_seeds = {"fock_" + spin: np.zeros((n, n)) for spin in spins}
    energies = []
    aa_cache: tuple[float, dict[str, np.ndarray]] | None = None
    for il, ir in ((0, 0), (0, 1), (1, 1)):
        left, right = spins[il], spins[ir]
        ol_act, or_act = active_occs[il], active_occs[ir]
        ol, orr = occs[il], occs[ir]
        if not ol_act or not or_act or (il == ir and ol_act == 1):
            energies.append(0.0)
            continue
        if same_spin_mp2 and il == 1 and ir == 1 and aa_cache is not None:
            e_val, cached_res = aa_cache
            energies.append(e_val)
            o, f = occs[1], frozen[1]
            seeds["beta"] += cached_res["bar_B_alpha"]
            if occupied_weights is None:
                fock_seeds["fock_beta"][f:o, f:o] += cached_res["bar_Fo_alpha"]
                fock_seeds["fock_beta"][o:, o:] += cached_res["bar_Fv_alpha"]
            else:
                diag_vec = np.zeros(n)
                diag_vec[f:o] = cached_res["bar_eo_alpha"]
                diag_vec[o:] = cached_res["bar_ev_alpha"]
                fock_seeds["fock_beta"] += np.diag(diag_vec)
            continue
        pair_inputs = {**controls[left], **controls[right]}
        if occupied_weights is None:
            graph = df_ump2_pair_response(
                left + "_" + right, q, ol_act, n - ol, or_act, n - orr, 1, source_weights=True
            )
            result = checked(
                graph, {**pair_inputs, "bar_force": np.array([-1.0])}
            ).outputs
            e_val = float(result["energy"])
        else:
            graph = df_ump2_pair_response(
                left + "_" + right, q, ol_act, n - ol, or_act, n - orr, 1, occupied_partition=True
            )
            reverse = transpose_program(
                graph, ("pair_energies",), max_elements=max(1_000_000, n * n)
            ).program
            seed = occupied_pair_energy_seed(occupied_weights[il], occupied_weights[ir])
            result = checked(
                Program({**graph.outputs, **reverse.outputs}),
                {**pair_inputs, "bar_pair_energies": seed},
            ).outputs
            e_val = float(np.vdot(seed, result["pair_energies"]))
        energies.append(e_val)
        if same_spin_mp2 and il == 0 and ir == 0:
            aa_cache = (e_val, result)
        for k in sorted({il, ir}):
            spin, o, f = spins[k], occs[k], frozen[k]
            seeds[spin] += result["bar_B_" + spin]
            if occupied_weights is None:
                fock_seeds["fock_" + spin][f:o, f:o] += result["bar_Fo_" + spin]
                fock_seeds["fock_" + spin][o:, o:] += result["bar_Fv_" + spin]
            else:
                diag_vec = np.zeros(n)
                diag_vec[f:o] = result["bar_eo_" + spin]
                diag_vec[o:] = result["bar_ev_" + spin]
                fock_seeds["fock_" + spin] += np.diag(diag_vec)

    mp2_raw_bar = np.zeros_like(mp2.three_center)
    mp2_metric_bar = np.zeros_like(mp2.metric)
    cbar = np.zeros_like(reference.coefficients)
    if (
        same_spin_mp2
        and "alpha" in graphs
        and "beta" in graphs
        and np.array_equal(seeds["alpha"], seeds["beta"])
    ):
        fwd, graph, inputs = graphs["alpha"]
        reverse = transpose_program(fwd, ("B",)).program
        result = checked(reverse, {**inputs, "bar_B": seeds["alpha"]}).outputs
        for k in (0, 1):
            cbar[spin_indices[k]] += result["bar_coefficients"]
        mp2_raw_bar += 2.0 * result["bar_ao_coulomb"]
        mg = Program({"metric_bar": graph.outputs["metric_bar"]})
        mp2_metric_bar += 2.0 * checked(
            mg, {**inputs, "factor_bar": seeds["alpha"]}
        ).outputs["metric_bar"]
    else:
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
    return FocalUMP2Unrelaxed(
        tuple(energies), fock_seeds, cbar, mp2_raw_bar, mp2_metric_bar
    )


def reference_df_ump2(
    request: FocalRequest,
    reference: FocalReference,
    jk: FocalCoulombSource,
    mp2: FocalCoulombSource,
    *,
    run: Callable | None = None,
    library: ctypes.CDLL | None = None,
    budget_mib: int = 512,
    occupied_weights: SpinOccupied | None = None,
    canonical_library: ctypes.CDLL | None = None,
    shell_library: Path | str | ctypes.CDLL | None = None,
) -> FocalUMP2Weights:
    """Calculate ordinary MP2 and relax its integral weights using the same canonical root.

    Independently declared JKFit and MP2Fit may have different dimensions.
    Return derivatives with respect to overlap, core, JKFit and MP2Fit raw
    three-center integrals and metrics. Nuclear forces are minus their
    contractions with the corresponding physical integral derivatives.
    Occupied weights select a linear combination of the canonical orbital
    increments. This observable also needs the same-space canonical Fock
    constraints, since unequal weights depend on the chosen occupied orbitals.
    """
    from generativeqc_compiler.tensor import Program, execute

    from ._f12_focal_reference import partition_focal_reference
    from ._f12_focal_singles_response import relaxed_reference_weights
    from ._f12_frozen_core import resolve_frozen_occupied
    from ._f12_spin_ao import _orbital_sources
    from ._f12_spin_fock import spin_fock, spin_fock_pullback
    from .elements import checked_integer

    reference = partition_focal_reference(request, reference, budget_mib)
    unrelaxed = ordinary_df_ump2_unrelaxed(
        request,
        reference,
        mp2,
        run=run,
        budget_mib=budget_mib,
        occupied_weights=occupied_weights,
        shell_library=shell_library,
    )
    cuda = request.backend == "cuda"
    n = len(reference.hcore)
    spin_indices = reference._pair_spin_indices(request)
    occs = tuple(reference.occupied[k] for k in spin_indices)
    frozen = resolve_frozen_occupied(request, occs)
    jk_fit = request.basis.jk_fit
    if (
        jk.executed_backend,
        jk.surface_id,
        jk.orbital_id,
        jk.fit_id,
    ) != (request.backend, request.surface_id, reference.orbital_id, jk_fit.identity):
        raise ValueError(
            "Coulomb source must match its declared surface, fit and backend"
        )
    qjk = len(jk.metric)
    for field, shape in (
        (jk.metric, (qjk, qjk)),
        (jk.three_center, (qjk, n, n)),
        (jk.whitener, (qjk, qjk)),
    ):
        if (
            not qjk
            or field.dtype != np.float64
            or field.shape != shape
            or not np.isfinite(field).all()
        ):
            raise ValueError("invalid FP64 molecular Coulomb source")
    if not np.allclose(
        jk.whitener @ jk.metric @ jk.whitener.T,
        np.eye(qjk),
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
    fock_seeds = unrelaxed.fock_seeds
    cbar = unrelaxed.obs_coefficients
    mp2_raw_bar = unrelaxed.mp2_three_center
    mp2_metric_bar = unrelaxed.mp2_metric
    energies = unrelaxed.spin_energies

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
    def fock_pullback(seeds_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        fseed = spin_fock_pullback(fspace, values, seeds_map, checked)
        _, fbars = _orbital_sources(
            space,
            ao,
            checked,
            {"source/" + k: v for k, v in fseed.items()},
            projected=projected,
            role="ordinary_fock",
        )
        assert fbars is not None
        return fbars

    direct_bars = fock_pullback(fock_seeds)
    restricted = len(reference.occupied) == 1
    for k, spin in enumerate(spins):
        idx = spin_indices[k]
        direct_bars["coefficients_" + spin] += cbar[idx] * (0.5 if restricted else 1.0)
    if occupied_weights is not None or any(frozen):
        from generativeqc_compiler.method.canonical_frame_response import (
            orbital_rotation_cotangent,
        )

        from ._canonical_frame import canonical_frame_seed
        from ._f12_focal_increment import _canonical_occupied_virtual_blocks

        canonical_seeds: dict[str, np.ndarray] = {}
        for k, spin in enumerate(spins):
            idx = spin_indices[k]
            rotation = checked(
                orbital_rotation_cotangent(n, n),
                {
                    "coefficients": reference.coefficients[idx],
                    "frame_bar": direct_bars["coefficients_" + spin],
                },
            ).outputs["rotation_gradient"]
            eps, occupied = reference.orbital_energies[idx], occs[k]
            canonical_seeds["fock_" + spin] = (
                canonical_frame_seed(
                    rotation, eps, occupied, maximum_bytes=budget << 20, library=canonical_library
                )[0]
                if cuda
                else _canonical_occupied_virtual_blocks(rotation, eps, occupied)
            )
        canonical_bars = fock_pullback(canonical_seeds)
        for key in direct_bars:
            direct_bars[key] += canonical_bars[key]

    total_cbar = np.zeros_like(reference.coefficients)
    for k, spin in enumerate(spins):
        total_cbar[spin_indices[k]] += direct_bars["coefficients_" + spin]
    direct = {
        "hcore": direct_bars["hcore_ao"],
        "three_center": direct_bars["ao_coulomb"],
        "metric": direct_bars["metric/coulomb"],
        "obs_coefficients": total_cbar,
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
