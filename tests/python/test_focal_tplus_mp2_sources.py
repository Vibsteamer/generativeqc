"""Occupied MP2/F12 partitions and their weighted analytic source response."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import test_focal_df_reference as reference_tests
from generativeqc._f12_focal_energy import FocalLowOrderEnergy
from generativeqc._f12_focal_tplus_sources import (
    FocalOccupiedTriples,
    OccupiedMP2F12,
    SpinOccupied,
    add_distinct_pair,
    add_occupied_pair_matrix,
    occupied_pair_energy_seed,
)
from generativeqc._f12_unrestricted import (
    UnrestrictedF12Energy,
    unrestricted_mp2_f12_ao_response,
)
from generativeqc.experimental.f12_focal import FocalBasis, FocalRequest
from generativeqc_compiler.mp2.unrestricted_df_response import df_ump2_pair_response
from generativeqc_compiler.tensor import execute, transpose_program
from test_f12_separate_fitting_spaces import inputs, whitener

hydrogen_bases = reference_tests.hydrogen_bases


@pytest.mark.parametrize("root", ["rhf", "uhf"])
def test_molecular_tplus_joins_only_the_exact_frame_and_fit(hydrogen_bases, root):
    from generativeqc._f12_focal_molecular import MolecularFocalProvider
    from generativeqc._f12_focal_reference import FocalReference
    from generativeqc.experimental.f12_focal import FocalSeries, plan_singlepoint

    rhf = root == "rhf"
    basis = hydrogen_bases[0]
    plan = plan_singlepoint(
        [("H", (0., 0., float(i))) for i in range(4 if rhf else 3)],
        FocalSeries(basis), FocalSeries(basis), None,
        multiplicity=1 if rhf else 4, root_id=root,
        correlation_space_id="all-active", backend="cpu_reference",
    )
    request = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    matrix = np.eye(4)
    blocks = matrix[None, :, :] if rhf else np.stack((matrix, matrix))
    ref = FocalReference(
        -1., None, matrix, matrix, blocks, np.zeros(blocks.shape[:2]), blocks,
        blocks, (2,) if rhf else (3, 0), 1, request.backend,
        request.surface_id, basis.basis_id, basis.orbital.identity, basis.jk_fit.identity,
    )
    bare = np.array([-.15, -.11] if rhf else [-.15, -.11, -.20])
    increment = np.array([-.012, -.023] if rhf else [-.012, -.023, -.005])
    conventional = np.array([-.001, -.003] if rhf else [-.001, -.003, -.002])
    parts = OccupiedMP2F12(
        (bare / 2, bare / 2) if rhf else (bare, np.empty(0)),
        (increment / 2, increment / 2) if rhf else (increment, np.empty(0)),
    )
    low = FocalLowOrderEnergy(ref, 6, None, (0., float(bare.sum()), 0.), float(increment.sum()), parts)
    triples = FocalOccupiedTriples(ref, "all-active", basis.mp2_fit.identity,
                                    conventional if rhf else (conventional, np.empty(0)))
    provider = MolecularFocalProvider(shell_library=Path("unused"))
    result = provider.tplus_from_occupied(request, low, triples)
    ratio = 1 + increment / bare
    assert result.energy == pytest.approx(float(np.dot(ratio, conventional)), abs=2e-16)
    np.testing.assert_allclose(result.mp2[0], -increment * conventional / bare**2)
    np.testing.assert_allclose(result.f12_increment[0], conventional / bare)
    if rhf:
        np.testing.assert_array_equal(result.mp2[0], result.mp2[1])
        np.testing.assert_allclose(result.triples, ratio)
    else:
        assert result.mp2[1].shape == result.triples[1].shape == (0,)
        np.testing.assert_allclose(result.triples[0], ratio)
    for different in (replace(triples, reference=replace(ref)), replace(triples, correlation_fit_id="other")):
        with pytest.raises(ValueError, match="same determinant, basis and correlation fit"):
            provider.tplus_from_occupied(request, low, different)
    with pytest.raises(ValueError, match="partitions disagree with their unscaled energies"):
        provider.tplus_from_occupied(request, replace(low, f12_increment=0.0), triples)

    cuda_request = replace(request, backend="cuda")
    cuda_ref = replace(ref, executed_backend="cuda")
    cuda_provider = replace(provider, run=execute, metric_library=Path("unused"), response_library=object())
    with pytest.raises(RuntimeError, match="FP64 CUDA executor"):
        cuda_provider.tplus_from_occupied(cuda_request, replace(low, reference=cuda_ref),
                                         replace(triples, reference=cuda_ref))


@pytest.mark.parametrize("channel", ["alpha_alpha", "alpha_beta", "beta_beta"])
def test_mp2_partition_and_source_transpose_match_explicit_spin_sum(
    channel: str,
) -> None:
    rng = np.random.default_rng(7215)
    dimensions = {"alpha": (3, 4), "beta": (2, 3)}
    left, right = channel.split("_")
    labels = list(dict.fromkeys((left, right)))
    state = {}
    for spin in labels:
        o, v = dimensions[spin]
        state["B_" + spin] = rng.normal(size=(3, o, v)) * 0.1
        state["eo_" + spin] = -np.arange(1, o + 1, dtype=float)
        state["ev_" + spin] = np.arange(1, v + 1, dtype=float)
    delta = {name: rng.normal(size=field.shape) * 0.15 for name, field in state.items()}
    weights = {spin: rng.normal(size=dimensions[spin][0]) for spin in labels}
    graph = df_ump2_pair_response(
        channel, 3, *dimensions[left], *dimensions[right], 1, occupied_partition=True
    )
    pair = execute(graph, state).outputs["pair_energies"]
    result = {spin: np.zeros(dimensions[spin][0]) for spin in labels}
    add_occupied_pair_matrix(result[left], result[right], pair)

    def literal_partitions(t: float) -> dict[str, np.ndarray]:
        fields = {name: field + t * delta[name] for name, field in state.items()}
        oracle = {spin: np.zeros(dimensions[spin][0]) for spin in labels}
        o1, v1 = dimensions[left]
        o2, v2 = dimensions[right]
        for i in range(o1):
            for j in range(o2):
                for a in range(v1):
                    for b in range(v2):
                        bl, br = fields["B_" + left], fields["B_" + right]
                        g = bl[:, i, a] @ br[:, j, b]
                        if left == right:
                            g -= bl[:, i, b] @ br[:, j, a]
                        d = (
                            fields["eo_" + left][i]
                            + fields["eo_" + right][j]
                            - fields["ev_" + left][a]
                            - fields["ev_" + right][b]
                        )
                        part = (0.125 if left == right else 0.5) * g * g / d
                        oracle[left][i] += part
                        oracle[right][j] += part
        return oracle

    for spin in labels:
        np.testing.assert_allclose(
            result[spin], literal_partitions(0)[spin], atol=2e-17
        )
    transpose = transpose_program(graph, ("pair_energies",)).program
    bars = execute(
        transpose,
        {
            **state,
            "bar_pair_energies": occupied_pair_energy_seed(
                weights[left], weights[right]
            ),
        },
    ).outputs
    predicted = sum(
        np.vdot(bars["bar_" + name], direction) for name, direction in delta.items()
    )

    def objective(t: float) -> float:
        return float(
            sum(
                np.vdot(weights[spin], value)
                for spin, value in literal_partitions(t).items()
            )
        )

    for h in (4e-4, 2e-4):
        assert predicted == pytest.approx(
            (objective(h) - objective(-h)) / (2 * h), abs=2e-9
        )


def test_distinct_spin_pair_marginal_and_empty_channel() -> None:
    alpha, beta = np.zeros(3), np.zeros(0)
    add_distinct_pair((alpha, beta), ("alpha", "alpha"), (0, 2), -0.7)
    add_distinct_pair((alpha, beta), ("alpha", "alpha"), (1, 2), 0.3)
    np.testing.assert_allclose(alpha, [-0.35, 0.15, -0.20])
    assert beta.size == 0


def test_weighted_f12_pair_sources_transpose_two_fitting_metrics() -> None:
    ao, jk = inputs()
    weights = np.array([0.13, -0.31]), np.array([1.7])
    rng = np.random.default_rng(618)

    def tangents(panel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = {}
        for name, x in panel.items():
            if name == "metric_whitener":
                continue
            d = rng.normal(size=x.shape) * 0.02
            result[name] = (
                d if name.startswith("coefficients") else (d + d.swapaxes(-1, -2)) / 2
            )
        return result

    da, dj = tangents(ao), tangents(jk)

    def moved(
        panel: dict[str, np.ndarray], direction: dict[str, np.ndarray], t: float
    ) -> dict[str, np.ndarray]:
        values = {name: panel[name] + t * x for name, x in direction.items()}
        values["metric_whitener"] = whitener(values["metric/coulomb"])
        return values

    def evaluate(
        t: float, seeds: SpinOccupied | None
    ) -> tuple[UnrestrictedF12Energy, dict[str, np.ndarray]]:
        return unrestricted_mp2_f12_ao_response(
            (2, 1, 3, 4, 3),
            moved(ao, da, t),
            execute,
            jkfit=moved(jk, dj, t),
            component="f12",
            occupied_f12_weights=seeds,
        )

    baseline, bb = evaluate(0.0, None)
    ones, ob = evaluate(0.0, (np.ones(2), np.ones(1)))
    assert baseline == ones
    for name in bb:
        np.testing.assert_allclose(bb[name], ob[name], atol=2e-15)
    _, bars = evaluate(0.0, weights)
    slope = sum(np.vdot(bars[k], v) for k, v in da.items())
    slope += sum(np.vdot(bars["jkfit/" + k], v) for k, v in dj.items())
    for h in (4e-4, 2e-4):
        assert slope == pytest.approx(
            (
                evaluate(h, weights)[0].f12_increment
                - evaluate(-h, weights)[0].f12_increment
            )
            / (2 * h),
            abs=2e-8,
        )


def test_molecular_weighted_mp2_f12_source_force_matches_recomputed_restricted_frame(
    hydrogen_bases: tuple[FocalBasis, dict, dict],
) -> None:
    from generativeqc._f12_focal_molecular import MolecularFocalProvider
    from generativeqc.experimental.f12_focal import FocalSeries, plan_singlepoint

    library = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not library:
        pytest.skip("requires the native molecular shell forward/derivative library")
    basis = hydrogen_bases[0]
    positions = np.array(
        [
            [0.0, 0.0, 0.1],
            [1.39, 0.11, 0.03],
            [0.08, 4.5, 0.46],
            [1.31, 4.71, 0.49],
        ]
    )
    direction = np.array(
        [
            [0.1, -0.4, 0.3],
            [0.3, 0.6, -0.2],
            [-0.2, 0.4, 0.1],
            [-0.2, -0.6, -0.2],
        ]
    )
    provider = MolecularFocalProvider(shell_library=Path(library))
    mp2_seed, f12_seed = np.array([1.2, -0.35]), np.array([-0.7, 0.6])

    def evaluate(
        t: float, rank: int | None = None
    ) -> tuple[FocalRequest, FocalLowOrderEnergy, float]:
        plan = plan_singlepoint(
            list(zip(["H"] * 4, positions + t * direction, strict=True)),
            FocalSeries(basis),
            FocalSeries(basis),
            basis,
            root_id="rhf",
            correlation_space_id="all-active",
            backend="cpu_reference",
        )
        req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
        state = provider.low_order_energy(
            req,
            include_hf_cabs=False,
            include_occupied_partitions=True,
            expected_cabs_rank=rank,
        )
        occupied = state.occupied_mp2_f12
        assert occupied is not None
        mp2, f12 = occupied.restricted_spatial()
        assert np.sum(mp2) + np.sum(f12) == pytest.approx(
            state.mp2_f12_correlation_energy, abs=2e-12
        )
        return req, state, float(np.vdot(mp2_seed, mp2) + np.vdot(f12_seed, f12))

    req, state, _ = evaluate(0)
    predicted = provider.tplus_scaling_forces(
        req,
        state.reference,
        (mp2_seed, mp2_seed),
        (f12_seed, f12_seed),
        expected_cabs_rank=state.cabs_rank,
    )
    for h in (4e-4, 2e-4):
        e1 = evaluate(-h, state.cabs_rank)[2]
        e2 = evaluate(h, state.cabs_rank)[2]
        assert np.vdot(predicted, direction) == pytest.approx(
            (e1 - e2) / (2 * h), abs=2e-7
        )
