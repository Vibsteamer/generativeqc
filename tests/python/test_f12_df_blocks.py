"""Regression checks for Coulomb-metric robust geminal fitting and its response."""

import numpy as np
import pytest
from generativeqc_compiler.mp2.f12_density_fitting import (
    coulomb_fit_response,
    fitted_integral_block,
    molecular_orbital_sources,
)
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks.f12_df_component import molecular_case, response_case


@pytest.mark.parametrize("case", ["h2", "water"])
def test_robust_occupied_pair_projection(case: str) -> None:
    molecular_case(case, execute)


def test_rectangular_block_and_adjoint() -> None:
    response_case(execute)


def test_metric_solve_response_across_independent_pair_tiles() -> None:
    rng = np.random.default_rng(184)
    a = rng.normal(size=(7, 7))
    metric = a.T @ a + np.eye(7)
    whitener = np.linalg.solve(np.linalg.cholesky(metric), np.eye(7))
    g, seed, dg = rng.normal(size=(3, 7, 5))
    dm = rng.normal(size=(7, 7))
    dm = (dm + dm.T) / 2
    slope = 0.0
    for columns in (slice(0, 3), slice(3, 5)):
        block = g[:, columns]
        result = execute(
            coulomb_fit_response(*block.shape),
            {
                "metric_whitener": whitener,
                "three_center": block,
                "fit_bar": seed[:, columns],
            },
        ).outputs
        np.testing.assert_allclose(
            result["fit"], np.linalg.solve(metric, block), atol=2e-14
        )
        np.testing.assert_allclose(
            result["metric_bar"], result["metric_bar"].T, atol=2e-14
        )
        slope += np.vdot(result["metric_bar"], dm) + np.vdot(
            result["three_center_bar"], dg[:, columns]
        )
    for h in (2e-5, 1e-5):
        displaced = [
            np.vdot(seed, np.linalg.solve(metric + t * dm, g + t * dg)) for t in (h, -h)
        ]
        assert slope == pytest.approx((displaced[0] - displaced[1]) / (2 * h), abs=2e-8)


@pytest.mark.parametrize("dimensions", [(0, 1, 2), (1, -1, 2), (5, 2, True)])
def test_positive_block_sizes(dimensions: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="positive dimensions"):
        fitted_integral_block(*dimensions)


@pytest.mark.parametrize("occupied", [1, 3])
def test_shared_orbital_frame_source_response(occupied: int) -> None:
    """Both AO indices use the same coefficient matrix, including the CABS columns."""
    rng = np.random.default_rng(5191)
    atomic, orbitals, auxiliaries = 7, 5, 4
    coefficients = rng.normal(size=(atomic, orbitals))
    sources = {
        "coefficients": coefficients,
        "hcore_ao": rng.normal(size=(atomic, atomic)),
        **{
            "ao_" + op: rng.normal(size=(auxiliaries, atomic, atomic))
            for op in (
                "coulomb",
                "f12",
                "f12_squared",
                "f12_over_coulomb",
                "double_commutator",
            )
        },
    }
    graph = molecular_orbital_sources(atomic, orbitals, occupied, auxiliaries)
    actual = execute(graph, sources).outputs
    reference = {"hcore": coefficients.T @ sources["hcore_ao"] @ coefficients}
    for name, value in sources.items():
        if name.startswith("ao_"):
            left = coefficients if name == "ao_coulomb" else coefficients[:, :occupied]
            reference["raw_" + name[3:]] = np.stack(
                [left.T @ matrix @ coefficients for matrix in value]
            )
    for name, value in reference.items():
        np.testing.assert_allclose(actual[name], value, atol=2e-12, rtol=2e-14)
    seeds = {name: rng.normal(size=value.shape) for name, value in reference.items()}
    reverse = transpose_program(graph, tuple(graph.outputs)).program
    bars = execute(
        reverse, {**sources, **{"bar_" + name: value for name, value in seeds.items()}}
    ).outputs
    assert set(bars) == {"bar_" + name for name in sources}
    for selected in ("coefficients", "hcore_ao", "ao_coulomb", "ao_double_commutator"):
        direction = rng.normal(size=sources[selected].shape)
        slope = np.vdot(bars["bar_" + selected], direction)
        for h in (2e-5, 1e-5):
            displaced = []
            for step in (h, -h):
                result = execute(
                    graph, {**sources, selected: sources[selected] + step * direction}
                ).outputs
                displaced.append(
                    sum(np.vdot(seeds[name], value) for name, value in result.items())
                )
            assert slope == pytest.approx(
                (displaced[0] - displaced[1]) / (2 * h), abs=2e-7
            )


@pytest.mark.parametrize(
    "dimensions", [(3, 4, 1, 2), (4, 4, 4, 2), (4, 4, -1, 2), (4, 4, 1, True)]
)
def test_orbital_source_dimensions(dimensions: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="positive AO"):
        molecular_orbital_sources(*dimensions)
