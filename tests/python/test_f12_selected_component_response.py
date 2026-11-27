"""Isolate three physical electronic observables before summing their moving-source response."""

import numpy as np
import pytest
from generativeqc._f12_unrestricted import unrestricted_mp2_f12_ao_response
from generativeqc_compiler.tensor import execute
from test_f12_separate_fitting_spaces import inputs, singles, whitener


def test_selected_f12_ordinary_and_singles_response_additivity() -> None:
    ao, jk = inputs()
    dims = (2, 1, 3, 4, 3)

    def calc(orbitals: dict, jks: dict, part: str) -> tuple:
        return unrestricted_mp2_f12_ao_response(
            dims, orbitals, execute, jkfit=jks, singles=singles, component=part
        )

    original, total = calc(ao, jk, "all")
    pieces = [calc(ao, jk, part) for part in ("ordinary", "f12", "singles")]
    assert original.active_pairs == 3
    assert pieces[0][0].ordinary_mp2 == pytest.approx(original.ordinary_mp2)
    assert pieces[1][0].f12_increment == pytest.approx(original.f12_increment)
    assert pieces[2][0].cabs_singles == pytest.approx(original.cabs_singles)
    assert (
        pieces[1][0].ordinary_mp2 == 0
        and pieces[1][0].cabs_singles == (0.0, 0.0)
        and pieces[1][0].active_pairs == 3
    )
    for k, v in total.items():
        np.testing.assert_allclose(
            sum(b[k] for _, b in pieces), v, atol=2e-13, rtol=1e-12
        )
    rng = np.random.default_rng(153)

    def tangent(src: dict) -> dict:
        result = {}
        for k, array in src.items():
            if k == "metric_whitener":
                continue
            value = rng.normal(size=array.shape) * 0.05
            result[k] = (
                value
                if k.startswith("coefficients")
                else (value + value.swapaxes(-1, -2)) / 2
            )
        return result

    da, dj = tangent(ao), tangent(jk)

    def moved(src: dict, direction: dict, h: float) -> dict:
        args = {k: src[k] + h * v for k, v in direction.items()}
        args["metric_whitener"] = whitener(args["metric/coulomb"])
        return args

    weights = pieces[1][1]
    slope = sum(np.vdot(weights[k], v) for k, v in da.items()) + sum(
        np.vdot(weights["jkfit/" + k], v) for k, v in dj.items()
    )
    for h in (1e-5, 3e-6):
        e = [
            calc(moved(ao, da, t), moved(jk, dj, t), "all")[0].f12_increment
            for t in (-h, h)
        ]
        assert slope == pytest.approx((e[1] - e[0]) / (2 * h), abs=2e-8)


def test_same_space_multiplier_uses_resolved_gaps_and_rejects_gauge_dependent_seed() -> (
    None
):
    from generativeqc._f12_focal_increment import _canonical_occupied_virtual_blocks
    from generativeqc_compiler.method.canonical_frame_response import (
        canonical_pair_terms,
    )

    e = np.array([-1.2, -0.6, 0.3, 0.3, 1.2])
    m = np.random.default_rng(83).normal(size=(5, 5))
    m[2, 3] = m[3, 2]
    seed = _canonical_occupied_virtual_blocks(m, e, 2)
    _, independent = canonical_pair_terms()
    for i, j in zip(*np.triu_indices(5, 1), strict=True):
        if (i < 2) != (j < 2) or e[i] == e[j]:
            assert seed[i, j] == 0.0
            continue
        expected = execute(
            independent,
            {
                "mpq": np.array(m[i, j]),
                "mqp": np.array(m[j, i]),
                "ep": np.array(e[i]),
                "eq": np.array(e[j]),
            },
        ).outputs["seed"]
        assert seed[i, j] == pytest.approx(float(expected), abs=1e-15, rel=0)
    m[2, 3] += 0.01
    with pytest.raises(ValueError, match="unresolved canonical gauge"):
        _canonical_occupied_virtual_blocks(m, e, 2)
