"""Surface identity and variationally consistent Cartesian derivatives of a mixed focal recipe."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from generativeqc.experimental.f12_focal import (
    FocalResult,
    FocalSeries,
    FocalTerm,
    Role,
    combine,
)


def point(role: Role, n: int, x: np.ndarray, scale: float, family: str) -> FocalTerm:
    """Quadratic energy surface with an exactly known inverse-cardinal error."""
    power = 5.0 if role == "hf_cabs" else 3.0
    factor = scale + (0 if role == "post_mp2" else n**-power)
    delta = x[1] - x[0]
    force = np.vstack((delta, -delta)) * factor
    return FocalTerm(
        "test geometry",
        family,
        f"{family}-{n}",
        n,
        role,
        float(delta @ delta) * factor / 2,
        force,
    )


@pytest.mark.parametrize("lo,hi", [(2, 3), (3, 4), (4, 5)])
def test_power_pairs_and_forces(lo: int, hi: int) -> None:
    xyz = np.array([[0.2, 0.0, 0.1], [0.7, -0.3, 0.8]])

    def run(x: np.ndarray) -> FocalResult:
        def series(role: Role, power: float, scale: float, family: str) -> FocalSeries:
            return FocalSeries(
                point(role, hi, x, scale, family),
                point(role, lo, x, scale, family),
                power,
            )

        return combine(
            series("hf_cabs", 5, 3.0, "parent"),
            series("mp2_f12_correlation", 3, 0.5, "ha"),
            point("post_mp2", 3, x, -0.25, "a"),
        )

    result = run(xyz)
    np.testing.assert_allclose(
        result.forces,
        point("post_mp2", 3, xyz, 3.25, "exact").forces,
        atol=1e-14,
        rtol=0,
    )
    for i, j in np.ndindex(xyz.shape):
        perturb = np.zeros_like(xyz)
        perturb[i, j] = 1e-5
        fd = -(run(xyz + perturb).energy - run(xyz - perturb).energy) / 2e-5
        assert abs(fd - result.forces[i, j]) < 1e-9


def test_single_endpoint_and_identity_mismatches() -> None:
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    hf = point("hf_cabs", 4, x, 3.0, "parent")
    corr = point("mp2_f12_correlation", 3, x, 0.5, "ha")
    high = point("post_mp2", 2, x, -0.25, "a")
    s = FocalSeries
    result = combine(s(hf), s(corr), high)
    assert result.energy == hf.energy + corr.energy + high.energy
    for bad in [
        replace(corr, surface_id="excited root"),
        replace(corr, role="hf_cabs"),
        replace(corr, forces=np.zeros((1, 3))),
        replace(corr, energy=np.nan),
    ]:
        with pytest.raises(ValueError):
            combine(s(hf), s(bad), high)
    with pytest.raises(ValueError):
        combine(
            s(hf, replace(hf, cardinal=3, series_id="other", basis_id="other-3"), 5),
            s(corr),
            high,
        )


def test_independent_ccsd_triples_and_exponential_hf() -> None:
    # Four surfaces with known limits and distinct errors; force is minus dE/dx.
    def component(role: Role, family: str, cardinal: int, x: float) -> FocalTerm:
        coefficients = {
            "hf_cabs": 2.75 + np.exp(-1.3 * cardinal),
            "mp2_f12_correlation": -0.6 + cardinal**-4.5,
            "ccsd_minus_mp2": 0.125 - 2 * cardinal**-3.0,
            "triples": -0.04 + 1.5 * cardinal**-2.0,
        }
        k = coefficients[role]
        return FocalTerm(
            "ordered",
            family,
            f"{family}-{cardinal}",
            cardinal,
            role,
            k * x**2 / 2,
            np.array([[k * x, 0.0, 0.0], [-k * x, 0.0, 0.0]]),
        )

    def run(x: float) -> FocalResult:
        def term(role: Role, family: str, n: int) -> FocalTerm:
            return component(role, family, n, x)

        return combine(
            FocalSeries(
                term("hf_cabs", "a", 5), term("hf_cabs", "a", 4), 1.3, "exponential"
            ),
            FocalSeries(
                term("mp2_f12_correlation", "ha", 4),
                term("mp2_f12_correlation", "ha", 3),
                4.5,
            ),
            FocalSeries(
                term("ccsd_minus_mp2", "parent", 3),
                term("ccsd_minus_mp2", "parent", 2),
                3.0,
            ),
            triples=FocalSeries(
                term("triples", "aug", 4), term("triples", "aug", 3), 2.0
            ),
        )

    x = 1.25
    result = run(x)
    k = 2.75 - 0.6 + 0.125 - 0.04
    assert result.energy == pytest.approx(k * x * x / 2, abs=1e-14, rel=0)
    np.testing.assert_allclose(result.forces[:, 0], [k * x, -k * x], atol=1e-14, rtol=0)
    for h in (4e-4, 2e-4):
        assert (
            abs(result.forces[1, 0] + (run(x + h).energy - run(x - h).energy) / (2 * h))
            < 5e-11
        )
    assert [w[2] for w in result.weights] == [4, 5, 3, 4, 2, 3, 3, 4]


def test_high_order_roles_cannot_double_count_triples() -> None:
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    h = point("hf_cabs", 4, x, 1.0, "h")
    c = point("mp2_f12_correlation", 3, x, 2.0, "c")
    p = point("post_mp2", 3, x, 1.0, "p")
    t = replace(p, role="triples")
    with pytest.raises(ValueError, match="role"):
        combine(FocalSeries(h), FocalSeries(c), p, triples=t)
    with pytest.raises(ValueError, match="role"):
        combine(FocalSeries(h), FocalSeries(c), replace(p, role="ccsd_minus_mp2"))
    with pytest.raises(ValueError, match="exponential"):
        combine(
            FocalSeries(h),
            FocalSeries(
                c, replace(c, cardinal=2, basis_id="other"), 2.0, "exponential"
            ),
            p,
        )
    with pytest.raises(ValueError, match="identity"):
        combine(
            FocalSeries(h),
            FocalSeries(c),
            replace(p, role="ccsd_minus_mp2"),
            triples=replace(t, surface_id="different root"),
        )
