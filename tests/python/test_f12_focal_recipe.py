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
