"""Physical H3 DF-HF + CABS singles, extended Fock and geometry response."""

import os
from pathlib import Path

import numpy as np
import pytest
import test_focal_df_reference as reference_tests
import test_focal_one_electron_sources as one_electron_tests
from generativeqc.experimental.f12_focal import (
    FocalSeries,
    df_reference,
    focal_cabs_projector,
    one_electron_sources,
    plan_singlepoint,
    reference_cabs_singles,
)

hydrogen_bases = reference_tests.hydrogen_bases
molecular_library = one_electron_tests.molecular_library




def test_ill_conditioned_ao_fock_congruence() -> None:
    from decimal import Decimal, localcontext

    from generativeqc._f12_cabs_fock import _project

    rng = np.random.default_rng(143)
    a = rng.normal(size=(24, 24))
    operator = (a + a.T) * 0.5
    frame = rng.normal(size=(24, 4)) * 2e4
    with localcontext() as ctx:
        ctx.prec = 60
        decimal_array = np.vectorize(Decimal.from_float, otypes=[object])
        c = decimal_array(frame)
        expected = np.asarray(c.T @ decimal_array(operator) @ c, dtype=np.float64)
    projected = _project(operator, frame, False, None)
    np.testing.assert_array_max_ulp(projected, expected, maxulp=2)
    np.testing.assert_allclose(projected, projected.T, rtol=0, atol=2e-8)
    bad = operator.copy()
    bad[0, 1] += 1e-6
    with pytest.raises(ValueError, match="physical AO DF Fock"):
        _project(bad, frame, False, None)


@pytest.mark.parametrize("occupied", [(2, 1), (2, 0)])
def test_physical_spin_exchange_and_fock_response(occupied: tuple[int, int]) -> None:
    from collections.abc import Callable

    from generativeqc._f12_cabs_fock import FockBars, Sources, physical_spin_operators
    from generativeqc_compiler.tensor import execute
    from scipy.linalg import solve_triangular

    rng = np.random.default_rng(563)

    def symmetric(x: np.ndarray) -> np.ndarray:
        return 0.5 * (x + x.swapaxes(-1, -2))

    n, q = 6, 4
    h, dh = (symmetric(rng.normal(size=(n, n))) for _ in range(2))
    raw, dr = (symmetric(rng.normal(size=(q, n, n))) for _ in range(2))
    a = rng.normal(size=(q, q))
    metric, dm = a.T @ a + np.eye(q), symmetric(rng.normal(size=(q, q)))
    frames, dc = (rng.normal(size=(2, n, n)) for _ in range(2))

    def calculate(
        t: float,
    ) -> tuple[dict[str, np.ndarray], Callable[[Sources], FockBars]]:
        w = solve_triangular(np.linalg.cholesky(metric + t * dm), np.eye(q), lower=True)
        return physical_spin_operators(
            h + t * dh, raw + t * dr, w, frames + t * dc, occupied, execute, False, None
        )

    operators, pullback = calculate(0)
    seeds = {key: rng.normal(size=value.shape) for key, value in operators.items()}
    inv = np.linalg.inv(metric)
    densities = [c[:, :o] @ c[:, :o].T for c, o in zip(frames, occupied, strict=True)]
    j = np.einsum("amn,ab,bij,ij->mn", raw, inv, raw, sum(densities))
    for name, c, density in zip(("alpha", "beta"), frames, densities, strict=True):
        k = np.einsum("ami,ab,bnj,ij->mn", raw, inv, raw, density)
        np.testing.assert_allclose(
            operators["exchange_" + name], c.T @ k @ c, atol=1e-12
        )
        np.testing.assert_allclose(
            operators["fock_" + name], c.T @ (h + j - k) @ c, atol=1e-12
        )
    coeff, hbar, rbar, mbar = pullback(seeds)
    analytic = (
        np.vdot(coeff, dc) + np.vdot(hbar, dh) + np.vdot(rbar, dr) + np.vdot(mbar, dm)
    )
    for step in (1e-5, 5e-6):
        plus, minus = (calculate(t)[0] for t in (step, -step))
        fd = sum(np.vdot(seed, plus[k] - minus[k]) for k, seed in seeds.items()) / (
            2 * step
        )
        assert analytic == pytest.approx(fd, abs=1.5e-6, rel=0)
