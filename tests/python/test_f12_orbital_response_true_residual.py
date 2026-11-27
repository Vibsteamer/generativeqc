"""Coupled DF-HF orbital response solves meet their tolerance on the true residual."""

import numpy as np
import pytest
from generativeqc._f12_focal_singles_response import _coupled_response_minres
from scipy.sparse.linalg import LinearOperator, minres


def hessian_like(seed: int = 7, rank: int = 286) -> np.ndarray:
    """SPD operator with the spectrum range of the HBr host orbital Hessian (0.30 .. 22.4)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((rank, rank)))
    eigenvalues = np.geomspace(0.296, 22.37, rank)
    return (q * eigenvalues) @ q.T


def driver(rank: int = 286) -> np.ndarray:
    b = np.random.default_rng(11).standard_normal(rank)
    return 0.413 * b / np.linalg.norm(b)


def relative_residual(h: np.ndarray, b: np.ndarray, z: np.ndarray) -> float:
    return float(np.linalg.norm(b - h @ z) / np.linalg.norm(b))


def test_unpreconditioned_solve_reaches_its_tolerance_on_the_true_residual() -> None:
    h, b = hessian_like(), driver()
    operator = LinearOperator(h.shape, matvec=lambda x: h @ x, dtype=np.float64)
    plain, status = minres(operator, b, rtol=1e-11, maxiter=4 * len(b))
    assert status == 0
    # SciPy's backward-error stop leaves the residual the host branch used to accept.
    assert relative_residual(h, b, plain) > 2e-11

    z = _coupled_response_minres(operator, b, None, False)
    exact = np.linalg.solve(h, b)
    assert relative_residual(h, b, z) <= 1e-11
    condition = 22.37 / 0.296
    assert np.linalg.norm(z - exact) <= condition * 1e-11 * np.linalg.norm(exact)
    assert np.linalg.norm(z - exact) < 0.1 * np.linalg.norm(plain - exact)


def test_rounding_level_null_mode_component_is_not_amplified() -> None:
    """UHF OH: a symmetry null mode (|lambda| 4.7e-13) with a 1e-15 driver component."""
    rank = 85
    q, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((rank, rank)))
    eigenvalues = np.concatenate(([4.7e-13], np.geomspace(0.168, 22.1, rank - 1)))
    h = (q * eigenvalues) @ q.T
    coefficients = np.random.default_rng(13).standard_normal(rank)
    coefficients[0] = 0.0
    coefficients *= 0.149 / np.linalg.norm(coefficients)
    coefficients[0] = 1.2e-15
    b = q @ coefficients
    operator = LinearOperator(h.shape, matvec=lambda x: h @ x, dtype=np.float64)

    z = _coupled_response_minres(operator, b, None, False)
    assert relative_residual(h, b, z) <= 1e-11
    null = q[:, 0]
    # Resolving the rounding-level component would put 1.2e-15 / 4.7e-13 along it.
    assert abs(null @ z) <= 1e-12
    physical = q[:, 1:] @ (coefficients[1:] / eigenvalues[1:])
    condition = 22.1 / 0.168
    error = z - null * (null @ z) - physical
    assert np.linalg.norm(error) <= condition * 1e-11 * np.linalg.norm(physical)


def test_unreachable_true_residual_fails_closed() -> None:
    h, b = hessian_like(), driver()
    noise = np.random.default_rng(3)

    def noisy(x: np.ndarray) -> np.ndarray:
        y = h @ x
        return y + 1e-9 * np.linalg.norm(y) * noise.standard_normal(len(y)) / np.sqrt(
            len(y)
        )

    operator = LinearOperator(h.shape, matvec=noisy, dtype=np.float64)
    with pytest.raises(RuntimeError, match="did not reach"):
        _coupled_response_minres(operator, b, None, False)


def test_matrix_free_solve_keeps_the_preconditioned_criterion() -> None:
    h, b = hessian_like(), driver()
    operator = LinearOperator(h.shape, matvec=lambda x: h @ x, dtype=np.float64)
    inverse_diagonal = 1.0 / np.diag(h)
    precond = LinearOperator(
        h.shape, matvec=lambda x: x * inverse_diagonal, dtype=np.float64
    )
    expected, status = minres(
        operator, b, M=precond, rtol=1e-12, maxiter=max(128, 4 * len(b))
    )
    assert status == 0
    assert np.array_equal(
        _coupled_response_minres(operator, b, precond, True), expected
    )
