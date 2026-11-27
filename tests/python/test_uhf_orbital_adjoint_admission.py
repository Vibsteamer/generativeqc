"""The dense DF-UHF orbital adjoint checks input sensitivity and forward error.

Each refinement correction must at least halve its predecessor. Publishing also requires a
posterior forward-error bound, even when the first correction is below the threshold. Exact
rational solutions are the oracle throughout.
"""

import ctypes
import typing
from fractions import Fraction

import numpy as np
import pytest
from generativeqc import _native

DOUBLE_P = ctypes.POINTER(ctypes.c_double)
UNAVAILABLE = ("no CUDA-capable device", "CUDA driver version is insufficient")
Rational = list[list[Fraction]]
FORCE_RELATIVE_TOLERANCE = 2.0**-38


def two_by_two(
    x: float, y: float, gap_alpha: float, gap_beta: float
) -> tuple[list[float], Rational]:
    """One alpha and one beta rotation with q = 1 and oo = vv = 0.

    The ABI then assembles H = [[g_a + x^2, 2xy], [2xy, g_b + y^2]].
    """
    factors = [x, 0.0, 0.0, gap_alpha, y, 0.0, 0.0, gap_beta]
    exact = [
        [Fraction(gap_alpha) + Fraction(x) ** 2, 2 * Fraction(x) * Fraction(y)],
        [2 * Fraction(x) * Fraction(y), Fraction(gap_beta) + Fraction(y) ** 2],
    ]
    return factors, exact


def exact_solution(h: Rational, driver: np.ndarray) -> list[Fraction]:
    d0, d1 = (Fraction(v) for v in driver)
    det = h[0][0] * h[1][1] - h[0][1] * h[1][0]
    return [(h[1][1] * d0 - h[0][1] * d1) / det, (h[0][0] * d1 - h[1][0] * d0) / det]


def assert_accurate(z: np.ndarray, exact: list[Fraction]) -> None:
    published = [Fraction(v) for v in z]
    error = max(abs(a - b) for a, b in zip(published, exact, strict=True))
    assert error <= Fraction(2) ** -39 * max(map(abs, published))


@pytest.fixture(scope="module")
def adjoint() -> typing.Callable[..., tuple[typing.Any, ...]]:
    try:
        library = _native.load_library(device="cuda")
        routine = library.generativeqc_uhf_orbital_adjoint_cuda_v1
    except (AttributeError, OSError, RuntimeError) as error:
        pytest.skip(f"CUDA DF-UHF orbital adjoint unavailable: {error}")
    routine.argtypes = [ctypes.c_int] * 7 + [
        DOUBLE_P,
        ctypes.c_size_t,
        ctypes.c_size_t,
        DOUBLE_P,
        DOUBLE_P,
        DOUBLE_P,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    routine.restype = ctypes.c_int

    def call(
        factors: list[float],
        driver: np.ndarray,
        directions: int = 0,
        auxiliary: int = 1,
        *,
        alpha: int = 1,
        beta: int = 1,
        orbitals: int = 2,
    ) -> tuple[typing.Any, ...]:
        rank = alpha * (orbitals - alpha) + beta * (orbitals - beta)
        rhs = np.linspace(0.3, 0.9, rank * directions)
        fixed = np.full(directions, 0.25)
        packed = np.ascontiguousarray(np.concatenate([factors, driver, rhs, fixed]))
        z, g, force = (
            np.full(rank, -37.0),
            np.full(directions, -37.0),
            np.full(directions, -37.0),
        )
        error = ctypes.create_string_buffer(256)
        status = routine(
            0,
            alpha,
            beta,
            orbitals,
            auxiliary,
            directions,
            0,
            packed.ctypes.data_as(DOUBLE_P),
            packed.size,
            1 << 20,
            z.ctypes.data_as(DOUBLE_P),
            g.ctypes.data_as(DOUBLE_P) if directions else None,
            force.ctypes.data_as(DOUBLE_P) if directions else None,
            error,
            len(error),
        )
        return status, error.value.decode(), z, g, force

    factors, _ = two_by_two(0.6, 0.7, 0.3, 0.9)
    status, detail, *_ = call(factors, np.array([1.0, 0.0]))
    if status and any(text in detail for text in UNAVAILABLE):
        pytest.skip(f"CUDA device unavailable: {detail}")
    return call


def scalar_gap(vo: list[float], pivot: Fraction, rate: Fraction) -> float:
    """The gap for a diagonal Hessian entry with a specified refinement rate.

    The alpha gap cancels sum vo_q^2 down to ``pivot`` while the FP64 assembly
    v = gap; v = fl(v + 2 vo_q^2); v = fl(v - vo_q^2) rounds at |v| ~ 1, so the LU pivot is
    pivot / (1 - rate) and each refinement step multiplies the alpha error by ``rate``. Every
    vo_q^2 is exact, so FMA contraction cannot change the assembly.
    """
    gap = float(pivot - sum(Fraction(x) ** 2 for x in vo))
    assembled = gap
    for x in vo:
        assembled += 2.0 * x * x
        assembled -= x * x
    assert Fraction(gap) + sum(Fraction(x) ** 2 for x in vo) == pivot
    assert 1 - pivot / Fraction(assembled) == rate
    return gap


def scalar_refinement(
    vo: list[float], pivot: Fraction, rate: Fraction
) -> tuple[list[float], Fraction]:
    q = len(vo)
    gap = scalar_gap(vo, pivot, rate)
    return [*vo, *[0.0] * (2 * q), gap, *[0.0] * (3 * q), 1.0], pivot


def not_halving() -> tuple[list[float], Fraction]:
    return scalar_refinement(
        [1.5 * 2.0**-28, 1.25 * 2.0**-27, 0.75], Fraction(2) ** -57, Fraction(15, 16)
    )


def contracting() -> tuple[list[float], Fraction]:
    return scalar_refinement([2.0**-28, 0.625], 5 * Fraction(2) ** -56, Fraction(3, 8))


def near_singular() -> tuple[list[float], Rational, np.ndarray]:
    x, y, gap_alpha = 0.6, 0.7, 0.3
    gap_beta = (2 * x * y) ** 2 / (gap_alpha + x * x) - y * y + 1e-12
    factors, h = two_by_two(x, y, gap_alpha, gap_beta)
    as_float = np.array([[float(v) for v in row] for row in h])
    return factors, h, np.linalg.eigh(as_float)[1]


def test_driver_exciting_a_soft_mode_is_rejected_without_outputs(
    adjoint: typing.Callable,
) -> None:
    factors, _, _ = near_singular()
    status, detail, z, g, force = adjoint(factors, np.array([1.0, 0.0]), directions=1)
    assert status == 1
    assert "ill-conditioned DF-UHF orbital Hessian: driver amplification" in detail
    assert np.all(z == -37.0) and np.all(g == -37.0) and np.all(force == -37.0)


def test_stiff_driver_on_a_near_singular_hessian_is_exact_in_every_output(
    adjoint: typing.Callable,
) -> None:
    # The LU solve alone leaves a soft-mode error of about 1e-4 that a force contraction
    # with a soft-mode component sees.
    factors, h, modes = near_singular()
    driver = modes[:, 1]
    status, detail, z, g, force = adjoint(factors, driver, directions=1)
    assert status == 0, detail
    exact = exact_solution(h, driver)
    assert_accurate(z, exact)
    gradient = Fraction(0.3) * exact[0] + Fraction(0.9) * exact[1]
    tolerance = FORCE_RELATIVE_TOLERANCE * abs(float(gradient))
    assert abs(g[0] - float(gradient)) <= tolerance
    assert abs(force[0] - float(Fraction(0.25) - gradient)) <= tolerance


def test_symmetry_protected_null_mode_with_symmetric_driver_is_exact(
    adjoint: typing.Callable,
) -> None:
    x = 0.6
    factors, h = two_by_two(x, x, x * x + 1e-12, x * x + 1e-12)
    driver = np.array([1.0, 1.0])
    status, detail, z, *_ = adjoint(factors, driver)
    assert status == 0, detail
    assert_accurate(z, exact_solution(h, driver))


def test_numerically_rank_deficient_hessian_fails_closed(
    adjoint: typing.Callable,
) -> None:
    # The exact Hessian is regular, but its smallest eigenvalue, fl(x^2) - x^2, is below the
    # rounding of its entries, so refinement from an FP64 factorization cannot resolve it.
    x = 0.6
    factors, _ = two_by_two(x, x, x * x, x * x)
    status, detail, z, g, force = adjoint(factors, np.array([1.0, 1.0]), directions=1)
    assert status == 1
    assert "DF-UHF orbital Hessian" in detail
    assert np.all(z == -37.0) and np.all(g == -37.0) and np.all(force == -37.0)


def test_well_conditioned_solve_is_admitted_and_exact(adjoint: typing.Callable) -> None:
    factors, h = two_by_two(0.6, 0.7, 0.3, 0.9)
    driver = np.array([0.8, -0.35])
    status, detail, z, g, force = adjoint(factors, driver, directions=2)
    assert status == 0, detail
    exact = [float(v) for v in exact_solution(h, driver)]
    assert np.linalg.norm(z - exact) <= 1e-13 * np.linalg.norm(exact)
    rhs = np.linspace(0.3, 0.9, 4).reshape(2, 2)
    np.testing.assert_allclose(g, rhs @ exact, rtol=1e-13, atol=0)
    np.testing.assert_allclose(force, 0.25 - rhs @ exact, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize(
    "z_alpha", [18 * 2.0**-40, 2.0**-36], ids=["second-correction", "first-correction"]
)
def test_refinement_that_does_not_halve_never_publishes_its_error(
    adjoint: typing.Callable, z_alpha: float
) -> None:
    # Each step keeps 15/16 of the alpha error and |z|_inf = 1. For z_alpha = 18 * 2^-40 the
    # corrections are 1.055 and 0.989 times 2^-40, for 2^-36 the first is 0.938 * 2^-40: the
    # final correction passes 2^-40, but the error it leaves is 16 times larger.
    factors, pivot = not_halving()
    driver = np.array([float(pivot * Fraction(z_alpha)), 1.0])
    status, detail, z, g, force = adjoint(factors, driver, directions=1, auxiliary=3)
    if status:
        assert "ill-conditioned DF-UHF orbital Hessian" in detail
        assert np.all(z == -37.0) and np.all(g == -37.0) and np.all(force == -37.0)
    else:
        assert_accurate(z, [Fraction(z_alpha), Fraction(1)])


def test_refinement_contracting_by_three_eighths_is_published_within_the_bound(
    adjoint: typing.Callable,
) -> None:
    # Corrections of 15, 5.6, 2.1 and 0.79 times 2^-40 leave an error of 1.27 * 2^-40.
    factors, pivot = contracting()
    z_alpha = 2.0**-34
    driver = np.array([float(pivot * Fraction(z_alpha)), 1.0])
    status, detail, z, *_ = adjoint(factors, driver, auxiliary=2)
    assert status == 0, detail
    assert_accurate(z, [Fraction(z_alpha), Fraction(1)])


def test_unexcited_noncontracting_rotation_publishes_the_exact_solve(
    adjoint: typing.Callable,
) -> None:
    # The LU solve is exact, so the correction and enclosed residual both vanish.
    factors, _ = not_halving()
    driver = np.array([0.0, 1.0])
    status, detail, z, g, force = adjoint(factors, driver, directions=1, auxiliary=3)
    assert status == 0, detail
    assert z.tolist() == [0.0, 1.0]
    np.testing.assert_allclose([g[0], force[0]], [0.9, 0.25 - 0.9], rtol=1e-15)


def test_fast_correction_cannot_mask_error_in_a_slow_rotation(
    adjoint: typing.Callable,
) -> None:
    # The FP64 solve misses fractions 15/16 and 3/8 of the first two rotations.
    # In units of 2^-40, the error in the first rotation is 7.5,
    # but its first correction is only 0.469. The second rotation has a
    # 0.938 first correction. The next correction's infinity norm is 0.440,
    # so a single normwise contraction probe would incorrectly admit the solve.
    slow_vo = [1.5 * 2.0**-28, 1.25 * 2.0**-27, 0.75]
    fast_vo = [2.0**-28, 0.625]
    slow_pivot, fast_pivot = Fraction(2) ** -57, 5 * Fraction(2) ** -56
    gaps = [
        scalar_gap(slow_vo, slow_pivot, Fraction(15, 16)),
        scalar_gap(fast_vo, fast_pivot, Fraction(3, 8)),
        1.0,
    ]
    q = len(slow_vo) + len(fast_vo)
    vo = np.zeros((q, 3))
    vo[: len(slow_vo), 0] = slow_vo
    vo[len(slow_vo) :, 1] = fast_vo
    factors = [*vo.ravel(), *[0.0] * (q + 9 * q), *gaps]
    exact = [8 * Fraction(2) ** -40, 4 * Fraction(2) ** -40, Fraction(1)]
    driver = np.array([float(slow_pivot * exact[0]), float(fast_pivot * exact[1]), 1.0])
    status, detail, z, g, force = adjoint(
        factors, driver, directions=1, auxiliary=q, alpha=1, beta=0, orbitals=4
    )
    if status:
        assert "ill-conditioned DF-UHF orbital Hessian" in detail
        assert np.all(z == -37.0) and np.all(g == -37.0) and np.all(force == -37.0)
    else:
        assert_accurate(z, exact)
