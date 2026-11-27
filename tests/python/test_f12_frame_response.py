"""Check invariant frame-to-projector cotangents without choosing an eigenvector gauge."""

from collections.abc import Callable

import numpy as np
import pytest
from generativeqc_compiler.mp2.f12_frame_response import frame_kernel_cotangent
from generativeqc_compiler.tensor import execute


@pytest.mark.parametrize("rank", [1, 3, 5])
def test_invariant_scalar_and_frame_rotations(rank: int) -> None:
    rng = np.random.default_rng(1987 + rank)
    atomic = 7
    a = rng.normal(size=(atomic, atomic))
    overlap = a.T @ a + np.eye(atomic)
    q, _ = np.linalg.qr(rng.normal(size=(atomic, rank)))
    frame = np.linalg.solve(np.linalg.cholesky(overlap).T, q)
    h = rng.normal(size=(atomic, atomic))
    h = (h + h.T) / 2

    def observable(c: np.ndarray) -> float:
        kernel = c @ c.T
        return float(np.vdot(h, kernel) + 0.2 * np.vdot(kernel, kernel))

    kernel = frame @ frame.T
    bar = 2 * h @ frame + 0.8 * kernel @ frame
    graph = frame_kernel_cotangent(atomic, rank)
    result = execute(
        graph, {"overlap": overlap, "frame": frame, "frame_bar": bar}
    ).outputs
    np.testing.assert_allclose(result["dual_gram"], np.eye(rank), atol=2e-14)
    np.testing.assert_allclose(result["gauge_skew"], 0, atol=2e-14)
    np.testing.assert_allclose(result["reconstruction_residual"], 0, atol=2e-14)
    np.testing.assert_allclose(result["kernel_bar"], result["kernel_bar"].T, atol=2e-14)
    direction = rng.normal(size=frame.shape)
    direction /= np.linalg.norm(direction)
    tangent = direction @ frame.T + frame @ direction.T
    slope = np.vdot(result["kernel_bar"], tangent)
    assert slope == pytest.approx(np.vdot(bar, direction), abs=2e-13)
    for step in (1e-4, 5e-5):
        fd = (
            observable(frame + step * direction) - observable(frame - step * direction)
        ) / (2 * step)
        assert slope == pytest.approx(fd, abs=1e-8)
    rotation, _ = np.linalg.qr(rng.normal(size=(rank, rank)))
    rotated = execute(
        graph,
        {"overlap": overlap, "frame": frame @ rotation, "frame_bar": bar @ rotation},
    ).outputs
    np.testing.assert_allclose(rotated["kernel_bar"], result["kernel_bar"], atol=2e-13)


def test_expose_gauge_dependence_and_bad_metric_dual() -> None:
    frame = np.eye(5)[:, :3]
    bar = np.zeros_like(frame)
    bar[0, 1] = 1
    graph = frame_kernel_cotangent(5, 3)
    result = execute(
        graph, {"overlap": np.eye(5), "frame": frame, "frame_bar": bar}
    ).outputs
    assert np.linalg.norm(result["gauge_skew"]) > 1
    assert np.linalg.norm(result["reconstruction_residual"]) > 0.5
    bad = execute(
        graph, {"overlap": np.eye(5) * 2, "frame": frame, "frame_bar": frame}
    ).outputs
    assert np.linalg.norm(bad["dual_gram"] - np.eye(3)) > 1


@pytest.mark.parametrize("dimensions", [(0, 1), (3, 0), (3, 4), (True, 1)])
def test_reject_invalid_frame_dimensions(dimensions: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="frame rank"):
        frame_kernel_cotangent(*dimensions)


@pytest.mark.parametrize("weak_overlap", [1.0, 2e-8])
def test_complete_cabs_physical_direction(weak_overlap: float) -> None:
    from benchmarks.f12_molecular_frame import constraint_weight

    rng = np.random.default_rng(18932)
    n, o = 13, 6
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = (q * np.geomspace(weak_overlap, 1, n)) @ q.T
    root = np.linalg.solve(np.linalg.cholesky(s).T, np.eye(n))
    orbitals, frame = root[:, :o], root[:, o:]
    u = s @ root[:, -2:]
    bar = 2 * u @ (u.T @ frame)
    d = rng.normal(size=(n, n))
    d = np.linalg.cholesky(s) @ (d + d.T) @ np.linalg.cholesky(s).T
    weight, _ = constraint_weight(
        s,
        orbitals,
        frame,
        bar,
        execute,
        metric_gram=lambda s, c: np.asarray(
            c.T.astype(np.longdouble) @ s.astype(np.longdouble) @ c, float
        ),
    )
    analytic = np.vdot(weight, d)
    solved = np.linalg.solve(s, u)
    ao = np.linalg.solve(s[:o, :o], u[:o])
    expected = -np.vdot(solved, d @ solved) + np.vdot(ao, d[:o, :o] @ ao)
    assert analytic == pytest.approx(expected, rel=2e-8, abs=2e-8)
    for h in (1e-4, 5e-5):

        def observable(metric: np.ndarray) -> float:
            return np.vdot(u, np.linalg.solve(metric, u)) - np.vdot(
                u[:o], np.linalg.solve(metric[:o, :o], u[:o])
            )

        slope = (observable(s + h * d) - observable(s - h * d)) / (2 * h)
        assert analytic == pytest.approx(slope, rel=2e-5, abs=2e-5)


def test_truncated_space_preserves_projector_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from benchmarks import f12_molecular_frame as molecular

    n, o, r = 6, 2, 3
    s = np.eye(n)
    frame = np.eye(n)[:, o : o + r]
    coeff = np.zeros((n, o + r))
    coeff[:o, :o] = np.eye(o)
    grad = np.ones_like(s) / 100

    def projector(
        overlap: np.ndarray,
        obs: int,
        *,
        expected_rank: int,
        cotangent: np.ndarray | None = None,
    ) -> SimpleNamespace:
        assert expected_rank == r
        return SimpleNamespace(frame=frame, rank=r, overlap_bar=grad)

    def evaluate(data: dict, run: Callable, **kwargs: object) -> tuple:
        return 0.5, {"coefficients": data["coefficients"].copy()}, {}

    monkeypatch.setattr(molecular, "evaluate", evaluate)
    data = {
        "overlap": s,
        "coefficients": coeff,
        "ao_permutation": np.arange(n),
        "space": [1, o, o + r],
    }
    _, original, _ = molecular.chain(data, execute, projector, None, pullback=True)
    _, optional, response = molecular.chain(
        data, execute, projector, None, pullback=True, full_rank_constraint=True
    )
    np.testing.assert_array_equal(original["overlap"], grad)
    np.testing.assert_array_equal(original["overlap"], optional["overlap"])
    assert response["cabs_response"] == {
        "requested_full_rank": True,
        "applied": "projector",
        "retained_rank": r,
        "complement_dimension": n - o,
    }


def test_compensated_full_rank_cabs_orthonormality_and_overlap_bar() -> None:
    """Verify Ozaki-4 compensated CABS dual_gram check and direct full-rank bar_S
    under diffuse ha-Q-F12-like overlap conditioning (||C||_F ~ 1e4, ||K||_F ~ 1e8)."""
    from scipy.linalg import cho_factor, cho_solve

    from generativeqc._f12_focal_projector import (
        compensated_cabs_frame_cotangent_checks,
        compensated_cabs_orthonormality,
        full_rank_cabs_overlap_bar,
    )

    rng = np.random.default_rng(20261125)
    n, o = 40, 20
    r = n - o
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.geomspace(2e-9, 4.0, n)
    s = (q * eigs) @ q.T
    s = 0.5 * (s + s.T)

    # Build a full-rank OBS-orthogonal CABS frame with one compensated Gram refinement step
    # (matching cabs_projector_cuda.cu).
    chol = cho_factor(s[:o, :o], lower=True)
    cross = s[:o, o:]
    x = -cho_solve(chol, cross)
    lift = np.vstack((x, np.eye(r)))
    gram = 0.5 * ((s[o:, o:] + cross.T @ x) + (s[o:, o:] + cross.T @ x).T)
    vals, vecs = np.linalg.eigh(gram)
    assert vals[0] > 0
    frame = lift @ (vecs / np.sqrt(vals))
    # Newton-Schulz refinement in longdouble to realize physical S-orthonormality.
    for _ in range(2):
        g_oz, _ = compensated_cabs_orthonormality(s, o, frame)
        frame = frame @ (1.5 * np.eye(r) - 0.5 * g_oz)

    gram_oz, obs_dual_oz = compensated_cabs_orthonormality(s, o, frame)
    assert np.max(np.abs(gram_oz - np.eye(r))) < 1e-10
    assert np.max(np.abs(obs_dual_oz)) < 1e-10

    # Gauge-invariant scalar f(C) = <A, C C^T> with symmetric A => G = 2 A C, C^T G symmetric.
    a_raw = rng.normal(scale=0.1, size=(n, n))
    a_sym = 0.5 * (a_raw + a_raw.T)
    # Use a low-norm physical AO operator in the dual space (like Fock * s_bar).
    u = s @ rng.normal(scale=0.1, size=(n, 3))
    bar_c = 2.0 * u @ (u.T @ frame)

    bar_s = full_rank_cabs_overlap_bar(s, o, frame, bar_c)
    assert np.allclose(bar_s, bar_s.T, atol=1e-15, rtol=0)

    # Independent closed-form oracle using K = S^{-1} - O and exact linear solves:
    # d<u u^T, S^{-1} - O> = -<S^{-1} u, dS S^{-1} u> + <S_{oo}^{-1} u_o, dS_{oo} S_{oo}^{-1} u_o>
    d_raw = rng.normal(size=(n, n))
    d_sym = 0.5 * (d_raw + d_raw.T)
    # Perturbation preserving S-conditioning scale.
    l_s = np.linalg.cholesky(s)
    d_phys = l_s @ d_sym @ l_s.T
    solved = np.linalg.solve(s, u)
    solved_o = cho_solve(chol, u[:o])
    expected_slope = float(
        -np.vdot(solved, d_phys @ solved) + np.vdot(solved_o, d_phys[:o, :o] @ solved_o)
    )
    analytic_slope = float(np.vdot(bar_s, d_phys))
    assert analytic_slope == pytest.approx(expected_slope, rel=1e-9, abs=1e-11)

    # Check compensated_cabs_frame_cotangent_checks accepts a valid frame even if FP64 dual_gram drifted,
    # while still rejecting genuinely non-orthonormal frames and genuinely skew frame_bar.
    transfer = execute(
        frame_kernel_cotangent(n, r),
        {"overlap": s, "frame": frame, "frame_bar": bar_c},
    ).outputs
    perturbed_transfer = {
        **transfer,
        "dual_gram": transfer["dual_gram"] + 5e-7 * np.eye(r),
    }
    assert compensated_cabs_frame_cotangent_checks(
        s, o, frame, bar_c, perturbed_transfer, check_dual_gram=True
    )

    # Genuine non-orthonormal frame (scaled by 1.001) must still be rejected.
    bad_frame = frame * 1.001
    bad_transfer = execute(
        frame_kernel_cotangent(n, r),
        {"overlap": s, "frame": bad_frame, "frame_bar": bar_c},
    ).outputs
    assert not compensated_cabs_frame_cotangent_checks(
        s, o, bad_frame, bar_c, bad_transfer, check_dual_gram=True
    )

    # Genuine gauge-dependent frame_bar (skew C^T G) must still be rejected.
    skew_bar = bar_c.copy()
    skew_bar[:, 0] += s @ frame[:, 1]
    skew_transfer = execute(
        frame_kernel_cotangent(n, r),
        {"overlap": s, "frame": frame, "frame_bar": skew_bar},
    ).outputs
    assert not compensated_cabs_frame_cotangent_checks(
        s, o, frame, skew_bar, skew_transfer, check_dual_gram=True
    )
