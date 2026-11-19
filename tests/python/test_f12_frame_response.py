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
