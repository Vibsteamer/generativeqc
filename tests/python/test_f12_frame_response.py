"""Check invariant frame-to-projector cotangents without choosing an eigenvector gauge."""

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
