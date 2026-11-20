"""CABS complement and weighted nuclear response on full and truncated branches."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc.experimental.f12_focal import FocalOneElectron, focal_cabs_projector


def independent_kernel(s, obs, cutoff):
    features = np.linalg.cholesky(s).T
    q, _ = np.linalg.qr(features[:, :obs])
    aux = features[:, obs:]
    left, singular, _ = np.linalg.svd(aux - q @ (q.T @ aux), full_matrices=False)
    frame = np.linalg.solve(features, left[:, singular**2 > cutoff * singular[0] ** 2])
    return frame @ frame.T


@pytest.mark.parametrize("small, rank", [(0.45, 3), (0.017, 2)])
def test_focal_cabs_fixed_rank_direction(small, rank):
    rng = np.random.default_rng(907)
    a = np.array([[1.0, 0.1], [0.1, 0.86]])
    b = rng.normal(size=(2, 3)) * 0.1
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    g = q @ np.diag([small, 0.9, 0.9]) @ q.T
    s = np.block([[a, b], [b.T, g + b.T @ np.linalg.solve(a, b)]])
    u = rng.normal(size=s.shape)
    ds = rng.normal(size=s.shape)
    ds = (ds + ds.T) * 0.05
    one = FocalOneElectron(s, s, None, None, 2, "geometry", "basis", "cpu_reference", 0)
    request = SimpleNamespace(
        role="hf_cabs",
        backend="cpu_reference",
        surface_id="geometry",
        basis=SimpleNamespace(basis_id="basis", cabs=object()),
    )
    result = focal_cabs_projector(
        request, one, relative_cutoff=0.1, cotangent=u, expected_rank=rank
    )
    np.testing.assert_allclose(result.kernel, independent_kernel(s, 2, 0.1), atol=5e-14)
    np.testing.assert_allclose(
        result.frame.T @ s @ result.frame, np.eye(rank), atol=2e-14
    )
    np.testing.assert_allclose(s[:2] @ result.frame, 0, atol=3e-15)
    for h in [0.0004, 0.0002]:
        diff = (
            independent_kernel(s + h * ds, 2, 0.1)
            - independent_kernel(s - h * ds, 2, 0.1)
        ) / (2 * h)
        assert np.vdot(u, diff) == pytest.approx(
            np.vdot(result.overlap_bar, ds), abs=2e-7
        )
    with pytest.raises(ValueError, match="rank"):
        focal_cabs_projector(request, one, expected_rank=5 - rank, relative_cutoff=0.1)
    with pytest.raises(ValueError, match="surfaces"):
        focal_cabs_projector(request, replace(one, surface_id="displaced"))
