"""Physical panel sharing and role-specific CUDA fitting factors."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc import _f12_focal_metric as metric
from generativeqc import _f12_focal_molecular as molecular
from test_focal_molecular_provider import XYZ, request


@pytest.fixture
def panel_shell(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    layouts = []

    def layout(*args: object, **kwargs: object) -> dict:
        result = {"ao_sizes": (2, 2), "ao_offsets": (0, 8)}
        layouts.append(result)
        return result

    def raw(*args: object, **kwargs: object) -> dict:
        return {
            "metric": np.array([[3.0, 1.0], [1.0, 2.0]]),
            "ao_coulomb": np.zeros((2, 2, 2)),
        }

    monkeypatch.setattr(molecular, "focal_shell_layout", layout)
    monkeypatch.setattr(molecular, "forward_integrals", lambda *a, **k: (raw(), {}))
    monkeypatch.setattr(molecular, "forward_metrics_only", raw)
    return layouts


@pytest.mark.parametrize("fits", [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, 2)])
@pytest.mark.parametrize("streaming", [False, True])
def test_cuda_role_factors_and_raw_panel_sharing(
    fits: tuple[int, int, int],
    streaming: bool,
    panel_shell: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def factor(
        m: np.ndarray, library: Path, *, budget_bytes: int, factorization: str = "symmetric"
    ) -> tuple[np.ndarray, dict]:
        assert library == Path("response.so") and budget_bytes == 3 << 20
        calls.append((m, factorization))
        if factorization == "inverse_cholesky":
            w = np.linalg.solve(np.linalg.cholesky(m), np.eye(len(m)))
        else:
            eig, u = np.linalg.eigh(m)
            w = (u / np.sqrt(eig)) @ u.T
        return w, {"factorization": factorization}

    monkeypatch.setenv("GENERATIVEQC_F12_STREAM_PANELS", str(int(streaming)))
    monkeypatch.setattr(molecular, "metric_whitener", factor)
    provider = molecular.MolecularFocalProvider(
        shell_library=Path("shell.so"), metric_library=Path("response.so"), budget_mib=3
    )
    req = replace(request(XYZ, fits), backend="cuda")
    jk, mp2, f12 = provider._pair_panels(req)
    assert len(panel_shell) == len(set(fits))
    factors = ["inverse_cholesky", "symmetric"]
    if fits[1] not in (fits[0], fits[2]):
        factors.append("symmetric")
    assert [kind for _, kind in calls] == factors
    assert (mp2 is f12) == (fits[1] == fits[2])
    assert (mp2 is jk) == (fits[1] == fits[0] and fits[1] != fits[2])
    assert (jk.data is f12.data) == (fits[0] == fits[2])
    assert jk is not f12 and jk.coulomb_only and not f12.coulomb_only
    assert jk.joint(req, explicit=False).whitener is jk.whitener
    assert f12.joint(req, explicit=True).whitener is f12.whitener
    assert jk.metric_diagnostics["factorization"] == "symmetric"
    assert f12.metric_diagnostics["factorization"] == "inverse_cholesky"
    assert f12.streaming == jk.streaming == streaming
    for panel in (jk, mp2, f12):
        np.testing.assert_allclose(
            panel.whitener @ panel.data["metric"] @ panel.whitener.T, np.eye(2), atol=1e-14, rtol=0
        )
    np.testing.assert_array_equal(np.triu(f12.whitener, 1), np.zeros((2, 2)))
    np.testing.assert_allclose(jk.whitener, jk.whitener.T, atol=1e-14, rtol=0)


def test_correlation_rank_refusal_propagates_without_retry(
    panel_shell: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def refused(*args: object, **kwargs: object) -> None:
        calls.append(kwargs)
        raise RuntimeError("unstable physical metric rank")

    monkeypatch.setattr(molecular, "metric_whitener", refused)
    provider = molecular.MolecularFocalProvider(
        shell_library=Path("shell.so"), metric_library=Path("response.so"), budget_mib=3
    )
    with pytest.raises(RuntimeError, match="unstable physical metric rank"):
        provider._pair_panels(replace(request(XYZ, (0, 0, 0)), backend="cuda"))
    assert calls == [{"budget_bytes": 3 << 20, "factorization": "inverse_cholesky"}]


def test_requested_inverse_cholesky_requires_matching_export(monkeypatch: pytest.MonkeyPatch) -> None:
    class LegacyLibrary:
        def gqc_f12_molecular_metric_whitener(self, *args: object) -> None:
            pytest.fail("unexpected symmetric ABI call for correlation fit")

    monkeypatch.setattr(metric.ct, "CDLL", lambda _: LegacyLibrary())
    with pytest.raises(RuntimeError, match="gqc_f12_molecular_metric_inverse_cholesky_v1"):
        metric.metric_whitener(np.eye(2), "legacy.so", factorization="inverse_cholesky")


def test_shared_raw_metric_direction_with_two_different_factors() -> None:
    from test_f12_separate_fitting_spaces import inputs, response, whitener

    ao, _ = inputs()
    m = ao["metric/coulomb"]
    direction = np.array([[0.1, -0.2, 0.3], [-0.2, 0.4, -0.2], [0.3, -0.2, -0.1]])

    def evaluate(step: float) -> tuple[float, np.ndarray]:
        raw = m + step * direction
        eig, vec = np.linalg.eigh(raw)
        corr = {**ao, "metric/coulomb": raw, "metric_whitener": whitener(raw)}
        jk = {
            "metric/coulomb": raw, "ao_coulomb": corr["ao_coulomb"],
            "metric_whitener": (vec / np.sqrt(eig)) @ vec.T,
        }
        e, weights = response(corr, jk)
        total = e.ordinary_mp2 + e.f12_increment + sum(e.cabs_singles)
        return total, weights["metric/coulomb"] + weights["jkfit/metric/coulomb"]

    _, bar = evaluate(0)
    for step in (2e-4, 1e-4):
        finite_difference = (evaluate(step)[0] - evaluate(-step)[0]) / (2 * step)
        np.testing.assert_allclose(np.vdot(bar, direction), finite_difference, atol=1e-8, rtol=0)
