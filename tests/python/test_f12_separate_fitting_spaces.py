"""Two independent Coulomb metrics for unrestricted Fock and F12 pair response."""

import numpy as np
from generativeqc._f12_unrestricted import (
    UnrestrictedF12Energy,
    unrestricted_mp2_f12_ao_response,
)
from generativeqc_compiler.tensor import execute
from scipy.linalg import solve_sylvester


def singles(
    fock: np.ndarray, occupied: int, *, spin_weight: int, pullback: bool
) -> tuple[float, np.ndarray]:
    if not occupied:
        return 0.0, np.zeros_like(fock)
    oo, vv, vo = (
        fock[:occupied, :occupied],
        fock[occupied:, occupied:],
        fock[occupied:, :occupied],
    )
    t = solve_sylvester(vv, -oo, vo)
    return -spin_weight * np.vdot(t, vo), spin_weight * np.block(
        [[-t.T @ t, -t.T], [-t, t @ t.T]]
    )


def whitener(metric: np.ndarray) -> np.ndarray:
    return np.linalg.solve(np.linalg.cholesky(metric), np.eye(len(metric)))


def inputs(seed: int = 52) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    n = 4

    def symmetric(shape: tuple[int, ...]) -> np.ndarray:
        raw = rng.normal(size=shape)
        return (raw + raw.swapaxes(-1, -2)) * 0.1

    def panel(q: int) -> dict:
        m = symmetric((q, q)) + 2 * np.eye(q)
        return {
            "ao_coulomb": symmetric((q, n, n)),
            "metric/coulomb": m,
            "metric_whitener": whitener(m),
        }

    jk, corr = panel(2), panel(3)
    ao = {
        **corr,
        "overlap_ao": np.eye(n),
        "hcore_ao": np.diag([-2.0, -1.0, 1.0, 3.0]) + symmetric((n, n)),
    }
    ao["coefficients_alpha"] = np.eye(n)
    ao["coefficients_beta"] = np.eye(n)[:, [1, 0, 2, 3]]
    for name, op in (
        ("f12", "geminal"),
        ("f12_squared", "squared_geminal"),
        ("f12_over_coulomb", "geminal_coulomb"),
        ("double_commutator", "double_commutator"),
    ):
        ao["ao_" + name] = symmetric((3, n, n))
        ao["metric/" + op] = symmetric((3, 3))
    return ao, jk


def response(
    ao: dict, jk: dict | None = None, *, frozen: tuple[int, int] = (0, 0)
) -> tuple[UnrestrictedF12Energy, dict]:
    return unrestricted_mp2_f12_ao_response(
        (2, 1, 3, 4, len(ao["metric/coulomb"])),
        ao,
        execute,
        jkfit=jk,
        singles=singles,
        frozen_occupied=frozen,
    )


def test_separate_metrics_match_four_index_oracle_and_gradients() -> None:
    ao, jk = inputs()
    terms, weights = response(ao, jk)
    rng = np.random.default_rng(916)

    def eri(p: dict) -> np.ndarray:
        raw, metric = p["ao_coulomb"], p["metric/coulomb"]
        fitted = np.linalg.solve(metric, raw.reshape(len(metric), -1)).reshape(
            raw.shape
        )
        return np.einsum("apq,ars->pqrs", raw, fitted)

    mo = [ao["coefficients_" + s] for s in ("alpha", "beta")]
    da, db = [c[:, :o] @ c[:, :o].T for c, o in zip(mo, (2, 1), strict=True)]
    g, v = eri(jk), eri(ao)
    j = np.einsum("pqrs,rs->pq", g, da + db)
    f = [
        c.T @ (ao["hcore_ao"] + j - np.einsum("prsq,rs->pq", g, d)) @ c
        for c, d in zip(mo, (da, db), strict=True)
    ]
    np.testing.assert_allclose(
        terms.cabs_singles,
        [
            singles(h, o, spin_weight=1, pullback=True)[0]
            for h, o in zip(f, (2, 1), strict=True)
        ],
        atol=2e-13,
    )
    # Alpha has one OBS virtual, hence its antisymmetric same-spin MP2 vanishes.
    eps = [x.diagonal() for x in f]
    expected = 0.0
    for i in range(2):
        for b in (1, 2):
            element = np.einsum(
                "pqrs,p,q,r,s", v, mo[0][:, i], mo[0][:, 2], mo[1][:, 0], mo[1][:, b]
            )
            expected += element**2 / (eps[0][i] + eps[1][0] - eps[0][2] - eps[1][b])
    np.testing.assert_allclose(terms.ordinary_mp2, expected, atol=2e-13)

    def energy(panel: str, source: dict, tangents: dict, step: float) -> float:
        changed = {
            **source,
            **{k: source[k] + step * d for k, d in tangents.items()},
        }
        changed["metric_whitener"] = whitener(changed["metric/coulomb"])
        e, _ = response(
            ao if panel == "jk" else changed, changed if panel == "jk" else jk
        )
        return e.ordinary_mp2 + e.f12_increment + sum(e.cabs_singles)

    for panel, names in (
        ("jk", ("ao_coulomb", "metric/coulomb")),
        ("corr", ("ao_coulomb", "metric/coulomb", "ao_f12", "metric/geminal")),
        (
            "frame",
            ("hcore_ao", "coefficients_alpha", "coefficients_beta", "overlap_ao"),
        ),
    ):
        source = jk if panel == "jk" else ao
        tangents = {k: rng.normal(size=source[k].shape) * 0.04 for k in names}
        for k in names:
            if not k.startswith("coefficients"):
                tangents[k] = (tangents[k] + tangents[k].swapaxes(-1, -2)) / 2
        predicted = sum(
            np.vdot(weights[("jkfit/" if panel == "jk" else "") + k], v)
            for k, v in tangents.items()
        )

        for step in (2e-4, 1e-4):
            np.testing.assert_allclose(
                predicted,
                (
                    energy(panel, source, tangents, step)
                    - energy(panel, source, tangents, -step)
                )
                / (2 * step),
                atol=2e-8,
                rtol=0,
            )


def test_identical_fit_recombines_once_and_singles_uses_only_jkfit() -> None:
    ao, _ = inputs()
    jk = {k: ao[k] for k in ("ao_coulomb", "metric/coulomb", "metric_whitener")}
    same, shared = response(ao)
    two, split = response(ao, jk)
    assert two == same
    for k, w in shared.items():
        total = (
            split[k] if "jkfit/" + k not in split else split[k] + split["jkfit/" + k]
        )
        np.testing.assert_allclose(total, w, atol=2e-12, rtol=0)
    e, w = response(ao, jk, frozen=(2, 1))
    assert e.active_pairs == 0 and e.ordinary_mp2 == e.f12_increment == 0
    for k in ("ao_coulomb", "metric/coulomb"):
        np.testing.assert_array_equal(w[k], np.zeros_like(w[k]))
        assert np.linalg.norm(w["jkfit/" + k]) > 1e-7
