"""Independent dense spin-pair finite differences for the stationary W adjoint."""

from itertools import combinations, product

import numpy as np
from generativeqc_compiler.tensor import execute

from tools.generativeqc_cc.uccsd_w_nuclear import unrestricted_cc_w_ao_weights


def test_open_shell_frozen_w_ao_weights() -> None:
    rng = np.random.default_rng(85217)
    n, obs, q = 6, 5, 4
    occupied = {"a": 3, "b": 2}
    frozen = {"a": 1, "b": 0}
    ao = {
        "coefficients_" + s: np.linalg.qr(rng.normal(size=(n, n)))[0]
        for s in ("alpha", "beta")
    }
    ao["overlap_ao"] = np.eye(n)
    operators = (
        ("coulomb", "coulomb"),
        ("geminal", "f12"),
        ("geminal_coulomb", "f12_over_coulomb"),
    )
    for op, field in operators:
        a = rng.normal(size=(q, n, n)) / 4
        ao["ao_" + field] = a + a.swapaxes(1, 2)
        a = rng.normal(size=(q, q))
        ao["metric/" + op] = a @ a.T + np.eye(q)
    ao["metric_whitener"] = np.linalg.inv(np.linalg.cholesky(ao["metric/coulomb"]))
    weights = {}
    for spins in product("ab", repeat=4):
        i, j, p, r = spins
        if sorted((i, j)) != sorted((p, r)):
            continue
        for label in ("vo", "vv"):
            name = "w_" + label + "_" + "".join(spins)
            if name == "w_vo_baab":
                continue
            shape = (
                occupied[i] - frozen[i],
                occupied[j] - frozen[j],
                obs - occupied[p],
                occupied[r] - frozen[r] if label == "vo" else obs - occupied[r],
            )
            weights[name] = rng.normal(size=shape)
    bars = unrestricted_cc_w_ao_weights(
        (3, 2, obs, n, q), ao, weights, execute, frozen_occupied=(1, 0)
    )

    def dense(fields: dict[str, np.ndarray]) -> float:
        coeff = {s[0]: fields["coefficients_" + s] for s in ("alpha", "beta")}
        raw = {
            (s, op): c.T @ fields["ao_" + key] @ c
            for s, c in coeff.items()
            for op, key in operators
        }
        inv = np.linalg.inv(fields["metric/coulomb"])
        fit = {
            s: (inv @ raw[s, "coulomb"].reshape(q, -1)).reshape(q, n, n) for s in coeff
        }
        overlap = coeff["a"].T @ fields["overlap_ao"] @ coeff["b"]
        tab = {key: np.zeros_like(v) for key, v in weights.items()}
        for l, r in (("a", "a"), ("a", "b"), ("b", "b")):
            li, ri = range(frozen[l], occupied[l]), range(frozen[r], occupied[r])
            for i, j in combinations(li, 2) if l == r else product(li, ri):
                f, g = fit[l][:, i], fit[r][:, j]
                gem = {}
                for op in ("geminal", "geminal_coulomb"):
                    m = (
                        f.T @ raw[r, op][:, j]
                        + raw[l, op][:, i].T @ g
                        - f.T @ fields["metric/" + op] @ g
                    )
                    gem[op] = (
                        (m - m.T) / 4
                        if l == r
                        else (3 * m + overlap @ m.T @ overlap) / 8
                    )
                coulomb = np.tensordot(
                    fit[l][:, :obs, :obs], raw[r, "coulomb"][:, :obs, :obs], axes=(0, 0)
                )
                w = gem["geminal_coulomb"][:obs, :obs] - np.einsum(
                    "urvs,rs->uv", coulomb, gem["geminal"][:obs, :obs]
                )
                for p, qspin, block in (
                    [(l, r, w)] if l == r else [(l, r, w), (r, l, -w.T)]
                ):
                    for label, sl in (
                        ("vo", slice(frozen[qspin], occupied[qspin])),
                        ("vv", slice(occupied[qspin], obs)),
                    ):
                        tile = block[occupied[p] :, sl]
                        for u, v, ii, jj, sign in ((l, r, i, j, 1), (r, l, j, i, -1)):
                            key = f"w_{label}_{u}{v}{p}{qspin}"
                            if key in tab:
                                tab[key][ii - frozen[u], jj - frozen[v]] = sign * tile
        return sum(float(np.vdot(w, tab[key])) for key, w in weights.items())

    for key, bar in bars.items():
        d = rng.normal(size=bar.shape)
        if not key.startswith("coefficients"):
            d = (d + d.swapaxes(-1, -2)) / 2
        h = 1e-5
        plus = dense({**ao, key: ao[key] + h * d})
        minus = dense({**ao, key: ao[key] - h * d})
        np.testing.assert_allclose(
            np.vdot(bar, d), (plus - minus) / (2 * h), atol=2e-8, rtol=2e-7
        )
