"""Compare six stationary C weight pullbacks with a dense frozen-core UHF calculation."""

from itertools import combinations, product

import numpy as np
from generativeqc_compiler.tensor import execute

from tools.generativeqc_cc.uccsd_c_nuclear import unrestricted_cc_c_ao_weights


def test_unrestricted_six_c_frozen_core() -> None:
    rng = np.random.default_rng(321)
    n, obs, na = 7, 5, 4
    occ = {"a": 3, "b": 2}
    core = {"a": 1, "b": 0}
    names = {"a": "alpha", "b": "beta"}
    ao = {
        "coefficients_" + s: np.linalg.qr(rng.normal(size=(n, n)))[0]
        for s in names.values()
    }

    def sym(shape: tuple[int, ...]) -> np.ndarray:
        m = rng.normal(size=shape) / 3
        return (m + m.swapaxes(-1, -2)) / 2

    ao.update(
        hcore_ao=sym((n, n)),
        overlap_ao=np.eye(n),
        ao_coulomb=sym((na, n, n)),
        ao_f12=sym((na, n, n)),
    )
    t = rng.normal(size=(na, na))
    ao["metric/coulomb"] = t @ t.T + np.eye(na)
    ao["metric/geminal"] = sym((na, na))
    ao["metric_whitener"] = np.linalg.inv(np.linalg.cholesky(ao["metric/coulomb"]))
    weights = {
        "c_f12_" + l + r + p + q: rng.normal(
            size=(occ[l] - core[l], occ[r] - core[r], obs - occ[p], obs - occ[q])
        )
        for l, r, p, q in product("ab", repeat=4)
        if sorted((l, r)) == sorted((p, q))
    }
    bars = unrestricted_cc_c_ao_weights(
        (3, 2, obs, n, na), ao, weights, execute, frozen_occupied=(1, 0)
    )

    def scalar(m: dict[str, np.ndarray]) -> float:
        cf = {sp: m["coefficients_" + s] for sp, s in names.items()}
        M = m["metric/geminal"]
        J = {s: c.T @ m["ao_coulomb"] @ c for s, c in cf.items()}
        R = {s: c.T @ m["ao_f12"] @ c for s, c in cf.items()}
        D = {
            s: np.linalg.solve(m["metric/coulomb"], a.reshape(na, -1)).reshape(a.shape)
            for s, a in J.items()
        }
        rho = sum(
            np.trace(D[s][:, :no, :no], axis1=1, axis2=2) for s, no in occ.items()
        )
        fock = {
            s: c.T @ m["hcore_ao"] @ c
            + np.einsum("Apq,A->pq", J[s], rho)
            - np.einsum("Aip,Aiq->pq", D[s][:, : occ[s]], J[s][:, : occ[s]])
            for s, c in cf.items()
        }
        cross = cf["a"].T @ m["overlap_ao"] @ cf["b"]
        blocks = {k: np.zeros_like(w) for k, w in weights.items()}
        for l, r in (("a", "a"), ("a", "b"), ("b", "b")):
            li, ri = range(core[l], occ[l]), range(core[r], occ[r])
            for i, j in combinations(li, 2) if l == r else product(li, ri):
                f, g = D[l][:, i], D[r][:, j]
                v = f.T @ R[r][:, j] + R[l][:, i].T @ g - f.T @ M @ g
                gem = (v - v.T) / 4 if l == r else (3 * v + cross @ v.T @ cross) / 8
                c = (
                    fock[l][occ[l] : obs, obs:] @ gem[obs:, occ[r] : obs]
                    + gem[occ[l] : obs, obs:] @ fock[r][occ[r] : obs, obs:].T
                )
                for p, q, arr in [(l, r, c)] if l == r else [(l, r, c), (r, l, -c.T)]:
                    blocks["c_f12_" + l + r + p + q][i - core[l], j - core[r]] = arr
                    blocks["c_f12_" + r + l + p + q][j - core[r], i - core[l]] = -arr
        return sum(float(np.vdot(v, weights[k])) for k, v in blocks.items())

    for k, bar in bars.items():
        dx = (
            rng.normal(size=bar.shape)
            if k.startswith("coefficients")
            else sym(bar.shape)
        )
        h = 1e-5
        numerical = (
            scalar({**ao, k: ao[k] + h * dx}) - scalar({**ao, k: ao[k] - h * dx})
        ) / (2 * h)
        np.testing.assert_allclose(np.vdot(dx, bar), numerical, atol=3e-9, rtol=2e-7)
