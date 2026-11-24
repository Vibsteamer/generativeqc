"""The occupied-leg UHF transforms and complete metric reverse against matrix algebra."""

import numpy as np
import pytest
from generativeqc._f12_spin_ao import unrestricted_orbital_sources
from generativeqc_compiler.tensor import execute


@pytest.mark.parametrize("occ", [(2, 1), (2, 0), (0, 1)])
def test_unrestricted_ao_sources_and_metric_response(occ: tuple[int, int]) -> None:
    rng = np.random.default_rng(25537)
    obs, ext, nc, aux = 3, 4, 5, 3
    space = (*occ, obs, ext, aux)
    ao = {}
    ops = {
        "geminal": "f12",
        "squared_geminal": "f12_squared",
        "geminal_coulomb": "f12_over_coulomb",
        "double_commutator": "double_commutator",
    }

    def sym(shape: tuple[int, ...]) -> np.ndarray:
        a = rng.normal(size=shape)
        return (a + a.swapaxes(-1, -2)) / 2

    for spin in ("alpha", "beta"):
        ao["coefficients_" + spin] = np.linalg.qr(rng.normal(size=(nc, ext)))[0]
    for op in ("overlap_ao", "hcore_ao"):
        ao[op] = sym((nc, nc))
    ao["ao_coulomb"] = sym((aux, nc, nc))
    ao["metric/coulomb"] = np.diag([1.0, 2.0, 3.0])
    for op, internal in ops.items():
        ao["ao_" + internal] = sym((aux, nc, nc))
        ao["metric/" + op] = sym((aux, aux))

    def reference(a: dict) -> dict:
        e = {
            "cross_overlap": a["coefficients_alpha"].T
            @ a["overlap_ao"]
            @ a["coefficients_beta"]
        }
        for x in ops:
            e["metric/" + x] = a["metric/" + x]
        for spin, occupied in zip(("alpha", "beta"), occ, strict=True):
            c = a["coefficients_" + spin]
            raw = c.T @ a["ao_coulomb"] @ c
            e["source/hcore_" + spin] = c.T @ a["hcore_ao"] @ c
            e["source/raw_coulomb_" + spin] = raw
            if occupied:
                e["source/fit_" + spin] = np.linalg.solve(
                    a["metric/coulomb"], raw[:, :occupied].reshape(aux, -1)
                ).reshape(aux, occupied, ext)
                for op, internal in ops.items():
                    e["raw/" + op + "/" + spin] = (
                        c[:, :occupied].T @ a["ao_" + internal] @ c
                    )
        return e

    independent = reference(ao)
    seeds = {k: rng.normal(size=v.shape) for k, v in independent.items()}
    direction = {
        k: (sym(a.shape) if k.startswith("metric/") else rng.normal(size=a.shape))
        for k, a in ao.items()
    }
    a = {
        **ao,
        "metric_whitener": np.linalg.inv(np.linalg.cholesky(ao["metric/coulomb"])),
    }
    result, bars = unrestricted_orbital_sources(
        space, a, execute, orbital_weights=seeds
    )
    for k, v in independent.items():
        np.testing.assert_allclose(result[k], v, atol=1e-12, rtol=0)
    actual = sum(np.vdot(bar, direction[k]) for k, bar in bars.items())

    def objective(step: float) -> float:
        val = reference({k: v + step * direction[k] for k, v in ao.items()})
        return sum(np.vdot(seeds[k], v) for k, v in val.items())

    for h in (1e-5, 3e-6):
        np.testing.assert_allclose(
            actual, (objective(h) - objective(-h)) / (2 * h), atol=6e-8, rtol=0
        )
