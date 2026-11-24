"""Check physical Hermitian AO-Fock reassociation and its TensorIR response."""

import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.f12_molecular_fock import compensated_fock
from generativeqc_compiler.mp2.f12_density_fitting import molecular_orbital_sources
from generativeqc_compiler.mp2.f12_molecular_pair import extended_df_fock
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.ad_program import transpose_program


def test_physical_fock_reassociation_and_response():
    rng = np.random.default_rng(36)
    na, n, aux, o = 7, 5, 3, 2
    c = rng.normal(size=(na, n))
    h = rng.normal(size=(na, na))
    h = (h + h.T) / 2
    g = rng.normal(size=(aux, na, na))
    g = (g + g.swapaxes(1, 2)) / 2
    mo_plan = molecular_orbital_sources(na, n, o, aux)
    feeds = {"coefficients": c, "hcore_ao": h, "ao_coulomb": g}
    for op in ("f12", "f12_squared", "f12_over_coulomb", "double_commutator"):
        feeds["ao_" + op] = g
    mapped = execute(mo_plan, feeds).outputs
    fit = rng.normal(size=(aux, o, n))
    fplan = extended_df_fock(o, n, aux)
    in_f = {
        "hcore": mapped["hcore"],
        "raw_coulomb_all": mapped["raw_coulomb"],
        "fit": fit,
    }
    f = execute(fplan, in_f).outputs
    seed = rng.normal(size=(n, n))
    bars = execute(
        transpose_program(fplan, ("fock", "exchange")).program,
        {**in_f, "bar_fock": seed, "bar_exchange": np.zeros_like(seed)},
    ).outputs
    reverse = transpose_program(mo_plan, ("hcore", "raw_coulomb")).program
    bar = execute(
        reverse,
        {
            **feeds,
            "bar_hcore": bars["bar_hcore"],
            "bar_raw_coulomb": bars["bar_raw_coulomb_all"],
        },
    ).outputs
    dc = rng.normal(size=c.shape)
    dh = rng.normal(size=h.shape)
    dh = (dh + dh.T) / 2
    dg = rng.normal(size=g.shape)
    dg = (dg + dg.swapaxes(1, 2)) / 2
    df = rng.normal(size=fit.shape)
    directional = sum(
        np.vdot(delta, bar[key])
        for key, delta in (
            ("bar_hcore_ao", dh),
            ("bar_ao_coulomb", dg),
            ("bar_coefficients", dc),
        )
    ) + np.vdot(df, bars["bar_fit"])

    def physical(t):
        b = c + t * dc
        integrals = g + t * dg
        d = fit + t * df
        tr = np.trace(d[:, :, :o], axis1=1, axis2=2)
        raw = (b[:, :o].T @ integrals) @ b
        k = np.einsum("aip,aiq->pq", d, raw)
        return b.T @ ((h + t * dh + 2 * np.einsum("a,auv->uv", tr, integrals)) @ b) - k

    np.testing.assert_allclose(physical(0), f["fock"], atol=2e-12, rtol=1e-12)
    for step in (1e-5, 5e-6):
        slope = np.vdot(seed, (physical(step) - physical(-step)) / (2 * step))
        np.testing.assert_allclose(slope, directional, atol=1e-7, rtol=2e-8)


def test_conditioned_cuda_fock():
    path = os.environ.get("GENERATIVEQC_TEST_FOCK_LIBRARY")
    if not path:
        pytest.skip("requires the physical CUDA F12 component")
    rng = np.random.default_rng(344)
    a, n = 16, 3
    c = rng.integers(-10, 11, size=(a, n)).astype(float) * 512
    h = rng.normal(size=(a, a))
    h = (h + h.T) / 2
    g = np.ascontiguousarray(-h[None])
    g[0, 0, 0] += 2.0**-26
    d = np.array([[[0.5, 0.0, 0.0]]])
    k = np.zeros((n, n))
    params = {"library": Path(path), "budget_bytes": 1 << 20}
    val, rec = compensated_fock(h, g, c, d, k, **params)
    expected = np.outer(c[0], c[0]) * 2.0**-26
    np.testing.assert_allclose(val, expected, rtol=0, atol=2e-13)
    assert rec["device_bytes"] > 0
    h[0, 1] += 0.1
    with pytest.raises(RuntimeError, match="Hermitian"):
        compensated_fock(h, g, c, d, k, **params)
    with pytest.raises(RuntimeError, match="budget"):
        compensated_fock(h, g, c, d, k, library=Path(path), budget_bytes=8)
