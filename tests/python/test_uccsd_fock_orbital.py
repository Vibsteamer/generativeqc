"""Independent coefficient and overlap derivatives of the ordinary spin Fock blocks."""

import os

import numpy as np
import pytest
from generativeqc_compiler.tensor import execute

from tools.generativeqc_cc.uccsd_fock_nuclear import unrestricted_cc_fock_jk_pullback
from tools.generativeqc_cc.uccsd_fock_orbital import (
    unrestricted_cc_fock_frame_weights,
    unrestricted_cc_fock_orbital_response,
)


def _inverse_sqrt(positive: np.ndarray) -> np.ndarray:
    eig, vec = np.linalg.eigh(positive)
    return (vec / np.sqrt(eig)) @ vec.T


def test_ordinary_fock_orbital_weights_in_nonorthogonal_ao_metric() -> None:
    rng = np.random.default_rng(12055)
    nao, nmo, naux = 6, 4, 3
    spins = ("alpha", "beta")
    occupied, frozen = (3, 1), (1, 0)
    x = rng.normal(size=(nao, nao))
    overlap = x @ x.T + np.eye(nao)
    h = rng.normal(size=(nao, nao))
    h = (h + h.T) / 2
    raw = rng.normal(size=(naux, nao, nao))
    raw = (raw + raw.swapaxes(1, 2)) / 2
    aux = rng.normal(size=(naux, naux))
    metric = aux @ aux.T + np.eye(naux)
    eri = np.einsum("Pab,PQ,Qcd->abcd", raw, np.linalg.inv(metric), raw)
    coefficients, weights = {}, {}
    for spin, no, nf in zip(spins, occupied, frozen, strict=True):
        columns = np.linalg.qr(rng.normal(size=(nao, nmo)))[0]
        coefficients[spin] = _inverse_sqrt(overlap) @ columns
        for block, dims in (
            ("oo", (no - nf, no - nf)),
            ("ov", (no - nf, nmo - no)),
            ("vv", (nmo - no, nmo - no)),
        ):
            weights["f" + block + "_" + spin[0] * 2] = rng.normal(size=dims)

    def energy(frame: dict[str, np.ndarray]) -> float:
        occupied_densities = {
            s: frame[s][:, :no] @ frame[s][:, :no].T
            for s, no in zip(spins, occupied, strict=True)
        }
        j = np.einsum("abkl,kl->ab", eri, sum(occupied_densities.values()))
        value = 0.0
        for s, no, nf in zip(spins, occupied, frozen, strict=True):
            k = np.einsum("akbl,kl->ab", eri, occupied_densities[s])
            f = frame[s].T @ (h + j - k) @ frame[s]
            for block, matrix in (
                ("oo", f[nf:no, nf:no]),
                ("ov", f[nf:no, no:]),
                ("vv", f[no:, no:]),
            ):
                value += np.vdot(weights["f" + block + "_" + s[0] * 2], matrix)
        return float(value)

    jk = unrestricted_cc_fock_jk_pullback(
        occupied,
        weights,
        coefficients,
        raw,
        metric,
        _inverse_sqrt(metric),
        execute,
        frozen_occupied=frozen,
    )
    bars, pulay = unrestricted_cc_fock_frame_weights(
        occupied,
        weights,
        {"coefficients_" + s: coefficients[s] for s in spins},
        h,
        jk,
        execute,
        frozen_occupied=frozen,
    )
    step = 2e-5
    for s in spins:
        direction = rng.normal(size=(nao, nmo))
        plus = {**coefficients, s: coefficients[s] + step * direction}
        minus = {**coefficients, s: coefficients[s] - step * direction}
        numeric = (energy(plus) - energy(minus)) / (2 * step)
        np.testing.assert_allclose(
            np.vdot(bars["coefficients_" + s], direction), numeric, atol=3e-8, rtol=2e-8
        )

    ds = rng.normal(size=(nao, nao))
    ds = (ds + ds.T) / 2

    def transported(delta: float) -> dict[str, np.ndarray]:
        displaced = overlap + delta * ds
        return {
            s: c @ _inverse_sqrt(c.T @ displaced @ c) for s, c in coefficients.items()
        }

    numeric = (energy(transported(step)) - energy(transported(-step))) / (2 * step)
    np.testing.assert_allclose(np.vdot(pulay, ds), numeric, atol=1e-8, rtol=1e-8)


@pytest.mark.skipif(
    os.getenv("GENERATIVEQC_TEST_CUDA") != "1",
    reason="requires native canonical-frame and unrestricted orbital-response CUDA libraries",
)
def test_native_orbital_weight_against_displaced_generalized_eigenvectors() -> None:
    """For a noninteracting open shell the exact frame solves one generalized eigenproblem."""
    rng = np.random.default_rng(9531)
    n = 4
    e = np.array([-2.3, -1.1, 0.6, 2.9])
    h = np.diag(e)
    central = {"whitener": np.eye(1)}
    energies, coeffs, weights, jk = {}, {}, {}, {}
    for s, no in (("alpha", 2), ("beta", 1)):
        central["coefficients_" + s] = coeffs["coefficients_" + s] = np.eye(n)
        central["df_" + s] = np.zeros((1, n, n))
        central["occupied_energies_" + s] = e[:no]
        energies[s] = e
        jk["coefficients_" + s] = np.zeros((n, n))
        for block, dims in (
            ("oo", (no, no)),
            ("ov", (no, n - no)),
            ("vv", (n - no, n - no)),
        ):
            weights["f" + block + "_" + s[0] * 2] = rng.normal(size=dims)
    bars, pulay = unrestricted_cc_fock_frame_weights(
        (2, 1), weights, coeffs, h, jk, execute
    )
    response = unrestricted_cc_fock_orbital_response(
        (2, 1, n, 1),
        central,
        energies,
        bars,
        pulay,
        execute,
    )
    dh, ds = rng.normal(size=(2, n, n))
    dh, ds = (dh + dh.T) / 2, (ds + ds.T) / 2

    def frame_energy(step: float) -> float:
        orth = _inverse_sqrt(np.eye(n) + step * ds)
        _, eigenvectors = np.linalg.eigh(orth @ (h + step * dh) @ orth)
        c = orth @ eigenvectors
        c *= np.sign(c.diagonal())
        # Hold the central H fixed to isolate the orbital motion.
        transformed = c.T @ h @ c
        energy = 0.0
        for s, no in (("aa", 2), ("bb", 1)):
            for block, matrix in (
                ("oo", transformed[:no, :no]),
                ("ov", transformed[:no, no:]),
                ("vv", transformed[no:, no:]),
            ):
                energy += np.vdot(weights["f" + block + "_" + s], matrix)
        return float(energy)

    step = 2e-5
    numeric = (frame_energy(step) - frame_energy(-step)) / (2 * step)
    analytic = np.vdot(response.weights["hcore_ao"], dh) + np.vdot(
        response.weights["overlap_ao"], ds
    )
    np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=2e-8)
