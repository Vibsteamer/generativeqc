"""DF Hamiltonian derivatives against displaced dense spin-orbital integrals."""

import numpy as np
import pytest
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.tensor import execute
from test_uccsd_equations import _feeds

from tools.generativeqc_cc.uccsd_df_direction import unrestricted_cc_df_hamiltonian_tangent


@pytest.mark.parametrize("occupied,frozen", [((3, 2), (1, 0)), ((0, 2), (0, 0))])
def test_molecular_df_hamiltonian_tangent(
    occupied: tuple[int, int], frozen: tuple[int, int]
) -> None:
    rng = np.random.default_rng(82136)
    obs, ext, aux = 4, 5, 3
    oa, ob = occupied
    fa, fb = frozen
    layout = UCCSDLayout(oa - fa, ob - fb, obs - oa, obs - ob)
    amplitudes = layout.unpack(np.zeros(layout.packed_size))
    x = rng.normal(size=(aux, aux))
    metric = np.eye(aux) + x @ x.T

    def symmetric(shape: tuple[int, ...]) -> np.ndarray:
        y = rng.normal(scale=0.2, size=shape)
        return (y + y.swapaxes(-1, -2)) / 2

    dm = symmetric((aux, aux))
    raw = {"source/raw_coulomb_" + s: symmetric((aux, ext, ext)) for s in ("alpha", "beta")}
    draw = {k: symmetric(v.shape) for k, v in raw.items()}
    df = {"fock_" + s: symmetric((ext, ext)) for s in ("alpha", "beta")}
    d = unrestricted_cc_df_hamiltonian_tangent(
        (*occupied, obs, ext, aux), raw, np.linalg.inv(np.linalg.cholesky(metric)),
        execute, data_tangent=draw, fock_tangent=df, coulomb_metric_tangent=dm,
        frozen_occupied=frozen,
    )
    # Fill one complete spin-orbital tensor, in the occupied-a, occupied-b,
    # virtual-a, virtual-b order used by the independent dense CC oracle.
    norb = 2 * obs - fa - fb
    active_occ = oa - fa + ob - fb
    positions = {
        "alpha": np.r_[0:oa - fa, active_occ:active_occ + obs - oa],
        "beta": np.r_[oa - fa:active_occ, active_occ + obs - oa:norb],
    }

    def displaced(h: float) -> dict[str, np.ndarray]:
        w = np.linalg.inv(np.linalg.cholesky(metric + h * dm))
        factors = np.zeros((norb, norb, aux))
        fock = np.zeros((norb, norb))
        for s, start in zip(("alpha", "beta"), frozen, strict=True):
            name = "source/raw_coulomb_" + s
            g = raw[name] + h * draw[name]
            tile = (w @ g.reshape(aux, -1)).reshape(aux, ext, ext)
            idx = np.ix_(positions[s], positions[s])
            factors[idx] = tile[:, start:obs, start:obs].transpose(1, 2, 0)
            fock[idx] = h * df["fock_" + s][start:obs, start:obs]
        direct = np.einsum("prx,qsx->pqrs", factors, factors)
        return _feeds(layout, fock, direct - direct.swapaxes(2, 3), amplitudes)

    for h in (2e-4, 1e-4):
        plus, minus = displaced(h), displaced(-h)
        assert d.keys() == plus.keys() - amplitudes.keys()
        for key, val in d.items():
            np.testing.assert_allclose(val, (plus[key] - minus[key]) / (2 * h), rtol=0, atol=1e-8)
