"""Check the six ordinary CC Fock weights against a moving-frame finite difference."""

import numpy as np
from generativeqc._f12_focal_one_electron import FocalOneElectron
from generativeqc_compiler.tensor import execute

from tools.generativeqc_cc.uccsd_fock_nuclear import unrestricted_cc_fock_nuclear_forces


def test_unrestricted_fock_derivative_with_frozen_orbitals() -> None:
    rng = np.random.default_rng(8131)
    nao, mo = 4, 3
    frozen, occ = (1, 0), (2, 1)
    h = rng.normal(size=(nao, nao))
    h = h + h.T
    dh = rng.normal(size=(1, 3, nao, nao))
    dh = dh + dh.swapaxes(-1, -2)
    c, dc, djk, weights = {}, {}, {}, {}
    for spin, no, nf in zip(("alpha", "beta"), occ, frozen, strict=True):
        c[spin] = rng.normal(size=(nao, mo))
        dc[spin] = rng.normal(size=(1, 3, nao, mo))
        djk[spin] = rng.normal(size=(1, 3, mo, mo))
        djk[spin] += djk[spin].swapaxes(-1, -2)
        o, v = no - nf, mo - no
        for name, shape in (("foo", (o, o)), ("fov", (o, v)), ("fvv", (v, v))):
            weights[name + "_" + spin[0] * 2] = rng.normal(size=shape)

    core = FocalOneElectron(
        np.eye(nao), h, None, dh, mo, "surface", "basis", "cpu_reference", h.nbytes
    )
    values = unrestricted_cc_fock_nuclear_forces(
        occ, weights, c, dc, djk, core, execute, frozen_occupied=frozen
    )

    def energy(axis: int, step: float) -> float:
        value = 0.0
        for s, no, nf in zip(("alpha", "beta"), occ, frozen, strict=True):
            frame = c[s] + step * dc[s][0, axis]
            f = frame.T @ (h + step * dh[0, axis]) @ frame + step * djk[s][0, axis]
            for n, block in (
                ("foo", f[nf:no, nf:no]),
                ("fov", f[nf:no, no:]),
                ("fvv", f[no:, no:]),
            ):
                value += np.vdot(weights[n + "_" + s[0] * 2], block)
        return float(value)

    eps = 1e-5
    fd = [(energy(k, -eps) - energy(k, eps)) / (2 * eps) for k in range(3)]
    np.testing.assert_allclose(values["ordinary_fock"][0], fd, rtol=1e-10, atol=2e-8)
    explicit = -np.einsum("Akij,ij->Ak", dh, values["ao_core_weight"])
    np.testing.assert_allclose(explicit, values["explicit_core"], atol=2e-12)
