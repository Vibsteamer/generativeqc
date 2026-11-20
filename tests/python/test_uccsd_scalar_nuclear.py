"""Test the unrestricted scalar adjoint with unequal frozen occupations and moving spin frames."""

import numpy as np
from generativeqc._f12_cc_spin_sources import unrestricted_cc_f12_fixed_source_feeds
from generativeqc._f12_spin_ao import unrestricted_orbital_sources
from generativeqc._f12_spin_fock import spin_fock
from generativeqc_compiler.tensor import execute

from tools.generativeqc_cc.uccsd_scalar_nuclear import unrestricted_cc_scalar_ao_weights


def test_frozen_scalar_pullback_to_all_ao_sources() -> None:
    rng = np.random.default_rng(1748)
    n = 6
    q = 4
    space = (3, 2, 5, n, q)
    frozen = (1, 0)

    def sym(shape: tuple[int, ...]) -> np.ndarray:
        m = rng.normal(size=shape)
        return (m + m.swapaxes(-1, -2)) * 0.2

    ao = {"hcore_ao": sym((n, n)), "overlap_ao": np.eye(n)}
    for spin in ("alpha", "beta"):
        ao["coefficients_" + spin] = np.linalg.qr(rng.normal(size=(n, n)))[0]
    for op, raw in (
        ("coulomb", "coulomb"),
        ("geminal", "f12"),
        ("geminal_coulomb", "f12_over_coulomb"),
        ("squared_geminal", "f12_squared"),
        ("double_commutator", "double_commutator"),
    ):
        ao["ao_" + raw] = sym((q, n, n))
        ao["metric/" + op] = sym((q, q)) + 2.0 * np.eye(q)

    def factor(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            **fields,
            "metric_whitener": np.linalg.inv(
                np.linalg.cholesky(fields["metric/coulomb"])
            ),
        }

    def energy(fields: dict[str, np.ndarray]) -> float:
        fields = factor(fields)
        data, _ = unrestricted_orbital_sources(space, fields, execute)
        src = {
            k.removeprefix("source/"): v
            for k, v in data.items()
            if k.startswith("source/")
        }
        ff = spin_fock((3, 2, n, q), src, execute)
        return float(
            unrestricted_cc_f12_fixed_source_feeds(
                space,
                data,
                ff,
                fields["metric_whitener"],
                execute,
                frozen_occupied=frozen,
            )["fixed_geminal_energy"]
        )

    weight = -0.73
    v, b, x, bars = unrestricted_cc_scalar_ao_weights(
        space, factor(ao), np.asarray(weight), execute, frozen_occupied=frozen
    )
    np.testing.assert_allclose(v + b - x, energy(ao), atol=2e-12, rtol=0)
    groups = [
        ("metric/coulomb",),
        ("coefficients_alpha", "coefficients_beta"),
        ("hcore_ao", "overlap_ao"),
        tuple(
            k
            for k in ao
            if k.startswith("ao_") or (k.startswith("metric/") and k != "metric/coulomb")
        ),
    ]
    for keys in groups:
        dx = {
            k: rng.normal(size=ao[k].shape) * 0.2
            if k.startswith("coefficients")
            else sym(ao[k].shape)
            for k in keys
        }
        h = 2.0e-5
        plus = {**ao, **{k: ao[k] + h * t for k, t in dx.items()}}
        minus = {**ao, **{k: ao[k] - h * t for k, t in dx.items()}}
        analytic = sum(np.vdot(bars[k], t) for k, t in dx.items())
        np.testing.assert_allclose(
            analytic,
            weight * (energy(plus) - energy(minus)) / (2 * h),
            rtol=3e-8,
            atol=2e-8,
        )
