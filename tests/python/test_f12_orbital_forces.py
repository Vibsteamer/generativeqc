import os

import numpy as np
import pytest
from generativeqc.f12_orbital_response import unrestricted_orbital_forces


def inputs(oa: int = 2, ob: int = 1) -> tuple[tuple, np.ndarray]:
    n, q = 4, 3
    rng = np.random.default_rng(32)
    factors = {}
    gaps = []
    for spin, o in (("alpha", oa), ("beta", ob)):
        if o:
            v = n - o
            for name, shape in (
                ("vo", (q, v, o)),
                ("oo", (q, o, o)),
                ("vv", (q, v, v)),
            ):
                factors[f"{name}_{spin}"] = np.zeros(shape)
            gap = rng.uniform(1, 2, (v, o))
            factors[f"gap_{spin}"] = gap
            gaps.extend(gap.ravel())
    driver = rng.normal(size=len(gaps))
    rhs = rng.normal(size=(2, 3, len(gaps)))
    reference = rng.normal(size=(2, 3))
    fixed = rng.normal(size=(2, 2, 3))
    return ((oa, ob, n, q), factors, driver, rhs, reference, fixed), np.array(gaps)


def test_molecular_adjoint_validates_budget_and_inputs_before_device_load() -> None:
    data, _ = inputs()
    with pytest.raises(ValueError, match="budget"):
        unrestricted_orbital_forces(*data, maximum_bytes=1)
    data[2][0] = np.nan
    with pytest.raises(ValueError, match="electronic driver"):
        unrestricted_orbital_forces(*data)


@pytest.mark.skipif(
    os.getenv("GENERATIVEQC_TEST_CUDA") != "1",
    reason="requires native CUDA library and GPU",
)
@pytest.mark.parametrize("oa,ob", [(2, 1), (2, 0), (0, 1)])
def test_native_transposed_solve_force_sign_and_single_spin(oa: int, ob: int) -> None:
    data, gaps = inputs(oa, ob)
    expected = data[2] / gaps
    gradient = data[3] @ expected
    result = unrestricted_orbital_forces(*data)
    np.testing.assert_allclose(result.rotations, expected, atol=1e-12)
    np.testing.assert_allclose(result.orbital_gradient, gradient, atol=1e-12)
    np.testing.assert_allclose(
        result.forces, data[4] - data[5].sum(axis=0) - gradient, atol=1e-12
    )


@pytest.mark.parametrize("oa,ob", [(2, 1), (2, 0), (0, 1)])
def test_ao_nuclear_equations_against_displaced_unwhitened_fock(
    oa: int, ob: int
) -> None:
    from generativeqc.f12_orbital_response import unrestricted_nuclear_source_terms
    from generativeqc_compiler.tensor import execute

    rng = np.random.default_rng(71)
    n, ao, q = 4, 6, 3

    def symmetric(x: np.ndarray) -> np.ndarray:
        return (x + x.swapaxes(-1, -2)) / 2

    raw = symmetric(rng.normal(size=(q, ao, ao)))
    dj = symmetric(rng.normal(size=(q, q)))
    j0 = rng.normal(size=(q, q))
    j0 = j0 @ j0.T + 2 * np.eye(q)
    w = np.linalg.qr(rng.normal(size=(q, q)))[0] @ np.linalg.inv(np.linalg.cholesky(j0))
    directions = [
        {
            "doverlap": symmetric(rng.normal(size=(ao, ao))),
            "dhcore": symmetric(rng.normal(size=(ao, ao))),
            "dmetric": dj,
            "dthree_center": symmetric(rng.normal(size=raw.shape)),
        }
        for _ in range(3)
    ]
    central = {"whitener": w}
    frame = []
    spins = ("alpha", "beta")
    for name, o in zip(spins, (oa, ob), strict=True):
        c = np.linalg.qr(rng.normal(size=(ao, n)))[0]
        eps = np.sort(rng.normal(size=n))
        r = c.T @ raw @ c
        fseed, sseed = (symmetric(rng.normal(size=(n, n))) for _ in range(2))
        frame.append((o, c, eps, fseed, sseed))
        for k, x in {
            "coefficients": c,
            "df": (w @ r.reshape(q, -1)).reshape(q, n, n),
            "occupied_energies": eps[:o],
            "fock_seed": fseed,
            "overlap_seed": sseed,
        }.items():
            central[k + "_" + name] = x
    built = unrestricted_nuclear_source_terms(
        (oa, ob, n, q), central, directions, 1, execute
    )
    for axis, tan in enumerate(directions):
        h = 1e-5
        sm = [c.T @ tan["doverlap"] @ c for _, c, *_ in frame]

        def focks(
            step: float,
            move_density: bool,
            tangent: dict[str, np.ndarray],
            overlaps: list[np.ndarray],
        ) -> list[np.ndarray]:
            # Invert the *displaced* Coulomb metric directly; allow the occupied
            # density to maintain its AO metric, or hold it exactly fixed.
            inv = np.linalg.inv(j0 + step * tangent["dmetric"])
            r = [
                c.T @ (raw + step * tangent["dthree_center"]) @ c for _, c, *_ in frame
            ]
            p = []
            for s, (o, *_) in zip(overlaps, frame, strict=True):
                d = np.zeros((n, n))
                d[:o, :o] = (
                    np.linalg.inv(np.eye(o) + step * s[:o, :o])
                    if move_density
                    else np.eye(o)
                )
                p.append(d)
            rho = sum(np.einsum("apq,pq->a", v, d) for v, d in zip(r, p, strict=True))
            return [
                c.T @ (step * tangent["dhcore"]) @ c
                + np.einsum("apq,a->pq", v, inv @ rho)
                - np.einsum("api,ab,ij,bqj->pq", v, inv, d, v)
                for (_, c, *_), v, d in zip(frame, r, p, strict=True)
            ]

        slopes = []
        for density in (False, True):
            plus, minus = focks(h, density, tan, sm), focks(-h, density, tan, sm)
            slopes.append([(a - b) / (2 * h) for a, b in zip(plus, minus, strict=True)])
        rhs = []
        for k, (name, (o, c, eps, weight, sw)) in enumerate(
            zip(spins, frame, strict=True)
        ):
            explicit, full = slopes[0][k], slopes[1][k]
            rhs.extend((sm[k][o:, :o] * eps[:o] - full[o:, :o]).ravel())
            for key, ref in [
                ("fixed_fock", np.vdot(weight, explicit)),
                ("metric_density", np.vdot(weight, full - explicit)),
                ("fixed_overlap", np.vdot(sw, sm[k])),
            ]:
                assert built.fixed_gradients[key + "_" + name][
                    0, axis
                ] == pytest.approx(ref, abs=2e-8)
        np.testing.assert_allclose(built.nuclear_rhs[0, axis], rhs, atol=2e-8, rtol=0)
