"""Two independent Coulomb metrics for unrestricted Fock and F12 pair response."""

import json
from pathlib import Path

import numpy as np
import pytest
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


def orbital_layout(q: int, manifest: dict) -> dict:
    return {
        "manifest": np.array(json.dumps(manifest)),
        "ao_sizes": np.array([4, q]),
        "orbital_shell_count": 4,
        "angular": np.zeros(4 + q, int),
        "atom": np.zeros(4 + q, int),
        "centers": np.zeros((4 + q, 3)),
        "prim_offsets": np.arange(5 + q),
        "exponents": np.ones(4 + q),
        "radial": np.ones(4 + q),
        "spherical_rows": np.arange(4 + q)[:, None],
        "transform_0": np.ones((1, 1)),
    }


def test_nuclear_response_uses_correlation_sink_and_jkfit_orbital_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    from generativeqc._one_electron_gradient import IntegralGradient
    from generativeqc.f12_orbital_response import unrestricted_nuclear_source_terms

    from benchmarks import f12_unrestricted_gaussian as gaussian

    ao, jk = inputs()
    _, direct = response(ao, jk)
    rng = np.random.default_rng(510)
    xyz = [[0.0, 0.0, 0.0], [0.2, 0.0, 1.1]]
    manifest = {
        "case": "test",
        "elements": ["H", "H"],
        "geometry_bohr": xyz,
        "geometry_sha256": "one",
        "options": {},
    }
    geom = {
        "combined_basis": {"elements": ["H", "H"], "coordinates_bohr": xyz},
        "layout": {"space": (2, 1, 3, 4, 3)},
        "geometry": {"geometry_sha256": "one"},
        "options": {},
        "molecule": "test",
    }

    corr_layout, jk_layout = orbital_layout(3, manifest), orbital_layout(2, manifest)
    invalid = {**jk_layout, "spherical_rows": jk_layout["spherical_rows"][::-1]}
    with pytest.raises(ValueError, match="ordering"):
        gaussian.check_shared_orbital_basis(corr_layout, invalid)
    central = {"whitener": jk["metric_whitener"]}
    df = (central["whitener"] @ jk["ao_coulomb"].reshape(2, -1)).reshape(2, 4, 4)
    for spin, occ in (("alpha", 2), ("beta", 1)):
        c = ao["coefficients_" + spin][:, :3]
        central["coefficients_" + spin] = c
        central["df_" + spin] = c.T @ df @ c
        central["occupied_energies_" + spin] = np.array([-2.0, -1.0])[:occ]
        for key in ("fock_seed_", "overlap_seed_"):
            raw = rng.normal(size=(3, 3))
            central[key + spin] = (raw + raw.T) / 2
    shared = ("hcore_ao", "overlap_ao")
    corr_ops = {
        **{v: k for k, v in gaussian._METRICS.items()},
        **{k: k for k in ao if k.startswith("ao_")},
    }
    jk_ops = {"metric": "jkfit/metric/coulomb", "ao_coulomb": "jkfit/ao_coulomb"}
    jets = {}
    for k in (*shared, *corr_ops.values(), *jk_ops.values()):
        d = rng.normal(size=(3, *direct[k].shape)) * 0.01
        d = (d + d.swapaxes(-1, -2)) / 2
        jets[k] = np.stack((d, -d))

    def contract(
        layout: dict, weights: dict, library: object, **kwargs: object
    ) -> dict:
        keys = jk_ops if layout is jk_layout else corr_ops
        gradient = sum(
            np.einsum("ax..., ...->ax", jets[name], weights[label], optimize=True)
            for label, name in keys.items()
        )
        return {
            "gradient_Eh_per_bohr": gradient.tolist(),
            "passes_contraction_checks": True,
        }

    class Core:
        def gradient(
            self, ws: np.ndarray, wh: np.ndarray, budget: int
        ) -> IntegralGradient:
            return IntegralGradient(
                np.einsum("axij,ij->ax", jets["overlap_ao"], ws)
                + np.einsum("axij,ij->ax", jets["hcore_ao"], wh),
                {},
            )

    monkeypatch.setattr(gaussian, "contract_live_weights", contract)
    reference = rng.normal(size=(2, 3))
    spec = (2, 1, 3, 2)
    z = rng.normal(size=4)
    forces, costs, _ = gaussian.gaussian_orbital_forces(
        spec,
        central,
        z,
        direct,
        {s: np.zeros((4, 4)) for s in ("alpha", "beta")},
        geom,
        corr_layout,
        reference,
        None,
        execute,
        cabs_h_shells=Core(),
        jkfit_layout=jk_layout,
    )
    directions = [
        {
            "dmetric": jets["jkfit/metric/coulomb"][idx],
            "dthree_center": jets["jkfit/ao_coulomb"][idx],
            "doverlap": jets["overlap_ao"][idx],
            "dhcore": jets["hcore_ao"][idx],
        }
        for idx in np.ndindex(2, 3)
    ]
    independent = unrestricted_nuclear_source_terms(
        spec, central, directions, 2, execute
    )
    explicit = sum(
        np.einsum("ax...,...->ax", d, direct[k], optimize=True) for k, d in jets.items()
    )
    expected = (
        reference
        - explicit
        - sum(independent.fixed_gradients.values())
        - independent.nuclear_rhs @ z
    )
    np.testing.assert_allclose(forces, expected, rtol=0, atol=4e-12)
    assert costs["JKFit"]["passes_contraction_checks"]


def test_gaussian_center_uses_jkfit_for_the_snapshot_and_returns_pair_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks import f12_unrestricted_gaussian_center as center

    ao, jk = inputs()
    h = np.diag([-2.0, -1.0, 1.0, 3.0])
    h[0, 3] = h[3, 0] = 0.08
    ao["hcore_ao"] = h
    for spin in ("alpha", "beta"):
        ao["coefficients_" + spin] = np.eye(4)
    diagonal = np.array([[0.1, 0.2, 0.4, 0.7], [0.3, 0.15, 0.05, 0.25]])
    jk["ao_coulomb"] = np.stack([np.diag(row) for row in diagonal])
    fitted = jk["metric_whitener"] @ diagonal
    occupations = (2, 1)
    charge = 2 * fitted[:, 0] + fitted[:, 1]
    snapshot = {
        "occupations": occupations,
        "nbf": 3,
        "overlap": np.eye(3),
        "hcore": h[:3, :3],
        "forces": np.zeros((2, 3)),
        "energy": -2.5,
    }
    for spin, occupied in zip(("alpha", "beta"), occupations, strict=True):
        eps = np.diag(h)[:3] + charge @ fitted[:, :3]
        eps[:occupied] -= (fitted[:, :occupied] ** 2).sum(axis=0)
        snapshot[spin] = {"mo": np.eye(3), "fock": np.diag(eps), "eps": eps}
    manifest = {
        "case": "replay",
        "elements": ["H", "H"],
        "geometry_bohr": [[0, 0, 0], [0, 0, 1]],
        "geometry_sha256": "same-center",
        "options": {},
    }
    basis = {"elements": ["H", "H"], "shells": [{"angular_momentum": 0}]}
    geometry = tmp_path / "geometry.json"
    geometry.write_text(
        json.dumps({"layout": {"space": [2, 1, 3, 4, 3]}, "combined_basis": basis})
    )
    paths = [tmp_path / "correlation.npz", tmp_path / "jkfit.npz"]
    for nfit, path in zip((3, 2), paths, strict=True):
        np.savez(path, **orbital_layout(nfit, manifest))

    def gaussian(*args: object, **kwargs: object) -> tuple:
        return (
            dict(ao),
            np.eye(4)[:, :3],
            np.eye(4)[:, 3:],
            np.arange(3),
            np.eye(3),
            h[:3, :3],
            {},
        )

    def fit(*args: object, **kwargs: object) -> tuple:
        return {"ao_coulomb": jk["ao_coulomb"], "metric": jk["metric/coulomb"]}, {}

    def gradient(
        space: tuple,
        central: dict,
        orbital: object,
        direct: dict,
        frame: dict,
        science: dict,
        layout: object,
        reference: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> tuple:
        assert space == (2, 1, 3, 2)
        assert (
            direct["ao_coulomb"].shape[0] == 3
            and direct["jkfit/ao_coulomb"].shape[0] == 2
        )
        return reference, {}, {}

    def solve_singles(
        fock: np.ndarray,
        occupied: int,
        *,
        library: object,
        spin_weight: int,
        pullback: bool,
    ) -> tuple:
        return singles(fock, occupied, spin_weight=spin_weight, pullback=pullback)

    monkeypatch.setattr(center, "gaussian_sources", gaussian)
    monkeypatch.setattr(center, "forward_integrals", fit)
    monkeypatch.setattr(
        center, "metric_whitener", lambda m, *args, **kwargs: (whitener(m), {})
    )
    monkeypatch.setattr(center, "gaussian_orbital_forces", gradient)
    monkeypatch.setattr(center, "cabs_singles", solve_singles)
    terms, _ = response(ao, jk)
    energy, _, costs = center.center_energy_forces(
        geometry,
        paths[0],
        tmp_path / "unused",
        snapshot,
        None,
        execute,
        sink_library=None,
        forward_library=None,
        jkfit_layout_path=paths[1],
        metric_library=tmp_path / "metric",
        projector_library=None,
        canonical_library=None,
        orbital_library=None,
    )
    assert (
        max(costs["shell_reference_fit"]["canonical_Fock_errors_Eh"].values()) < 1e-12
    )
    assert abs(energy["UMP2_Eh"] - terms.ordinary_mp2) < 2e-13
    assert abs(energy["F12_Eh"] - terms.f12_increment) < 2e-13
    np.testing.assert_allclose(
        energy["CABS_singles_Eh"], terms.cabs_singles, atol=2e-13, rtol=0
    )
