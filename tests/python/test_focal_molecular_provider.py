"""Public geometry provider with coincident and independent auxiliary bases."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalRequest,
    FocalSeries,
    MolecularFocalProvider,
    plan_singlepoint,
)
from generativeqc.profiles import canonical_hash


def basis(name: str, exponents: tuple[float, ...]) -> BasisSet:
    shells = tuple(BasisShell(0, (e,), ((1.0,),)) for e in exponents)
    return BasisSet(
        name,
        (ElementBasis(1, shells),),
        BasisProvenance(
            "test", name, "uncontracted hydrogen shells", canonical_hash(exponents)
        ),
        representation="spherical",
    )


def request(
    x: np.ndarray, fits: tuple[int, int, int], root: str = "uhf"
) -> FocalRequest:
    obs = basis("obs", (1.5, 0.32))
    cabs = basis("cabs", (0.09,))
    a, b, c = [
        basis(f"aux-{i}", (5.0 + i, 1.1 + 0.1 * i, 0.28 + 0.02 * i, 0.07 + 0.01 * i))
        for i in fits
    ]
    card = FocalBasis(
        2, "model", obs, jk_fit=a, mp2_fit=b, f12_fit=c, cabs=cabs, beta=0.9
    )
    geom = list(zip(["H"] * len(x), x, strict=True))
    steps = plan_singlepoint(
        geom,
        FocalSeries(card),
        FocalSeries(card),
        card,
        root_id=root,
        correlation_space_id="all-active",
        multiplicity=1 if root == "rhf" else 2,
        backend="cpu_reference",
    ).steps
    return next(r for r, _ in steps if r.role == "mp2_f12_correlation")


XYZ = np.array([[0.1, -0.2, 0.15], [1.6, 0.1, -0.2], [0.47, 1.57, 0.24]])


def test_molecular_provider_requires_declared_backend_and_high_order(
    tmp_path: Path,
) -> None:
    p = MolecularFocalProvider(shell_library=tmp_path / "unused.so")
    q = request(XYZ, (0, 1, 2))
    with pytest.raises(ValueError, match="high-order evaluator"):
        p(replace(q, role="post_mp2"))
    with pytest.raises(ValueError, match="CUDA"):
        p(replace(q, backend="cuda"))
    with pytest.raises(ValueError, match="all-active UHF"):
        p(replace(q, root_id="rhf"))


@pytest.mark.parametrize(
    ("root", "fits"),
    [
        ("uhf", (0, 0, 0)),
        ("uhf", (0, 0, 1)),
        ("uhf", (0, 1, 0)),
        ("uhf", (0, 1, 1)),
        ("uhf", (0, 1, 2)),
        ("rhf", (0, 1, 2)),
    ],
)
def test_molecular_pair_all_fit_assignments(
    root: str, fits: tuple[int, int, int]
) -> None:
    shell = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shell:
        pytest.skip("requires the built native F12 host shell library")
    from benchmarks.f12_focal_increment import molecular_f12_increment
    from benchmarks.f12_focal_ump2 import molecular_df_ump2

    lib = Path(shell)
    p = MolecularFocalProvider(shell_library=lib)
    xyz = XYZ[:2] if root == "rhf" else XYZ
    q = request(xyz, fits, root)
    val = p(q)
    ordinary = molecular_df_ump2(q, shell_library=lib, one_electron_library=lib)
    explicit = molecular_f12_increment(q, shell_library=lib, one_electron_library=lib)
    assert (val.term.surface_id, val.term.basis_id, val.term.role) == (
        q.surface_id,
        q.basis.basis_id,
        q.role,
    )
    assert val.term.energy == pytest.approx(
        sum(ordinary.spin_energies) + explicit.energy, abs=2e-10, rel=0
    )
    np.testing.assert_allclose(
        val.term.forces,
        ordinary.correlation_forces + explicit.forces,
        atol=3e-9,
        rtol=0,
    )
    direction = np.array([[0.1, 0.4, -0.2], [-0.2, 0.3, 0.5], [0.2, -0.1, 0.3]])
    direction = direction[: len(xyz)]
    direction /= np.linalg.norm(direction)
    for h in (4e-4, 2e-4):
        plus, minus = [
            p(request(xyz + sign * h * direction, fits, root)).term.energy
            for sign in (1, -1)
        ]
        assert np.vdot(val.term.forces, direction) == pytest.approx(
            (minus - plus) / (2 * h), abs=2e-7, rel=0
        )


def test_uhf_fixed_cc_feed_fitting_roles() -> None:
    shells = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shells:
        pytest.skip("set GENERATIVEQC_F12_HOST_SINK for molecular host integrals")
    p = MolecularFocalProvider(shell_library=Path(shells))
    outputs = [
        p.ucc_fixed_sources(request(XYZ, fits))
        for fits in ((0, 1, 2), (0, 2, 2), (2, 2, 2))
    ]
    a, b, same_fit = outputs
    assert a.cabs_rank == b.cabs_rank == same_fit.cabs_rank == 3
    assert a.reference.occupied == (2, 1) and a.reference.forces is None
    assert a.reference.jk_fit_id != a.f12_fit_id == same_fit.reference.jk_fit_id
    spins = ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
    assert set(a.feeds) == {
        prefix + s for prefix in ("w_vo_", "w_vv_", "c_f12_") for s in spins
    } | {"fixed_geminal_energy"}
    for k in a.feeds:
        np.testing.assert_array_equal(a.feeds[k], b.feeds[k])
    assert not np.isclose(
        a.feeds["fixed_geminal_energy"],
        same_fit.feeds["fixed_geminal_energy"],
        rtol=0,
        atol=1e-7,
    )
    for suffix in ("aaaa", "bbbb"):
        for prefix in ("w_vo_", "w_vv_", "c_f12_"):
            v = a.feeds[prefix + suffix]
            np.testing.assert_allclose(v, -v.swapaxes(0, 1), rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        a.feeds["c_f12_abab"], -a.feeds["c_f12_baab"].swapaxes(0, 1), rtol=0, atol=1e-11
    )
