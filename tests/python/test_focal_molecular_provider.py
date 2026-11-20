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


@pytest.mark.parametrize("fits", [(0, 1, 2), (0, 0, 2), (0, 2, 2)])
def test_uhf_equation_sources_against_independent_df_chemist_blocks(
    fits: tuple[int, int, int],
) -> None:
    shell = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shell:
        pytest.skip("set GENERATIVEQC_F12_HOST_SINK for the molecular shell library")
    pyscf = pytest.importorskip("pyscf")
    from pyscf import df, gto

    req = request(XYZ, fits)
    provider = MolecularFocalProvider(shell_library=Path(shell))
    full = provider.ucc_equation_sources(req)
    fixed = provider.ucc_fixed_sources(req)
    assert full.fixed.f12_fit_id == fixed.f12_fit_id
    assert full.mp2_fit_id == req.basis.mp2_fit.identity
    assert len(full.ordinary) == 38 and len(fixed.feeds) == 19
    for k in fixed.feeds:
        np.testing.assert_allclose(
            full.fixed.feeds[k], fixed.feeds[k], rtol=0, atol=1e-12
        )

    def shells(exps: tuple[float, ...]) -> dict:
        return {"H": [[0, [e, 1.0]] for e in exps]}

    mol = gto.M(
        atom=list(zip(["H"] * 3, XYZ, strict=True)),
        unit="Bohr",
        basis=shells((1.5, 0.32)),
        spin=1,
        verbose=0,
    )

    def factors(index: int) -> np.ndarray:
        aux = df.addons.make_auxmol(
            mol,
            shells(
                (
                    5.0 + index,
                    1.1 + 0.1 * index,
                    0.28 + 0.02 * index,
                    0.07 + 0.01 * index,
                )
            ),
        )
        m = aux.intor("int2c2e")
        raw = df.incore.aux_e2(mol, aux, intor="int3c2e", aosym="s1").transpose(2, 0, 1)
        return np.linalg.solve(np.linalg.cholesky(m), raw.reshape(len(m), -1)).reshape(
            raw.shape
        )

    jk, cc = factors(fits[0]), factors(fits[1])
    c = dict(zip("ab", full.fixed.reference.coefficients, strict=True))
    density = {
        s: k[:, :o] @ k[:, :o].T for (s, k), o in zip(c.items(), (2, 1), strict=True)
    }
    h = pyscf.scf.hf.get_hcore(mol)
    j = np.einsum("Qpq,Qrs,rs->pq", jk, jk, density["a"] + density["b"], optimize=True)
    fmo = {
        s: k.T
        @ (h + j - np.einsum("Qpr,Qst,rt->ps", jk, jk, density[s], optimize=True))
        @ k
        for s, k in c.items()
    }
    mo = {s: k.T @ cc @ k for s, k in c.items()}
    for name, value in full.ordinary.items():
        domains, spin = name[1:].split("_")
        ix = tuple(
            slice(None, 2 if s == "a" else 1)
            if d == "o"
            else slice(2 if s == "a" else 1, None)
            for s, d in zip(spin, domains, strict=True)
        )
        if name.startswith("f"):
            gold = fmo[spin[0]][ix]
        else:
            i, j, k, l = spin
            eri = np.zeros((mol.nao,) * 4)
            if i == k and j == l:
                eri += np.einsum("Qik,Qjl->ijkl", mo[i], mo[j])
            if i == l and j == k:
                eri -= np.einsum("Qil,Qjk->ijkl", mo[i], mo[j])
            gold = eri[ix]
        np.testing.assert_allclose(value, gold, rtol=0, atol=3e-9, err_msg=name)
