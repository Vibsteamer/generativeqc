"""Canonical DF-RHF and DF-UHF frames and analytic reference forces on a distorted H3 geometry."""

import os
from dataclasses import replace

import numpy as np
import pytest
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalRequest,
    FocalSeries,
    df_reference,
    plan_singlepoint,
)
from generativeqc.profiles import canonical_hash


@pytest.fixture(scope="module")
def hydrogen_bases() -> tuple[FocalBasis, dict, dict]:
    gto = pytest.importorskip("pyscf.gto")

    def record(name: str) -> tuple[BasisSet, dict]:
        raw = gto.basis.load(name, "H")
        shells = []
        for angular, *rows in raw:
            exponents, *coefficients = zip(*rows, strict=True)
            shells.append(BasisShell(angular, exponents, tuple(coefficients)))
        return BasisSet(
            name,
            (ElementBasis(1, tuple(shells)),),
            BasisProvenance(
                "PySCF basis library",
                name,
                "basis data for testing",
                canonical_hash(raw),
            ),
            representation="spherical",
        ), {"H": raw}

    orbital, obs = record("cc-pvdz-f12")
    jkfit, jk = record("aug-cc-pvtz-jkfit")
    optri, _ = record("cc-pvdz-f12-optri")
    ri, _ = record("aug-cc-pvtz-ri")
    basis = FocalBasis(
        2,
        "VnZ-F12/parent-OptRI",
        {"H": orbital},
        jk_fit=jkfit,
        mp2_fit=ri,
        f12_fit=ri,
        cabs=optri,
        beta=0.9,
    )
    return basis, obs, jk


@pytest.mark.parametrize(
    "backend",
    [
        "cpu_reference",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                os.environ.get("GENERATIVEQC_TEST_CUDA") != "1",
                reason="explicit CUDA qualification",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "root,charge,multiplicity,occupied", [("rhf", 1, 1, (1,)), ("uhf", 0, 2, (2, 1))]
)
def test_physical_df_frame_and_nuclear_forces(
    hydrogen_bases: tuple[FocalBasis, dict, dict],
    root: str,
    charge: int,
    multiplicity: int,
    occupied: tuple[int, ...],
    backend: str,
) -> None:
    from pyscf import gto, scf

    basis, obs, jk = hydrogen_bases
    atoms = [
        ("H", (0.10, -0.23, 0.2)),
        ("H", (1.38, 0.27, -0.1)),
        ("H", (0.67, 1.45, 0.31)),
    ]
    plan = plan_singlepoint(
        atoms,
        FocalSeries(basis),
        FocalSeries(basis),
        basis,
        root_id=root,
        correlation_space_id="all-active",
        charge=charge,
        multiplicity=multiplicity,
        backend=backend,
    )
    req = plan.steps[0][0]
    mol = gto.M(
        atom=atoms,
        unit="Bohr",
        basis=obs,
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    mf = (scf.UHF(mol) if root == "uhf" else scf.RHF(mol)).density_fit(auxbasis=jk)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-9, 150
    mf.kernel()
    assert mf.converged
    reference = df_reference(req)
    assert reference.executed_backend == backend and reference.iterations > 0
    assert (
        reference.basis_id == basis.basis_id and reference.surface_id == req.surface_id
    )
    assert (
        reference.orbital_id == basis.orbital.identity
        and reference.jk_fit_id == basis.jk_fit.identity
    )
    assert reference.occupied == occupied
    assert reference.energy == pytest.approx(mf.e_tot, abs=2e-9, rel=0)
    np.testing.assert_allclose(
        reference.forces, -mf.nuc_grad_method().kernel(), atol=3e-8, rtol=0
    )
    np.testing.assert_allclose(reference.overlap, mf.get_ovlp(), atol=3e-10, rtol=0)
    np.testing.assert_allclose(reference.hcore, mf.get_hcore(), atol=3e-10, rtol=0)
    np.testing.assert_allclose(
        reference.orbital_energies, np.atleast_2d(mf.mo_energy), atol=5e-8, rtol=0
    )
    np.testing.assert_allclose(
        reference.density,
        mf.make_rdm1().reshape(reference.density.shape),
        atol=5e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        reference.fock, mf.get_fock().reshape(reference.fock.shape), atol=3e-8, rtol=0
    )
    for occ, coeff, density, fock, eps in zip(
        occupied,
        reference.coefficients,
        reference.density,
        reference.fock,
        reference.orbital_energies,
        strict=True,
    ):
        np.testing.assert_allclose(
            coeff.T @ reference.overlap @ coeff, np.eye(len(eps)), atol=3e-8, rtol=0
        )
        np.testing.assert_allclose(
            fock @ coeff, (reference.overlap @ coeff) * eps, atol=3e-8, rtol=0
        )
        np.testing.assert_allclose(
            (2 if root == "rhf" else 1) * coeff[:, :occ] @ coeff[:, :occ].T,
            density,
            atol=5e-8,
            rtol=0,
        )
    if root == "rhf":
        energy_only = df_reference(req, forces=False)
        assert energy_only.forces is None and energy_only.energy == pytest.approx(
            reference.energy, abs=2e-10, rel=0
        )
        with pytest.raises(ValueError, match="canonical uhf root"):
            df_reference(replace(req, root_id="uhf:excited"))


@pytest.mark.parametrize("root,charge,multiplicity", [("rhf", 1, 1), ("uhf", 0, 2)])
def test_displaced_reference_seed_retains_occupied_space(
    hydrogen_bases: tuple[FocalBasis, dict, dict],
    root: str,
    charge: int,
    multiplicity: int,
) -> None:
    from generativeqc.experimental.f12_focal import FocalReferenceSeed
    from pyscf import gto, scf

    basis, obs, jk = hydrogen_bases
    xyz = np.array([[0.10, -0.23, 0.2], [1.38, 0.27, -0.1], [0.67, 1.45, 0.31]])

    def request(coords: np.ndarray) -> FocalRequest:
        return plan_singlepoint(
            list(zip(("H",) * 3, coords, strict=True)),
            FocalSeries(basis),
            FocalSeries(basis),
            post_mp2=None,
            root_id=root,
            correlation_space_id="all-active",
            charge=charge,
            multiplicity=multiplicity,
            backend="cpu_reference",
        ).steps[0][0]

    base = request(xyz)
    initial = df_reference(base, forces=False)
    seed = FocalReferenceSeed(base, initial)
    xyz[1, 0] -= 4e-4
    moved = request(xyz)
    final = df_reference(moved, seed=seed)
    mol = gto.M(
        atom=list(zip(("H",) * 3, xyz, strict=True)),
        basis=obs,
        charge=charge,
        spin=multiplicity - 1,
        unit="Bohr",
        verbose=0,
    )
    mf = (scf.RHF(mol) if root == "rhf" else scf.UHF(mol)).density_fit(auxbasis=jk)
    mf.conv_tol = 1e-13
    mf.kernel()
    assert mf.converged and final.occupied_overlaps is not None
    assert min(final.occupied_overlaps) > 0.99999
    assert abs(final.energy - mf.e_tot) < 3e-9
    np.testing.assert_allclose(
        final.forces, -mf.nuc_grad_method().kernel(), rtol=0, atol=3e-7
    )
    with pytest.raises(ValueError, match="identities differ"):
        FocalReferenceSeed(moved, initial)
