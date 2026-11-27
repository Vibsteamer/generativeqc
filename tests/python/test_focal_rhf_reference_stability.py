"""A failed focal DF-RHF primary settles to the lowest internally stable RHF determinant.

Stretched N2 has several RHF stationary points. From the default core guess,
first-order DIIS and the level-shifted recovery do not converge. The
independent oracle is PySCF DF-RHF with the same orbital and fitting basis
records and real internal RHF stability analysis from several guesses.
"""

from types import ModuleType

import pytest
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    conventional_mp2_request,
    df_reference,
)
from generativeqc.profiles import canonical_hash

Atoms = list[tuple[str, tuple[float, float, float]]]

STRETCHED_N2: Atoms = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 3.7795))]
ORBITAL, FIT = "cc-pvdz", "def2-universal-jkfit"


def _record(gto: ModuleType, name: str, symbols: list[str]) -> tuple[BasisSet, dict]:
    elements, raw = [], {}
    for symbol in sorted(set(symbols), key=gto.charge):
        raw[symbol] = gto.basis.load(name, symbol)
        shells = []
        for angular, *rows in raw[symbol]:
            exponents, *coefficients = zip(*rows, strict=True)
            shells.append(BasisShell(angular, exponents, tuple(coefficients)))
        elements.append(ElementBasis(gto.charge(symbol), tuple(shells)))
    provenance = BasisProvenance(
        "PySCF basis library", name, "basis data for testing", canonical_hash(raw)
    )
    return BasisSet(name, tuple(elements), provenance, representation="spherical"), raw


def _lowest_stable_pyscf(atoms: Atoms, orbital: dict, fit: dict) -> float:
    from pyscf import gto, scf

    mol = gto.M(atom=atoms, unit="Bohr", basis=orbital, verbose=0)
    energies = []
    for guess in ("minao", "atom", "huckel", "1e"):
        mf = scf.RHF(mol).density_fit(auxbasis=fit)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle, mf.init_guess = (
            1e-12,
            1e-8,
            300,
            guess,
        )
        mf.kernel()
        for _ in range(6):
            if not mf.converged:
                break
            orbitals, _, stable, _ = mf.stability(return_status=True)
            if stable:
                energies.append(mf.e_tot)
                break
            mf.kernel(dm0=mf.make_rdm1(orbitals, mf.mo_occ))
    assert energies, "PySCF found no internally stable DF-RHF determinant"
    return min(energies)


def test_failed_primary_reaches_lowest_stable_rhf() -> None:
    gto = pytest.importorskip("pyscf.gto")
    symbols = [symbol for symbol, _ in STRETCHED_N2]
    orbital, orbital_raw = _record(gto, ORBITAL, symbols)
    fit, fit_raw = _record(gto, FIT, symbols)
    basis = FocalBasis(2, "f3/rhf-stability", orbital, jk_fit=fit, mp2_fit=fit)
    request = conventional_mp2_request(
        tuple(STRETCHED_N2),
        basis,
        root_id="rhf",
        correlation_space_id="all-active",
        charge=0,
        multiplicity=1,
        backend="cpu_reference",
        protocol_id="f3/rhf-stability",
    )
    reference = df_reference(request, forces=False)
    assert reference.occupied == (7,)
    assert reference.energy == pytest.approx(
        _lowest_stable_pyscf(STRETCHED_N2, orbital_raw, fit_raw), abs=1e-9, rel=0
    )
