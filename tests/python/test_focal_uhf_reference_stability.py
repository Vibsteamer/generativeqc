"""Focal DF-UHF references reach the lowest stable UHF determinant from the default guess.

A spectator next to OH(2Pi) lifts the pi-hole degeneracy. From the default core
guess, first-order DIIS either stalls on the soft hole-rotation mode (Ne) or
converges to an internally unstable UHF stationary point several mEh too high
(Li+). The independent oracle is PySCF DF-UHF with the same orbital and fitting
basis records and real internal stability analysis from several guesses.
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

OH = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.834))]
CASES = {
    "neon_3A": (0, [*OH, ("Ne", (5.669, 0.0, 0.917))]),
    "lithium_cation_4A": (1, [*OH, ("Li", (7.559, 0.0, 0.0))]),
}
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


def _lowest_stable_pyscf(atoms: Atoms, charge: int, orbital: dict, fit: dict) -> float:
    from pyscf import gto, scf

    mol = gto.M(
        atom=atoms, unit="Bohr", basis=orbital, charge=charge, spin=1, verbose=0
    )
    energies = []
    for guess in ("minao", "atom", "huckel", "1e"):
        mf = scf.UHF(mol).density_fit(auxbasis=fit)
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
    assert energies, "PySCF found no stable DF-UHF determinant"
    return min(energies)


@pytest.mark.parametrize("case", sorted(CASES))
def test_default_guess_reaches_lowest_stable_uhf(case: str) -> None:
    gto = pytest.importorskip("pyscf.gto")
    charge, atoms = CASES[case]
    symbols = [symbol for symbol, _ in atoms]
    orbital, orbital_raw = _record(gto, ORBITAL, symbols)
    fit, fit_raw = _record(gto, FIT, symbols)
    basis = FocalBasis(2, "f3/uhf-stability", orbital, jk_fit=fit, mp2_fit=fit)
    request = conventional_mp2_request(
        tuple(atoms),
        basis,
        root_id="uhf",
        correlation_space_id="all-active",
        charge=charge,
        multiplicity=2,
        backend="cpu_reference",
        protocol_id="f3/uhf-stability",
    )
    reference = df_reference(request, forces=False)
    assert reference.occupied == (
        (sum(gto.charge(s) for s in symbols) - charge + 1) // 2,
        (sum(gto.charge(s) for s in symbols) - charge - 1) // 2,
    )
    assert reference.energy == pytest.approx(
        _lowest_stable_pyscf(atoms, charge, orbital_raw, fit_raw), abs=1e-9, rel=0
    )
