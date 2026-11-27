"""Generate independent PySCF DF-HF canonical spectra of NaBr and NaI.

The halogen orbital and JKFit records are read from the packaged methyl-halide
JSON files without GenerativeQC; sodium records come from the Basis Set
Exchange and the Br/I scalar ECPs from the PySCF library. Each case starts from
the PySCF default guess, so no spectrum depends on another case. Threaded SCF
runs are not bitwise reproducible; compare a regenerated file numerically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import basis_set_exchange as bse
import numpy as np
import pyscf
from pyscf import gto, scf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "python/generativeqc/data/f12_methyl_halide"
OUTPUT = ROOT / "tests/data/frozen_core_gap/nax_df_hf_spectra.json"
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
# Huber and Herzberg (1979) X 1Sigma+ equilibrium distances, NIST Chemistry WebBook.
R_E_ANGSTROM = {"Br": 2.502038, "I": 2.711452}
HALOGEN_Z = {"Br": 35, "I": 53}
CARDINAL = {2: "D", 3: "T"}
SCAN_FACTORS = (0.80, 0.85, 0.90, 0.95, 1.10, 1.25, 1.50, 2.00)
CHARGES = (0, 1, -1)
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-8


def packaged_shells(halogen: str, cardinal: int, role: str) -> tuple[list, str]:
    path = DATA / f"{halogen.lower()}-{CARDINAL[cardinal].lower()}" / f"{role}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    element = next(
        e for e in record["elements"] if e["atomic_number"] == HALOGEN_Z[halogen]
    )
    shells = [
        [shell["angular_momentum"]]
        + [[float(e), float(c)] for e, c in zip(shell["exponents"], row, strict=True)]
        for shell in element["shells"]
        for row in shell["coefficients"]
    ]
    return shells, hashlib.sha256(path.read_bytes()).hexdigest()


def bse_shells(name: str, z: int) -> list:
    element = bse.get_basis(name, elements=[z], header=False)["elements"][str(z)]
    shells = []
    for shell in element["electron_shells"]:
        angular = shell["angular_momentum"]
        rows = shell["coefficients"]
        pairs = (
            zip(angular, rows, strict=True)
            if len(angular) > 1
            else ((angular[0], r) for r in rows)
        )
        for am, row in pairs:
            shells.append(
                [am]
                + [
                    [float(e), float(c)]
                    for e, c in zip(shell["exponents"], row, strict=True)
                ]
            )
    return shells


def dominant_labels(mol: gto.Mole, coefficients: np.ndarray, count: int) -> list[str]:
    """Atom symbol and angular momentum holding the largest Mulliken population."""
    overlap = mol.intor("int1e_ovlp")
    keys = [
        f"{symbol} {shell.lstrip('0123456789')[0]}"
        for _, symbol, shell, _ in mol.ao_labels(fmt=False)
    ]
    labels = []
    for k in range(count):
        c = coefficients[:, k]
        population: dict[str, float] = {}
        for key, value in zip(keys, c * (overlap @ c), strict=True):
            population[key] = population.get(key, 0.0) + float(value)
        labels.append(max(population, key=population.__getitem__))
    return labels


def spectrum(
    halogen: str, cardinal: int, factor: float, charge: int, sources: dict
) -> dict:
    orbital, orbital_sha = packaged_shells(halogen, cardinal, "orbital")
    jkfit, jkfit_sha = packaged_shells(halogen, cardinal, "jkfit")
    sources[f"{halogen.lower()}-{CARDINAL[cardinal].lower()}"] = {
        "orbital_sha256": orbital_sha,
        "jkfit_sha256": jkfit_sha,
    }
    r_bohr = R_E_ANGSTROM[halogen] * factor * BOHR_PER_ANGSTROM
    mol = gto.M(
        atom=[("Na", (0.0, 0.0, 0.0)), (halogen, (0.0, 0.0, r_bohr))],
        basis={
            "Na": bse_shells(f"cc-pV{CARDINAL[cardinal]}Z-F12", 11),
            halogen: orbital,
        },
        ecp={halogen: gto.basis.load_ecp("cc-pvdz-pp", halogen)},
        unit="Bohr",
        charge=charge,
        spin=0 if charge == 0 else 1,
        cart=False,
        verbose=0,
    )
    aux = {"Na": bse_shells("def2-universal-JKFIT", 11), halogen: jkfit}
    mf = (scf.RHF(mol) if charge == 0 else scf.UHF(mol)).density_fit(auxbasis=aux)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 300
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"Na{halogen} {factor} r_e charge {charge} did not converge")
    if charge == 0:
        blocks = [("rhf", mf.mo_energy, mf.mo_coeff, mf.mo_occ)]
    else:
        blocks = [
            (name, mf.mo_energy[s], mf.mo_coeff[s], mf.mo_occ[s])
            for s, name in enumerate(("alpha", "beta"))
        ]
    channels = []
    for name, energies, coefficients, occupations in blocks:
        occupied = int(np.count_nonzero(occupations > 0))
        channels.append(
            {
                "channel": name,
                "occupied": occupied,
                "occupied_energies_Eh": [float(e) for e in energies[:occupied]],
                "dominant_character": dominant_labels(mol, coefficients, occupied),
            }
        )
    return {
        "halogen": halogen,
        "cardinal": cardinal,
        "bond_factor": factor,
        "bond_length_bohr": r_bohr,
        "charge": charge,
        "multiplicity": 1 if charge == 0 else 2,
        "root_id": "rhf" if charge == 0 else "uhf",
        "total_energy_Eh": float(mf.e_tot),
        "channels": channels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sources: dict = {}
    cases = []
    for halogen in ("Br", "I"):
        for cardinal in (2, 3):
            for charge in CHARGES:
                cases.append(spectrum(halogen, cardinal, 1.0, charge, sources))
        for factor in SCAN_FACTORS:
            for charge in CHARGES:
                cases.append(spectrum(halogen, 2, factor, charge, sources))
    document = {
        "schema": "generativeqc.tests.frozen_core_gap.v1",
        "generator": "tools/generate_frozen_core_gap_references.py",
        "oracle": {
            "pyscf": pyscf.__version__,
            "basis_set_exchange": bse.__version__,
            "numpy": np.__version__,
            "method": "PySCF DF-RHF (neutral) and DF-UHF (doublet ions), default guess",
            "conv_tol": CONV_TOL,
            "conv_tol_grad": CONV_TOL_GRAD,
        },
        "geometry": {
            "layout": "Na at the origin, halogen on +z",
            "r_e_angstrom": R_E_ANGSTROM,
            "r_e_source": "Huber and Herzberg (1979), NIST Chemistry WebBook",
            "bohr_per_angstrom": BOHR_PER_ANGSTROM,
        },
        "basis": {
            "Na": "Basis Set Exchange cc-pV{D,T}Z-F12; JKFit def2-universal-JKFIT",
            "halogen": "packaged methyl-halide orbital.json and jkfit.json",
            "halogen_ecp": "PySCF cc-pvdz-pp (ECP10MDF for Br, ECP28MDF for I)",
            "packaged_files": sources,
        },
        "cases": cases,
    }
    text = re.sub(
        r"\[\s+([^\[\]{}]*?)\s+\]",
        lambda match: "[" + " ".join(match.group(1).split()) + "]",
        json.dumps(document, indent=1),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
