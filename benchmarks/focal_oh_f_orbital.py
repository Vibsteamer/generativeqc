"""Conventional (T) control on OH with an explicitly retained oxygen f shell."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np
from generativeqc._f12_focal_ucc_triples import molecular_triples_on_reference
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, load_basis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalReference,
    FocalReferenceSeed,
    FocalSeries,
    df_reference,
    plan_singlepoint,
)
from generativeqc.profiles import canonical_hash

if TYPE_CHECKING:
    from collections.abc import Iterator

COORDINATES = np.array([[0.12, -0.28, 0.21], [1.75, 0.26, -0.36]])
SYMBOLS = ("O", "H")
STEPS = (0.0004, 0.0002)
OXYGEN_F_EXPONENT = 1.4956


def through_f(path: Path) -> BasisSet:
    original = load_basis(path)
    elements = tuple(
        replace(
            element,
            shells=tuple(
                shell for shell in element.shells if shell.angular_momentum <= 3
            ),
        )
        for element in original.elements
    )
    return replace(original, name=original.name + " through f", elements=elements)


def basis(data: Path, bounded_host_fit: bool) -> FocalBasis:
    load_fit = through_f if bounded_host_fit else load_basis
    jk, ri = (load_fit(data / name) for name in ("jkfit.json", "mp2fit.json"))
    small = FocalBasis(2, "compact OH", "sto-3g", jk_fit=jk)
    oxygen = small.orbital.by_element[8]
    augmented = replace(
        oxygen,
        shells=(*oxygen.shells, BasisShell(3, (OXYGEN_F_EXPONENT,), ((1.0,),))),
    )
    orbital = BasisSet(
        "OH STO-3G plus one oxygen f shell",
        (small.orbital.by_element[1], augmented),
        BasisProvenance(
            "STO-3G and the tight oxygen f shell of the cc-pVTZ-F12 record",
            "1",
            small.orbital.provenance.license,
            canonical_hash([small.orbital.identity, OXYGEN_F_EXPONENT]),
        ),
        representation="spherical",
    )
    return FocalBasis(
        2,
        "compact OH oxygen-f control",
        orbital,
        jk_fit=jk,
        mp2_fit=ri,
        f12_fit=ri,
        cabs=orbital,
        beta=0.9,
    )


def targets() -> Iterator[tuple[str, np.ndarray]]:
    yield "center", COORDINATES.copy()
    for atom, axis in np.ndindex(COORDINATES.shape):
        for step in STEPS:
            for delta in (-step, step):
                xyz = COORDINATES.copy()
                xyz[atom, axis] += delta
                yield f"{atom}-{axis}-{delta:+.4f}", xyz


def pyscf_oracle(card: FocalBasis) -> dict:
    from pyscf import ao2mo, df, gto, lib, scf
    from pyscf.cc import uccsd

    lib.num_threads(1)

    def shells(record: BasisSet) -> dict[str, list]:
        def entry(shell: BasisShell) -> list:
            return [
                shell.angular_momentum,
                *[
                    [
                        float(exponent),
                        *(float(column[i]) for column in shell.coefficients),
                    ]
                    for i, exponent in enumerate(shell.exponents)
                ],
            ]

        return {
            s: [entry(sh) for sh in record.by_element[z].shells]
            for s, z in zip(SYMBOLS, (8, 1), strict=True)
        }

    orbital = shells(card.orbital)
    jk, ri = shells(card.jk_fit), shells(card.mp2_fit)
    points, seed = {}, None
    for label, xyz in targets():
        mol = gto.M(
            atom=list(zip(SYMBOLS, xyz, strict=True)),
            basis=orbital,
            unit="Bohr",
            spin=1,
            verbose=0,
        )
        mf = scf.UHF(mol).density_fit(auxbasis=jk)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 3e-10, 150
        guess = (
            None
            if seed is None
            else scf.addons.project_dm_nr2nr(
                seed.mol,
                seed.make_rdm1(),
                mol,
            )
        )
        mf.kernel(dm0=guess)
        assert mf.converged and mol.nelec == (5, 4)
        fit = df.DF(mol, auxbasis=ri)
        factors = np.concatenate(list(fit.loop()))
        mf._eri = ao2mo.restore(8, factors.T @ factors, mol.nao)
        cc = uccsd.UCCSD(mf, frozen=None)
        cc.incore_complete, cc.direct = True, False
        cc.conv_tol, cc.conv_tol_normt, cc.max_cycle = 2e-13, 3e-10, 140
        eris = cc.ao2mo()
        cc.kernel(eris=eris)
        assert cc.converged
        row = {
            "hf": mf.e_tot,
            "triples": cc.ccsd_t(eris=eris),
            "coefficients": np.asarray(mf.mo_coeff).tolist(),
            "geometry_bohr": xyz.tolist(),
        }
        if seed is None:
            seed = mf
            row.update(
                hcore=mf.get_hcore().tolist(),
                orbital_energies=np.asarray(mf.mo_energy).tolist(),
                density=np.asarray(mf.make_rdm1()).tolist(),
                fock=np.asarray(mf.get_fock()).tolist(),
                overlap=mf.get_ovlp().tolist(),
            )
        points[label] = row
    return {"basis_id": card.basis_id, "points": points, "occupations": [5, 4]}


def measure(card: FocalBasis, independent: dict, backend: str) -> dict:
    start = monotonic()
    assert independent["basis_id"] == card.basis_id
    series = FocalSeries(card)
    root = None
    values, overlaps = {}, []
    for label, xyz in targets():
        request = next(
            q
            for q, _ in plan_singlepoint(
                list(zip(SYMBOLS, xyz, strict=True)),
                series,
                series,
                None,
                f12_doubles=card,
                triples=card,
                root_id="uhf",
                multiplicity=2,
                correlation_space_id="all-active",
                backend=backend,
                protocol_id="OH compact spherical oxygen-f",
            ).steps
            if q.role == "triples"
        )
        oracle = independent["points"][label]
        assert xyz.tolist() == oracle["geometry_bohr"]
        if root is None:
            model = FocalReference(
                oracle["hf"],
                None,
                *(
                    np.asarray(oracle[k])
                    for k in (
                        "overlap",
                        "hcore",
                        "coefficients",
                        "orbital_energies",
                        "density",
                        "fock",
                    )
                ),
                (5, 4),
                0,
                "cpu_reference",
                request.surface_id,
                card.basis_id,
                card.orbital.identity,
                card.jk_fit.identity,
            )
            root = FocalReferenceSeed(request, model, minimum_overlap=0.999)
        ref = df_reference(request, forces=False, seed=root, budget_mib=1536)
        assert (
            ref.executed_backend == backend
            and ref.forces is None
            and ref.occupied == (5, 4)
        )
        assert abs(ref.energy - oracle["hf"]) < 3e-8
        for n, external, live in zip(
            ref.occupied, oracle["coefficients"], ref.coefficients, strict=True
        ):
            singular = np.linalg.svd(
                np.asarray(external)[:, :n].T @ ref.overlap @ live[:, :n],
                compute_uv=False,
            )
            overlaps.append(float(singular.min()))
        value, force = molecular_triples_on_reference(
            request,
            ref,
            budget_mib=1536,
            forces=label == "center",
        )
        values[label] = value
        if force is not None:
            center_force = force
            root = FocalReferenceSeed(request, ref, minimum_overlap=0.999)
        assert (force is not None) == (label == "center")
    energies = {label: row["triples"] for label, row in independent["points"].items()}
    errors = {}
    for step in STEPS:
        ref_force = np.zeros_like(center_force)
        fd_force = np.zeros_like(center_force)
        for atom, axis in np.ndindex(COORDINATES.shape):
            minus, plus = [f"{atom}-{axis}-{sign * step:+.4f}" for sign in (-1, 1)]
            ref_force[atom, axis] = (energies[minus] - energies[plus]) / (2 * step)
            fd_force[atom, axis] = (values[minus] - values[plus]) / (2 * step)
        errors[str(step)] = {
            "independent": float(abs(center_force - ref_force).max()),
            "native": float(abs(center_force - fd_force).max()),
        }
    max_energy_error = max(abs(values[k] - energies[k]) for k in values)
    assert abs(values["center"]) > 1e-8 and min(overlaps) > 0.999
    assert max_energy_error < 3e-9
    assert max(e for pair in errors.values() for e in pair.values()) < 3e-8
    return {
        "passed": True,
        "backend": backend,
        "science_seconds": monotonic() - start,
        "basis_id": card.basis_id,
        "nominal_orbital": "STO-3G plus one real spherical oxygen f",
        "occupied": [5, 4],
        "triples_energies_Eh": values,
        "forces_Eh_per_bohr": center_force.tolist(),
        "independent_max_energy_Eh": max_energy_error,
        "root_overlap_min": min(overlaps),
        "errors_by_step_Eh_per_bohr": errors,
        "analytic_centers": 1,
        "energy_only_points": 24,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--backend", choices=("oracle", "cpu_reference", "cuda"), required=True
    )
    parser.add_argument(
        "--bounded-host-fit",
        action="store_true",
        help="Use only s through f fitting shells in the host-only control.",
    )
    args = parser.parse_args()
    card = basis(args.data, args.bounded_host_fit)
    result = (
        pyscf_oracle(card)
        if args.backend == "oracle"
        else measure(
            card,
            json.loads((args.data / "independent.json").read_bytes()),
            args.backend,
        )
    )
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
