"""Scalar-ECP F12 basis selections and NIST geometries for CH₃Br and CH₃I.

The heavy-augmented D and T selections augment carbon and the halogen and use
the parent hydrogen orbital basis. The JK fit is diagnostic def2-universal-JKFit.
The packaged manifest names the orbital, MP2/F12 fitting and OptRI source for
each element and identifies the scalar ECP and the geometry reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..elements import checked_integer
from .f12_focal import FocalBasis, FocalSeries, plan_singlepoint

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .f12_focal import FocalPlan

_DATA = Path(__file__).resolve().parents[1] / "data" / "f12_methyl_halide"


def methyl_halide_basis(halogen: str, cardinal: int) -> FocalBasis:
    """Select the D (2) or T (3) C/H/Br or C/H/I orbital and all fitting roles."""
    if halogen not in ("Br", "I"):
        raise ValueError("Choose Br or I")
    cardinal = checked_integer(cardinal, "methyl-halide cardinal", low=2, high=3)
    index = json.loads((_DATA / "manifest.json").read_text())
    key = f"{halogen.lower()}-{'d' if cardinal == 2 else 't'}"
    folder = _DATA / key
    basis = FocalBasis(
        cardinal,
        index["series_id"],
        folder / "orbital.json",
        jk_fit=folder / "jkfit.json",
        mp2_fit=folder / "ri.json",
        f12_fit=folder / "ri.json",
        cabs=folder / "cabs.json",
        beta=index["beta_bohr_inverse"],
    )
    if basis.basis_id != index["basis_sets"][key]["id"]:
        raise ValueError(f"The packaged {key} basis identity differs from its manifest")
    return basis


def methyl_halide_plan(
    halogen: str,
    cardinal: int,
    *,
    root: str = "rhf",
    coordinates_bohr: Iterable[Iterable[float]] | None = None,
    backend: str = "cuda",
    f12_doubles: bool = False,
    conventional_triples: bool = False,
) -> FocalPlan:
    """Plan CH₃X neutral RHF or vertical cation UHF at NIST or caller coordinates.

    Positions are in bohr, ordered C, halogen, H, H, H. The optional high-order
    terms request all-active CCSD-F12b minus MP2-F12 and conventional (T) in
    this exact cardinal basis. Execution uses the supplied public provider.
    """
    if root not in ("rhf", "uhf"):
        raise ValueError("Choose rhf (neutral singlet) or uhf (cation doublet)")
    basis = methyl_halide_basis(halogen, cardinal)
    g = json.loads((_DATA / "manifest.json").read_text())["geometries"][halogen]
    coords = g["coordinates_bohr"] if coordinates_bohr is None else coordinates_bohr
    atoms = tuple(zip(g["symbols"], coords, strict=True))
    series = FocalSeries(basis)
    charge = int(root == "uhf")
    return plan_singlepoint(
        atoms,
        series,
        series,
        None,
        root_id=root,
        charge=charge,
        multiplicity=charge + 1,
        correlation_space_id="all-active",
        backend=backend,
        protocol_id=f"CH3{halogen}/ha-PP-F12/split-fit/all-active",
        f12_doubles=basis if f12_doubles else None,
        triples=basis if conventional_triples else None,
    )
