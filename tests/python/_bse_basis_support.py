"""Redistributable test bases read from basis_set_exchange with pinned content.

MolSSI Basis Set Exchange data are BSD-3-Clause and arrive with the
``reference-test`` extra. Every record used here is checked against the
element content hash below, so a BSE release that changes a record fails
loudly instead of silently moving a numerical reference.
"""

from collections.abc import Iterable
from types import ModuleType

import pytest
from generativeqc.basis import (
    BasisProvenance,
    BasisSet,
    ElementBasis,
    element_content_sha256,
)
from generativeqc.basis_import import bse_element_records
from generativeqc.profiles import canonical_hash

CONTENT_SHA256 = {
    "cc-pVDZ-PP": {
        35: "637231493b296aa5018b5eb2bd49a586e6cfddbe00df5f0f73856b19437e0c92",
    },
    "aug-cc-pVDZ-PP": {
        35: "d7029142b5338caba7288e8ceaaa90042979a0c49d0d0cd761a2825d360eb27e",
        53: "274f6347c0798bc1c48d773b28fa3d8a67bd022308df7c4326a33e70960bc8b0",
    },
    "aug-cc-pVTZ-PP": {
        35: "afb818a239a819c215e88729acb9c69a82e9393d727760df5daa9a96d8aaf3f1",
        53: "b0eae37f2439d9206708217c6516a6fcf69ac483cffee807347fe29e97d9db31",
    },
    "aug-cc-pVDZ": {
        1: "f2dd7edad423fbb3b771915f809e93f4570c12660534a8109b7e87691a7b1672",
        6: "4e7e767e5fc948c15930b67b898d85bde7c6e7503a63cfe324949acb4e20bd12",
        8: "6ef52aef357ca4744137f9f1df34dbcb55fe5cecd6c3f5d4f851f74d358ac78e",
    },
    "aug-cc-pVTZ": {
        1: "18ac9303c7aa2aabbac294d7b23a2d577a255c512ec6af66d60051c4041b5413",
        6: "a4de2ff3ce6e260b006289fa9af1ed1e41976c9c6ee2814e9865738f23c0862c",
    },
    "def2-universal-JKFIT": {
        1: "bf11c70bfe971463f3530ca810e50932a844bba80f0b54d1ea14cc6925fdef70",
        6: "8237aff18f432b9b36bd5f5cac3b5c9edbeee500441772e9eeed89ea283d3e4e",
        8: "a44f330642b01247ee9ae6e134564ef0de46dd49d0a959abc71d1888fb64dc9c",
        35: "e9b8c7a2c33b45ed7e18a8864485a7d3be77d856188bd039e40acc6b69631118",
        53: "805d83e482a851b19c8e6ceef1cf19144aec9b0424942d069c985430dd09e46b",
    },
}
PP_ORBITAL = {
    2: ("aug-cc-pVDZ-PP", "aug-cc-pVDZ"),
    3: ("aug-cc-pVTZ-PP", "aug-cc-pVTZ"),
}


def _bse() -> ModuleType:
    return pytest.importorskip(
        "basis_set_exchange",
        reason="MolSSI BSE test bases need the reference-test extra",
    )


def bse_element(name: str, atomic_number: int) -> ElementBasis:
    """Return one pinned spherical BSE element record."""
    bse = _bse()
    data = bse.get_basis(name, elements=[atomic_number], header=False)
    element = bse_element_records(data)[atomic_number]
    found = element_content_sha256(element, "spherical")
    assert found == CONTENT_SHA256[name][atomic_number], (
        f"basis_set_exchange {bse.get_version()} changed {name} Z={atomic_number}"
    )
    return element


def bse_basis(names: dict[int, str], label: str) -> BasisSet:
    """Assemble pinned BSE records named per element, in atomic-number order."""
    bse = _bse()
    records = tuple(bse_element(names[z], z) for z in sorted(names))
    return BasisSet(
        label,
        records,
        BasisProvenance(
            "MolSSI Basis Set Exchange",
            f"basis_set_exchange {bse.get_version()}",
            "BSD-3-Clause",
            canonical_hash([element_content_sha256(e, "spherical") for e in records]),
        ),
        "spherical",
    )


def pp_orbital_basis(elements: Iterable[int], cardinal: int = 2) -> BasisSet:
    """aug-cc-pVnZ-PP for Br/I (ECP10MDF/ECP28MDF) and aug-cc-pVnZ elsewhere.

    The triple-zeta sets reach f shells on C, Br and I and d shells on H.
    """
    pp, all_electron = PP_ORBITAL[cardinal]
    names = {z: pp if z in (35, 53) else all_electron for z in set(elements)}
    return bse_basis(names, f"{pp}/{all_electron}")


def jk_fit_basis(elements: Iterable[int]) -> BasisSet:
    """def2-universal-JKFIT for the requested elements."""
    return bse_basis(
        {z: "def2-universal-JKFIT" for z in set(elements)}, "def2-universal-JKFIT"
    )


def br_ecp10mdf() -> ElementBasis:
    """Br ECP10MDF as distributed with BSE cc-pVDZ-PP; use its ECP fields only."""
    return bse_element("cc-pVDZ-PP", 35)
