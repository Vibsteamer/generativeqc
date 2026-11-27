"""Compare the reconstructed methyl-halide basis records with PySCF's library."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
from generativeqc.basis_import import bse_element_records

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "python/generativeqc/data/f12_methyl_halide_external.json"
)
Shell = tuple[int, Iterable[float], Iterable[Iterable[float]]]
ContractedFunction = tuple[tuple[float, float], ...]


def _contracted_functions(
    shells: Iterable[Shell],
) -> dict[int, list[ContractedFunction]]:
    """Canonicalize each contraction independently of shell grouping and order."""
    functions: dict[int, list[ContractedFunction]] = {}
    for angular, exponents, coefficients in shells:
        for column in coefficients:
            pairs = tuple(
                sorted(
                    (float(exponent), float(coefficient))
                    for exponent, coefficient in zip(exponents, column, strict=True)
                    if float(coefficient) != 0.0
                )
            )
            functions.setdefault(angular, []).append(pairs)
    return {angular: sorted(columns) for angular, columns in functions.items()}


def _pyscf_shells(rows_by_shell: list) -> list[Shell]:
    return [
        (
            angular,
            [row[0] for row in rows],
            list(zip(*(row[1:] for row in rows), strict=True)),
        )
        for angular, *rows in rows_by_shell
    ]


@pytest.mark.parametrize(("halogen", "halogen_z"), [("Br", 35), ("I", 53)])
@pytest.mark.parametrize("cardinal", ["d", "t"])
def test_methyl_halide_bse_records_match_pyscf(
    halogen: str, halogen_z: int, cardinal: str
) -> None:
    bse = pytest.importorskip("basis_set_exchange")
    gto = pytest.importorskip("pyscf.gto")
    key = f"{halogen.lower()}-{cardinal}"
    roles = json.loads(MANIFEST.read_text(encoding="utf-8"))["series"][key]["roles"]
    # Select exactly the redistributable records used by this reconstruction.
    comparisons = (
        ("orbital", 1, "H", f"cc-pv{cardinal}z-f12"),
        ("ri", 6, "C", "aug-cc-pvtz-ri"),
        ("jkfit", 1, "H", "def2-universal-jkfit"),
        ("jkfit", 6, "C", "def2-universal-jkfit"),
        ("jkfit", halogen_z, halogen, "def2-universal-jkfit"),
    )
    for role, z, symbol, pyscf_name in comparisons:
        spec = next(
            entry for entry in roles[role]["elements"] if entry["atomic_number"] == z
        )
        (record,) = spec["records"]
        data = bse.get_basis(record["bse"], elements=[z], header=False)
        element = bse_element_records(data)[z]
        actual = _contracted_functions(
            (shell.angular_momentum, shell.exponents, shell.coefficients)
            for shell in element.shells
        )
        expected = _contracted_functions(
            _pyscf_shells(gto.basis.load(pyscf_name, symbol))
        )
        assert actual.keys() == expected.keys(), (key, role, symbol)
        for angular in actual:
            assert len(actual[angular]) == len(expected[angular]), (
                key,
                role,
                symbol,
                angular,
            )
            for left, right in zip(actual[angular], expected[angular], strict=True):
                np.testing.assert_allclose(
                    left,
                    right,
                    rtol=1e-12,
                    atol=0,
                    err_msg=f"{key} {role} {symbol} l={angular}",
                )

    spec = next(
        entry
        for entry in roles["orbital"]["elements"]
        if entry["atomic_number"] == halogen_z
    )
    data = bse.get_basis(spec["ecp"]["bse"], elements=[halogen_z], header=False)
    element = bse_element_records(data)[halogen_z]
    # The same scalar pseudopotential is used at D and T cardinalities.
    core, terms = gto.basis.load_ecp("cc-pvdz-pp", halogen)
    assert element.ecp_core_electrons == core == {"Br": 10, "I": 28}[halogen]
    potentials = json.loads(element.ecp_data)
    actual_terms = sorted(
        (potential["angular_momentum"][0], power, float(exponent), float(coefficient))
        for potential in potentials
        for power, exponent, coefficient in zip(
            potential["r_exponents"],
            potential["gaussian_exponents"],
            potential["coefficients"][0],
            strict=True,
        )
        if float(coefficient) != 0.0
    )
    # PySCF labels the local channel -1; BSE stores it above the nonlocal channels.
    local = max(term[0] for term in actual_terms)
    expected_terms = sorted(
        (local if angular < 0 else angular, power, exponent, coefficient)
        for angular, by_power in terms
        for power, primitives in enumerate(by_power)
        for exponent, coefficient in primitives
        if coefficient != 0.0
    )
    assert actual_terms == expected_terms
