"""Frozen-core shell-gap guard against independent NaBr and NaI DF-HF spectra."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pytest
from _bse_basis_support import bse_element
from generativeqc import Atom
from generativeqc._f12_frozen_core import (
    DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH,
    check_frozen_core_shell_gap,
    resolve_frozen_occupied_for_atoms,
    select_frozen_occupied_frame,
)
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis

ROOT = Path(__file__).resolve().parents[2]
ORACLE = json.loads(
    (ROOT / "tests/data/frozen_core_gap/nax_df_hf_spectra.json").read_text(
        encoding="utf-8"
    )
)
HALOGEN_Z = {"Br": 35, "I": 53}
# Na 1s2s2p and the halogen (n-1)sp outside the ECP core.
FORMER_FC4 = 9
# The same orbitals and the halogen (n-1)d shell.
FORMER_FC9 = 14


def _case_id(case: dict) -> str:
    sign = {0: "", 1: "+", -1: "-"}[case["charge"]]
    cardinal = "DT"[case["cardinal"] - 2]
    return f"Na{case['halogen']}{sign}-{cardinal}Z-{case['bond_factor']:.2f}re"


CASES = [pytest.param(case, id=_case_id(case)) for case in ORACLE["cases"]]
CHANNELS = [channel for case in ORACLE["cases"] for channel in case["channels"]]


def _oracle_case(halogen: str, charge: int) -> dict:
    return next(
        case
        for case in ORACLE["cases"]
        if case["halogen"] == halogen
        and case["charge"] == charge
        and case["cardinal"] == 2
        and case["bond_factor"] == 1.0
    )


def _atoms(case: dict) -> tuple[Atom, Atom]:
    z = HALOGEN_Z[case["halogen"]]
    return (Atom(11, (0.0, 0.0, 0.0)), Atom(z, (0.0, 0.0, case["bond_length_bohr"])))


@functools.cache
def _counting_basis(halogen: str) -> BasisSet:
    """Combine a BSE halogen ECP with a placeholder sodium record for counting."""
    sodium = ElementBasis(11, (BasisShell(0, ("1.0",), (("1.0",),)),))
    halogen_record = bse_element("aug-cc-pVDZ-PP", HALOGEN_Z[halogen])
    return BasisSet(
        f"na{halogen.lower()}-counting",
        (sodium, halogen_record),
        BasisProvenance("Synthetic sodium and Basis Set Exchange", "test", "BSD-3-Clause", "0" * 64),
        "spherical",
    )


def _resolve(
    case: dict, space: str, min_gap_eh: float = DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH
) -> tuple[int, int]:
    channels = case["channels"]
    energies = [np.asarray(c["occupied_energies_Eh"]) for c in channels]
    return resolve_frozen_occupied_for_atoms(
        _atoms(case),
        _counting_basis(case["halogen"]),
        (channels[0]["occupied"], channels[-1]["occupied"]),
        space,
        root_id=case["root_id"],
        orbital_energies=energies[0] if len(energies) == 1 else energies,
        min_gap_eh=min_gap_eh,
    )


def _cut_gap(channel: dict, n_frozen: int) -> float:
    energies = channel["occupied_energies_Eh"]
    return energies[n_frozen] - energies[n_frozen - 1]


@pytest.mark.parametrize("case", CASES)
def test_historical_explicit_cut_inside_the_halogen_d_shell_is_refused(case: dict) -> None:
    d_shell = f"{case['halogen']} d"
    for channel in case["channels"]:
        character = channel["dominant_character"]
        assert character[FORMER_FC4 - 1] == character[FORMER_FC4] == d_shell
    with pytest.raises(ValueError, match="splits a near-degenerate shell"):
        _resolve(case, "frozen:9")


@pytest.mark.parametrize("case", CASES)
def test_historical_explicit_cut_between_complete_shells_is_accepted(case: dict) -> None:
    for channel in case["channels"]:
        character = channel["dominant_character"]
        assert character[FORMER_FC9 - 1] == "Na p"
        assert character[FORMER_FC9] == f"{case['halogen']} s"
    assert _resolve(case, "frozen:14") == (FORMER_FC9, FORMER_FC9)


def test_default_gap_lies_inside_the_decision_invariant_window() -> None:
    widest_split = max(_cut_gap(c, FORMER_FC4) for c in CHANNELS)
    narrowest_boundary = min(_cut_gap(c, FORMER_FC9) for c in CHANNELS)
    assert widest_split < DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH < narrowest_boundary

    for threshold in (
        float(np.nextafter(widest_split, np.inf)),
        DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH,
        float(np.nextafter(narrowest_boundary, 0.0)),
    ):
        for channel in CHANNELS:
            energies = channel["occupied_energies_Eh"]
            n_occupied = channel["occupied"]
            with pytest.raises(ValueError, match="splits a near-degenerate shell"):
                check_frozen_core_shell_gap(
                    energies, FORMER_FC4, threshold, n_occupied=n_occupied
                )
            assert check_frozen_core_shell_gap(
                energies, FORMER_FC9, threshold, n_occupied=n_occupied
            ) == _cut_gap(channel, FORMER_FC9)

    widest = max(CHANNELS, key=lambda c: _cut_gap(c, FORMER_FC4))
    assert (
        check_frozen_core_shell_gap(
            widest["occupied_energies_Eh"],
            FORMER_FC4,
            widest_split,
            n_occupied=widest["occupied"],
        )
        == widest_split
    )


def _native_basis(halogen: str) -> tuple[BasisSet, BasisSet]:
    bse = pytest.importorskip("basis_set_exchange")
    z = HALOGEN_Z[halogen]
    try:
        external = methyl_halide_basis(halogen, 2)
    except FileNotFoundError as error:
        pytest.skip(str(error))

    def sodium(name: str) -> ElementBasis:
        element = bse.get_basis(name, elements=[11], header=False)["elements"]["11"]
        shells = []
        for shell in element["electron_shells"]:
            exponents = tuple(str(e) for e in shell["exponents"])
            rows = tuple(tuple(str(c) for c in row) for row in shell["coefficients"])
            (am,) = shell["angular_momentum"]
            shells.append(BasisShell(am, exponents, rows))
        return ElementBasis(11, tuple(shells))

    def combined(role: str, sodium_name: str, record: BasisSet) -> BasisSet:
        return BasisSet(
            f"na{halogen.lower()}-{role}",
            (sodium(sodium_name), record.by_element[z]),
            BasisProvenance(
                "Basis Set Exchange sodium and verified external methyl-halide records",
                "test", record.provenance.license, "0" * 64,
            ),
            "spherical",
        )

    return (
        combined("orbital", "cc-pVDZ-F12", external.orbital),
        combined("jkfit", "def2-universal-JKFIT", external.jk_fit),
    )
