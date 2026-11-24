"""Per-element focal bases bind the selected shells, ECPs and fitted operators."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc.basis import load_basis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalSeries,
    conventional_df_post_mp2,
    plan_singlepoint,
)
from test_cbs_open_shell_native import independent_components


def test_element_selections_snapshot_each_operator_and_ecp(tmp_path: Path) -> None:
    small = FocalBasis(2, "test", "sto-3g", jk_fit="def2-svp")
    sr = load_basis(
        Path(__file__).resolve().parents[1] / "reference_data/ks-scf-2984/basis.json"
    )
    source = tmp_path / "sr.json"
    sr.write(source)
    selections = {"H": "sto-3g", "Sr": source}
    auxiliary = {1: small.jk_fit, 38: source}
    mixed = FocalBasis(
        2,
        "test",
        selections,
        jk_fit=auxiliary,
        mp2_fit=auxiliary,
        f12_fit=selections,
        cabs=auxiliary,
        beta=1.0,
    )
    assert mixed.orbital.by_element == {
        1: small.orbital.by_element[1],
        38: sr.by_element[38],
    }
    assert mixed.orbital.by_element[38].ecp_core_electrons == 28
    assert mixed.f12_fit == mixed.orbital
    assert mixed.jk_fit == mixed.cabs == mixed.mp2_fit
    assert mixed.jk_fit.by_element[1] == small.jk_fit.by_element[1]
    assert mixed.jk_fit.identity != mixed.orbital.identity
    manifest = json.loads(mixed.orbital.provenance.source)
    assert [entry["basis_id"] for entry in manifest] == [
        small.orbital.identity,
        sr.identity,
    ]
    original_id = mixed.basis_id
    source.unlink()
    selections.clear()
    same = FocalBasis(
        2,
        "test",
        {38: sr, 1: small.orbital},
        jk_fit={"Sr": sr, "H": small.jk_fit},
        mp2_fit=mixed.mp2_fit,
        f12_fit=mixed.f12_fit,
        cabs=mixed.cabs,
        beta=1.0,
    )
    assert same.basis_id == mixed.basis_id == original_id


def test_bad_element_maps_fail_at_construction() -> None:
    src = FocalBasis(2, "test", "sto-3g", jk_fit="def2-svp")
    only_h = replace(src.orbital, elements=(src.orbital.by_element[1],))
    for mapping, problem in [
        ({}, "at least one"),
        ({"H": only_h, 1: src.jk_fit}, "duplicate"),
        ({"He": only_h}, "lacks He"),
        (
            {1: only_h, 2: replace(src.jk_fit, representation="cartesian")},
            "representation",
        ),
    ]:
        with pytest.raises(ValueError, match=problem):
            FocalBasis(2, "test", src.orbital, jk_fit=mapping)


def test_cardinal_pair_must_change_an_orbital_basis_used_by_this_geometry() -> None:
    low = FocalBasis(2, "mixed", "sto-3g", jk_fit="sto-3g", cabs="def2-svp")
    upper = FocalBasis(
        3,
        "mixed",
        {"H": "sto-3g", "He": "def2-svp"},
        jk_fit="sto-3g",
        cabs="def2-svp",
    )
    assert low.orbital.identity != upper.orbital.identity
    with pytest.raises(ValueError, match="distinct orbital bases"):
        plan_singlepoint(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            FocalSeries(upper, low, 3.0),
            FocalSeries(low),
            low,
            root_id="rhf",
            correlation_space_id="all-active",
        )


def test_mixed_basis_doublet_increment_against_independent_energies() -> None:
    """Compare the HeH native correction and its Cartesian forces to displaced PySCF."""
    basis = FocalBasis(
        2,
        "mixed HeH",
        {"H": "sto-3g", "He": "def2-svp"},
        jk_fit={"H": "def2-svp", "He": "def2-svp"},
        mp2_fit={2: "def2-svp", 1: "def2-svp"},
        f12_fit="def2-svp",
        cabs="def2-svp",
        beta=1.0,
    )
    vector = np.array([1.6, 0.3, -0.2])
    direction = vector / np.linalg.norm(vector)
    atoms = (("He", (0.0, 0.0, 0.0)), ("H", tuple(vector)))
    plan = plan_singlepoint(
        atoms,
        FocalSeries(basis),
        FocalSeries(basis),
        basis,
        multiplicity=2,
        root_id="uhf",
        correlation_space_id="all-active",
        backend="cpu_reference",
    )
    result = conventional_df_post_mp2(
        next(r for r, _ in plan.steps if r.role == "post_mp2")
    )
    # Assemble the independent PySCF records directly from the two source families.
    source = FocalBasis(2, "test", "sto-3g", jk_fit="def2-svp")
    orbital = replace(
        source.orbital,
        elements=(source.orbital.by_element[1], source.jk_fit.by_element[2]),
    )

    def oracle(displacement: float) -> float:
        geometry = (atoms[0], ("H", tuple(vector + direction * displacement)))
        energies = independent_components(
            orbital, 0, 2, atoms=geometry, auxiliary=source.jk_fit, full_reference=True
        )
        return energies["uccsd(t)"] - energies["ump2"]

    assert result.converged and result.executed_backend == "cpu_reference"
    assert result.term.energy == pytest.approx(oracle(0), abs=3e-9, rel=0)
    for step in (4e-4, 2e-4):
        derivative = (oracle(step) - oracle(-step)) / (2 * step)
        np.testing.assert_allclose(
            result.term.forces,
            [derivative * direction, -derivative * direction],
            atol=3e-8,
            rtol=0,
        )
