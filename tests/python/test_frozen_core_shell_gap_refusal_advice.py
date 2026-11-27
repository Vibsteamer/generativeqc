"""Frozen-core refusal diagnostics reflect the information provided by the caller."""

from dataclasses import replace

import numpy as np
import pytest

from generativeqc import Atom
from generativeqc import _f12_frozen_core as core
from generativeqc._f12_frozen_core import resolve_frozen_occupied_for_atoms
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis


def test_explicit_count_refusal_lists_nearest_admissible_counts_by_spin():
    shell = BasisShell(0, ("1.0",), (("1.0",),))
    atoms = (Atom(6, (0.0, 0.0, 0.0)),)
    basis = BasisSet(
        "diagnostic", (ElementBasis(6, (shell,)),),
        BasisProvenance("synthetic", "1", "MIT", "0" * 64), "spherical",
    )
    alpha = np.array([-11.0, -2.5, -1.74, -1.50, -1.10, -0.77])
    beta = np.array([-11.0, -2.5, -1.74, -1.73, -1.10, -0.77])
    with pytest.raises(ValueError, match=r"beta spin.*'frozen:3,3'.*are 2 and 4"):
        resolve_frozen_occupied_for_atoms(
            atoms, basis, (6, 5), "frozen:3,3", root_id="uhf", orbital_energies=(alpha, beta),
        )


def test_chemical_core_needs_coefficients_to_decide_which_orbitals_to_freeze():
    shell = BasisShell(0, ("1.0",), (("1.0",),))
    atoms = (Atom(11, (0.0, 0.0, 0.0)),)
    basis = BasisSet(
        "diagnostic", (ElementBasis(11, (shell,)),),
        BasisProvenance("synthetic", "1", "MIT", "0" * 64), "spherical",
    )
    assert resolve_frozen_occupied_for_atoms(atoms, basis, (6, 5), "frozen-core-spd") == (1, 1)
    with pytest.raises(ValueError, match="requires orbital coefficients and overlap"):
        resolve_frozen_occupied_for_atoms(
            atoms, basis, (6, 5), "frozen-core-spd",
            orbital_energies=[-40.5, -2.8, -1.51, -1.50, -1.49, -0.25],
        )


@pytest.fixture
def interleaved_nai(monkeypatch):
    """A canonical NaI spectrum with occupied Na 2s below occupied I 4d."""
    s, p, d = (BasisShell(l, ("1",), (("1",),)) for l in range(3))
    na = ElementBasis(11, (s, s, s, p))
    iodine = ElementBasis(
        53, (s, p, d, s, p), ecp_core_electrons=28,
        ecp_data='[{"ecp_type":"scalar_ecp","angular_momentum":[0],"r_exponents":[0],'
                 '"gaussian_exponents":["1"],"coefficients":[["1"]]}]',
    )
    basis = BasisSet("NaI test orbital basis", (na, iodine),
                     BasisProvenance("synthetic", "1", "MIT", "0" * 64), "spherical")
    atoms = (Atom(11, (0, 0, 0)), Atom(53, (0, 0, 5)))
    monkeypatch.setattr(
        core, "_atomic_core_orbitals",
        lambda basis, element: ((np.eye(6)[:, :1], np.arange(2.0)) if element == na else
                                (np.eye(13)[:, :9], np.arange(10.0))),
    )
    cols = (0, 6, 7, 8, 9, 1, 10, 11, 12, 13, 14, 3, 4, 5, 15, 16, 17, 18, 2)
    c = np.eye(19)[:, cols]
    eps = np.array([-40., -8., -7.2, -7.1, -7., -2.55, -2.28, -2.27, -2.26,
                    -2.25, -2.24, -1.56, -1.55, -1.54, -.90, -.36, -.35, -.34, .2])
    return atoms, basis, np.eye(19), c, eps


def test_selects_i_4d_by_identity_while_correlating_na_2s(interleaved_nai):
    atoms, basis, overlap, c, eps = interleaved_nai
    frame = core.select_frozen_occupied_frame(
        atoms, basis, overlap, (c,), (eps,), (18,),
        "cv-active", "frozen-core-spd",
    )
    assert frame.counts("cv-active") == (1, 1)
    assert frame.counts("frozen-core-spd") == (10, 10)
    assert frame.channels("frozen-core-spd")[0].indices == (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)
    assert frame.occupied_order[0] == (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 5, 11, 12, 13, 14, 15, 16, 17)
    assert frame.channels("frozen-core-spd")[0].minimum_gap_eh == pytest.approx(.27)

    fc = core.select_frozen_occupied_frame(atoms, basis, overlap, (c,), (eps,), (18,), "frozen-core")
    assert fc.channels("frozen-core")[0].indices == (0, 1, 2, 3, 4)
    assert fc.occupied_order[0] == tuple(range(18))


def test_rejects_actual_interleaved_gap_and_a_nonnested_cv_space(interleaved_nai):
    atoms, basis, overlap, c, eps = interleaved_nai
    beta = eps.copy()
    beta[5] = -2.29
    with pytest.raises(ValueError, match=r"beta spin.*near-degenerate frozen-active pair.*0\.0500 Eh"):
        core.select_frozen_occupied_frame(
            atoms, basis, overlap, (c, c), (eps, beta), (18, 17), "frozen-core-spd",
        )
    with pytest.raises(ValueError, match="cv-active frozen selection to be a subset"):
        core.select_frozen_occupied_frame(
            atoms, basis, overlap, (c,), (eps,), (18,), "cv-active", "frozen:0",
        )


def test_interleaved_halogen_subshell_requires_spd_policy(interleaved_nai):
    atoms, basis, overlap, c, eps = interleaved_nai
    displaced_frame = c.copy()
    displaced_frame[:, [3, 6]] = displaced_frame[:, [6, 3]]
    with pytest.raises(ValueError, match="interleaved .* core shell active.*frozen-core-spd"):
        core.select_frozen_occupied_frame(
            atoms, basis, overlap, (displaced_frame,), (eps,), (18,), "frozen-core",
        )


def test_no_active_occupied_still_refused_after_sodium_switch(interleaved_nai):
    atoms, basis, overlap, c, eps = interleaved_nai
    na2 = (atoms[0], replace(atoms[0], position=(0, 0, 5.0)))
    assert core.resolve_frozen_occupied_for_atoms(na2, basis, (10, 10), "frozen-core-spd") == (2, 2)
    assert core.resolve_frozen_occupied_for_atoms(na2, basis, (10, 9), "frozen-core-spd") == (2, 2)
    with pytest.raises(ValueError, match="leaves no active occupied orbital in a spin channel"):
        core.select_frozen_occupied_frame(
            na2, basis, overlap, (c, c), (eps, eps), (3, 2), "frozen-core-spd",
        )
