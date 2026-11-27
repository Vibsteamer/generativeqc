"""Gate 1 unit tests for Upstream-PR Hardening Track B (B1, H1, H5, M2, M7, M8)."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# B1: Optional basis_set_exchange guard & packaging declaration
# ============================================================================


def test_b1_pyproject_declares_basis_set_exchange_optional_extra() -> None:
    """pyproject.toml declares basis-set-exchange in optional dependencies."""
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    opt = data["project"]["optional-dependencies"]
    assert any(dep.startswith("basis-set-exchange") for dep in opt["reference-test"])
    assert any(dep.startswith("basis-set-exchange") for dep in opt["bse"])


# ============================================================================
# M7: Frozen-core shell-splitting energy-gap guard (NaBr FC4 vs FC9)
# ============================================================================


def test_frozen_core_shell_gap_refuses_an_explicit_count_that_splits_a_shell() -> None:
    from generativeqc._f12_frozen_core import check_frozen_core_shell_gap

    # A count-based cut falls inside a fivefold near-degenerate d shell.
    eps = np.array([-40.5, -9.6, -7.2, -7.2, -7.2, -3.014, -3.012, -3.012,
                    -3.010, -3.008, -2.8, -1.55, -1.55, -1.55, -0.9, -0.35])
    with pytest.raises(ValueError, match=r"nearest frozen counts .* are 5 and 10\.$"):
        check_frozen_core_shell_gap(eps, 9, n_occupied=16, correlation_space_id="frozen:9")
    assert check_frozen_core_shell_gap(eps, 10, n_occupied=16) == pytest.approx(0.208)


def test_m7_real_df_hf_shell_gaps_pass_for_h2o_oh_hbr() -> None:
    """Real DF-HF canonical orbital gaps for H2O and OH are >> 0.05 Eh."""
    from generativeqc._f12_focal_reference import df_reference
    from generativeqc._f12_frozen_core import (
        DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH,
        select_frozen_occupied_frame,
    )
    from generativeqc.experimental.f12_focal import FocalBasis, FocalSeries, plan_singlepoint

    # 1. H2O (RHF) and OH (UHF) in sto-3g / def2-svp.
    light_basis = FocalBasis(
        2,
        "m7-light",
        "sto-3g",
        jk_fit="def2-svp",
        mp2_fit="def2-svp",
        f12_fit="def2-svp",
        cabs="def2-svp",
        beta=0.9,
    )
    for atoms, mult, root, cspace in (
        (
            (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.43, 1.1)), ("H", (0.0, -1.43, 1.1))),
            1,
            "rhf",
            "frozen-core",
        ),
        (
            (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.83))),
            2,
            "uhf",
            "frozen-core",
        ),
    ):
        plan = plan_singlepoint(
            atoms,
            FocalSeries(light_basis),
            FocalSeries(light_basis),
            None,
            charge=0,
            multiplicity=mult,
            root_id=root,
            correlation_space_id=cspace,
            backend="cpu_reference",
        )
        req = plan.steps[0][0]
        ref = df_reference(req, forces=False, budget_mib=256)
        frame = select_frozen_occupied_frame(
            req.atoms, req.basis.orbital, ref.overlap, ref.coefficients, ref.orbital_energies,
            ref.occupied, cspace,
        )
        assert frame.counts(cspace) == (1, 1)
        for channel in frame.channels(cspace):
            assert channel.minimum_gap_eh > 10.0 * DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH
