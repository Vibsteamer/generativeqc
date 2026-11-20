"""Method identity and molecular checks for an independently weighted F12 doubles correction."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc.experimental.f12_focal import (
    FocalEvaluation,
    FocalReference,
    FocalRequest,
    FocalSeries,
    MolecularFocalProvider,
    combine,
    plan_singlepoint,
)
from test_f12_focal_entry import basis, term
from test_focal_molecular_provider import XYZ, request


def test_f12_doubles_role_has_its_own_series_and_optional_triples() -> None:
    d, t = basis(2, "def2-svp"), basis(3, "def2-tzvp")
    hf, pair = FocalSeries(t), FocalSeries(t)
    doubles = FocalSeries(t, d, 3.0)
    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))]
    args = {
        "root_id": "rhf",
        "correlation_space_id": "all-active",
        "backend": "cpu_reference",
    }
    plan = plan_singlepoint(
        atoms, hf, pair, post_mp2=None, f12_doubles=doubles, triples=d, **args
    )
    values = {
        "hf_cabs": -1.0,
        "mp2_f12_correlation": -0.12,
        "ccsd_f12b_minus_mp2_f12": 0.02,
        "triples": -0.003,
    }
    measured = []
    for req, _ in plan.steps:
        e = values[req.role]
        if req.role == "ccsd_f12b_minus_mp2_f12":
            e += 0.03 * req.basis.cardinal**-3
        measured.append(
            FocalEvaluation(term(req, e, np.eye(2, 3) * e), req.backend, True)
        )
    result = plan.collect(reversed(measured))
    h, c, lo, hi, tr = (x.term for x in measured)
    doubled = FocalSeries(hi, lo, 3.0)
    explicit = combine(
        FocalSeries(h), FocalSeries(c), post_mp2=None, f12_doubles=doubled, triples=tr
    )
    assert result.energy == explicit.energy == pytest.approx(-1.103, abs=2e-14)
    np.testing.assert_array_equal(result.forces, explicit.forces)
    without_triples = combine(
        FocalSeries(h), FocalSeries(c), post_mp2=None, f12_doubles=doubled
    )
    assert without_triples.energy == pytest.approx(-1.1, abs=2e-14)
    conventional = replace(measured[2], term=replace(lo, role="ccsd_minus_mp2"))
    with pytest.raises(ValueError, match="unexpected focal component"):
        plan.collect((conventional, *measured[:2], *measured[3:]))
    with pytest.raises(ValueError, match="choose either"):
        plan_singlepoint(atoms, hf, pair, d, f12_doubles=doubles, **args)


def test_uhf_f12_doubles_share_one_reference_and_three_distinct_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import generativeqc._f12_focal_reference as reference

    shell = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shell:
        pytest.skip("requires the compiled native molecular integral library")
    req = request(XYZ, (0, 1, 2))
    provider = MolecularFocalProvider(shell_library=Path(shell))
    observed = []
    real = reference.df_reference

    def track(
        req: FocalRequest, *, forces: bool = True, budget_mib: int = 512
    ) -> FocalReference:
        state = real(req, forces=forces, budget_mib=budget_mib)
        observed.append(state)
        return state

    monkeypatch.setattr(reference, "df_reference", track)
    result = provider(replace(req, role="ccsd_f12b_minus_mp2_f12"))
    (root,) = observed
    assert root.occupied == (2, 1) and root.forces is None
    assert (
        len({root.jk_fit_id, req.basis.f12_fit.identity, req.basis.mp2_fit.identity})
        == 3
    )
    assert result.converged and result.executed_backend == "cpu_reference"
    assert result.term.role == "ccsd_f12b_minus_mp2_f12"
    mp2 = provider(req).term
    # Independent SciPy CC root and two five-point H3 stationary control stencils.
    assert (result.term.energy + mp2.energy) == pytest.approx(
        -0.06269423582838254, abs=2e-9, rel=0
    )
    expected = np.array(
        [
            [-0.015595598786942675, -0.00544913766399822, 0.003396720797919454],
            [0.01152223908004619, 0.007655165905132155, -0.0021322076194293473],
            [0.0040733597096373475, -0.0022060282405961715, -0.0012645131768652489],
        ]
    )
    np.testing.assert_allclose(
        result.term.forces + mp2.forces, expected, rtol=0, atol=2e-7
    )
