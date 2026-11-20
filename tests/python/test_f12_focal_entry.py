"""Geometry-first F12 focal dispatch with explicit component execution."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalEvaluation,
    FocalRequest,
    FocalRunResult,
    FocalSeries,
    FocalTerm,
    conventional_df_post_mp2,
    plan_singlepoint,
    singlepoint,
)

if TYPE_CHECKING:
    from pathlib import Path

    from generativeqc.experimental.f12_focal import Backend


def basis(cardinal: int, orbital: str = "sto-3g", series: str = "test") -> FocalBasis:
    return FocalBasis(
        cardinal,
        series,
        orbital,
        jk_fit="sto-3g",
        mp2_fit="sto-3g",
        f12_fit="sto-3g",
        cabs="sto-3g",
        beta=0.9 + 0.1 * (cardinal - 2),
    )


def term(request: FocalRequest, energy: float, forces: np.ndarray) -> FocalTerm:
    return FocalTerm(
        request.surface_id,
        request.basis.series_id,
        request.basis.basis_id,
        request.basis.cardinal,
        request.role,
        energy,
        forces,
    )


def test_independent_components_match_serial_and_reject_incomplete_or_stale_results() -> None:
    atoms = [("H", (0, 0, 0)), ("H", (0, 0, 1.4))]
    low, high = basis(2, "def2-svp"), basis(3, "def2-tzvp")
    series = FocalSeries(high, low, 3.0)
    options = dict(root_id="rhf", correlation_space_id="all", backend="cpu_reference")
    plan = plan_singlepoint(atoms, series, series, low, **options)

    def evaluate(request: FocalRequest) -> FocalEvaluation:
        offsets = {"hf_cabs": -1.0, "mp2_f12_correlation": -0.02, "post_mp2": -0.001}
        energy = offsets[request.role] + request.basis.cardinal ** -3
        forces = np.outer([1.0, -1.0], [0.0, 0.0, energy])
        return FocalEvaluation(term(request, energy, forces), request.backend, True)

    returns = [evaluate(request) for request, _ in reversed(plan.steps)]
    result = plan.collect(returns)
    serial = singlepoint(atoms, series, series, low, provider=evaluate, **options)
    assert result.weights == serial.weights
    assert result.energy == serial.energy
    np.testing.assert_array_equal(result.forces, serial.forces)
    assert [e.term.basis_id for e in result.components] == [
        e.term.basis_id for e in serial.components
    ]
    with pytest.raises(ValueError, match="missing.*post_mp2"):
        plan.collect(returns[1:])
    with pytest.raises(ValueError, match="duplicate"):
        plan.collect([*returns, returns[0]])
    changed = plan_singlepoint(
        [atoms[0], ("H", (0, 0, 1.4001))], series, series, low, **options
    )
    with pytest.raises(ValueError, match="surface"):
        changed.collect(returns)
    new_protocol = plan_singlepoint(
        atoms, series, series, low, protocol_id="method-v2-build-b", **options
    )
    with pytest.raises(ValueError, match="surface"):
        new_protocol.collect(returns)
    with pytest.raises(RuntimeError, match="backend"):
        plan.collect([replace(returns[0], executed_backend="cuda"), *returns[1:]])


def test_focal_dispatch_uses_ordered_nuclei_per_role_and_cardinal() -> None:
    low, high = basis(2, "def2-svp"), basis(3, "def2-tzvp")
    assert low.beta != high.beta
    spec = FocalSeries(high, low, 3.0)
    calls = []
    reused = np.empty((2, 3))

    def run(x: np.ndarray) -> FocalRunResult:
        def provider(request: FocalRequest) -> FocalEvaluation:
            xyz = np.array([a.position for a in request.atoms])
            np.testing.assert_array_equal(xyz, x)
            assert [a.atomic_number for a in request.atoms] == [1, 2]
            assert request.charge == 0 and request.multiplicity == 2
            assert request.backend == "cuda"
            assert (
                request.root_id == "doublet" and request.correlation_space_id == "full"
            )
            calls.append((request.role, request.basis.cardinal, request.surface_id))
            target = {"hf_cabs": 1.5, "mp2_f12_correlation": -0.2, "post_mp2": 0.4}
            scale = target[request.role]
            if request.role != "post_mp2":
                scale += request.basis.cardinal**-3
            displacement = xyz[1] - xyz[0]
            reused[:] = scale * np.stack((displacement, -displacement))
            return FocalEvaluation(
                term(request, float(scale * (displacement @ displacement) / 2), reused),
                "cuda",
                converged=True,
            )

        return singlepoint(
            zip(("H", "He"), x, strict=True),
            spec,
            spec,
            low,
            charge=0,
            multiplicity=2,
            root_id="doublet",
            correlation_space_id="full",
            provider=provider,
        )

    geometry = np.array([[0.15, -0.3, 0.6], [0.7, 0.25, -0.1]])
    result = run(geometry)
    assert [(role, cardinal) for role, cardinal, _ in calls] == [
        ("hf_cabs", 2),
        ("hf_cabs", 3),
        ("mp2_f12_correlation", 2),
        ("mp2_f12_correlation", 3),
        ("post_mp2", 2),
    ]
    assert len({surface for _, _, surface in calls}) == 1
    assert all(item.executed_backend == "cuda" for item in result.components)
    assert not np.shares_memory(result.components[0].term.forces, reused)
    for point in np.ndindex(geometry.shape):
        moved = np.zeros_like(geometry)
        moved[point] = 2e-4
        fd = -(run(geometry + moved).energy - run(geometry - moved).energy) / (4e-4)
        assert result.forces[point] == pytest.approx(fd, abs=2e-12, rel=0)
    assert calls[-1][2] != calls[0][2]


@pytest.mark.parametrize(
    "problem", ["surface", "basis", "backend", "force", "convergence"]
)
def test_focal_rejects_mismatched_result_before_next_request(problem: str) -> None:
    b = basis(2)
    calls = []

    def provider(request: FocalRequest) -> FocalEvaluation:
        calls.append(request.role)
        result = term(request, 0.0, np.zeros((2, 3)))
        if problem == "surface":
            result = replace(result, surface_id="another geometry")
        elif problem == "basis":
            result = replace(result, basis_id="different auxiliary role")
        elif problem == "force":
            result = replace(result, forces=np.zeros((1, 3)))
        return FocalEvaluation(
            result,
            "cpu_reference" if problem == "backend" else "cuda",
            converged=problem != "convergence",
        )

    with pytest.raises(
        RuntimeError if problem in ("backend", "convergence") else ValueError
    ):
        singlepoint(
            [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
            FocalSeries(b),
            FocalSeries(b),
            b,
            root_id="RHF",
            correlation_space_id="all",
            provider=provider,
        )
    assert calls == ["hf_cabs"]


def test_beta_change_alone_is_not_a_cardinal_pair() -> None:
    low, high = basis(2), basis(3)
    assert (
        low.basis_id != high.basis_id and low.orbital.identity == high.orbital.identity
    )

    def provider(request: FocalRequest) -> FocalEvaluation:
        pytest.fail(f"unexpected electronic call {request.role}")

    with pytest.raises(ValueError, match="distinct orbital bases"):
        singlepoint(
            [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
            FocalSeries(high, low, 3.0),
            FocalSeries(low),
            low,
            root_id="RHF",
            correlation_space_id="all",
            provider=provider,
        )


def test_basis_snapshot_and_ecp_mismatch_before_electronic_work(tmp_path: Path) -> None:
    b = basis(2)
    original = b.orbital.by_element[1]
    adjusted = replace(
        b.orbital,
        name="test-ECP",
        elements=(
            original,
            replace(
                original,
                atomic_number=3,
                nuclear_charge=3,
                ecp_core_electrons=2,
                ecp_data=json.dumps(
                    [
                        {
                            "ecp_type": "scalar_ecp",
                            "angular_momentum": [0],
                            "r_exponents": [2],
                            "gaussian_exponents": ["1.0"],
                            "coefficients": [["1.0"]],
                        },
                    ]
                ),
            ),
        ),
    )
    ecp = adjusted.by_element[3]
    assert ecp.ecp_data is not None
    changed_ecp = json.loads(ecp.ecp_data)
    changed_ecp[0]["coefficients"][0][0] = "2.0"
    changed = replace(
        adjusted,
        elements=(
            original,
            replace(ecp, ecp_data=json.dumps(changed_ecp)),
        ),
    )
    source = tmp_path / "orbital.json"
    adjusted.write(source)
    a = FocalBasis(
        2,
        "pp",
        source,
        jk_fit=adjusted,
        mp2_fit=adjusted,
        f12_fit=adjusted,
        cabs=adjusted,
        beta=1.0,
    )
    changed.write(source)
    c = FocalBasis(
        2,
        "pp",
        source,
        jk_fit=adjusted,
        mp2_fit=adjusted,
        f12_fit=adjusted,
        cabs=adjusted,
        beta=1.0,
    )
    assert a.orbital.identity == adjusted.identity
    assert c.orbital.identity == changed.identity and a.basis_id != c.basis_id

    def provider(request: FocalRequest) -> FocalEvaluation:
        pytest.fail(f"unexpected electronic call {request.role}")

    with pytest.raises(ValueError, match="same ECP Hamiltonian"):
        singlepoint(
            [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.0))],
            FocalSeries(a),
            FocalSeries(a),
            c,
            root_id="RHF",
            correlation_space_id="valence",
            provider=provider,
        )


@pytest.mark.parametrize(
    "backend",
    [
        "cpu_reference",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
                reason="requires an explicit CUDA allocation opt-in",
            ),
        ),
    ],
)
def test_entry_provider_reuses_native_same_reference_high_order(
    backend: Backend,
) -> None:
    b = basis(2)

    def provider(request: FocalRequest) -> FocalEvaluation:
        if request.role != "post_mp2":
            return FocalEvaluation(
                term(request, 0.0, np.zeros((2, 3))), backend, converged=True
            )
        return conventional_df_post_mp2(request)

    def run(radius: float) -> FocalRunResult:
        return singlepoint(
            [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, radius))],
            FocalSeries(b),
            FocalSeries(b),
            b,
            root_id="uhf",
            correlation_space_id="all-active",
            backend=backend,
            provider=provider,
        )

    result = run(1.4)
    assert result.energy != 0.0
    h = 2e-4
    fd = -(run(1.4 + h).energy - run(1.4 - h).energy) / (2 * h)
    assert result.forces[1, 2] == pytest.approx(fd, abs=3e-8, rel=0)


def test_native_grouped_increment_refuses_split_role() -> None:
    b = basis(2)

    def provider(request: FocalRequest) -> FocalEvaluation:
        if request.role in ("hf_cabs", "mp2_f12_correlation"):
            return FocalEvaluation(term(request, 0.0, np.zeros((2, 3))), "cuda", True)
        return conventional_df_post_mp2(request)

    with pytest.raises(ValueError, match="only returns the post_mp2 role"):
        singlepoint(
            [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
            FocalSeries(b),
            FocalSeries(b),
            b,
            triples=b,
            root_id="uhf",
            correlation_space_id="all-active",
            provider=provider,
        )
