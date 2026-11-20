"""Conventional (T) has a separately relaxed force on an open-shell surface."""

from dataclasses import replace

import numpy as np
import pytest
from generativeqc import Calculator
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalRequest,
    FocalSeries,
    conventional_df_ucc_triples,
    plan_singlepoint,
)


def triples_request() -> FocalRequest:
    atoms = [
        ("N", (0.0, 0.0, 0.0)),
        ("H", (1.51, 0.0, 1.08)),
        ("H", (-1.58, 0.0, 1.24)),
    ]
    low = FocalBasis(
        2,
        "parent",
        "sto-3g",
        jk_fit="def2-svp",
        mp2_fit="def2-svp",
        f12_fit="def2-svp",
        cabs="def2-svp",
        beta=0.9,
    )
    high = FocalBasis(
        2, "conventional", "sto-3g", jk_fit="def2-svp", mp2_fit="def2-svp"
    )
    plan = plan_singlepoint(
        atoms,
        FocalSeries(low),
        FocalSeries(low),
        high,
        triples=high,
        root_id="uhf",
        correlation_space_id="all-active",
        multiplicity=2,
        backend="cpu_reference",
    )
    return next(req for req, _ in plan.steps if req.role == "triples")


def test_native_triples_energy_and_signed_force() -> None:
    req = triples_request()
    out = conventional_df_ucc_triples(req)
    assert out.converged and out.executed_backend == "cpu_reference"
    assert out.term.role == "triples" and out.term.basis_id == req.basis.basis_id
    assert out.term.surface_id == req.surface_id and abs(out.term.energy) > 1e-6
    np.testing.assert_allclose(out.term.forces.sum(axis=0), 0.0, atol=2e-9)

    q = Calculator(
        method="uccsd(t)",
        basis=req.basis.orbital,
        auxiliary_basis=req.basis.mp2_fit,
        density_fitting="full_cuda",
        device="cpu",
        initial_guess=None,
    )
    try:

        def energy(dz: float) -> float:
            atoms = list(req.atoms)
            x, y, z = atoms[1].position
            atoms[1] = replace(atoms[1], position=(x, y, z + dz))
            state = q.singlepoint(atoms, multiplicity=2, properties=("energy",))
            assert state.correlation is not None
            return state.correlation.ccsd_t_triples_energy

        assert energy(0) == pytest.approx(out.term.energy, abs=2e-12, rel=0)
        for h in (2e-4, 1e-4):
            assert (energy(-h) - energy(h)) / (2 * h) == pytest.approx(
                out.term.forces[1, 2],
                abs=2e-9,
                rel=0,
            )
    finally:
        q.clear_cache()


def test_native_triples_preserves_declared_root_and_fits() -> None:
    req = triples_request()
    with pytest.raises(ValueError, match="triples role and UHF root"):
        conventional_df_ucc_triples(replace(req, root_id="rhf"))
    with pytest.raises(ValueError, match="all-active"):
        conventional_df_ucc_triples(replace(req, correlation_space_id="frozen-core"))
    separate = FocalBasis(2, "different", "sto-3g", jk_fit="def2-svp", mp2_fit="sto-3g")
    with pytest.raises(ValueError, match="same JKFit and MP2Fit"):
        conventional_df_ucc_triples(replace(req, basis=separate))
