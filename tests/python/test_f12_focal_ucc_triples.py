"""Conventional (T) has a separately relaxed force on an open-shell surface."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from pathlib import Path

    from generativeqc.experimental.f12_focal import (
        FocalLowOrderEnergy,
        FocalLowOrderForces,
        FocalReferenceSeed,
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


@pytest.mark.parametrize("separate", [False, True])
def test_native_triples_energy_and_signed_force(separate: bool) -> None:
    req = triples_request()
    if separate:
        basis = FocalBasis(
            2, "independent", "sto-3g", jk_fit="def2-svp", mp2_fit="sto-3g"
        )
        req = replace(req, basis=basis)
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
    reference_energy = None
    if separate:
        reference = Calculator(
            method="uhf",
            basis=req.basis.orbital,
            auxiliary_basis=req.basis.jk_fit,
            density_fitting="cpu",
            device="cpu",
            initial_guess=None,
        )
        try:
            reference_energy = reference.singlepoint(
                req.atoms, multiplicity=2, properties=("energy",)
            ).energy
        finally:
            reference.clear_cache()
    try:

        def energy(dz: float) -> float:
            atoms = list(req.atoms)
            x, y, z = atoms[1].position
            atoms[1] = replace(atoms[1], position=(x, y, z + dz))
            state = q.singlepoint(
                atoms,
                multiplicity=2,
                properties=("energy",),
                reference_auxiliary_basis=req.basis.jk_fit if separate else None,
            )
            if separate:
                assert (
                    state.basis_metadata["reference_auxiliary"]["basis_identity"]
                    == req.basis.jk_fit.identity
                )
                assert (
                    state.basis_metadata["auxiliary"]["basis_identity"]
                    == req.basis.mp2_fit.identity
                )
            assert state.correlation is not None
            if dz == 0 and reference_energy is not None:
                assert state.correlation.reference_energy == pytest.approx(
                    reference_energy, abs=1e-9, rel=0
                )
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
    missing = FocalBasis(2, "missing", "sto-3g", jk_fit="def2-svp")
    with pytest.raises(ValueError, match="requires a correlation fitting basis"):
        conventional_df_ucc_triples(replace(req, basis=missing))


def test_triples_from_retained_uhf_and_rhf_references() -> None:
    from generativeqc._f12_focal_ucc_triples import molecular_triples_on_reference
    from generativeqc.experimental.f12_focal import df_reference

    req = triples_request()
    ref = df_reference(req, forces=False)
    analytic = molecular_triples_on_reference(req, ref, budget_mib=1024, forces=True)
    energy, forces = molecular_triples_on_reference(
        req, ref, budget_mib=1024, forces=False
    )
    assert forces is None and energy == pytest.approx(analytic[0], abs=2e-13)
    independent = conventional_df_ucc_triples(req).term
    assert energy == pytest.approx(independent.energy, abs=3e-11)
    np.testing.assert_allclose(analytic[1], independent.forces, rtol=0, atol=2e-9)
    with pytest.raises(ValueError, match="matching molecular reference"):
        molecular_triples_on_reference(
            replace(req, surface_id="other"), ref, budget_mib=1024, forces=False
        )

    # Independent PySCF DF-RCCSD(T) on LiH/STO-3G with def2-SVP fitting.
    # The force uses the +/- 0.0004-bohr signed energies on the hydrogen z axis.
    b = req.basis
    low = FocalBasis(
        2,
        "calibration",
        b.orbital,
        jk_fit=b.jk_fit,
        mp2_fit=b.mp2_fit,
        f12_fit=b.mp2_fit,
        cabs=b.jk_fit,
        beta=1.0,
    )
    atoms = [("Li", (0.0, 0.0, 0.0)), ("H", (0.1, 0.2, 3.0))]
    rhf = next(
        q
        for q, _ in plan_singlepoint(
            atoms,
            FocalSeries(low),
            FocalSeries(low),
            None,
            f12_doubles=low,
            triples=b,
            root_id="rhf",
            correlation_space_id="all-active",
            backend="cpu_reference",
        ).steps
        if q.role == "triples"
    )
    r = df_reference(rhf, forces=False)
    assert len(r.occupied) == 1
    e, f = molecular_triples_on_reference(rhf, r, budget_mib=1024, forces=True)
    assert e == pytest.approx(-7.4691686931e-6, abs=8e-12)
    assert f[1, 2] == pytest.approx(5.63559527e-6, abs=1e-11)


def test_public_split_fit_triples_share_the_reference_at_center_and_displacement(
    tmp_path: Path,
) -> None:
    from generativeqc.experimental.f12_focal import (
        FocalLowOrderEnergy,
        FocalLowOrderForces,
        MolecularFocalProvider,
        df_reference,
        singlepoint,
        singlepoint_energy,
    )

    def low_order(
        request: FocalRequest,
        seed_options: dict[str, FocalReferenceSeed],
        expected_rank: int | None = None,
    ) -> FocalLowOrderEnergy:
        ref = df_reference(request, forces=False, **seed_options)
        assert expected_rank in (None, 1)
        return FocalLowOrderEnergy(ref, 1, (0.0, 0.0), (0.0, 0.0, 0.0), 0.0)

    class RetainedTriplesProvider(MolecularFocalProvider):
        # Exercise the native SCF and (T) paths while isolating the unrelated F12 graph.
        def low_order_energy_and_forces(
            self, request: FocalRequest, *, include_hf_cabs: bool = True
        ) -> FocalLowOrderForces:
            low = low_order(request, self._seed_options(request))
            zeros = np.zeros((len(request.atoms), 3))
            return FocalLowOrderForces(low, zeros, zeros)

        def low_order_energy(
            self,
            request: FocalRequest,
            *,
            expected_cabs_rank: int | None = None,
            include_hf_cabs: bool = True,
        ) -> FocalLowOrderEnergy:
            return low_order(request, self._seed_options(request), expected_cabs_rank)

        def f12_doubles_energy_and_forces(
            self, request: FocalRequest, *, low_order: FocalLowOrderForces
        ) -> tuple[float, np.ndarray]:
            return 0.0, np.zeros_like(low_order.mp2_f12_correlation_forces)

        def f12_doubles_energy(
            self,
            request: FocalRequest,
            *,
            low_order: FocalLowOrderEnergy,
            expected_cabs_rank: int | None = None,
        ) -> float:
            return 0.0

    orb, jk = "sto-3g", "def2-svp"
    low = FocalBasis(
        2, "f12", orb, jk_fit=jk, mp2_fit=orb, f12_fit=jk, cabs=jk, beta=1.0
    )
    high = FocalBasis(2, "conventional", orb, jk_fit=jk, mp2_fit=jk)
    assert low.basis_id != high.basis_id
    provider = RetainedTriplesProvider(shell_library=tmp_path / "unused.so")
    series = FocalSeries(low)
    options = dict(
        provider=provider,
        root_id="rhf",
        correlation_space_id="all-active",
        backend="cpu_reference",
        f12_doubles=low,
        triples=high,
    )

    def atoms(z: float) -> list[tuple[str, tuple[float, float, float]]]:
        return [("Li", (0.0, 0.0, 0.0)), ("H", (0.1, 0.2, z))]

    center = singlepoint(atoms(3.0), series, series, None, **options)
    assert center.cabs_ranks == {low.basis_id: 1}
    assert center.references[high.basis_id].reference.basis_id == high.basis_id
    assert (
        center.references[high.basis_id].reference.coefficients
        is center.references[low.basis_id].reference.coefficients
    )
    triple = next(p.term for p in center.components if p.term.role == "triples")
    assert triple.energy == pytest.approx(-7.4691686931e-6, abs=8e-12)

    def forward(z: float) -> float:
        result = singlepoint_energy(atoms(z), series, series, center=center, **options)
        assert (
            result.references[high.basis_id].reference.coefficients
            is result.references[low.basis_id].reference.coefficients
        )
        return next(p.energy for p in result.components if p.request.role == "triples")

    for step in (4e-4, 2e-4):
        force = (forward(3.0 - step) - forward(3.0 + step)) / (2 * step)
        assert force == pytest.approx(triple.forces[1, 2], abs=2e-9, rel=0)

    stricter_center = replace(
        center,
        references={
            **center.references,
            high.basis_id: replace(center.references[high.basis_id], minimum_overlap=1.0),
        },
    )
    with pytest.raises(ValueError, match="Target SCF changed occupied subspace"):
        singlepoint_energy(
            atoms(3.0004), series, series, center=stricter_center, **options
        )
