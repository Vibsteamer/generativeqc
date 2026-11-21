"""Focal alternatives share an input frame while each total uses one triples role."""

from pathlib import Path

import numpy as np
import pytest
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalLowOrderEnergy,
    FocalLowOrderForces,
    FocalReference,
    FocalRequest,
    FocalSeries,
    FocalTriplesAlternatives,
    FocalTriplesEvaluation,
    MolecularFocalProvider,
    plan_singlepoint,
    singlepoint,
    singlepoint_energy,
)


def test_molecular_triples_alternatives_and_frozen_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = FocalBasis(
        2,
        "ha-DZ-F12",
        "sto-3g",
        jk_fit="sto-3g",
        mp2_fit="sto-3g",
        f12_fit="sto-3g",
        cabs="sto-3g",
        beta=0.9,
    )
    series = FocalSeries(card)
    water = [("O", (0, 0, 0)), ("H", (0, 0, 1.8)), ("H", (1.75, 0, -0.4))]
    moved = [water[0], ("H", (0, 0, 1.8004)), water[2]]
    calls = []

    def low(
        provider: MolecularFocalProvider,
        request: FocalRequest,
        *,
        expected_cabs_rank: int | None = None,
        include_hf_cabs: bool = True,
        include_occupied_partitions: bool = False,
    ) -> FocalLowOrderEnergy:
        calls.append(("reference", expected_cabs_rank, include_occupied_partitions))
        if expected_cabs_rank is not None:
            assert provider._seed_options(request)["seed"].reference.occupied == (5,)
        eye = np.eye(7)[None, ...]
        reference = FocalReference(
            -75.0,
            None,
            eye[0],
            eye[0],
            eye,
            np.arange(7.0)[None, :],
            eye,
            eye,
            (5,),
            3,
            request.backend,
            request.surface_id,
            request.basis.basis_id,
            request.basis.orbital.identity,
            request.basis.jk_fit.identity,
        )
        return FocalLowOrderEnergy(reference, 1, (0.0, 0.0), (0.0, -0.15, 0.0), -0.05)

    def low_forces(
        provider: MolecularFocalProvider,
        request: FocalRequest,
        *,
        include_hf_cabs: bool,
    ) -> FocalLowOrderForces:
        return FocalLowOrderForces(
            low(provider, request, include_hf_cabs=include_hf_cabs),
            np.zeros((3, 3)),
            np.full((3, 3), 0.1),
        )

    def models(
        requests: tuple[FocalRequest, ...],
        *,
        reference: FocalReference,
        low_order: FocalLowOrderEnergy | FocalLowOrderForces,
        forces: bool,
        expected_cabs_rank: int | None,
    ) -> dict[str, FocalTriplesEvaluation]:
        calls.append(("models", tuple(request.role for request in requests), forces))
        assert reference is (
            low_order.energies.reference if forces else low_order.reference
        )
        assert expected_cabs_rank == (None if forces else 1)
        corrections = {"triples_plus": -0.04, "triples_f12": -0.07}
        return {
            q.role: FocalTriplesEvaluation(
                corrections[q.role],
                np.full((3, 3), corrections[q.role]) if forces else None,
                1,
            )
            for q in requests
        }

    monkeypatch.setattr(MolecularFocalProvider, "low_order_energy", low)
    monkeypatch.setattr(
        MolecularFocalProvider, "low_order_energy_and_forces", low_forces
    )
    monkeypatch.setattr(
        MolecularFocalProvider, "f12_doubles_energy", lambda *a, **kw: -0.02
    )
    monkeypatch.setattr(
        MolecularFocalProvider,
        "f12_doubles_energy_and_forces",
        lambda *a, **kw: (-0.02, np.full((3, 3), 0.02)),
    )
    monkeypatch.setattr(
        MolecularFocalProvider,
        "triples_energy_and_forces",
        lambda *a, **kw: (-0.03, np.full((3, 3), 0.03)),
    )
    provider = MolecularFocalProvider(
        shell_library=Path("unused"),
        triples_models=models,
        qualified_tplus=lambda q: q.root_id == "rhf",
    )
    common = {
        "provider": provider,
        "root_id": "rhf",
        "correlation_space_id": "all-active",
        "f12_doubles": card,
        "backend": "cpu_reference",
    }
    methods = {"tplus": card, "f12": card}
    result = singlepoint(water, series, series, None, triples=methods, **common)
    assert isinstance(result, FocalTriplesAlternatives)
    plus, f12 = result.variants.values()
    assert plus.energy == pytest.approx(-75.26)
    assert f12.energy == pytest.approx(-75.29)
    np.testing.assert_allclose(plus.forces, 0.08)
    np.testing.assert_allclose(f12.forces, 0.05)
    assert len(plus.components) == len(f12.components) == 4
    assert plus.components[-1].term.role == "triples_plus"
    assert f12.components[-1].term.role == "triples_f12"
    assert plus.references[card.basis_id] is f12.references[card.basis_id]
    assert calls == [
        ("reference", None, False),
        ("models", ("triples_plus", "triples_f12"), True),
    ]

    calls.clear()
    displaced = singlepoint_energy(
        moved, series, series, triples=methods, center=result, **common
    )
    assert isinstance(displaced, FocalTriplesAlternatives)
    assert [v.energy for v in displaced.variants.values()] == pytest.approx(
        [-75.26, -75.29]
    )
    assert calls == [
        ("reference", 1, True),
        ("models", ("triples_plus", "triples_f12"), False),
    ]
    with pytest.raises(ValueError, match="same triples alternatives"):
        singlepoint_energy(
            moved, series, series, triples={"tplus": card}, center=result, **common
        )

    assert singlepoint(
        water, series, series, None, triples=card, **common
    ).energy == pytest.approx(-75.26)
    frozen = singlepoint(
        water, series, series, None, triples=card, protocol_id="archived", **common
    )
    assert frozen.energy == pytest.approx(-75.25)
    assert frozen.components[-1].term.role == "triples"
    unqualified = {
        **common,
        "provider": MolecularFocalProvider(shell_library=Path("unused")),
    }
    with pytest.raises(NotImplementedError, match="no native tplus or f12"):
        singlepoint(water, series, series, None, triples=methods, **unqualified)
    with pytest.raises(ValueError, match="unknown focal triples method"):
        plan_singlepoint(
            water,
            series,
            series,
            None,
            root_id="rhf",
            correlation_space_id="all-active",
            f12_doubles=card,
            triples={"tstar": card},
        )
