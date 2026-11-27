"""Public geometry provider with coincident and independent auxiliary bases."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    DEFAULT_SLATER_FIT_NORMALIZATION,
    FocalBasis,
    FocalRequest,
    FocalSeries,
    MolecularFocalProvider,
    plan_singlepoint,
)
from generativeqc.profiles import canonical_hash

if TYPE_CHECKING:
    from collections.abc import Mapping

    from generativeqc_compiler.tensor import Execution, Program


def basis(name: str, exponents: tuple[float, ...]) -> BasisSet:
    shells = tuple(BasisShell(0, (e,), ((1.0,),)) for e in exponents)
    return BasisSet(
        name,
        (ElementBasis(1, shells),),
        BasisProvenance(
            "test", name, "uncontracted hydrogen shells", canonical_hash(exponents)
        ),
        representation="spherical",
    )


def request(
    x: np.ndarray,
    fits: tuple[int, int, int],
    root: str = "uhf",
    *,
    jk_fit: BasisSet | None = None,
    slater_normalization: str = DEFAULT_SLATER_FIT_NORMALIZATION,
) -> FocalRequest:
    obs = basis("obs", (1.5, 0.32))
    cabs = basis("cabs", (0.09,))
    a, b, c = [
        basis(f"aux-{i}", (5.0 + i, 1.1 + 0.1 * i, 0.28 + 0.02 * i, 0.07 + 0.01 * i))
        for i in fits
    ]
    card = FocalBasis(
        2, "model", obs, jk_fit=jk_fit if jk_fit is not None else a,
        mp2_fit=b, f12_fit=c, cabs=cabs, beta=0.9,
    )
    geom = list(zip(["H"] * len(x), x, strict=True))
    steps = plan_singlepoint(
        geom,
        FocalSeries(card),
        FocalSeries(card),
        card,
        root_id=root,
        correlation_space_id="all-active",
        multiplicity=1 if root == "rhf" else 2,
        backend="cpu_reference",
        slater_normalization=slater_normalization,
    ).steps
    return next(r for r, _ in steps if r.role == "mp2_f12_correlation")


XYZ = np.array([[0.1, -0.2, 0.15], [1.6, 0.1, -0.2], [0.47, 1.57, 0.24]])


def test_molecular_provider_requires_declared_backend_and_high_order(
    tmp_path: Path,
) -> None:
    p = MolecularFocalProvider(shell_library=tmp_path / "unused.so")
    q = request(XYZ, (0, 1, 2))
    with pytest.raises(ValueError, match="high-order evaluator"):
        p(replace(q, role="post_mp2"))
    with pytest.raises(ValueError, match="CUDA"):
        p(replace(q, backend="cuda"))
    with pytest.raises(ValueError, match="singlet RHF"):
        p(replace(q, root_id="rhf"))


@pytest.mark.parametrize(
    ("root", "fits"),
    [
        ("uhf", (0, 0, 0)),
        ("uhf", (0, 0, 1)),
        ("uhf", (0, 1, 0)),
        ("uhf", (0, 1, 1)),
        ("uhf", (0, 1, 2)),
        ("rhf", (0, 1, 2)),
    ],
)
def test_molecular_pair_all_fit_assignments(
    root: str, fits: tuple[int, int, int]
) -> None:
    shell = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shell:
        pytest.skip("requires the built native F12 host shell library")
    from benchmarks.f12_focal_increment import molecular_f12_increment
    from benchmarks.f12_focal_ump2 import molecular_df_ump2

    lib = Path(shell)
    p = MolecularFocalProvider(shell_library=lib)
    xyz = XYZ[:2] if root == "rhf" else XYZ
    q = request(xyz, fits, root)
    val = p(q)
    ordinary = molecular_df_ump2(q, shell_library=lib, one_electron_library=lib)
    explicit = molecular_f12_increment(q, shell_library=lib, one_electron_library=lib)
    assert (val.term.surface_id, val.term.basis_id, val.term.role) == (
        q.surface_id,
        q.basis.basis_id,
        q.role,
    )
    assert val.term.energy == pytest.approx(
        sum(ordinary.spin_energies) + explicit.energy, abs=2e-10, rel=0
    )
    np.testing.assert_allclose(
        val.term.forces,
        ordinary.correlation_forces + explicit.forces,
        atol=3e-9,
        rtol=0,
    )
    direction = np.array([[0.1, 0.4, -0.2], [-0.2, 0.3, 0.5], [0.2, -0.1, 0.3]])
    direction = direction[: len(xyz)]
    direction /= np.linalg.norm(direction)
    for h in (4e-4, 2e-4):
        plus, minus = [
            p(request(xyz + sign * h * direction, fits, root)).term.energy
            for sign in (1, -1)
        ]
        assert np.vdot(val.term.forces, direction) == pytest.approx(
            (minus - plus) / (2 * h), abs=2e-7, rel=0
        )


@pytest.mark.parametrize(
    ("root", "fits"),
    [("rhf", (0, 2, 2)), ("uhf", (0, 1, 2)), ("uhf", (0, 0, 0)), ("uhf", (0, 0, 2))],
)
def test_one_root_low_order_energies_and_forces(
    root: str, fits: tuple[int, int, int]
) -> None:
    from unittest.mock import patch

    from generativeqc import _f12_focal_molecular as owner
    from generativeqc import _f12_focal_reference as reference_owner
    from generativeqc_compiler.tensor import execute

    shell = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shell:
        pytest.skip("requires the molecular host shell library")
    q = request(XYZ[:2] if root == "rhf" else XYZ, fits, root)
    provider = MolecularFocalProvider(shell_library=Path(shell))
    independent_roles = [
        provider(replace(q, role=role)).term for role in ("hf_cabs", q.role)
    ]
    with (
        patch.object(reference_owner, "df_reference", wraps=reference_owner.df_reference) as scf,
        patch.object(
            owner, "one_electron_sources", wraps=owner.one_electron_sources
        ) as one,
        patch.object(
            MolecularFocalProvider,
            "_panel",
            autospec=True,
            side_effect=MolecularFocalProvider._panel,
        ) as panel,
    ):
        center = provider.low_order_energy_and_forces(q)
    assert scf.call_count == one.call_count == 1
    assert scf.call_args.kwargs["forces"] and one.call_args.kwargs["derivatives"]
    assert panel.call_count == len(set(fits))
    for actual, term in zip(
        (center.hf_cabs_forces, center.mp2_f12_correlation_forces),
        independent_roles,
        strict=True,
    ):
        np.testing.assert_allclose(actual, term.forces, atol=2e-9, rtol=0)

    def forward_only(graph: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        assert not any(k.startswith(("bar_", "dB_", "dFo_", "dFv_")) for k in inputs)
        assert not any(
            "force" in k or "gradient" in k or k.startswith("bar_")
            for k in graph.outputs
        )
        return execute(graph, inputs)

    with (
        patch.object(owner, "df_reference", wraps=owner.df_reference) as scf,
        patch.object(
            owner, "one_electron_sources", wraps=owner.one_electron_sources
        ) as one,
        patch.object(
            MolecularFocalProvider,
            "_gradient",
            side_effect=AssertionError("unexpected derivative"),
        ),
    ):
        energy = replace(provider, run=forward_only).low_order_energy(
            q, expected_cabs_rank=center.energies.cabs_rank
        )
    assert energy.reference.forces is None
    assert scf.call_count == one.call_count == 1
    assert (
        not scf.call_args.kwargs["forces"] and not one.call_args.kwargs["derivatives"]
    )
    np.testing.assert_allclose(
        energy.cabs_spin_energies,
        center.energies.cabs_spin_energies,
        atol=2e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        energy.ordinary_mp2_spin_energies,
        center.energies.ordinary_mp2_spin_energies,
        atol=2e-10,
        rtol=0,
    )
    assert energy.f12_increment == pytest.approx(
        center.energies.f12_increment, abs=2e-10, rel=0
    )
    for x, term in zip(
        (energy.hf_cabs_energy, energy.mp2_f12_correlation_energy),
        independent_roles,
        strict=True,
    ):
        assert x == pytest.approx(term.energy, abs=2e-10, rel=0)


@pytest.mark.parametrize("separation", [0.00081, 0.001])
def test_ordinary_energy_uses_one_canonical_spectrum_with_or_without_partitions(
    separation: float,
) -> None:
    from conftest import f12_host_libraries

    shell, _ = f12_host_libraries()
    provider = MolecularFocalProvider(shell_library=shell)
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    # Both JK primitives survive the fit cutoff. Their near dependence makes the
    # independently constructed extended Fock distinguishable from the SCF spectrum.
    jk = basis("close-jk", (5.0, 1.1, 0.28, 0.28 * (1 + separation), 0.07))
    q = request(xyz, (0, 1, 1), "rhf", jk_fit=jk)
    center = provider.low_order_energy_and_forces(q)
    ref = center.energies.reference
    for partitioned in (False, True):
        result = provider.low_order_energy(q, reference=ref, include_occupied_partitions=partitioned)
        np.testing.assert_allclose(
            result.ordinary_mp2_spin_energies, center.energies.ordinary_mp2_spin_energies,
            atol=5e-13, rtol=0,
        )
        assert result.f12_increment == pytest.approx(center.energies.f12_increment, abs=5e-13, rel=0)
        if partitioned:
            assert result.occupied_mp2_f12 is not None
            assert sum(map(np.sum, result.occupied_mp2_f12.mp2)) == pytest.approx(
                sum(result.ordinary_mp2_spin_energies), abs=5e-13, rel=0,
            )
