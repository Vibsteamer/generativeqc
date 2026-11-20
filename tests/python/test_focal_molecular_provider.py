"""Public geometry provider with coincident and independent auxiliary bases."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalRequest,
    FocalSeries,
    MolecularFocalProvider,
    plan_singlepoint,
)
from generativeqc.profiles import canonical_hash


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
    x: np.ndarray, fits: tuple[int, int, int], root: str = "uhf"
) -> FocalRequest:
    obs = basis("obs", (1.5, 0.32))
    cabs = basis("cabs", (0.09,))
    a, b, c = [
        basis(f"aux-{i}", (5.0 + i, 1.1 + 0.1 * i, 0.28 + 0.02 * i, 0.07 + 0.01 * i))
        for i in fits
    ]
    card = FocalBasis(
        2, "model", obs, jk_fit=a, mp2_fit=b, f12_fit=c, cabs=cabs, beta=0.9
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
    with pytest.raises(ValueError, match="all-active UHF"):
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
