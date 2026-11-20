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


def test_uhf_fixed_cc_feed_fitting_roles() -> None:
    shells = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shells:
        pytest.skip("set GENERATIVEQC_F12_HOST_SINK for molecular host integrals")
    p = MolecularFocalProvider(shell_library=Path(shells))
    outputs = [
        p.ucc_fixed_sources(request(XYZ, fits))
        for fits in ((0, 1, 2), (0, 2, 2), (2, 2, 2))
    ]
    a, b, same_fit = outputs
    assert a.cabs_rank == b.cabs_rank == same_fit.cabs_rank == 3
    assert a.reference.occupied == (2, 1) and a.reference.forces is None
    assert a.reference.jk_fit_id != a.f12_fit_id == same_fit.reference.jk_fit_id
    spins = ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
    assert set(a.feeds) == {
        prefix + s for prefix in ("w_vo_", "w_vv_", "c_f12_") for s in spins
    } | {"fixed_geminal_energy"}
    for k in a.feeds:
        np.testing.assert_array_equal(a.feeds[k], b.feeds[k])
    assert not np.isclose(
        a.feeds["fixed_geminal_energy"],
        same_fit.feeds["fixed_geminal_energy"],
        rtol=0,
        atol=1e-7,
    )
    for suffix in ("aaaa", "bbbb"):
        for prefix in ("w_vo_", "w_vv_", "c_f12_"):
            v = a.feeds[prefix + suffix]
            np.testing.assert_allclose(v, -v.swapaxes(0, 1), rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        a.feeds["c_f12_abab"], -a.feeds["c_f12_baab"].swapaxes(0, 1), rtol=0, atol=1e-11
    )


@pytest.mark.parametrize("fits", [(0, 1, 2), (0, 0, 2), (0, 2, 2)])
def test_uhf_equation_sources_against_independent_df_chemist_blocks(
    fits: tuple[int, int, int],
) -> None:
    shell = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shell:
        pytest.skip("set GENERATIVEQC_F12_HOST_SINK for the molecular shell library")
    pyscf = pytest.importorskip("pyscf")
    from pyscf import df, gto

    req = request(XYZ, fits)
    provider = MolecularFocalProvider(shell_library=Path(shell))
    full = provider.ucc_equation_sources(req)
    fixed = provider.ucc_fixed_sources(req)
    assert full.fixed.f12_fit_id == fixed.f12_fit_id
    assert full.mp2_fit_id == req.basis.mp2_fit.identity
    assert len(full.ordinary) == 38 and len(fixed.feeds) == 19
    for k in fixed.feeds:
        np.testing.assert_allclose(
            full.fixed.feeds[k], fixed.feeds[k], rtol=0, atol=1e-12
        )

    def shells(exps: tuple[float, ...]) -> dict:
        return {"H": [[0, [e, 1.0]] for e in exps]}

    mol = gto.M(
        atom=list(zip(["H"] * 3, XYZ, strict=True)),
        unit="Bohr",
        basis=shells((1.5, 0.32)),
        spin=1,
        verbose=0,
    )

    def factors(index: int) -> np.ndarray:
        aux = df.addons.make_auxmol(
            mol,
            shells(
                (
                    5.0 + index,
                    1.1 + 0.1 * index,
                    0.28 + 0.02 * index,
                    0.07 + 0.01 * index,
                )
            ),
        )
        m = aux.intor("int2c2e")
        raw = df.incore.aux_e2(mol, aux, intor="int3c2e", aosym="s1").transpose(2, 0, 1)
        return np.linalg.solve(np.linalg.cholesky(m), raw.reshape(len(m), -1)).reshape(
            raw.shape
        )

    jk, cc = factors(fits[0]), factors(fits[1])
    c = dict(zip("ab", full.fixed.reference.coefficients, strict=True))
    density = {
        s: k[:, :o] @ k[:, :o].T for (s, k), o in zip(c.items(), (2, 1), strict=True)
    }
    h = pyscf.scf.hf.get_hcore(mol)
    j = np.einsum("Qpq,Qrs,rs->pq", jk, jk, density["a"] + density["b"], optimize=True)
    fmo = {
        s: k.T
        @ (h + j - np.einsum("Qpr,Qst,rt->ps", jk, jk, density[s], optimize=True))
        @ k
        for s, k in c.items()
    }
    mo = {s: k.T @ cc @ k for s, k in c.items()}
    for name, value in full.ordinary.items():
        domains, spin = name[1:].split("_")
        ix = tuple(
            slice(None, 2 if s == "a" else 1)
            if d == "o"
            else slice(2 if s == "a" else 1, None)
            for s, d in zip(spin, domains, strict=True)
        )
        if name.startswith("f"):
            gold = fmo[spin[0]][ix]
        else:
            i, j, k, l = spin
            eri = np.zeros((mol.nao,) * 4)
            if i == k and j == l:
                eri += np.einsum("Qik,Qjl->ijkl", mo[i], mo[j])
            if i == l and j == k:
                eri -= np.einsum("Qil,Qjk->ijkl", mo[i], mo[j])
            gold = eri[ix]
        np.testing.assert_allclose(value, gold, rtol=0, atol=3e-9, err_msg=name)


def test_public_uhf_stationary_lagrangian_on_same_molecular_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shell:
        pytest.skip("set GENERATIVEQC_F12_HOST_SINK for the molecular shell library")
    from generativeqc import _f12_focal_reference
    from generativeqc.experimental.f12_focal import FocalUCCSDResult, FocalUCCStationary
    from generativeqc.experimental.uccsd_f12 import SolverOptions
    from generativeqc_compiler.cc.uccsd_equations import build_f12_program
    from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
    from generativeqc_compiler.tensor import execute

    provider = MolecularFocalProvider(shell_library=Path(shell), budget_mib=256)
    q = request(XYZ, (0, 1, 2))
    solve_reference = _f12_focal_reference.df_reference
    reference_calls = []

    def counted_reference(
        request: FocalRequest, *, forces: bool, budget_mib: int
    ) -> _f12_focal_reference.FocalReference:
        reference_calls.append((request.surface_id, forces))
        return solve_reference(request, forces=forces, budget_mib=budget_mib)

    monkeypatch.setattr(_f12_focal_reference, "df_reference", counted_reference)
    result = provider.uccsd_f12b(
        q, options=SolverOptions(residual_tolerance=3e-12, energy_tolerance=5e-13)
    )
    assert reference_calls == [(q.surface_id, True)]
    assert isinstance(result, FocalUCCSDResult)
    solution = result.stationary
    assert solution.sources.fixed.reference.forces is not None
    assert result.energy == pytest.approx(-1.5204305761129642, rel=0, abs=3e-9)
    np.testing.assert_allclose(
        result.energy,
        result.hf_cabs_energy + solution.primal.correlation_energy,
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        result.forces,
        result.hf_cabs_forces + result.ordinary_cc_forces + result.fixed_cc_forces,
        rtol=0,
        atol=1e-13,
    )
    nuclear_direction = np.array([[0.1, 0.4, -0.2], [-0.2, 0.3, 0.5], [0.2, -0.1, 0.3]])
    nuclear_direction /= np.linalg.norm(nuclear_direction)
    # Fully reoptimized signed energies from PySCF DF-UHF, Sylvester CABS
    # singles and an independent dense SciPy root of the 59 CC equations.
    for signed in (-0.07236072075644806, -0.07236076799699287):
        assert np.vdot(result.forces, nuclear_direction) == pytest.approx(
            signed, rel=0, abs=3e-7
        )
    assert isinstance(solution, FocalUCCStationary)
    sources = solution.sources
    ref = sources.fixed.reference
    assert (ref.surface_id, ref.basis_id, ref.executed_backend) == (
        q.surface_id,
        q.basis.basis_id,
        "cpu_reference",
    )
    assert len({ref.jk_fit_id, sources.mp2_fit_id, sources.fixed.f12_fit_id}) == 3
    assert solution.primal.converged and solution.dual.converged
    # Independent dense SciPy root for the same 59 equations.
    assert solution.primal.correlation_energy == pytest.approx(
        -0.06269423582838254, rel=0, abs=2e-11
    )
    np.testing.assert_allclose(
        solution.primal.total_energy,
        ref.energy + solution.primal.correlation_energy,
        rtol=0,
        atol=1e-13,
    )
    layout = UCCSDLayout(2, 1, 4, 5)
    program = build_f12_program(layout)
    multiplier = layout.unpack_dual(solution.dual.packed_multipliers)
    inputs = {**sources.ordinary, **sources.fixed.feeds}

    def lagrangian(t: np.ndarray, feeds: dict[str, np.ndarray]) -> float:
        out = execute(program, {**feeds, **layout.unpack(t)}).outputs
        return float(out["correlation_energy"]) + sum(
            float(np.vdot(multiplier[b.name], out["r" + b.name[1:]]))
            for b in layout.blocks
        )

    t = solution.primal.packed_amplitudes
    direction = np.cos(np.arange(len(t), dtype=float))
    direction /= np.linalg.norm(direction)
    for h in (4e-5, 2e-5):
        derivative = (
            lagrangian(t + h * direction, inputs)
            - lagrangian(t - h * direction, inputs)
        ) / (2 * h)
        assert abs(derivative) < 2e-9

    weights = solution.source_cotangents
    assert weights is not None
    assert weights.ordinary_weights.keys() == sources.ordinary.keys()
    assert sources.fixed.feeds.keys() - weights.fixed_source_weights.keys() == {
        "w_vo_baab"
    }
    assert weights.correlation_energy == pytest.approx(
        solution.primal.correlation_energy, rel=0, abs=1e-13
    )
    # Perturb every active ordinary and geminal input in one reproducible
    # direction. Contract the packed dual in the independent NumPy evaluator.
    bars = {**weights.ordinary_weights, **weights.fixed_source_weights}
    rng = np.random.default_rng(23014)
    directions = {k: np.asarray(rng.standard_normal(v.shape)) for k, v in bars.items()}
    predicted = sum(float(np.vdot(bars[k], u)) for k, u in directions.items())
    for h in (4e-5, 2e-5):
        plus = {**inputs, **{k: inputs[k] + h * u for k, u in directions.items()}}
        minus = {**inputs, **{k: inputs[k] - h * u for k, u in directions.items()}}
        derivative = (lagrangian(t, plus) - lagrangian(t, minus)) / (2 * h)
        assert derivative == pytest.approx(predicted, rel=0, abs=5e-10)

    # Five-point derivatives of the phase-aligned, displaced native ordinary
    # inputs at fixed stationary source weights, for all nine Cartesian axes.
    ordinary_force = provider.ucc_ordinary_source_forces(q, solution)
    expected = np.array(
        [
            [-0.013754271792271061, -0.005831284569426017, 0.002889057415733758],
            [0.010868287982620494, 0.005496551045856025, -0.002190452201020557],
            [0.0028859840007737273, 0.0003347334201024103, -0.0006986052157193409],
        ]
    )
    np.testing.assert_allclose(ordinary_force, expected, rtol=0, atol=2e-8)
    np.testing.assert_allclose(ordinary_force.sum(axis=0), 0, rtol=0, atol=2e-10)
    with pytest.raises(ValueError, match="stationary root and force request"):
        provider.ucc_ordinary_source_forces(
            replace(q, surface_id="different"), solution
        )
    with pytest.raises(ValueError, match="source_cotangents=True"):
        provider.ucc_ordinary_source_forces(
            q, replace(solution, source_cotangents=None)
        )

    with pytest.raises(ValueError, match="MolecularFocalProvider.cuda"):
        provider.ucc_stationary(replace(q, backend="cuda"))


@pytest.mark.parametrize(
    ("root", "fits"),
    [("rhf", (0, 2, 2)), ("uhf", (0, 1, 2)), ("uhf", (0, 0, 0)), ("uhf", (0, 0, 2))],
)
def test_one_root_low_order_energies_and_forces(
    root: str, fits: tuple[int, int, int]
) -> None:
    from unittest.mock import patch

    from generativeqc import _f12_focal_molecular as owner
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
        patch.object(owner, "df_reference", wraps=owner.df_reference) as scf,
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


def test_rhf_focal_center_and_forward_displacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A two-electron RHF recipe has an empty triples space on each geometry."""
    from generativeqc.experimental.f12_focal import singlepoint, singlepoint_energy

    shell = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shell:
        pytest.skip("set GENERATIVEQC_F12_HOST_SINK for molecular host integrals")
    x = XYZ[:2].copy()
    q = request(x, (0, 1, 1), "rhf")
    p = MolecularFocalProvider(shell_library=Path(shell), budget_mib=800)
    ser = FocalSeries(q.basis)
    args = {
        "hf_cabs": ser,
        "correlation": ser,
        "f12_doubles": ser,
        "triples": ser,
        "post_mp2": None,
        "root_id": "rhf",
        "correlation_space_id": "all-active",
        "provider": p,
        "backend": "cpu_reference",
        "protocol_id": "h2",
    }
    center = singlepoint(list(zip(("H", "H"), x, strict=True)), **args)
    assert len(center.references) == 1
    triple = center.components[-1].term
    assert triple.role == "triples" and triple.energy == 0 and not np.any(triple.forces)

    def forbid_gradient(*args: object, **kwargs: object) -> None:
        pytest.fail("a forward-only geometry called the integral gradient engine")

    monkeypatch.setattr(MolecularFocalProvider, "_gradient", forbid_gradient)
    for step in (2.0e-4, 4.0e-4):
        energies = []
        for h in (-step, step):
            y = x.copy()
            y[0, 0] += h
            result = singlepoint_energy(
                list(zip(("H", "H"), y, strict=True)), center=center, **args
            )
            assert result.references[q.basis.basis_id].reference.forces is None
            assert result.cabs_ranks == center.cabs_ranks
            assert result.components[-1].energy == 0.0
            energies.append(result.energy)
        assert (
            abs((energies[0] - energies[1]) / (2 * step) - center.forces[0, 0]) < 3e-6
        )


def test_cc_only_basis_skips_hf_singles_and_reference_gradient() -> None:
    from unittest.mock import patch

    from generativeqc import _f12_cabs_fock as fock
    from generativeqc import _f12_focal_molecular as owner
    from generativeqc.experimental.f12_focal import singlepoint, singlepoint_energy

    shell = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shell:
        pytest.skip("requires the molecular host shell library")
    xyz = XYZ[:2]
    low = request(xyz, (0, 1, 1), "rhf").basis
    high = request(xyz, (0, 2, 2), "rhf").basis
    provider = MolecularFocalProvider(shell_library=Path(shell), budget_mib=800)
    args = {
        "hf_cabs": FocalSeries(low),
        "correlation": FocalSeries(low),
        "f12_doubles": FocalSeries(high),
        "triples": FocalSeries(high),
        "post_mp2": None,
        "provider": provider,
        "root_id": "rhf",
        "correlation_space_id": "all-active",
        "backend": "cpu_reference",
        "protocol_id": "h2-mixed-fits",
    }
    atoms = list(zip(("H", "H"), xyz, strict=True))
    moved = xyz.copy()
    moved[0, 0] += 2e-4
    displaced = list(zip(("H", "H"), moved, strict=True))
    center_method = MolecularFocalProvider.low_order_energy_and_forces
    energy_method = MolecularFocalProvider.low_order_energy
    with (
        patch.object(
            MolecularFocalProvider,
            "low_order_energy_and_forces",
            lambda self, q, **_: center_method(self, q),
        ),
        patch.object(
            MolecularFocalProvider,
            "low_order_energy",
            lambda self, q, *, expected_cabs_rank, **_: energy_method(
                self, q, expected_cabs_rank=expected_cabs_rank
            ),
        ),
    ):
        full = singlepoint(atoms, **args)
        full_signed = singlepoint_energy(displaced, center=full, **args)
    with (
        patch.object(owner, "df_reference", wraps=owner.df_reference) as scf,
        patch.object(
            MolecularFocalProvider,
            "_singles_components",
            autospec=True,
            side_effect=MolecularFocalProvider._singles_components,
        ) as singles,
    ):
        center = singlepoint(atoms, **args)
    assert [
        (c.args[0].basis.basis_id, c.kwargs["forces"]) for c in scf.call_args_list
    ] == [(low.basis_id, True), (high.basis_id, False)]
    assert (
        singles.call_count == 1
        and singles.call_args.args[1].basis.basis_id == low.basis_id
    )
    assert center.energy == pytest.approx(full.energy, abs=1e-12, rel=0)
    np.testing.assert_allclose(center.forces, full.forces, atol=1e-11, rtol=0)
    for actual, original in zip(center.components, full.components, strict=True):
        assert actual.term.energy == pytest.approx(
            original.term.energy, abs=1e-12, rel=0
        )
        np.testing.assert_allclose(
            actual.term.forces, original.term.forces, atol=1e-11, rtol=0
        )
    assert center.cabs_ranks == full.cabs_ranks
    for key, seed in center.references.items():
        ref = seed.reference
        previous = full.references[key].reference
        occupied_overlap = (
            previous.coefficients[0, :, 0] @ ref.overlap @ ref.coefficients[0, :, 0]
        )
        assert abs(occupied_overlap) == pytest.approx(1.0, abs=1e-10, rel=0)
    with (
        patch.object(owner, "df_reference", wraps=owner.df_reference) as scf,
        patch.object(fock, "physical_fock", wraps=fock.physical_fock) as singles_fock,
    ):
        signed = singlepoint_energy(displaced, center=center, **args)
    assert scf.call_count == 2 and all(
        not c.kwargs["forces"] for c in scf.call_args_list
    )
    assert singles_fock.call_count == 1
    assert signed.cabs_ranks == center.cabs_ranks
    assert all(
        min(s.reference.occupied_overlaps) > 0.999 for s in signed.references.values()
    )
    assert [p.energy for p in signed.components] == pytest.approx(
        [p.energy for p in full_signed.components], abs=1e-12, rel=0
    )
