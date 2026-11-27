"""Gate 1 local CPU qualification for RHF DF-RCCSD(T)/(T+) triples_fp32_w.

Verifies:
1. All-electron H2O, CH4, and HBr with split JK/MP2 fitting bases for both
   (T) and weighted (T+) energy and analytic forces:
   - repeating the FP64 calculation gives bit-identical results;
   - the single-pass schedule emulates FP32 forward W on the host. The split3
     schedule executes FP32 products in both directions with FP64 recombination;
   - the accuracy limits are 1e-9 Eh for energies and 1e-8 Eh/bohr for forces;
   - occupied partition scalar closure holds to <= 1e-11 Eh and translational/rotational
     force invariants hold cleanly.
2. Central finite differences of same-mode and FP64 energies at 2e-4 and 1e-4
   bohr independently check each force on H2O and HBr for (T) and fixed-weight (T+).
3. Direct closed-shell spatial DF-(T)/(T+) mixed-precision equation verification
   (FP32 panels + FP32 forward-W with FP64 assembly + deferred FP64 pullback_panel +
   chunked-k FP32->FP64 pullback_w + exact canonical triangular (T+) fock_vv) against
   pure FP64 and independent PySCF DF-RCCSD(T) on H2O.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import f12_host_libraries

from generativeqc._f12_focal_molecular import MolecularFocalProvider
from generativeqc._f12_focal_reference import df_reference
from generativeqc._f12_focal_ucc_triples import spatial_rhf_triples_v2
from generativeqc._f12_frozen_core import resolve_frozen_occupied
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalSeries,
    plan_singlepoint,
)


def _make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
    prov = BasisProvenance(source=name, version="1", license="MIT", checksum="0" * 64)
    return BasisSet(name, elems, prov, representation="spherical")


def _sh(l: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
    return BasisShell(l, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),))


def _compact_split_basis(elements: tuple[int, ...]) -> FocalBasis:
    """Compact real-spherical split-fit basis for H, C, O, and all-electron Br (3-21G+d)."""
    orb_map = {
        1: ElementBasis(
            1,
            (
                _sh(0, (3.42525091, 0.62391373), (0.15432897, 0.53532814)),
                _sh(0, (0.16885540,), (1.0,)),
            ),
        ),
        6: ElementBasis(
            6,
            (
                _sh(
                    0,
                    (71.616837, 13.045096, 3.5305122),
                    (0.15432897, 0.53532814, 0.44463454),
                ),
                _sh(0, (2.9412494, 0.6834831), (-0.09996723, 0.39951283)),
                _sh(0, (0.2222899,), (1.0,)),
                _sh(1, (2.9412494, 0.6834831), (0.15591627, 0.60768372)),
                _sh(1, (0.2222899,), (1.0,)),
            ),
        ),
        8: ElementBasis(
            8,
            (
                _sh(
                    0,
                    (130.70932, 23.808861, 6.4436083),
                    (0.15432897, 0.53532814, 0.44463454),
                ),
                _sh(0, (5.0331513, 1.1695961), (-0.09996723, 0.39951283)),
                _sh(0, (0.3803890,), (1.0,)),
                _sh(1, (5.0331513, 1.1695961), (0.15591627, 0.60768372)),
                _sh(1, (0.3803890,), (1.0,)),
            ),
        ),
        35: ElementBasis(
            35,
            (
                _sh(0, (6103.2899, 923.69743, 202.52031), (0.062418, 0.372041, 0.685873)),
                _sh(0, (270.6015, 58.25357, 18.46933), (-0.112149, 0.093145, 0.961679)),
                _sh(0, (19.76142, 6.821752, 2.291629), (-0.29387, 0.280266, 0.902036)),
                _sh(0, (2.131206, 0.499354), (-0.651803, 1.336012)),
                _sh(0, (0.164764,), (1.0,)),
                _sh(1, (270.6015, 58.25357, 18.46933), (0.147751, 0.601056, 0.41287)),
                _sh(1, (19.76142, 6.821752, 2.291629), (0.025007, 0.48661, 0.582423)),
                _sh(1, (2.131206, 0.499354), (0.028708, 0.98407)),
                _sh(1, (0.164764,), (1.0,)),
                _sh(2, (33.965097, 8.900831, 2.428436), (0.149667, 0.511748, 0.575915)),
                _sh(2, (0.45,), (1.0,)),
            ),
        ),
    }
    jk_map = {
        1: ElementBasis(
            1,
            (
                _sh(0, (6.0,), (1.0,)),
                _sh(0, (1.5,), (1.0,)),
                _sh(0, (0.4,), (1.0,)),
                _sh(1, (1.4,), (1.0,)),
            ),
        ),
        6: ElementBasis(
            6,
            (
                _sh(0, (36.0,), (1.0,)),
                _sh(0, (9.0,), (1.0,)),
                _sh(0, (2.2,), (1.0,)),
                _sh(0, (0.55,), (1.0,)),
                _sh(1, (4.5,), (1.0,)),
                _sh(1, (1.1,), (1.0,)),
                _sh(2, (1.6,), (1.0,)),
            ),
        ),
        8: ElementBasis(
            8,
            (
                _sh(0, (60.0,), (1.0,)),
                _sh(0, (15.0,), (1.0,)),
                _sh(0, (4.0,), (1.0,)),
                _sh(0, (1.1,), (1.0,)),
                _sh(1, (7.0,), (1.0,)),
                _sh(1, (1.8,), (1.0,)),
                _sh(2, (2.2,), (1.0,)),
            ),
        ),
        35: ElementBasis(
            35,
            (
                _sh(0, (1800.0,), (1.0,)),
                _sh(0, (450.0,), (1.0,)),
                _sh(0, (120.0,), (1.0,)),
                _sh(0, (32.0,), (1.0,)),
                _sh(0, (9.0,), (1.0,)),
                _sh(0, (2.5,), (1.0,)),
                _sh(0, (0.7,), (1.0,)),
                _sh(0, (0.22,), (1.0,)),
                _sh(1, (40.0,), (1.0,)),
                _sh(1, (12.0,), (1.0,)),
                _sh(1, (3.2,), (1.0,)),
                _sh(1, (0.85,), (1.0,)),
                _sh(2, (16.0,), (1.0,)),
                _sh(2, (4.5,), (1.0,)),
                _sh(2, (1.2,), (1.0,)),
            ),
        ),
    }
    mp2_map = {
        1: ElementBasis(
            1,
            (
                _sh(0, (5.2,), (1.0,)),
                _sh(0, (1.2,), (1.0,)),
                _sh(0, (0.32,), (1.0,)),
                _sh(1, (1.2,), (1.0,)),
            ),
        ),
        6: ElementBasis(
            6,
            (
                _sh(0, (28.0,), (1.0,)),
                _sh(0, (7.2,), (1.0,)),
                _sh(0, (1.8,), (1.0,)),
                _sh(0, (0.45,), (1.0,)),
                _sh(1, (3.8,), (1.0,)),
                _sh(1, (0.95,), (1.0,)),
                _sh(2, (1.3,), (1.0,)),
            ),
        ),
        8: ElementBasis(
            8,
            (
                _sh(0, (48.0,), (1.0,)),
                _sh(0, (11.0,), (1.0,)),
                _sh(0, (2.8,), (1.0,)),
                _sh(0, (0.75,), (1.0,)),
                _sh(1, (5.8,), (1.0,)),
                _sh(1, (1.4,), (1.0,)),
                _sh(2, (1.9,), (1.0,)),
            ),
        ),
        35: ElementBasis(
            35,
            (
                _sh(0, (1400.0,), (1.0,)),
                _sh(0, (350.0,), (1.0,)),
                _sh(0, (90.0,), (1.0,)),
                _sh(0, (26.0,), (1.0,)),
                _sh(0, (7.5,), (1.0,)),
                _sh(0, (2.1,), (1.0,)),
                _sh(0, (0.58,), (1.0,)),
                _sh(0, (0.18,), (1.0,)),
                _sh(1, (32.0,), (1.0,)),
                _sh(1, (9.5,), (1.0,)),
                _sh(1, (2.6,), (1.0,)),
                _sh(1, (0.70,), (1.0,)),
                _sh(2, (12.0,), (1.0,)),
                _sh(2, (3.8,), (1.0,)),
                _sh(2, (1.0,), (1.0,)),
            ),
        ),
    }
    cabs_map = {
        1: ElementBasis(1, (_sh(0, (0.8,), (1.0,)), _sh(1, (0.9,), (1.0,)))),
        6: ElementBasis(
            6, (_sh(0, (1.4,), (1.0,)), _sh(1, (1.4,), (1.0,)), _sh(2, (0.8,), (1.0,)))
        ),
        8: ElementBasis(
            8, (_sh(0, (2.0,), (1.0,)), _sh(1, (2.0,), (1.0,)), _sh(2, (1.2,), (1.0,)))
        ),
        35: ElementBasis(
            35,
            (_sh(0, (1.1,), (1.0,)), _sh(1, (1.1,), (1.0,)), _sh(2, (1.4,), (1.0,))),
        ),
    }
    uniq = tuple(sorted(set(elements)))
    orb = _make_basis("gate1-rhf-fp32w-orb", tuple(orb_map[z] for z in uniq))
    jk = _make_basis("gate1-rhf-fp32w-jk", tuple(jk_map[z] for z in uniq))
    mp2 = _make_basis("gate1-rhf-fp32w-mp2", tuple(mp2_map[z] for z in uniq))
    cabs = _make_basis("gate1-rhf-fp32w-cabs", tuple(cabs_map[z] for z in uniq))
    return FocalBasis(
        2, "gate1-rhf-fp32w-split", orb, jk_fit=jk, mp2_fit=mp2, f12_fit=mp2, cabs=cabs, beta=1.0
    )


MOLECULES = {
    "H2O": {
        "elements": (1, 8),
        "atoms": (
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.10, -1.43, 1.11)),
            ("H", (-0.04, 1.39, 1.15)),
        ),
    },
    "CH4": {
        "elements": (1, 6),
        "atoms": (
            ("C", (0.0, 0.0, 0.0)),
            ("H", (1.18, 1.18, 1.18)),
            ("H", (-1.18, -1.18, 1.18)),
            ("H", (-1.18, 1.18, -1.18)),
            ("H", (1.18, -1.18, -1.18)),
        ),
    },
    "HBr": {
        "elements": (1, 35),
        "atoms": (
            ("Br", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 2.67)),
        ),
    },
}


def _degeneracy_safe_weights(eps_active: np.ndarray) -> np.ndarray:
    """Assign distinct physical (T+)-like weights that remain equal across degenerate MOs."""
    o_act = len(eps_active)
    palette = [1.18, 0.91, 1.27, 0.84, 1.12, 0.96, 1.23, 0.89, 1.07, 1.15, 0.93, 1.04]
    weights = np.zeros(o_act, dtype=np.float64)
    group_idx = 0
    for i in range(o_act):
        assigned = False
        for j in range(i):
            if abs(eps_active[i] - eps_active[j]) < 1e-6:
                weights[i] = weights[j]
                assigned = True
                break
        if not assigned:
            weights[i] = palette[group_idx % len(palette)]
            group_idx += 1
    return weights


def _check_force_invariants(atoms, forces: np.ndarray) -> None:
    coords = np.asarray([a.position for a in atoms], dtype=np.float64)
    net_force = np.max(np.abs(np.sum(forces, axis=0)))
    torque = np.max(np.abs(np.sum(np.cross(coords, forces), axis=0)))
    assert net_force < 1e-11, f"Net force invariant violated: {net_force}"
    assert torque < 1e-10, f"Torque invariant violated: {torque}"


def _w_options(schedule: str) -> dict[str, bool | str]:
    if schedule == "fp32_split3":
        return {"triples_w_schedule": schedule, "triples_fp32_w": False}
    return {"triples_fp32_w": schedule == "fp32_emulated"}


def _check_w_execution(
    telemetry, schedule: str, *, forces: bool, prefix: str, record_property
) -> None:
    mixed = schedule != "fp64"
    assert telemetry["triples_fp32_w"] is (schedule == "fp32_emulated")
    assert telemetry["triples_fp32_w_executed"] is mixed
    assert telemetry["triples_fp32_w_resource_fallback"] is False
    forward = "fp32" if schedule == "fp32_split3" else schedule
    reverse = "fp32" if schedule == "fp32_split3" else "fp64"
    assert telemetry["forward_w_precision"] == forward
    assert telemetry["reverse_w_precision"] == (reverse if forces else "n/a")
    assert telemetry["w_precision_schedule"] == (
        "fp32_single" if schedule == "fp32_emulated" else schedule
    )
    nforward = telemetry["triples_fp32_forward_gemms"]
    nreverse = telemetry["triples_fp32_reverse_gemms"]
    assert (nforward > 0) is mixed
    assert (nreverse > 0) is (forces and schedule == "fp32_split3")
    assert nforward + nreverse == telemetry["triples_fp32_gemms"]
    for key in (
        "w_precision_schedule", "forward_w_precision", "reverse_w_precision",
        "triples_fp32_forward_gemms", "triples_fp32_reverse_gemms", "triples_fp64_gemms",
        "triples_fp32_w_resource_fallback",
    ):
        record_property(f"{prefix}_{key}", telemetry[key])


@pytest.mark.parametrize("mol_name", ["H2O", "CH4", "HBr"])
@pytest.mark.parametrize("mode", ["T", "Tplus"])
@pytest.mark.parametrize("mixed_w", ["fp32_emulated", "fp32_split3"])
def test_rhf_triples_fp32w_parity_and_bitwise_baseline(
    mol_name: str, mode: str, mixed_w: str, record_property
) -> None:
    """Check each arithmetic schedule against the repeated FP64 energy and force controls."""
    shell_lib, one_lib = f12_host_libraries()
    spec = MOLECULES[mol_name]
    basis = _compact_split_basis(spec["elements"])
    plan = plan_singlepoint(
        spec["atoms"],
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=0,
        multiplicity=1,
        root_id="rhf",
        correlation_space_id="frozen-core",
        backend="cpu_reference",
    )
    req = replace(
        next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation"),
        role="triples",
    )
    provider = MolecularFocalProvider(
        shell_library=shell_lib,
        one_electron_library=one_lib,
        budget_mib=1024,
        preconvergence="fp64",
    )
    ref = df_reference(req, forces=False, budget_mib=1024)
    o_full = ref.occupied[0]
    frozen = resolve_frozen_occupied(req, (o_full, o_full))[0]
    eps_act = ref.orbital_energies[0][frozen:o_full]
    weights = _degeneracy_safe_weights(eps_act) if mode == "Tplus" else None

    base_a = spatial_rhf_triples_v2(
        req,
        ref,
        budget_mib=1024,
        forces=True,
        occupied_partition=True,
        occupied_weights=weights,
        preconvergence="fp64",
        triples_fp32_w=False,
        provider=provider,
    )
    base_b = spatial_rhf_triples_v2(
        req,
        ref,
        budget_mib=1024,
        forces=True,
        occupied_partition=True,
        occupied_weights=weights,
        preconvergence="fp64",
        triples_fp32_w=False,
        provider=provider,
    )
    fp32w = spatial_rhf_triples_v2(
        req,
        ref,
        budget_mib=1024,
        forces=True,
        occupied_partition=True,
        occupied_weights=weights,
        preconvergence="fp64",
        provider=provider,
        **_w_options(mixed_w),
    )

    assert base_a.forces is not None and base_b.forces is not None and fp32w.forces is not None
    assert base_a.occupied is not None and base_b.occupied is not None and fp32w.occupied is not None

    # 1) triples_fp32_w=False must be bit-identical (0.0) to pure FP64 baseline
    assert base_a.energy == base_b.energy
    assert float(np.max(np.abs(base_a.forces - base_b.forces))) == 0.0
    assert float(np.max(np.abs(base_a.occupied - base_b.occupied))) == 0.0
    assert base_a.telemetry["triples_fp32_w"] is False
    assert base_a.telemetry["triples_fp32_w_executed"] is False
    assert base_a.telemetry["triples_fp32_gemms"] == 0
    assert base_a.telemetry["triples_fp64_gemms"] > 0

    _check_w_execution(
        fp32w.telemetry, mixed_w, forces=True, prefix="center", record_property=record_property
    )

    if weights is None:
        closure = abs(float(np.sum(fp32w.occupied)) - fp32w.energy)
    else:
        closure = abs(float(np.dot(weights, fp32w.occupied)) - fp32w.energy)
    assert closure <= 1e-11 * (1.0 + abs(fp32w.energy)), f"Closure failed: {closure}"

    # 3) Strict accuracy gates vs FP64
    e_diff = abs(fp32w.energy - base_a.energy)
    f_diff = float(np.max(np.abs(fp32w.forces - base_a.forces)))
    record_property("energy_difference_eh", e_diff)
    record_property("max_force_difference_eh_bohr", f_diff)
    assert e_diff <= 1e-9, f"{mol_name} ({mode}) |dE|={e_diff:.3e} > 1e-9 Eh"
    assert f_diff <= 1e-8, f"{mol_name} ({mode}) max|dF|={f_diff:.3e} > 1e-8 Eh/bohr"
    _check_force_invariants(req.atoms, fp32w.forces)


@pytest.mark.parametrize("mol_name,atom_idx,axis", [("H2O", 1, 1), ("HBr", 1, 2)])
@pytest.mark.parametrize("mode", ["T", "Tplus"])
@pytest.mark.parametrize("fd_fp32_w", [True, False], ids=["same-mode", "fp64-control"])
@pytest.mark.parametrize("h", [2e-4, 1e-4], ids=["h", "h-over-2"])
@pytest.mark.parametrize("mixed_w", ["fp32_emulated", "fp32_split3"])
def test_rhf_triples_fp32w_central_fd_forces(
    mol_name: str, atom_idx: int, axis: int, mode: str, fd_fp32_w: bool, h: float,
    mixed_w: str, record_property
) -> None:
    """Check the analytic force of each mixed schedule against central differences.

    The same-mode leg differences fp32_split3 energies directly. Single-pass W is
    accepted against FP64: its displaced energies within 1e-9 Eh and its force
    against the FP64 central difference. The difference of its own energies is a
    recorded diagnostic.
    """
    shell_lib, one_lib = f12_host_libraries()
    spec = MOLECULES[mol_name]
    basis = _compact_split_basis(spec["elements"])
    provider = MolecularFocalProvider(
        shell_library=shell_lib,
        one_electron_library=one_lib,
        budget_mib=1024,
        preconvergence="fp64",
    )

    def make_req(coords):
        plan = plan_singlepoint(
            coords,
            FocalSeries(basis),
            FocalSeries(basis),
            None,
            charge=0,
            multiplicity=1,
            root_id="rhf",
            correlation_space_id="frozen-core",
            backend="cpu_reference",
        )
        return replace(
            next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation"),
            role="triples",
        )

    req0 = make_req(spec["atoms"])
    ref0 = df_reference(req0, forces=False, budget_mib=1024)
    o_full = ref0.occupied[0]
    frozen = resolve_frozen_occupied(req0, (o_full, o_full))[0]
    eps_act = ref0.orbital_energies[0][frozen:o_full]
    weights = _degeneracy_safe_weights(eps_act) if mode == "Tplus" else None

    analytic = spatial_rhf_triples_v2(
        req0,
        ref0,
        budget_mib=1024,
        forces=True,
        occupied_partition=True,
        occupied_weights=weights,
        preconvergence="fp64",
        provider=provider,
        **_w_options(mixed_w),
    )
    assert analytic.forces is not None
    _check_w_execution(
        analytic.telemetry, mixed_w, forces=True, prefix="center", record_property=record_property
    )

    def displaced_energy(delta: float, schedule: str) -> float:
        moved = [(sym, list(pos)) for sym, pos in spec["atoms"]]
        moved[atom_idx][1][axis] += delta
        req_d = make_req([(sym, tuple(pos)) for sym, pos in moved])
        ref_d = df_reference(req_d, forces=False, budget_mib=1024)
        res_d = spatial_rhf_triples_v2(
            req_d,
            ref_d,
            budget_mib=1024,
            forces=False,
            occupied_weights=weights,
            preconvergence="fp64",
            **_w_options(schedule),
        )
        _check_w_execution(
            res_d.telemetry, schedule, forces=False,
            prefix=f"{'plus' if delta > 0 else 'minus'}_{schedule}",
            record_property=record_property,
        )
        return res_d.energy

    fd_schedule = mixed_w if fd_fp32_w else "fp64"
    energy_plus, energy_minus = displaced_energy(+h, fd_schedule), displaced_energy(-h, fd_schedule)
    if fd_schedule == "fp32_emulated":
        # Single-pass W energies carry rounding noise that central differences amplify
        # by 1/(2h), so the difference of its own energies is not an acceptance oracle.
        fp64_plus, fp64_minus = displaced_energy(+h, "fp64"), displaced_energy(-h, "fp64")
        deviation = max(abs(energy_plus - fp64_plus), abs(energy_minus - fp64_minus))
        record_property("single_pass_energy_deviation_eh", deviation)
        self_consistent_fd = -(energy_plus - energy_minus) / (2.0 * h)
        record_property(
            "diagnostic_self_consistent_fd_error_eh_bohr",
            abs(float(analytic.forces[atom_idx, axis]) - self_consistent_fd),
        )
        assert deviation <= 1e-9, f"{mol_name} ({mode}) displaced |dE|={deviation:.3e} > 1e-9 Eh"
        energy_plus, energy_minus = fp64_plus, fp64_minus
    fd_force = -(energy_plus - energy_minus) / (2.0 * h)
    err = abs(float(analytic.forces[atom_idx, axis]) - fd_force)
    record_property("step_bohr", h)
    record_property("energy_plus_eh", energy_plus)
    record_property("energy_minus_eh", energy_minus)
    record_property("analytic_force_eh_bohr", float(analytic.forces[atom_idx, axis]))
    record_property("fd_force_eh_bohr", fd_force)
    record_property("fd_error_eh_bohr", err)
    assert err < 1e-8, (
        f"{mol_name} ({mode}) FD force mismatch at atom={atom_idx}, axis={axis}: "
        f"analytic={analytic.forces[atom_idx, axis]:.12f}, fd={fd_force:.12f}, err={err:.3e}"
    )


def test_spatial_closed_shell_fp32w_equations_and_canonical_fock_vv() -> None:
    """Directly verify closed-shell spatial FP32-W equations matching df_triples_cuda.cu."""
    pyscf_gto = pytest.importorskip("pyscf.gto")
    pyscf_scf = pytest.importorskip("pyscf.scf")
    pyscf_cc = pytest.importorskip("pyscf.cc")
    pyscf_df = pytest.importorskip("pyscf.df")
    from generativeqc_compiler.cc.occupied_triples import PERMUTATIONS

    # Closed-shell H2O in cc-pVDZ (frozen-core)
    mol = pyscf_gto.M(
        atom="O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11",
        basis="cc-pvdz",
        unit="Bohr",
        cart=False,
        verbose=0,
    )
    mf = pyscf_scf.RHF(mol).density_fit(auxbasis="cc-pvdz-ri").run(conv_tol=1e-13)
    mycc = pyscf_cc.RCCSD(mf, frozen=1).run(conv_tol=1e-12, conv_tol_normt=1e-10)
    t1, t2 = mycc.t1, mycc.t2
    o, v = t1.shape
    nocc_tot = mol.nelectron // 2
    Co, Cv = mf.mo_coeff[:, 1:nocc_tot], mf.mo_coeff[:, nocc_tot:]
    eps_o, eps_v = mf.mo_energy[1:nocc_tot], mf.mo_energy[nocc_tot:]
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis="cc-pvdz-ri")
    j2c = auxmol.intor("int2c2e")
    j3c = pyscf_df.incore.aux_e2(mol, auxmol, intor="int3c2e")
    Winv = np.linalg.solve(np.linalg.cholesky(j2c), np.eye(len(j2c)))
    B_ao = np.einsum("pq,mnq->pmn", Winv, j3c)
    q = B_ao.shape[0]
    bov = np.einsum("qmn,mi,na->qia", B_ao, Co, Cv, optimize=True)
    bvv = np.einsum("qmn,ma,nb->qab", B_ao, Cv, Cv, optimize=True)
    boo = np.einsum("qmn,mi,nj->qij", B_ao, Co, Co, optimize=True)
    ovoo = np.einsum("qia,qjk->iajk", bov, boo, optimize=True)
    ovov = np.einsum("qia,qjb->iajb", bov, bov, optimize=True)
    weights = _degeneracy_safe_weights(eps_o)

    panels_f64 = np.einsum("qia,qbf->iabf", bov, bvv, optimize=True)
    bov_f32, bvv_f32 = bov.astype(np.float32), bvv.astype(np.float32)
    ovoo_f32, t2_f32 = ovoo.astype(np.float32), t2.astype(np.float32)
    panels_f32 = np.empty((o, v, v, v), dtype=np.float32)
    for i in range(o):
        panels_f32[i] = (bov_f32[:, i, :].T @ bvv_f32.reshape(q, v * v)).reshape(v, v, v)

    def r3(x: np.ndarray) -> np.ndarray:
        return (
            4.0 * x
            - 2.0 * x.transpose(0, 2, 1)
            - 2.0 * x.transpose(2, 1, 0)
            - 2.0 * x.transpose(1, 0, 2)
            + x.transpose(1, 2, 0)
            + x.transpose(2, 0, 1)
        )

    def eval_spatial(fp32_w: bool):
        e_t = 0.0
        e_tp = 0.0
        bar_panel = np.zeros((o, v, v, v), dtype=np.float64)
        bar_t2 = np.zeros((o, o, v, v), dtype=np.float64)
        fvv = np.zeros((v, v), dtype=np.float64)
        for i in range(o):
            for j in range(i + 1):
                for k in range(j + 1):
                    occ = (i, j, k)
                    mult = 6.0 if i == k else (2.0 if (i == j or j == k) else 1.0)
                    tile_w = (weights[i] + weights[j] + weights[k]) / 3.0
                    Ws, Vs = [], []
                    for perm in PERMUTATIONS:
                        I, J, K = occ[perm[0]], occ[perm[1]], occ[perm[2]]
                        if fp32_w:
                            w1 = (
                                panels_f32[I].reshape(v * v, v) @ t2_f32[K, J].T
                            ).reshape(v, v, v).astype(np.float64)
                            w2 = (
                                ovoo_f32[I, :, J] @ t2_f32[:, K].reshape(o, v * v)
                            ).reshape(v, v, v).astype(np.float64)
                            w = w1 - w2
                        else:
                            w = (
                                panels_f64[I].reshape(v * v, v) @ t2[K, J].T
                            ).reshape(v, v, v) - (
                                ovoo[I, :, J] @ t2[:, K].reshape(o, v * v)
                            ).reshape(v, v, v)
                        vv = np.einsum("ab,c->abc", ovov[I, :, J], t1[K])
                        Ws.append(w)
                        Vs.append(vv)
                    W_tot = sum(
                        Ws[m].transpose(tuple(PERMUTATIONS[m].index(x) for x in range(3)))
                        for m in range(6)
                    )
                    V_tot = sum(
                        Vs[m].transpose(tuple(PERMUTATIONS[m].index(x) for x in range(3)))
                        for m in range(6)
                    )
                    Z_tot = W_tot + 0.5 * V_tot
                    D = (
                        eps_o[i]
                        + eps_o[j]
                        + eps_o[k]
                        - eps_v[:, None, None]
                        - eps_v[None, :, None]
                        - eps_v[None, None, :]
                    )
                    X = W_tot / D
                    Y = r3(Z_tot) / D
                    # Exact closed-shell (T) factor for canonical i >= j >= k: (6/mult) * (1/3) = 2/mult
                    e_tile = (2.0 / mult) * float(np.sum(W_tot * Y))
                    e_t += e_tile
                    e_tp += tile_w * e_tile

                    # Canonical triangular fock_vv (exact identity matching df_triples_cuda.cu)
                    M = (
                        np.einsum("abc,dbc->ad", X, Y)
                        + np.einsum("bca,bcd->ad", X, Y)
                        + np.einsum("bac,bdc->ad", X, Y)
                    )
                    fvv += (tile_w / mult) * (M + M.T)

                    bw_tot = (2.0 * tile_w / mult) * (Y + r3(W_tot) / D)
                    for m, perm in enumerate(PERMUTATIONS):
                        I, J, K = occ[perm[0]], occ[perm[1]], occ[perm[2]]
                        bw = np.ascontiguousarray(bw_tot.transpose(perm))
                        bw_2d = bw.reshape(v * v, v)
                        if fp32_w:
                            bw32 = bw_2d.astype(np.float32)
                            bar_panel[I] += (bw32 @ t2_f32[K, J]).reshape(v, v, v).astype(
                                np.float64
                            )
                            for row0 in range(0, v * v, 4096):
                                chunk = slice(row0, min(v * v, row0 + 4096))
                                bar_t2[K, J] += (
                                    bw32[chunk].T @ panels_f32[I].reshape(v * v, v)[chunk]
                                ).astype(np.float64)
                        else:
                            bar_panel[I] += (bw_2d @ t2[K, J]).reshape(v, v, v)
                            bar_t2[K, J] += bw_2d.T @ panels_f64[I].reshape(v * v, v)
                        bar_t2[:, K] -= (ovoo[I, :, J].T @ bw.reshape(v, v * v)).reshape(o, v, v)
        bar_bov = np.einsum("iabf,qbf->qia", bar_panel, bvv, optimize=True)
        bar_bvv = np.einsum("iabf,qia->qbf", bar_panel, bov, optimize=True)
        return e_t, e_tp, bar_bov, bar_bvv, bar_t2, fvv

    e_t64, e_tp64, bo64, bv64, bt64, fvv64 = eval_spatial(False)
    e_t32, e_tp32, bo32, bv32, bt32, fvv32 = eval_spatial(True)
    assert abs(e_t64 - mycc.ccsd_t()) < 1e-11
    assert abs(e_t32 - e_t64) <= 1e-9
    assert abs(e_tp32 - e_tp64) <= 1e-9
    assert np.max(np.abs(bo32 - bo64)) <= 1e-8
    assert np.max(np.abs(bv32 - bv64)) <= 1e-8
    assert np.max(np.abs(bt32 - bt64)) <= 1e-8
    assert np.max(np.abs(fvv32 - fvv64)) <= 1e-8


def test_vendor_boundaries_remain_clean() -> None:
    """Confirm modified RHF triples files add zero direct vendor BLAS symbols."""
    from tools.check_vendor_boundaries import audit_vendor_boundaries

    report = audit_vendor_boundaries(Path(__file__).resolve().parents[2])
    owned = (
        "src/cc/df_triples_cuda.cu",
        "src/cc/df_triples.hpp",
        "src/cc/df_triples_traversal.hpp",
        "src/methods/df_rccsdt_method.hpp",
        "src/methods/df_rccsdt_method.cpp",
        "src/api/c_api_f12_focal_reference.cpp",
    )
    for rel in owned:
        assert rel not in report["inventory"], f"Unexpected vendor reference in {rel}"
        assert not any(rel in err for err in report["errors"]), report["errors"]
