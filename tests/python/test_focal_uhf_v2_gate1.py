"""Energy & Force Gate 1 verification for native spin-blocked DF-UCCSD(T)/(T+) v2 (CPU).

Verifies:
1. Closed-shell H2O (AE & FC, split JK/MP2 fit, both RHF and UHF roots):
   DF-UCCSD & DF-UCCSD(T) energies match PySCF 2.14.0 DF-RCCSD(T) to < 1e-12 Eh,
   and occupied triples partitions satisfy sum(e_a) + sum(e_b) == E_(T) and e_a == e_b
   to < 1e-12 Eh.
2. Open-shell radicals OH (doublet), NH2 (doublet), O2 (triplet), CH3 (doublet)
   (AE & FC, split JK/MP2 fit):
   DF-UCCSD & DF-UCCSD(T) energies match PySCF 2.14.0 DF-UCCSD(T) to < 1e-11 Eh.
3. Weighted (T+) energy closure on OH & NH2 (AE & FC):
   sum_i w_a[i] * e_a[i] + sum_j w_b[j] * e_b[j] == E_(T+) to < 1e-13 Eh.
4. Force Gate 1 (CPU, split-fit, AE & FC):
   - Closed-shell H2O: UHF v2 (T) & weighted (T+) analytic forces match two-step central FD
     (h = 4e-4, 2e-4 bohr) to < 2e-8 Eh/bohr, match RHF spatial relaxation to < 1e-11 Eh/bohr,
     and match legacy retained all-active (T) forces to < 1e-11 Eh/bohr.
   - Open-shell OH & NH2: unweighted (T), fixed unequal-weighted (T+), and full end-to-end
     unrestricted_tplus_energy_and_forces_on_reference analytic forces match two-step central
     FD (h = 4e-4, 2e-4 bohr) to < 2e-8 Eh/bohr with net force & torque < 1e-12.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import f12_host_shell

from generativeqc._f12_focal_molecular import MolecularFocalProvider
from generativeqc._f12_focal_reference import df_reference
from generativeqc._f12_focal_ucc_triples import (
    molecular_occupied_triples_on_reference,
    molecular_triples_on_reference,
    relax_spatial_rhf_force_sources,
    spin_blocked_uhf_triples_v2,
)
from generativeqc._f12_frozen_core import resolve_frozen_occupied
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalSeries,
    plan_singlepoint,
)


def _pyscf_basis_dict(basis: BasisSet, atoms) -> dict[str, list]:
    """Convert a GenerativeQC BasisSet to PySCF internal gto.basis format."""
    out: dict[str, list] = {}
    symbols = {1: "H", 6: "C", 7: "N", 8: "O"}
    for atom in atoms:
        sym = symbols[atom.atomic_number]
        if sym in out:
            continue
        eb = basis.by_element[atom.atomic_number]
        shells = []
        for sh in eb.shells:
            for crow in sh.coefficients:
                row = [sh.angular_momentum]
                for exp, coeff in zip(sh.exponents, crow, strict=True):
                    row.append([float(exp), float(coeff)])
                shells.append(row)
        out[sym] = shells
    return out


def _build_split_hcno_bases() -> FocalBasis:
    """Build compact spherical orbital, JKFit and distinct MP2Fit bases for H, C, N, O."""

    def make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
        prov = BasisProvenance(
            source=name,
            version="1",
            license="MIT",
            checksum="0" * 64,
        )
        return BasisSet(name, elems, prov, representation="spherical")

    def shell(l: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
        return BasisShell(
            l, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),)
        )

    # 3-21G-like real-spherical orbital bases for H, C, N, O
    h_orb = ElementBasis(
        1,
        (
            shell(0, (5.4471780, 0.82454724), (0.1562850, 0.9046910)),
            shell(0, (0.18319158,), (1.0,)),
        ),
    )
    c_orb = ElementBasis(
        6,
        (
            shell(
                0,
                (172.25600, 25.910900, 5.5333500),
                (0.0617669, 0.3587940, 0.7007130),
            ),
            shell(0, (3.6649800, 0.7705450), (-0.3958970, 1.2158400)),
            shell(0, (0.1958570,), (1.0,)),
            shell(1, (3.6649800, 0.7705450), (0.2364600, 0.8606190)),
            shell(1, (0.1958570,), (1.0,)),
        ),
    )
    n_orb = ElementBasis(
        7,
        (
            shell(
                0,
                (242.76600, 36.485100, 7.8144900),
                (0.0598657, 0.3529550, 0.7065130),
            ),
            shell(0, (5.4252200, 1.1491500), (-0.4133010, 1.2244200)),
            shell(0, (0.2832050,), (1.0,)),
            shell(1, (5.4252200, 1.1491500), (0.2379720, 0.8589530)),
            shell(1, (0.2832050,), (1.0,)),
        ),
    )
    o_orb = ElementBasis(
        8,
        (
            shell(
                0,
                (322.03700, 48.430800, 10.420600),
                (0.0592394, 0.3515000, 0.7076580),
            ),
            shell(0, (7.4029400, 1.5762000), (-0.4044530, 1.2215600)),
            shell(0, (0.3736840,), (1.0,)),
            shell(1, (7.4029400, 1.5762000), (0.2445860, 0.8539550)),
            shell(1, (0.3736840,), (1.0,)),
        ),
    )

    def heavy_aux(z: int, scale: float) -> ElementBasis:
        return ElementBasis(
            z,
            (
                shell(0, (55.0 * scale,), (1.0,)),
                shell(0, (14.0 * scale,), (1.0,)),
                shell(0, (3.6 * scale,), (1.0,)),
                shell(0, (0.95 * scale,), (1.0,)),
                shell(0, (0.28 * scale,), (1.0,)),
                shell(1, (7.5 * scale,), (1.0,)),
                shell(1, (2.0 * scale,), (1.0,)),
                shell(1, (0.55 * scale,), (1.0,)),
                shell(2, (2.8 * scale,), (1.0,)),
                shell(2, (0.75 * scale,), (1.0,)),
            ),
        )

    h_jk = ElementBasis(
        1,
        (
            shell(0, (6.0,), (1.0,)),
            shell(0, (1.5,), (1.0,)),
            shell(0, (0.4,), (1.0,)),
            shell(1, (1.8,), (1.0,)),
            shell(1, (0.5,), (1.0,)),
        ),
    )
    h_mp2 = ElementBasis(
        1,
        (
            shell(0, (5.2,), (1.0,)),
            shell(0, (1.25,), (1.0,)),
            shell(0, (0.33,), (1.0,)),
            shell(1, (1.45,), (1.0,)),
        ),
    )

    orb = make_basis("uhf-v2-gate1-orb", (h_orb, c_orb, n_orb, o_orb))
    jk = make_basis(
        "uhf-v2-gate1-jk",
        (h_jk, heavy_aux(6, 0.72), heavy_aux(7, 0.86), heavy_aux(8, 1.0)),
    )
    mp2 = make_basis(
        "uhf-v2-gate1-mp2",
        (h_mp2, heavy_aux(6, 0.58), heavy_aux(7, 0.70), heavy_aux(8, 0.82)),
    )
    return FocalBasis(
        2, "uhf-v2-gate1-split", orb, jk_fit=jk, mp2_fit=mp2, f12_fit=mp2, cabs=jk, beta=1.0
    )


def _pyscf_df_ccsd_t(
    atoms_spec,
    basis: FocalBasis,
    *,
    root: str,
    multiplicity: int,
    frozen: int,
) -> tuple[float, float, float]:
    """Run independent PySCF 2.14.0 split-fit DF-HF + DF-CCSD(T)."""
    from pyscf import cc as pyscf_cc, gto, scf as pyscf_scf

    plan = plan_singlepoint(
        atoms_spec,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=0,
        multiplicity=multiplicity,
        root_id=root,
        correlation_space_id="all-active",
        backend="cpu_reference",
    )
    req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    mol = gto.M(
        atom=[(s, c) for s, c in atoms_spec],
        basis=_pyscf_basis_dict(basis.orbital, req.atoms),
        unit="Bohr",
        cart=False,
        charge=0,
        spin=multiplicity - 1,
        verbose=0,
    )
    jk_aux = _pyscf_basis_dict(basis.jk_fit, req.atoms)
    mp2_aux = _pyscf_basis_dict(basis.mp2_fit, req.atoms)
    mf = (
        pyscf_scf.RHF(mol) if root == "rhf" else pyscf_scf.UHF(mol)
    ).density_fit(auxbasis=jk_aux)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 150
    mf.kernel()
    assert mf.converged

    mycc = (
        pyscf_cc.CCSD(mf, frozen=frozen)
        if root == "rhf"
        else pyscf_cc.UCCSD(mf, frozen=frozen)
    ).density_fit(auxbasis=mp2_aux)
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-11
    mycc.max_cycle = 150
    mycc.kernel()
    assert mycc.converged
    e_t = float(mycc.ccsd_t())
    return float(mf.e_tot), float(mycc.e_corr), e_t


def _make_triples_request(
    atoms_spec,
    basis: FocalBasis,
    *,
    root: str,
    multiplicity: int,
    correlation_space_id: str,
):
    plan = plan_singlepoint(
        atoms_spec,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=0,
        multiplicity=multiplicity,
        root_id=root,
        correlation_space_id=correlation_space_id,
        backend="cpu_reference",
    )
    pair_req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    ref = df_reference(pair_req, forces=False, budget_mib=512)
    triples_req = replace(pair_req, role="triples")
    return triples_req, ref


H2O_GEOM = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43, 1.11)),
    ("H", (0.0, 1.43, 1.11)),
)
OH_GEOM = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 1.84)),
)
NH2_GEOM = (
    ("N", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.52, 1.18)),
    ("H", (0.0, 1.52, 1.18)),
)
O2_GEOM = (
    ("O", (0.0, 0.0, -1.14)),
    ("O", (0.0, 0.0, 1.14)),
)
CH3_GEOM = (
    ("C", (0.0, 0.0, 0.0)),
    ("H", (2.04, 0.0, 0.0)),
    ("H", (-1.02, 1.76669, 0.0)),
    ("H", (-1.02, -1.76669, 0.0)),
)


@pytest.mark.parametrize("correlation_space_id", ["all-active", "frozen-core"])
@pytest.mark.parametrize("root", ["rhf", "uhf"])
def test_closed_shell_h2o_matches_pyscf_rccsd_t_and_occupied_closure(
    correlation_space_id: str, root: str
):
    """Closed-shell H2O: DF-UCCSD(T) v2 matches PySCF DF-RCCSD(T) and e_a == e_b to <1e-12 Eh."""
    pytest.importorskip("pyscf")
    basis = _build_split_hcno_bases()
    req, ref = _make_triples_request(
        H2O_GEOM,
        basis,
        root=root,
        multiplicity=1,
        correlation_space_id=correlation_space_id,
    )
    spins = ref._pair_spin_indices(req)
    frozen = resolve_frozen_occupied(req, tuple(ref.occupied[k] for k in spins))[0]
    e_hf_ref, e_ccsd_ref, e_t_ref = _pyscf_df_ccsd_t(
        H2O_GEOM, basis, root="rhf", multiplicity=1, frozen=frozen
    )
    assert abs(ref.energy - e_hf_ref) < 1e-11

    res = spin_blocked_uhf_triples_v2(
        req, ref, budget_mib=512, forces=False, occupied_partition=True
    )
    assert res.occupied_alpha is not None and res.occupied_beta is not None
    assert abs(res.ccsd_energy - e_ccsd_ref) < 1e-12, (
        f"H2O {root} {correlation_space_id} CCSD mismatch: {res.ccsd_energy} vs {e_ccsd_ref}"
    )
    assert abs(res.energy - e_t_ref) < 1e-12, (
        f"H2O {root} {correlation_space_id} (T) mismatch: {res.energy} vs {e_t_ref}"
    )
    assert abs((res.ccsd_energy + res.energy) - (e_ccsd_ref + e_t_ref)) < 1e-12
    assert np.max(np.abs(res.occupied_alpha - res.occupied_beta)) < 1e-12
    occ_sum = float(np.sum(res.occupied_alpha) + np.sum(res.occupied_beta))
    assert abs(occ_sum - res.energy) < 1e-12
    assert res.telemetry["ccsd_iterations"] > 0
    assert res.telemetry["triples_fp64_gemms"] > 0
    assert res.telemetry["frozen_alpha"] == frozen
    assert res.telemetry["frozen_beta"] == frozen

    # Verify focal wrapper routing matches identically
    e_hook, f_hook = molecular_triples_on_reference(
        req, ref, budget_mib=512, forces=False
    )
    assert f_hook is None
    assert abs(e_hook - res.energy) < 1e-14
    e_occ_hook, occ_obj = molecular_occupied_triples_on_reference(
        replace(req, role="triples_plus"), ref, budget_mib=512
    )
    assert abs(e_occ_hook - res.energy) < 1e-14
    if root == "rhf":
        assert isinstance(occ_obj.values, np.ndarray)
        assert np.max(np.abs(occ_obj.values - (res.occupied_alpha + res.occupied_beta))) < 1e-14
    else:
        assert isinstance(occ_obj.values, tuple)
        assert np.max(np.abs(occ_obj.values[0] - res.occupied_alpha)) < 1e-14
        assert np.max(np.abs(occ_obj.values[1] - res.occupied_beta)) < 1e-14


@pytest.mark.parametrize(
    "name,geom,mult",
    [
        ("OH", OH_GEOM, 2),
        ("NH2", NH2_GEOM, 2),
        ("O2", O2_GEOM, 3),
        ("CH3", CH3_GEOM, 2),
    ],
)
@pytest.mark.parametrize("correlation_space_id", ["all-active", "frozen-core"])
def test_open_shell_radicals_match_pyscf_df_uccsd_t(
    name: str, geom, mult: int, correlation_space_id: str
):
    """Open-shell OH, NH2, O2, CH3 (AE & FC, split-fit): match PySCF DF-UCCSD(T) to <1e-11 Eh."""
    pytest.importorskip("pyscf")
    basis = _build_split_hcno_bases()
    req, ref = _make_triples_request(
        geom,
        basis,
        root="uhf",
        multiplicity=mult,
        correlation_space_id=correlation_space_id,
    )
    frozen = resolve_frozen_occupied(req, (ref.occupied[0], ref.occupied[1]))[0]
    e_hf_ref, e_ccsd_ref, e_t_ref = _pyscf_df_ccsd_t(
        geom, basis, root="uhf", multiplicity=mult, frozen=frozen
    )
    assert abs(ref.energy - e_hf_ref) < 1e-11, (
        f"{name} UHF reference mismatch: {ref.energy} vs {e_hf_ref}"
    )

    res = spin_blocked_uhf_triples_v2(
        req, ref, budget_mib=512, forces=False, occupied_partition=True
    )
    assert res.occupied_alpha is not None and res.occupied_beta is not None
    assert abs(res.ccsd_energy - e_ccsd_ref) < 1e-11, (
        f"{name} {correlation_space_id} DF-UCCSD mismatch: "
        f"{res.ccsd_energy} vs {e_ccsd_ref} (diff={res.ccsd_energy - e_ccsd_ref:.3e})"
    )
    assert abs(res.energy - e_t_ref) < 1e-11, (
        f"{name} {correlation_space_id} DF-UCCSD(T) mismatch: "
        f"{res.energy} vs {e_t_ref} (diff={res.energy - e_t_ref:.3e})"
    )
    occ_sum = float(np.sum(res.occupied_alpha) + np.sum(res.occupied_beta))
    assert abs(occ_sum - res.energy) < 1e-13


@pytest.mark.parametrize(
    "name,geom,mult",
    [
        ("OH", OH_GEOM, 2),
        ("NH2", NH2_GEOM, 2),
    ],
)
@pytest.mark.parametrize("correlation_space_id", ["all-active", "frozen-core"])
def test_weighted_tplus_energy_closure_open_shell(
    name: str, geom, mult: int, correlation_space_id: str
):
    """Weighted (T+) energy closure: sum_i w_a[i]*e_a[i] + sum_j w_b[j]*e_b[j] == E_(T+) to <1e-13 Eh."""
    basis = _build_split_hcno_bases()
    req, ref = _make_triples_request(
        geom,
        basis,
        root="uhf",
        multiplicity=mult,
        correlation_space_id=correlation_space_id,
    )
    fa, fb = resolve_frozen_occupied(req, (ref.occupied[0], ref.occupied[1]))
    oa_act = ref.occupied[0] - fa
    ob_act = ref.occupied[1] - fb

    # Construct non-trivial positive (T+)-like weights that respect orbital degeneracies
    def make_weights(eps: np.ndarray, seed_offset: float) -> np.ndarray:
        w = np.empty_like(eps)
        for idx, e in enumerate(eps):
            matched = False
            for prev in range(idx):
                if abs(e - eps[prev]) < 1e-5:
                    w[idx] = w[prev]
                    matched = True
                    break
            if not matched:
                w[idx] = 1.0 + seed_offset + 0.07 * (idx + 1)
        return w

    eps_a = ref.orbital_energies[0][fa : ref.occupied[0]]
    eps_b = ref.orbital_energies[1][fb : ref.occupied[1]]
    wa = make_weights(eps_a, 0.05)
    wb = make_weights(eps_b, 0.13)
    assert wa.shape == (oa_act,) and wb.shape == (ob_act,)

    unweighted = spin_blocked_uhf_triples_v2(
        req, ref, budget_mib=512, forces=False, occupied_partition=True
    )
    weighted = spin_blocked_uhf_triples_v2(
        req,
        ref,
        budget_mib=512,
        forces=False,
        occupied_partition=True,
        occupied_weights=(wa, wb),
    )
    assert unweighted.occupied_alpha is not None and unweighted.occupied_beta is not None
    assert weighted.occupied_alpha is not None and weighted.occupied_beta is not None
    # Unweighted per-orbital e_a, e_b are invariant under weighting
    assert np.max(np.abs(weighted.occupied_alpha - unweighted.occupied_alpha)) < 1e-14
    assert np.max(np.abs(weighted.occupied_beta - unweighted.occupied_beta)) < 1e-14

    expected_tplus = float(
        np.dot(wa, unweighted.occupied_alpha) + np.dot(wb, unweighted.occupied_beta)
    )
    assert abs(weighted.energy - expected_tplus) < 1e-13, (
        f"{name} {correlation_space_id} weighted (T+) closure mismatch: "
        f"{weighted.energy} vs {expected_tplus} (diff={weighted.energy - expected_tplus:.3e})"
    )


def _host_focal_provider() -> MolecularFocalProvider:
    """Resolve the CPU molecular integral sink library for Force Gate 1."""
    return MolecularFocalProvider(shell_library=f12_host_shell(), budget_mib=512)


def _make_degeneracy_safe_weights(eps: np.ndarray, seed_offset: float) -> np.ndarray:
    """Build unequal positive (T+)-like weights that respect orbital degeneracies."""
    w = np.empty_like(eps)
    for idx, e in enumerate(eps):
        matched = False
        for prev in range(idx):
            if abs(e - eps[prev]) < 1e-5:
                w[idx] = w[prev]
                matched = True
                break
        if not matched:
            w[idx] = 1.0 + seed_offset + 0.07 * (idx + 1)
    return w


def _shift_geom(geom, atom_idx: int, axis: int, delta: float):
    shifted = []
    for idx, (sym, pos) in enumerate(geom):
        if idx == atom_idx:
            coords = list(pos)
            coords[axis] += delta
            shifted.append((sym, tuple(coords)))
        else:
            shifted.append((sym, pos))
    return tuple(shifted)


def _assert_net_force_and_torque(geom, forces: np.ndarray, *, tol: float = 1e-12) -> None:
    coords = np.array([pos for _, pos in geom], dtype=np.float64)
    net_force = np.max(np.abs(np.sum(forces, axis=0)))
    net_torque = np.max(np.abs(np.sum(np.cross(coords, forces), axis=0)))
    assert net_force < tol, f"Net force violation: {net_force:.3e} >= {tol:.1e}"
    assert net_torque < tol, f"Net torque violation: {net_torque:.3e} >= {tol:.1e}"


def _extract_raw_v2_force_sources(
    req, ref, *, occupied_weights=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Call generativeqc_focal_ucc_triples_v2 directly to inspect raw spin-blocked cotangents."""
    import ctypes as ct
    from contextlib import ExitStack

    from generativeqc import _native
    from generativeqc.calculator import Calculator

    spins = ref._pair_spin_indices(req)
    oa_full, ob_full = (ref.occupied[k] for k in spins)
    n = len(ref.overlap)
    fa, fb = resolve_frozen_occupied(req, (oa_full, ob_full))
    owner = ref._native_owner
    assert owner is not None
    fn = owner.library.generativeqc_focal_ucc_triples_v2
    wa = (
        np.ascontiguousarray(occupied_weights[0], dtype=np.float64)
        if occupied_weights is not None
        else None
    )
    wb = (
        np.ascontiguousarray(occupied_weights[1], dtype=np.float64)
        if occupied_weights is not None
        else None
    )
    calc = Calculator(
        method="uccsd(t)",
        device="cpu",
        basis=req.basis.orbital,
        auxiliary_basis=req.basis.mp2_fit,
        density_fitting="full_cuda",
        initial_guess=None,
        correlation_memory_budget_bytes=512 << 20,
    )
    with ExitStack() as stack:
        stack.callback(calc.clear_cache)
        lib = calc._library
        ctx = ct.c_void_p()
        _native.check(
            lib,
            lib.generativeqc_context_create(
                ct.byref(calc._context_descriptor()), ct.byref(ctx)
            ),
        )
        stack.callback(lib.generativeqc_context_destroy, ctx)
        geometry = (ctx, req.atoms, req.charge, req.multiplicity)
        system = calc._create_native_system(*geometry)
        stack.callback(lib.generativeqc_system_destroy, system)
        corr_fit = calc._create_native_system(*geometry, req.basis.mp2_fit)
        stack.callback(lib.generativeqc_system_destroy, corr_fit)

        ccsd_e = ct.c_double()
        tri_e = ct.c_double()
        grad = np.empty(3 * len(req.atoms), dtype=np.float64)
        bar_fa = np.empty(n * n, dtype=np.float64)
        bar_fb = np.empty(n * n, dtype=np.float64)
        bar_ca = np.empty(n * n, dtype=np.float64)
        bar_cb = np.empty(n * n, dtype=np.float64)
        tel = np.zeros(10, dtype=np.float64)

        def ptr(arr: np.ndarray | None):
            return (
                arr.ctypes.data_as(ct.POINTER(ct.c_double))
                if arr is not None and arr.size > 0
                else None
            )

        status = fn(
            ctx,
            system,
            corr_fit,
            owner.handle,
            512 << 20,
            1,
            0,
            fa,
            fb,
            ptr(wa),
            0 if wa is None else wa.size,
            ptr(wb),
            0 if wb is None else wb.size,
            3,
            0,
            ct.byref(ccsd_e),
            ct.byref(tri_e),
            None,
            0,
            None,
            0,
            ptr(grad),
            grad.size,
            ptr(bar_fa),
            bar_fa.size,
            ptr(bar_fb),
            bar_fb.size,
            ptr(bar_ca),
            bar_ca.size,
            ptr(bar_cb),
            bar_cb.size,
            ptr(tel),
            tel.size,
        )
        _native.check(lib, status, context=ctx)
    return (
        bar_fa.reshape(n, n),
        bar_fb.reshape(n, n),
        bar_ca.reshape(n, n),
        bar_cb.reshape(n, n),
        grad,
    )


H2O_ASYM_GEOM = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.09, -1.43, 1.11)),
    ("H", (-0.05, 1.38, 1.16)),
)
NH2_ASYM_GEOM = (
    ("N", (0.0, 0.0, 0.0)),
    ("H", (0.11, -1.52, 1.18)),
    ("H", (-0.06, 1.46, 1.24)),
)


@pytest.mark.parametrize("correlation_space_id", ["all-active", "frozen-core"])
def test_force_gate1_closed_shell_h2o_fd_and_rhf_spatial_parity(
    correlation_space_id: str,
):
    """Closed-shell H2O (AE & FC, split-fit): UHF v2 (T) & (T+) forces match FD & RHF spatial."""
    provider = _host_focal_provider()
    basis = _build_split_hcno_bases()

    def eval_point(geom, *, root: str, weights=None, forces: bool = False):
        req, ref = _make_triples_request(
            geom,
            basis,
            root=root,
            multiplicity=1,
            correlation_space_id=correlation_space_id,
        )
        res = spin_blocked_uhf_triples_v2(
            req,
            ref,
            budget_mib=512,
            forces=forces,
            occupied_partition=True,
            occupied_weights=weights,
            provider=provider if forces else None,
        )
        return req, ref, res

    req_uhf, ref_uhf, res_uhf_unw = eval_point(H2O_ASYM_GEOM, root="uhf", forces=True)
    req_rhf, ref_rhf, res_rhf_unw = eval_point(H2O_ASYM_GEOM, root="rhf", forces=True)
    assert res_uhf_unw.forces is not None and res_rhf_unw.forces is not None
    assert res_uhf_unw.telemetry["lambda_iterations"] > 0
    _assert_net_force_and_torque(H2O_ASYM_GEOM, res_uhf_unw.forces)
    _assert_net_force_and_torque(H2O_ASYM_GEOM, res_rhf_unw.forces)

    # UHF v2 vs RHF v2 and direct spatial RHF relaxation parity (< 1e-11 Eh/bohr)
    assert np.max(np.abs(res_uhf_unw.forces - res_rhf_unw.forces)) < 1e-11
    bfa, bfb, bca, bcb, cgrad = _extract_raw_v2_force_sources(req_rhf, ref_rhf)
    assert np.max(np.abs(bfa - bfb)) < 1e-12
    assert np.max(np.abs(bca - bcb)) < 1e-12
    spatial_unw = relax_spatial_rhf_force_sources(
        provider, req_rhf, ref_rhf, bfa + bfb, bca + bcb, cgrad
    )
    assert np.max(np.abs(res_uhf_unw.forces - spatial_unw)) < 1e-11

    # Focal wrapper routing parity
    e_hook, f_hook = molecular_triples_on_reference(
        req_uhf, ref_uhf, budget_mib=512, forces=True, provider=provider
    )
    assert f_hook is not None
    assert abs(e_hook - res_uhf_unw.energy) < 1e-14
    assert np.max(np.abs(f_hook - res_uhf_unw.forces)) < 1e-14

    # Independent legacy retained all-active (T) force oracle parity (< 1e-11 Eh/bohr)
    if correlation_space_id == "all-active":
        e_leg, f_leg = molecular_triples_on_reference(
            req_rhf, ref_rhf, budget_mib=512, forces=True, provider=None
        )
        assert f_leg is not None
        assert abs(res_uhf_unw.energy - e_leg) < 1e-12
        assert np.max(np.abs(res_uhf_unw.forces - f_leg)) < 1e-11

    # Weighted (T+) with unequal spin-symmetric weights
    fa, _ = resolve_frozen_occupied(req_rhf, (ref_rhf.occupied[0], ref_rhf.occupied[0]))
    eps_o = ref_rhf.orbital_energies[0][fa : ref_rhf.occupied[0]]
    w_spatial = _make_degeneracy_safe_weights(eps_o, 0.06)
    weights = (w_spatial, w_spatial.copy())

    _, _, res_uhf_w = eval_point(H2O_ASYM_GEOM, root="uhf", weights=weights, forces=True)
    _, _, res_rhf_w = eval_point(H2O_ASYM_GEOM, root="rhf", weights=weights, forces=True)
    assert res_uhf_w.forces is not None and res_rhf_w.forces is not None
    _assert_net_force_and_torque(H2O_ASYM_GEOM, res_uhf_w.forces)
    assert np.max(np.abs(res_uhf_w.forces - res_rhf_w.forces)) < 1e-11

    bfa_w, bfb_w, bca_w, bcb_w, cgrad_w = _extract_raw_v2_force_sources(
        req_rhf, ref_rhf, occupied_weights=weights
    )
    spatial_w = relax_spatial_rhf_force_sources(
        provider, req_rhf, ref_rhf, bfa_w + bfb_w, bca_w + bcb_w, cgrad_w
    )
    assert np.max(np.abs(res_uhf_w.forces - spatial_w)) < 1e-11

    # Two-step central FD (h = 4e-4, 2e-4 bohr) for both unweighted (T) and weighted (T+)
    for atom_idx, axis in ((0, 2), (1, 0), (1, 1)):
        for h in (4e-4, 2e-4):
            gp = _shift_geom(H2O_ASYM_GEOM, atom_idx, axis, +h)
            gm = _shift_geom(H2O_ASYM_GEOM, atom_idx, axis, -h)

            ep = eval_point(gp, root="uhf")[2].energy
            em = eval_point(gm, root="uhf")[2].energy
            fd_unw = -(ep - em) / (2.0 * h)
            err_unw = abs(res_uhf_unw.forces[atom_idx, axis] - fd_unw)
            assert err_unw < 2e-8, (
                f"H2O {correlation_space_id} (T) FD mismatch at ({atom_idx},{axis}) h={h}: "
                f"an={res_uhf_unw.forces[atom_idx, axis]:.12f} fd={fd_unw:.12f} err={err_unw:.3e}"
            )

            ep_w = eval_point(gp, root="uhf", weights=weights)[2].energy
            em_w = eval_point(gm, root="uhf", weights=weights)[2].energy
            fd_w = -(ep_w - em_w) / (2.0 * h)
            err_w = abs(res_uhf_w.forces[atom_idx, axis] - fd_w)
            assert err_w < 2e-8, (
                f"H2O {correlation_space_id} weighted (T+) FD mismatch at ({atom_idx},{axis}) h={h}: "
                f"an={res_uhf_w.forces[atom_idx, axis]:.12f} fd={fd_w:.12f} err={err_w:.3e}"
            )


@pytest.mark.parametrize(
    "name,geom,coords_to_check",
    [
        ("OH", OH_GEOM, ((1, 2),)),
        ("NH2", NH2_ASYM_GEOM, ((0, 2), (1, 0), (1, 1))),
    ],
)
@pytest.mark.parametrize("correlation_space_id", ["all-active", "frozen-core"])
def test_force_gate1_open_shell_unweighted_and_weighted_two_step_fd(
    name: str, geom, coords_to_check, correlation_space_id: str
):
    """Open-shell OH & NH2 (AE & FC, split-fit): (T), weighted (T+), & live (T+) vs two-step FD."""
    provider = _host_focal_provider()
    basis = _build_split_hcno_bases()

    def eval_triples(g, *, weights=None, forces: bool = False):
        req, ref = _make_triples_request(
            g,
            basis,
            root="uhf",
            multiplicity=2,
            correlation_space_id=correlation_space_id,
        )
        res = spin_blocked_uhf_triples_v2(
            req,
            ref,
            budget_mib=512,
            forces=forces,
            occupied_partition=True,
            occupied_weights=weights,
            provider=provider if forces else None,
        )
        return req, ref, res

    req0, ref0, res_unw = eval_triples(geom, forces=True)
    assert res_unw.forces is not None
    assert res_unw.telemetry["lambda_iterations"] > 0
    _assert_net_force_and_torque(geom, res_unw.forces)

    fa, fb = resolve_frozen_occupied(req0, (ref0.occupied[0], ref0.occupied[1]))
    wa = _make_degeneracy_safe_weights(ref0.orbital_energies[0][fa : ref0.occupied[0]], 0.05)
    wb = _make_degeneracy_safe_weights(ref0.orbital_energies[1][fb : ref0.occupied[1]], 0.13)
    weights = (wa, wb)

    _, _, res_w = eval_triples(geom, weights=weights, forces=True)
    assert res_w.forces is not None
    _assert_net_force_and_torque(geom, res_w.forces)

    # Full end-to-end unrestricted_tplus_energy_and_forces_on_reference (live MP2-F12 weights!)
    def eval_live_tplus(g, *, forces: bool = False, cabs_rank: int | None = None):
        plan = plan_singlepoint(
            g,
            FocalSeries(basis),
            FocalSeries(basis),
            basis,
            charge=0,
            multiplicity=2,
            root_id="uhf",
            correlation_space_id=correlation_space_id,
            backend="cpu_reference",
        )
        pair_req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
        tplus_req = replace(pair_req, role="triples_plus")
        low = provider.low_order_energy(
            pair_req,
            include_hf_cabs=False,
            include_occupied_partitions=True,
            expected_cabs_rank=cabs_rank,
        )
        if forces:
            return provider.unrestricted_tplus_energy_and_forces_on_reference(
                tplus_req, low_order=low, expected_cabs_rank=cabs_rank
            )
        conv, ev = provider.unrestricted_tplus_energy_on_reference(
            tplus_req, low_order=low, expected_cabs_rank=cabs_rank
        )
        return conv, ev, {}

    conv_e, ev_tplus, tel_tplus = eval_live_tplus(geom, forces=True)
    assert ev_tplus.forces is not None
    assert abs(conv_e - res_unw.energy) < 1e-12
    assert tel_tplus["lambda_iterations"] > 0
    _assert_net_force_and_torque(geom, ev_tplus.forces)

    for atom_idx, axis in coords_to_check:
        for h in (4e-4, 2e-4):
            gp = _shift_geom(geom, atom_idx, axis, +h)
            gm = _shift_geom(geom, atom_idx, axis, -h)

            # 1) Unweighted (T)
            fd_unw = -(eval_triples(gp)[2].energy - eval_triples(gm)[2].energy) / (2.0 * h)
            err_unw = abs(res_unw.forces[atom_idx, axis] - fd_unw)
            assert err_unw < 2e-8, (
                f"{name} {correlation_space_id} (T) FD mismatch at ({atom_idx},{axis}) h={h}: "
                f"an={res_unw.forces[atom_idx, axis]:.12f} fd={fd_unw:.12f} err={err_unw:.3e}"
            )

            # 2) Fixed unequal-weighted (T+)
            fd_w = -(
                eval_triples(gp, weights=weights)[2].energy
                - eval_triples(gm, weights=weights)[2].energy
            ) / (2.0 * h)
            err_w = abs(res_w.forces[atom_idx, axis] - fd_w)
            assert err_w < 2e-8, (
                f"{name} {correlation_space_id} weighted (T+) FD mismatch at ({atom_idx},{axis}) "
                f"h={h}: an={res_w.forces[atom_idx, axis]:.12f} fd={fd_w:.12f} err={err_w:.3e}"
            )

            # 3) Live dynamic-weight (T+) including tplus_scaling_forces
            _, ev_p, _ = eval_live_tplus(gp, cabs_rank=ev_tplus.cabs_rank)
            _, ev_m, _ = eval_live_tplus(gm, cabs_rank=ev_tplus.cabs_rank)
            fd_live = -(ev_p.energy - ev_m.energy) / (2.0 * h)
            err_live = abs(ev_tplus.forces[atom_idx, axis] - fd_live)
            assert err_live < 2e-8, (
                f"{name} {correlation_space_id} live (T+) FD mismatch at ({atom_idx},{axis}) "
                f"h={h}: an={ev_tplus.forces[atom_idx, axis]:.12f} fd={fd_live:.12f} err={err_live:.3e}"
            )


def test_provider_pair_source_cache_isolates_surface_id():
    """Reusing one MolecularFocalProvider across RHF/UHF or AE/FC surfaces must not collide."""
    provider = _host_focal_provider()
    basis = _build_split_hcno_bases()
    energies = []
    for root_id, corr_space in (
        ("rhf", "all-active"),
        ("uhf", "all-active"),
        ("uhf", "frozen-core"),
    ):
        plan = plan_singlepoint(
            H2O_ASYM_GEOM,
            FocalSeries(basis),
            FocalSeries(basis),
            basis,
            charge=0,
            multiplicity=1,
            root_id=root_id,
            correlation_space_id=corr_space,
            backend="cpu_reference",
        )
        pair_req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
        low = provider.low_order_energy(
            pair_req, include_hf_cabs=True, include_occupied_partitions=True
        )
        energies.append((root_id, corr_space, low.hf_cabs_energy, low.mp2_f12_correlation_energy))
    assert abs(energies[0][2] - energies[1][2]) < 1e-12
    assert abs(energies[0][3] - energies[1][3]) < 1e-12
