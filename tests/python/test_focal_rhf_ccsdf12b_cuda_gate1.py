"""Gate 1 local CPU qualification for spatial RHF CCSD-F12b primal, Lambda, and fixed forces.

Verifies:
1. Compact spatial fixed-source builder matches unrestricted_cc_f12_fixed_source_feeds (< 1e-13).
2. Spatial RHF CCSD-F12b primal + Lambda + Fock cotangents + 17 fixed-source cotangents
   match retained _f12_ucc_solver.py to < 1e-11 on H2 (o_act=1), H2O (AE & FC),
   CH4 (AE & FC), and HBr (ECP10MDF, AE & FC).
3. Batched/RHF-symmetric spatial_rhf_fixed_cc_forces (both dense and streaming F12 AO panels)
   matches fixed_ucc_source_forces to < 1e-11 Eh/bohr.
"""

from __future__ import annotations


import numpy as np
import pytest

from conftest import f12_host_libraries

from generativeqc._f12_focal_cc_fixed import fixed_ucc_source_forces
from generativeqc._f12_focal_cc_sources import molecular_ucc_sources
from generativeqc._f12_focal_one_electron import one_electron_sources
from generativeqc._f12_focal_rcc_f12b import (
    _build_spatial_rhf_fixed_sources,
    spatial_rhf_ccsdf12b_cpu_solve,
    spatial_rhf_fixed_cc_forces,
)
from generativeqc._f12_focal_reference import FocalReference, df_reference
from generativeqc._f12_ucc_solver import (
    SolverOptions,
    UCCSDLambdaOptions,
    solve_focal_uccsd_f12b_with_cotangents,
)
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalSeries,
    MolecularFocalProvider,
    plan_singlepoint,
)
from generativeqc.response_solver import GMRESOptions
from _bse_basis_support import br_ecp10mdf

def _make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
    prov = BasisProvenance(source=name, version="1", license="MIT", checksum="0" * 64)
    return BasisSet(name, elems, prov, representation="spherical")


def _sh(l: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
    return BasisShell(l, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),))


def _compact_split_basis(elements: tuple[int, ...]) -> FocalBasis:
    """Compact spherical split-fit (JKFit != MP2Fit == F12Fit, OptRI) basis for H, C, O, Br(ECP10MDF)."""
    br_ecp = br_ecp10mdf()
    # Compact orbital shells (minimal-plus-split) for fast exact CPU CC tests.
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
                _sh(0, (71.616837, 13.045096, 3.5305122), (0.15432897, 0.53532814, 0.44463454)),
                _sh(0, (2.9412494, 0.6834831), (-0.09996723, 0.39951283)),
                _sh(0, (0.2222899,), (1.0,)),
                _sh(1, (2.9412494, 0.6834831), (0.15591627, 0.60768372)),
                _sh(1, (0.2222899,), (1.0,)),
            ),
        ),
        8: ElementBasis(
            8,
            (
                _sh(0, (130.70932, 23.808861, 6.4436083), (0.15432897, 0.53532814, 0.44463454)),
                _sh(0, (5.0331513, 1.1695961), (-0.09996723, 0.39951283)),
                _sh(0, (0.3803890,), (1.0,)),
                _sh(1, (5.0331513, 1.1695961), (0.15591627, 0.60768372)),
                _sh(1, (0.3803890,), (1.0,)),
            ),
        ),
        35: ElementBasis(
            35,
            (
                # Compact 3s3p2d valence+outer-core basis paired with real ECP10MDF (25 explicit e-).
                _sh(0, (18.4, 5.2, 1.8), (0.22, 0.55, 0.35)),
                _sh(0, (0.85, 0.31), (0.45, 0.62)),
                _sh(0, (0.12,), (1.0,)),
                _sh(1, (14.1, 4.1, 1.4), (0.25, 0.52, 0.38)),
                _sh(1, (0.62, 0.22), (0.48, 0.58)),
                _sh(1, (0.095,), (1.0,)),
                _sh(2, (6.8, 2.3), (0.42, 0.65)),
                _sh(2, (0.68,), (1.0,)),
            ),
            ecp_core_electrons=br_ecp.ecp_core_electrons,
            ecp_data=br_ecp.ecp_data,
        ),
    }
    jk_map = {
        1: ElementBasis(1, (_sh(0, (6.0,), (1.0,)), _sh(0, (1.5,), (1.0,)), _sh(0, (0.4,), (1.0,)), _sh(1, (1.4,), (1.0,)))),
        6: ElementBasis(6, (_sh(0, (36.0,), (1.0,)), _sh(0, (9.0,), (1.0,)), _sh(0, (2.2,), (1.0,)), _sh(0, (0.55,), (1.0,)), _sh(1, (4.5,), (1.0,)), _sh(1, (1.1,), (1.0,)), _sh(2, (1.6,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (60.0,), (1.0,)), _sh(0, (15.0,), (1.0,)), _sh(0, (4.0,), (1.0,)), _sh(0, (1.1,), (1.0,)), _sh(1, (7.0,), (1.0,)), _sh(1, (1.8,), (1.0,)), _sh(2, (2.2,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (32.0,), (1.0,)), _sh(0, (9.0,), (1.0,)), _sh(0, (2.5,), (1.0,)), _sh(0, (0.7,), (1.0,)), _sh(0, (0.22,), (1.0,)), _sh(1, (12.0,), (1.0,)), _sh(1, (3.2,), (1.0,)), _sh(1, (0.85,), (1.0,)), _sh(2, (4.5,), (1.0,)), _sh(2, (1.2,), (1.0,)))),
    }
    mp2_map = {
        1: ElementBasis(1, (_sh(0, (5.2,), (1.0,)), _sh(0, (1.2,), (1.0,)), _sh(0, (0.32,), (1.0,)), _sh(1, (1.2,), (1.0,)))),
        6: ElementBasis(6, (_sh(0, (28.0,), (1.0,)), _sh(0, (7.2,), (1.0,)), _sh(0, (1.8,), (1.0,)), _sh(0, (0.45,), (1.0,)), _sh(1, (3.8,), (1.0,)), _sh(1, (0.95,), (1.0,)), _sh(2, (1.3,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (48.0,), (1.0,)), _sh(0, (11.0,), (1.0,)), _sh(0, (2.8,), (1.0,)), _sh(0, (0.75,), (1.0,)), _sh(1, (5.8,), (1.0,)), _sh(1, (1.4,), (1.0,)), _sh(2, (1.9,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (26.0,), (1.0,)), _sh(0, (7.5,), (1.0,)), _sh(0, (2.1,), (1.0,)), _sh(0, (0.58,), (1.0,)), _sh(0, (0.18,), (1.0,)), _sh(1, (9.5,), (1.0,)), _sh(1, (2.6,), (1.0,)), _sh(1, (0.70,), (1.0,)), _sh(2, (3.8,), (1.0,)), _sh(2, (1.0,), (1.0,)))),
    }
    cabs_map = {
        1: ElementBasis(1, (_sh(0, (0.8,), (1.0,)), _sh(1, (0.9,), (1.0,)))),
        6: ElementBasis(6, (_sh(0, (1.4,), (1.0,)), _sh(1, (1.4,), (1.0,)), _sh(2, (0.8,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (2.0,), (1.0,)), _sh(1, (2.0,), (1.0,)), _sh(2, (1.2,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (1.1,), (1.0,)), _sh(1, (1.1,), (1.0,)), _sh(2, (1.4,), (1.0,)))),
    }
    uniq = tuple(sorted(set(elements)))
    orb = _make_basis("gate1-f12b-orb", tuple(orb_map[z] for z in uniq))
    jk = _make_basis("gate1-f12b-jk", tuple(jk_map[z] for z in uniq))
    mp2 = _make_basis("gate1-f12b-mp2", tuple(mp2_map[z] for z in uniq))
    cabs = _make_basis("gate1-f12b-cabs", tuple(cabs_map[z] for z in uniq))
    return FocalBasis(2, "gate1-f12b-split", orb, jk_fit=jk, mp2_fit=mp2, f12_fit=mp2, cabs=cabs, beta=1.0)


def _spatial_chemists_blocks_from_mp2_panel(provider, req, ref, sources, frozen: int):
    """Build exact spatial active chemists' ERI blocks (pq|rs) from the MP2Fit panel."""
    mp = provider._panel(req, "mp2_fit", orbital_only=True)
    ao3 = mp.data["ao_coulomb"]  # (Q, n, n)
    w = mp.whitener  # (Q, Q)
    c = ref.coefficients[0]  # (n, n)
    o_full = ref.occupied[0]
    c_occ = c[:, frozen:o_full]
    c_vir = c[:, o_full:]

    # B_PQ = W @ raw_Q -> (Q, p, q)
    raw_oo = np.einsum("Qmn,mi,nj->Qij", ao3, c_occ, c_occ, optimize=True)
    raw_ov = np.einsum("Qmn,mi,na->Qia", ao3, c_occ, c_vir, optimize=True)
    raw_vv = np.einsum("Qmn,ma,nb->Qab", ao3, c_vir, c_vir, optimize=True)
    boo = np.einsum("PQ,Qij->Pij", w, raw_oo, optimize=True)
    bov = np.einsum("PQ,Qia->Pia", w, raw_ov, optimize=True)
    bvv = np.einsum("PQ,Qab->Pab", w, raw_vv, optimize=True)
    bvo = bov.swapaxes(1, 2)

    ovov = np.einsum("Pia,Pjb->iajb", bov, bov, optimize=True)
    ovvo = np.einsum("Pia,Pbj->iabj", bov, bvo, optimize=True)
    oovv = np.einsum("Pij,Pab->ijab", boo, bvv, optimize=True)
    ovvv = np.einsum("Pia,Pbc->iabc", bov, bvv, optimize=True)
    ovoo = np.einsum("Pia,Pjk->iajk", bov, boo, optimize=True)
    oooo = np.einsum("Pij,Pkl->ijkl", boo, boo, optimize=True)
    vvvv = np.einsum("Pab,Pcd->abcd", bvv, bvv, optimize=True)
    eps = ref.orbital_energies[0]
    eps_o = eps[frozen:o_full]
    eps_v = eps[o_full:]
    return {
        "foo": np.asarray(sources.ordinary["foo_aa"], dtype=np.float64),
        "fov": np.asarray(sources.ordinary["fov_aa"], dtype=np.float64),
        "fvv": np.asarray(sources.ordinary["fvv_aa"], dtype=np.float64),
        "ovov": ovov,
        "ovvo": ovvo,
        "oovv": oovv,
        "ovvv": ovvv,
        "ovoo": ovoo,
        "oooo": oooo,
        "vvvv": vvvv,
        "eps_o": eps_o,
        "eps_v": eps_v,
    }


MOLECULES = [
    pytest.param(
        "H2",
        (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.42))),
        (1,),
        "all-active",
        id="H2-AE",
    ),
    pytest.param(
        "H2O",
        (("O", (0.0, 0.0, 0.0)), ("H", (0.0, -1.43, 1.11)), ("H", (0.0, 1.43, 1.11))),
        (1, 8),
        "all-active",
        id="H2O-AE",
    ),
    pytest.param(
        "H2O",
        (("O", (0.0, 0.0, 0.0)), ("H", (0.0, -1.43, 1.11)), ("H", (0.0, 1.43, 1.11))),
        (1, 8),
        "frozen-core",
        id="H2O-FC",
    ),
    pytest.param(
        "CH4",
        (
            ("C", (0.0, 0.0, 0.0)),
            ("H", (1.19, 1.19, 1.19)),
            ("H", (-1.19, -1.19, 1.19)),
            ("H", (-1.19, 1.19, -1.19)),
            ("H", (1.19, -1.19, -1.19)),
        ),
        (1, 6),
        "all-active",
        id="CH4-AE",
    ),
    pytest.param(
        "CH4",
        (
            ("C", (0.0, 0.0, 0.0)),
            ("H", (1.19, 1.19, 1.19)),
            ("H", (-1.19, -1.19, 1.19)),
            ("H", (-1.19, 1.19, -1.19)),
            ("H", (1.19, -1.19, -1.19)),
        ),
        (1, 6),
        "frozen-core",
        id="CH4-FC",
    ),
    pytest.param(
        "HBr",
        (("Br", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.68))),
        (1, 35),
        "all-active",
        id="HBr-ECP-AE",
    ),
    pytest.param(
        "HBr",
        (("Br", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.68))),
        (1, 35),
        "frozen-core",
        id="HBr-ECP-FC",
    ),
]


def _canonical_rhf_reference_for_test(provider, req) -> FocalReference:
    """Use native df_reference when supported on CPU, or solve exact DF-RHF on native ECP integrals."""
    has_ecp = any(
        req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons for a in req.atoms
    )
    if not has_ecp:
        return df_reference(req, forces=False, budget_mib=provider.budget_mib)

    # Exact canonical DF-RHF on native one-electron ECP + JKFit integrals (for CPU ECP tests).
    from scipy.linalg import eigh

    one = one_electron_sources(
        req,
        library=provider.one_electron_library or provider.shell_library,
        orbital_only=True,
        derivatives=False,
        budget_mib=provider.budget_mib,
    )
    jk = provider._panel(req, "jk_fit", orbital_only=True)
    S = np.asarray(one.overlap, dtype=np.float64)
    H = np.asarray(one.hcore, dtype=np.float64)
    ao3 = np.asarray(jk.data["ao_coulomb"], dtype=np.float64)  # (Q, n, n)
    B = np.einsum("PQ,Qmn->Pmn", jk.whitener, ao3, optimize=True)
    n_elec = sum(
        a.atomic_number - req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons
        for a in req.atoms
    ) - req.charge
    nocc = n_elec // 2
    enuc = 0.0
    for i, a in enumerate(req.atoms):
        zi = a.atomic_number - req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons
        for j in range(i + 1, len(req.atoms)):
            zj = req.atoms[j].atomic_number - req.basis.orbital.by_element[req.atoms[j].atomic_number].ecp_core_electrons
            r = np.linalg.norm(np.array(a.position) - np.array(req.atoms[j].position))
            enuc += zi * zj / r

    eps, C = eigh(H, S)
    D = 2.0 * (C[:, :nocc] @ C[:, :nocc].T)
    f_hist, e_hist = [], []
    F = H.copy()
    for it in range(1, 120):
        c_occ = C[:, :nocc]
        q_vec = np.einsum("Pmn,mn->P", B, D, optimize=True)
        J = np.einsum("P,Pmn->mn", q_vec, B, optimize=True)
        Bo = np.einsum("Pmn,ni->Pmi", B, c_occ, optimize=True)
        K = np.einsum("Pmi,Pni->mn", Bo, Bo, optimize=True)
        F = H + J - K
        F = 0.5 * (F + F.T)
        err = F @ D @ S - S @ D @ F
        if np.max(np.abs(err)) < 1e-12:
            break
        f_hist.append(F.copy())
        e_hist.append(err.ravel().copy())
        if len(f_hist) > 8:
            f_hist.pop(0)
            e_hist.pop(0)
        if len(f_hist) >= 2:
            m = len(f_hist)
            A = np.zeros((m + 1, m + 1))
            for a_i in range(m):
                for b_i in range(m):
                    A[a_i, b_i] = float(np.dot(e_hist[a_i], e_hist[b_i]))
                A[a_i, m] = A[m, a_i] = -1.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                coeff = np.linalg.solve(A, rhs)[:m]
                F_use = sum(c * fm for c, fm in zip(coeff, f_hist, strict=True))
            except np.linalg.LinAlgError:
                F_use = F
        else:
            F_use = F
        eps, C = eigh(F_use, S)
        D = 2.0 * (C[:, :nocc] @ C[:, :nocc].T)
    eps, C = eigh(F, S)
    D = 2.0 * (C[:, :nocc] @ C[:, :nocc].T)
    e_elec = 0.5 * float(np.sum(D * (H + F)))
    return FocalReference(
        energy=enuc + e_elec,
        forces=None,
        overlap=S,
        hcore=H,
        coefficients=C[None, :, :].copy(),
        orbital_energies=eps[None, :].copy(),
        density=D[None, :, :].copy(),
        fock=F[None, :, :].copy(),
        occupied=(nocc,),
        iterations=it,
        executed_backend=req.backend,
        surface_id=req.surface_id,
        basis_id=req.basis.basis_id,
        orbital_id=req.basis.orbital.identity,
        jk_fit_id=req.basis.jk_fit.identity,
    )


@pytest.mark.parametrize("label,atoms_spec,elements,corr_space", MOLECULES)
def test_spatial_rhf_ccsdf12b_primal_lambda_and_weights_parity(
    label: str,
    atoms_spec: tuple[tuple[str, tuple[float, float, float]], ...],
    elements: tuple[int, ...],
    corr_space: str,
) -> None:
    """Verify exact < 1e-11 parity of spatial C++ CCSD-F12b vs legacy _f12_ucc_solver.py."""
    shell_lib, one_lib = f12_host_libraries()
    basis = _compact_split_basis(elements)
    provider = MolecularFocalProvider(
        shell_library=shell_lib,
        one_electron_library=one_lib,
        budget_mib=1536,
    )
    plan = plan_singlepoint(
        atoms_spec,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=0,
        multiplicity=1,
        root_id="rhf",
        correlation_space_id=corr_space,
        backend="cpu_reference",
    )
    req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    ref = _canonical_rhf_reference_for_test(provider, req)
    low = None
    if not any(req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons for a in req.atoms):
        low = provider.low_order_energy(req, include_hf_cabs=False)
        ref = low.reference

    # 1) Legacy spin-resolved UCCSD-F12b reference oracle
    sources = molecular_ucc_sources(provider, req, include_ordinary=True, reference=ref)
    frozen = sources.fixed.frozen_occupied[0]
    assert sources.fixed.frozen_occupied == (frozen, frozen)
    if label == "HBr":
        assert frozen == (4 if corr_space == "frozen-core" else 0)
    elif label in ("H2O", "CH4"):
        assert frozen == (1 if corr_space == "frozen-core" else 0)

    ref_state, ref_dual, ref_pullback = solve_focal_uccsd_f12b_with_cotangents(
        sources,
        options=SolverOptions(
            residual_tolerance=1e-12,
            energy_tolerance=1e-13,
            max_bytes=1536 << 20,
        ),
        lambda_options=UCCSDLambdaOptions(
            cc_tolerance=1e-10,
            lambda_tolerance=1e-10,
            max_bytes=1536 << 20,
            max_total_work=1 << 50,
            gmres=GMRESOptions(rtol=0.0, atol=1e-13, max_iterations=400, max_workspace_bytes=1536 << 20),
        ),
    )

    # 2) Compact spatial fixed-source builder exact parity
    w_vo, w_vv, c_f12, fixed_e, rank, sp_frozen = _build_spatial_rhf_fixed_sources(
        provider, req, ref, expected_cabs_rank=sources.fixed.cabs_rank, derivatives=False
    )
    assert rank == sources.fixed.cabs_rank and sp_frozen == frozen
    assert abs(fixed_e - float(sources.fixed.feeds["fixed_geminal_energy"])) < 1e-13
    np.testing.assert_allclose(w_vo, sources.fixed.feeds["w_vo_abab"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(w_vv, sources.fixed.feeds["w_vv_abab"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(c_f12, sources.fixed.feeds["c_f12_abab"], rtol=0, atol=1e-13)

    # 3) Spatial C++ CCSD-F12b primal + Lambda + Fock/fixed cotangents
    blocks = _spatial_chemists_blocks_from_mp2_panel(provider, req, ref, sources, frozen)
    spatial = spatial_rhf_ccsdf12b_cpu_solve(
        **blocks,
        w_vo=w_vo,
        w_vv=w_vv,
        c_f12=c_f12,
        fixed_geminal_energy=fixed_e,
        solve_lambda=True,
        max_bytes=1536 << 20,
    )

    # Exact energy & delta (CCSD-F12b - MP2-F12) parity (< 1e-11 Eh)
    assert ref_state.correlation_energy is not None
    assert abs(spatial["correlation_energy"] - ref_state.correlation_energy) < 1e-11
    if low is not None:
        sp_delta = spatial["correlation_energy"] - low.mp2_f12_correlation_energy
        ref_delta = ref_state.correlation_energy - low.mp2_f12_correlation_energy
        assert abs(sp_delta - ref_delta) < 1e-11

    # Exact primal amplitude parity (< 1e-11)
    np.testing.assert_allclose(spatial["t1"], ref_state.amplitudes["t1a"], rtol=0, atol=1e-11)
    np.testing.assert_allclose(spatial["t2"], ref_state.amplitudes["t2ab"], rtol=0, atol=1e-11)

    # Exact Lambda multiplier parity (< 1e-11) across all 5 spin blocks
    for blk_name, sp_blk in spatial["uccsd_lambda_blocks"].items():
        np.testing.assert_allclose(
            sp_blk,
            ref_dual.multipliers[blk_name],
            rtol=0,
            atol=1e-11,
            err_msg=f"{label}/{corr_space} Lambda block {blk_name} mismatch",
        )

    # Exact fixed-source cotangent parity (< 1e-11) across all 17 active blocks
    assert set(spatial["fixed_weights"]) == set(ref_pullback.fixed_source_weights)
    for key, sp_w in spatial["fixed_weights"].items():
        np.testing.assert_allclose(
            sp_w,
            ref_pullback.fixed_source_weights[key],
            rtol=0,
            atol=1e-11,
            err_msg=f"{label}/{corr_space} fixed cotangent {key} mismatch",
        )

    # Exact ordinary Fock cotangent parity (< 1e-11; foo/fvv are symmetrized on the symmetric Fock space)
    for fkey, sp_key in (("foo", "bar_foo"), ("fov", "bar_fov"), ("fvv", "bar_fvv")):
        ref_f = ref_pullback.ordinary_weights[f"{fkey}_aa"] + ref_pullback.ordinary_weights[f"{fkey}_bb"]
        if fkey in ("foo", "fvv"):
            ref_f = 0.5 * (ref_f + ref_f.T)
        np.testing.assert_allclose(
            spatial[sp_key],
            ref_f,
            rtol=0,
            atol=1e-11,
            err_msg=f"{label}/{corr_space} ordinary Fock cotangent {fkey} mismatch",
        )


@pytest.mark.parametrize(
    "label,atoms_spec,elements,corr_space",
    [
        pytest.param(
            "H2O",
            (("O", (0.0, 0.0, 0.0)), ("H", (0.12, -1.43, 1.11)), ("H", (-0.05, 1.38, 1.16))),
            (1, 8),
            "all-active",
            id="H2O-AE",
        ),
        pytest.param(
            "H2O",
            (("O", (0.0, 0.0, 0.0)), ("H", (0.12, -1.43, 1.11)), ("H", (-0.05, 1.38, 1.16))),
            (1, 8),
            "frozen-core",
            id="H2O-FC",
        ),
        pytest.param(
            "HBr",
            (("Br", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.68))),
            (1, 35),
            "frozen-core",
            id="HBr-ECP-FC",
        ),
    ],
)
def test_spatial_rhf_fixed_cc_forces_matches_reference_and_streaming(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    atoms_spec: tuple[tuple[str, tuple[float, float, float]], ...],
    elements: tuple[int, ...],
    corr_space: str,
) -> None:
    """Verify spatial_rhf_fixed_cc_forces (dense & streaming) matches fixed_ucc_source_forces < 1e-11."""
    shell_lib, one_lib = f12_host_libraries()
    basis = _compact_split_basis(elements)
    plan = plan_singlepoint(
        atoms_spec,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=0,
        multiplicity=1,
        root_id="rhf",
        correlation_space_id=corr_space,
        backend="cpu_reference",
    )
    req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")

    provider_ref = MolecularFocalProvider(
        shell_library=shell_lib, one_electron_library=one_lib, budget_mib=1536
    )
    ref = _canonical_rhf_reference_for_test(provider_ref, req)
    sources = molecular_ucc_sources(provider_ref, req, include_ordinary=True, reference=ref)
    frozen = sources.fixed.frozen_occupied[0]

    w_vo, w_vv, c_f12, fixed_e, cabs_rank, _ = _build_spatial_rhf_fixed_sources(
        provider_ref, req, ref, expected_cabs_rank=sources.fixed.cabs_rank, derivatives=True
    )
    blocks = _spatial_chemists_blocks_from_mp2_panel(provider_ref, req, ref, sources, frozen)
    spatial = spatial_rhf_ccsdf12b_cpu_solve(
        **blocks,
        w_vo=w_vo,
        w_vv=w_vv,
        c_f12=c_f12,
        fixed_geminal_energy=fixed_e,
        solve_lambda=True,
        max_bytes=1536 << 20,
    )

    ref_fixed_forces = fixed_ucc_source_forces(
        provider_ref, req, sources, spatial["fixed_weights"]
    )

    # Dense non-streaming spatial fixed forces
    monkeypatch.setenv("GENERATIVEQC_F12_STREAM_PANELS", "0")
    provider_dense = MolecularFocalProvider(
        shell_library=shell_lib, one_electron_library=one_lib, budget_mib=1536
    )
    sp_dense = spatial_rhf_fixed_cc_forces(
        provider_dense,
        req,
        ref,
        cabs_rank=cabs_rank,
        fixed_geminal_energy=fixed_e,
        t1=spatial["t1"],
        t2=spatial["t2"],
        lambda1=spatial["lambda1"],
        lambda2=spatial["lambda2"],
        frozen_occupied=frozen,
    )
    np.testing.assert_allclose(sp_dense, ref_fixed_forces, rtol=0, atol=1e-11)

    # Streaming F12 AO panels spatial fixed forces
    monkeypatch.setenv("GENERATIVEQC_F12_STREAM_PANELS", "1")
    provider_stream = MolecularFocalProvider(
        shell_library=shell_lib, one_electron_library=one_lib, budget_mib=1536
    )
    sp_stream = spatial_rhf_fixed_cc_forces(
        provider_stream,
        req,
        ref,
        cabs_rank=cabs_rank,
        fixed_geminal_energy=fixed_e,
        t1=spatial["t1"],
        t2=spatial["t2"],
        lambda1=spatial["lambda1"],
        lambda2=spatial["lambda2"],
        frozen_occupied=frozen,
    )
    np.testing.assert_allclose(sp_stream, ref_fixed_forces, rtol=0, atol=1e-11)


def _basis_set_to_pyscf(basis: BasisSet) -> tuple[dict, dict]:
    """Convert a GenerativeQC BasisSet (orbital or auxiliary, with optional BSE scalar ECP) to PySCF dicts."""
    import json
    from pyscf.data.elements import ELEMENTS

    pyscf_basis: dict[str, list] = {}
    pyscf_ecp: dict[str, list] = {}
    for elem in basis.elements:
        sym = ELEMENTS[elem.atomic_number]
        shells = []
        for sh in elem.shells:
            for col in sh.coefficients:
                rows = [[float(e), float(c)] for e, c in zip(sh.exponents, col, strict=True)]
                shells.append([int(sh.angular_momentum), *rows])
        pyscf_basis[sym] = shells
        if elem.ecp_core_electrons and elem.ecp_data:
            raw_ecp = json.loads(elem.ecp_data)
            max_l = max(int(item["angular_momentum"][0]) for item in raw_ecp)
            channels = []
            for item in raw_ecp:
                l_val = int(item["angular_momentum"][0])
                ch = -1 if l_val == max_l else l_val
                r_exps = [int(r) for r in item["r_exponents"]]
                max_r = max(r_exps)
                powers: list[list[list[float]]] = [[] for _ in range(max_r + 1)]
                for r, e, c in zip(
                    r_exps, item["gaussian_exponents"], item["coefficients"][0], strict=True
                ):
                    powers[r].append([float(e), float(c)])
                channels.append([ch, powers])
            channels.sort(key=lambda x: x[0])
            pyscf_ecp[sym] = [int(elem.ecp_core_electrons), channels]
    return pyscf_basis, pyscf_ecp


@pytest.mark.parametrize(
    ("label", "atoms_spec", "elements", "corr_space"),
    [
        pytest.param(
            "H2O",
            (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.43, 1.11)), ("H", (0.0, -1.43, 1.11))),
            (1, 8),
            "all-active",
            id="H2O-AE",
        ),
        pytest.param(
            "H2O",
            (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.43, 1.11)), ("H", (0.0, -1.43, 1.11))),
            (1, 8),
            "frozen-core",
            id="H2O-FC",
        ),
        pytest.param(
            "CH4",
            (
                ("C", (0.0, 0.0, 0.0)),
                ("H", (1.19, 1.19, 1.19)),
                ("H", (-1.19, -1.19, 1.19)),
                ("H", (-1.19, 1.19, -1.19)),
                ("H", (1.19, -1.19, -1.19)),
            ),
            (1, 6),
            "frozen-core",
            id="CH4-FC",
        ),
        pytest.param(
            "HBr",
            (("Br", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.68))),
            (1, 35),
            "frozen-core",
            id="HBr-ECP-FC",
        ),
    ],
)
def test_spatial_rhf_ccsdf12b_zero_f12_matches_pyscf_df_rccsd(
    label: str,
    atoms_spec: tuple[tuple[str, tuple[float, float, float]], ...],
    elements: tuple[int, ...],
    corr_space: str,
) -> None:
    """Zero-F12 degeneration: CCSD-F12b with C=W=V+B-X=0 matches external PySCF DF-RCCSD < 1e-11 Eh."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import cc, df, gto, scf

    shell_lib, one_lib = f12_host_libraries()
    basis = _compact_split_basis(elements)
    plan = plan_singlepoint(
        atoms_spec,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=0,
        multiplicity=1,
        root_id="rhf",
        correlation_space_id=corr_space,
        backend="cpu_reference",
    )
    req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    provider = MolecularFocalProvider(
        shell_library=shell_lib, one_electron_library=one_lib, budget_mib=1536
    )
    ref = _canonical_rhf_reference_for_test(provider, req)
    sources = molecular_ucc_sources(provider, req, include_ordinary=True, reference=ref)
    frozen = sources.fixed.frozen_occupied[0]
    blocks = _spatial_chemists_blocks_from_mp2_panel(provider, req, ref, sources, frozen)

    o_act, n_vir = blocks["fov"].shape
    zero_wvo = np.zeros((o_act, o_act, n_vir, o_act), dtype=np.float64)
    zero_wvv = np.zeros((o_act, o_act, n_vir, n_vir), dtype=np.float64)
    zero_c = np.zeros((o_act, o_act, n_vir, n_vir), dtype=np.float64)
    spatial = spatial_rhf_ccsdf12b_cpu_solve(
        **blocks,
        w_vo=zero_wvo,
        w_vv=zero_wvv,
        c_f12=zero_c,
        fixed_geminal_energy=0.0,
        solve_lambda=True,
        max_bytes=1536 << 20,
    )

    pyscf_orb, pyscf_ecp = _basis_set_to_pyscf(basis.orbital)
    pyscf_jk, _ = _basis_set_to_pyscf(basis.jk_fit)
    assert basis.mp2_fit is not None
    pyscf_mp2, _ = _basis_set_to_pyscf(basis.mp2_fit)

    mol = gto.M(
        atom=list(atoms_spec),
        unit="Bohr",
        basis=pyscf_orb,
        ecp=pyscf_ecp or None,
        spin=0,
        charge=0,
        cart=False,
        verbose=0,
    )
    mf = scf.RHF(mol).density_fit(auxbasis=pyscf_jk)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    assert abs(float(mf.e_tot) - float(ref.energy)) < 1e-11, (
        f"{label} ({corr_space}) RHF mismatch vs PySCF {pyscf.__version__}: "
        f"{mf.e_tot} vs {ref.energy}"
    )

    mycc = cc.RCCSD(mf, frozen=frozen if frozen > 0 else None).density_fit(
        auxbasis=pyscf_mp2
    )
    mycc.conv_tol = 1e-13
    mycc.conv_tol_normt = 1e-12
    mycc.max_cycle = 200
    mycc.kernel()
    assert mycc.converged
    e_diff = abs(float(spatial["correlation_energy"]) - float(mycc.e_corr))
    etot_diff = abs(
        (float(ref.energy) + float(spatial["correlation_energy"])) - float(mycc.e_tot)
    )
    print(
        f"[PYSCF {pyscf.__version__} ZERO-F12] {label} ({corr_space}): "
        f"e_corr_diff={e_diff:.3e} Eh, e_tot_diff={etot_diff:.3e} Eh"
    )
    assert e_diff < 1e-11, (
        f"{label} ({corr_space}) zero-F12 DF-RCCSD e_corr mismatch vs PySCF "
        f"{pyscf.__version__}: diff={e_diff:.3e} Eh "
        f"(spatial={spatial['correlation_energy']}, pyscf={mycc.e_corr})"
    )
    assert etot_diff < 1e-11

    mycc.solve_lambda()
    assert mycc.converged_lambda
    assert abs(np.linalg.norm(spatial["t1"]) - np.linalg.norm(mycc.t1)) < 1e-11
    assert abs(np.linalg.norm(spatial["t2"]) - np.linalg.norm(mycc.t2)) < 1e-11
    # Exact spin-adapted Lambda correspondence:
    # GenerativeQC solves for lambda1 = 2 * l1_pyscf and lambda2 = 2 * l2_pyscf - P_{ab} l2_pyscf
    # (equivalently PySCF's l2 = L2_ab = (2 * lambda2 + P_{ab} lambda2) / 3).
    pyscf_l2_in_gqc_basis = 2.0 * mycc.l2 - mycc.l2.transpose(0, 1, 3, 2)
    l1_diff = abs(np.linalg.norm(spatial["lambda1"]) - 2.0 * np.linalg.norm(mycc.l1))
    l2_diff = abs(np.linalg.norm(spatial["lambda2"]) - np.linalg.norm(pyscf_l2_in_gqc_basis))
    print(
        f"  t1_diff={abs(np.linalg.norm(spatial['t1']) - np.linalg.norm(mycc.t1)):.3e}, "
        f"t2_diff={abs(np.linalg.norm(spatial['t2']) - np.linalg.norm(mycc.t2)):.3e}, "
        f"l1_diff={l1_diff:.3e}, l2_diff={l2_diff:.3e}"
    )
    assert l1_diff < 1e-11, f"lambda1 norm mismatch: {l1_diff:.3e}"
    assert l2_diff < 1e-11, f"lambda2 norm mismatch: {l2_diff:.3e}"
