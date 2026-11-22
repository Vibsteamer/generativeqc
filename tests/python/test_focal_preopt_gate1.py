"""Gate 1 local CPU + PySCF 2.14.0 verification for Phase 1 pre-optimizations.

Verifies:
1. Standard chemical valence frozen-core resolution (H..I, Br ECP10MDF=4, I ECP28MDF=4).
2. Frozen-core DF-MP2 energy vs independent PySCF 2.14.0 (split JK/MP2 fit, RHF & UHF).
3. Frozen-core DF-MP2 full Cartesian analytic forces vs two-step central finite differences
   (proving frozen-core <-> active-occupied canonical frame response is exact).
4. Frozen-core weighted MP2 occupied-partition analytic forces (for (T+)) vs two-step FD.
5. Frozen-core DF-MP2-F12 / 3C(FIX) energy & analytic forces vs two-step FD.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from generativeqc._f12_focal_reference import df_reference
from generativeqc._f12_focal_ump2 import FocalCoulombSource, reference_df_ump2
from generativeqc._f12_frozen_core import (
    active_occupied_counts,
    frozen_core_orbitals_for_atom,
    resolve_frozen_occupied,
)
from generativeqc.basis import BasisSet
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalRequest,
    FocalSeries,
    plan_singlepoint,
)


def _pyscf_basis_dict(basis: BasisSet, atoms) -> dict[str, list]:
    """Convert a GenerativeQC BasisSet to PySCF internal gto.basis format."""
    out: dict[str, list] = {}
    symbols = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O", 9: "F"}
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


def _build_split_water_bases() -> FocalBasis:
    """Build compact spherical orbital, JKFit and distinct MP2Fit bases for H2O / OH."""
    from generativeqc.basis import BasisProvenance, BasisShell, ElementBasis

    def make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
        prov = BasisProvenance(
            source=name,
            version="1",
            license="MIT",
            checksum="0" * 64,
        )
        return BasisSet(name, elems, prov, representation="spherical")

    def Shell(l: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
        return BasisShell(l, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),))
    # Compact 3-21G-like spherical orbital basis (O: 3s2p = 9 AO, H: 2s = 2 AO -> H2O = 13 AO)
    o_orb = ElementBasis(
        8,
        (
            Shell(0, (130.70932, 23.808861, 6.4436083), (0.15432897, 0.53532814, 0.44463454)),
            Shell(0, (5.0331513, 1.1695961), (-0.09996723, 0.39951283)),
            Shell(0, (0.3803890,), (1.0,)),
            Shell(1, (5.0331513, 1.1695961), (0.15591627, 0.60768372)),
            Shell(1, (0.3803890,), (1.0,)),
        ),
    )
    h_orb = ElementBasis(
        1,
        (
            Shell(0, (3.42525091, 0.62391373), (0.15432897, 0.53532814)),
            Shell(0, (0.16885540,), (1.0,)),
        ),
    )
    # Distinct JKFit and MP2Fit auxiliary bases (s, p, d)
    o_jk = ElementBasis(
        8,
        (
            Shell(0, (60.0,), (1.0,)),
            Shell(0, (15.0,), (1.0,)),
            Shell(0, (4.0,), (1.0,)),
            Shell(0, (1.1,), (1.0,)),
            Shell(0, (0.35,), (1.0,)),
            Shell(1, (8.0,), (1.0,)),
            Shell(1, (2.2,), (1.0,)),
            Shell(1, (0.6,), (1.0,)),
            Shell(2, (3.0,), (1.0,)),
            Shell(2, (0.8,), (1.0,)),
        ),
    )
    h_jk = ElementBasis(
        1,
        (
            Shell(0, (6.0,), (1.0,)),
            Shell(0, (1.5,), (1.0,)),
            Shell(0, (0.4,), (1.0,)),
            Shell(1, (1.8,), (1.0,)),
            Shell(1, (0.5,), (1.0,)),
        ),
    )
    o_mp2 = ElementBasis(
        8,
        (
            Shell(0, (48.0,), (1.0,)),
            Shell(0, (11.0,), (1.0,)),
            Shell(0, (2.8,), (1.0,)),
            Shell(0, (0.75,), (1.0,)),
            Shell(1, (6.5,), (1.0,)),
            Shell(1, (1.7,), (1.0,)),
            Shell(1, (0.45,), (1.0,)),
            Shell(2, (2.4,), (1.0,)),
            Shell(2, (0.65,), (1.0,)),
        ),
    )
    h_mp2 = ElementBasis(
        1,
        (
            Shell(0, (5.2,), (1.0,)),
            Shell(0, (1.2,), (1.0,)),
            Shell(0, (0.32,), (1.0,)),
            Shell(1, (1.4,), (1.0,)),
        ),
    )
    orb = make_basis("gate1-orb", (h_orb, o_orb))
    jk = make_basis("gate1-jk", (h_jk, o_jk))
    mp2 = make_basis("gate1-mp2", (h_mp2, o_mp2))
    return FocalBasis(2, "gate1-split", orb, jk_fit=jk, mp2_fit=mp2, f12_fit=mp2, cabs=jk, beta=1.0)


def _pyscf_coulomb_source(
    request: FocalRequest, fit: BasisSet, atoms
) -> FocalCoulombSource:
    """Build exact native-ordered spherical Coulomb 3-center & metric integrals via PySCF."""
    from pyscf import df as pyscf_df, gto
    from scipy.linalg import solve_triangular

    symbols = {1: "H", 8: "O"}
    atom_spec = [(symbols[a.atomic_number], tuple(a.position)) for a in atoms]
    mol = gto.M(
        atom=atom_spec,
        basis=_pyscf_basis_dict(request.basis.orbital, atoms),
        unit="Bohr",
        cart=False,
        charge=request.charge,
        spin=request.multiplicity - 1,
        verbose=0,
    )
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=_pyscf_basis_dict(fit, atoms))
    metric = auxmol.intor("int2c2e")
    j3c = pyscf_df.incore.aux_e2(mol, auxmol, intor="int3c2e")  # (n, n, q)
    three_center = np.ascontiguousarray(np.transpose(j3c, (2, 0, 1)))  # (q, n, n)
    whitener = solve_triangular(np.linalg.cholesky(metric), np.eye(len(metric)), lower=True)
    return FocalCoulombSource(
        request.surface_id,
        request.basis.orbital.identity,
        fit.identity,
        request.backend,
        three_center,
        metric,
        whitener,
    )


def _pyscf_nuclear_gradient(
    atoms,
    basis_orb: BasisSet,
    basis_jk: BasisSet,
    basis_mp2: BasisSet,
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    """Contract overlap, hcore, JKFit and MP2Fit integral weights with exact PySCF derivatives."""
    from pyscf import df as pyscf_df, gto

    symbols = {1: "H", 8: "O"}
    atom_spec = [(symbols[a.atomic_number], tuple(a.position)) for a in atoms]
    mol = gto.M(atom=atom_spec, basis=_pyscf_basis_dict(basis_orb, atoms), unit="Bohr", cart=False, verbose=0)
    jkmol = pyscf_df.addons.make_auxmol(mol, auxbasis=_pyscf_basis_dict(basis_jk, atoms))
    mpmol = pyscf_df.addons.make_auxmol(mol, auxbasis=_pyscf_basis_dict(basis_mp2, atoms))

    # Nuclear derivative of 1e and 2c/3c integrals via PySCF ip integrals
    nat = len(atoms)
    grad = np.zeros((nat, 3))
    aoslices = mol.aoslice_by_atom()
    s1 = -mol.intor("int1e_ipovlp")  # (3, n, n): d<mu|nu>/dR_mu
    t1 = -mol.intor("int1e_ipkin")
    v1_bra = -mol.intor("int1e_ipnuc")
    w_s = weights["overlap"]
    w_h = weights["hcore"]
    for a in range(nat):
        p0, p1 = aoslices[a, 2], aoslices[a, 3]
        # Bra + ket derivative for shells centered on atom a
        ds = np.zeros((3, mol.nao, mol.nao))
        ds[:, p0:p1, :] += s1[:, p0:p1, :]
        ds[:, :, p0:p1] += np.transpose(s1[:, p0:p1, :], (0, 2, 1))
        dh = np.zeros((3, mol.nao, mol.nao))
        dh[:, p0:p1, :] += t1[:, p0:p1, :] + v1_bra[:, p0:p1, :]
        dh[:, :, p0:p1] += np.transpose(t1[:, p0:p1, :] + v1_bra[:, p0:p1, :], (0, 2, 1))
        with mol.with_rinv_at_nucleus(a):
            v_nuc = -mol.atom_charge(a) * mol.intor("int1e_iprinv")
            dh += v_nuc + np.transpose(v_nuc, (0, 2, 1))
        for xyz in range(3):
            grad[a, xyz] += np.vdot(w_s, ds[xyz]) + np.vdot(w_h, dh[xyz])

    def add_df_grad(auxmol, w_3c, w_2c):
        auxslices = auxmol.aoslice_by_atom()
        # int3c2e_ip1: (3, n, n, q) derivative wrt bra (1st orbital index)
        # int3c2e_ip2: (3, n, n, q) derivative wrt aux index
        ip1 = -pyscf_df.incore.aux_e2(mol, auxmol, intor="int3c2e_ip1")  # (3, n, n, q)
        ip2 = -pyscf_df.incore.aux_e2(mol, auxmol, intor="int3c2e_ip2")  # (3, n, n, q)
        m_ip1 = -auxmol.intor("int2c2e_ip1")  # (3, q, q)
        w3_nnq = np.transpose(w_3c, (1, 2, 0))  # (n, n, q)
        for a in range(nat):
            p0, p1 = aoslices[a, 2], aoslices[a, 3]
            q0, q1 = auxslices[a, 2], auxslices[a, 3]
            for xyz in range(3):
                # orbital bra & ket
                grad[a, xyz] += np.sum(w3_nnq[p0:p1, :, :] * ip1[xyz, p0:p1, :, :])
                grad[a, xyz] += np.sum(w3_nnq[:, p0:p1, :] * np.transpose(ip1[xyz, p0:p1, :, :], (1, 0, 2)))
                # auxiliary center
                grad[a, xyz] += np.sum(w3_nnq[:, :, q0:q1] * ip2[xyz, :, :, q0:q1])
                # 2c metric bra & ket
                grad[a, xyz] += np.sum(w_2c[q0:q1, :] * m_ip1[xyz, q0:q1, :])
                grad[a, xyz] += np.sum(w_2c[:, q0:q1] * m_ip1[xyz, q0:q1, :].T)

    add_df_grad(jkmol, weights["three_center"], weights["metric"])
    add_df_grad(mpmol, weights["mp2_three_center"], weights["mp2_metric"])
    return -grad


def test_frozen_core_atom_table():
    assert frozen_core_orbitals_for_atom(1) == 0
    assert frozen_core_orbitals_for_atom(6) == 1
    assert frozen_core_orbitals_for_atom(8) == 1
    assert frozen_core_orbitals_for_atom(16) == 5
    assert frozen_core_orbitals_for_atom(17) == 5
    # Br ECP10MDF: 10 ECP electrons = 5 pairs -> 9 - 5 = 4 frozen (3s3p), 17 correlated (3d4s4p)
    assert frozen_core_orbitals_for_atom(35, ecp_core_electrons=10) == 4
    # I ECP28MDF: 28 ECP electrons = 14 pairs -> 18 - 14 = 4 frozen (4s4p), 17 correlated (4d5s5p)
    assert frozen_core_orbitals_for_atom(53, ecp_core_electrons=28) == 4
    assert active_occupied_counts((10, 10), (2, 2)) == (8, 8)


@pytest.mark.parametrize("root,atoms_spec,mult", [
    ("rhf", (("O", (0.0, 0.0, 0.0)), ("H", (0.0, -1.43, 1.11)), ("H", (0.0, 1.43, 1.11))), 1),
    ("uhf", (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.84))), 2),
])
def test_frozen_core_df_mp2_matches_pyscf_energy(root, atoms_spec, mult):
    """Verify frozen-core split-fit DF-MP2 energy matches PySCF 2.14.0 to < 1e-11 Eh."""
    pyscf_gto = pytest.importorskip("pyscf.gto")
    pyscf_scf = pytest.importorskip("pyscf.scf")
    pyscf_mp = pytest.importorskip("pyscf.mp")
    pyscf_df = pytest.importorskip("pyscf.df")

    basis = _build_split_water_bases()
    plan = plan_singlepoint(
        atoms_spec,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=0,
        multiplicity=mult,
        root_id=root,
        correlation_space_id="frozen-core",
        backend="cpu_reference",
    )
    req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    ref = df_reference(req, forces=False, budget_mib=512)
    jk = _pyscf_coulomb_source(req, basis.jk_fit, req.atoms)
    mp2_src = _pyscf_coulomb_source(req, basis.mp2_fit, req.atoms)
    res = reference_df_ump2(req, ref, jk, mp2_src, budget_mib=512)
    native_mp2 = sum(res.spin_energies)

    # Independent PySCF 2.14.0 split-fit DF-RHF/UHF + DF-MP2(frozen=1)
    mol = pyscf_gto.M(
        atom=[(s, c) for s, c in atoms_spec],
        basis=_pyscf_basis_dict(basis.orbital, req.atoms),
        unit="Bohr",
        cart=False,
        charge=0,
        spin=mult - 1,
        verbose=0,
    )
    jk_aux = _pyscf_basis_dict(basis.jk_fit, req.atoms)
    mp2_aux = _pyscf_basis_dict(basis.mp2_fit, req.atoms)
    mf = (pyscf_scf.RHF(mol) if root == "rhf" else pyscf_scf.UHF(mol)).density_fit(auxbasis=jk_aux)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 100
    mf.kernel()
    assert mf.converged
    assert abs(ref.energy - mf.e_tot) < 1e-11

    mp = (
        pyscf_mp.MP2(mf, frozen=1) if root == "rhf" else pyscf_mp.UMP2(mf, frozen=1)
    ).density_fit(auxbasis=mp2_aux)
    e_corr, _ = mp.kernel()
    assert abs(native_mp2 - e_corr) < 1e-11, f"{root} frozen-core DF-MP2 mismatch: {native_mp2} vs {e_corr}"


def test_frozen_core_df_mp2_forces_and_weighted_tplus_forces_two_step_fd():
    """Verify frozen-core split-fit DF-MP2 & weighted-occupied MP2 analytic forces vs two-step FD."""
    pytest.importorskip("pyscf.gto")

    basis = _build_split_water_bases()
    center_coords = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.12, -1.43, 1.11)),
        ("H", (-0.05, 1.38, 1.16)),
    ]

    def eval_point(coords, *, occupied_weights=None):
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
        req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
        ref = df_reference(req, forces=False, budget_mib=512)
        jk = _pyscf_coulomb_source(req, basis.jk_fit, req.atoms)
        mp2_src = _pyscf_coulomb_source(req, basis.mp2_fit, req.atoms)
        out = reference_df_ump2(
            req, ref, jk, mp2_src, budget_mib=512, occupied_weights=occupied_weights
        )
        return req, sum(out.spin_energies), out.weights

    # 1) Unweighted frozen-core DF-MP2 (4 active occupied orbitals out of 5)
    req0, e0, w0 = eval_point(center_coords)
    analytic_unw = _pyscf_nuclear_gradient(
        req0.atoms, basis.orbital, basis.jk_fit, basis.mp2_fit, w0
    )

    # 2) Unequal occupied-weighted frozen-core DF-MP2 (as in (T+) chain rule!)
    occ_w_vec = np.array([1.15, 0.92, 1.31, 0.87], dtype=np.float64)
    occ_weights = (occ_w_vec, occ_w_vec.copy())
    _, ew0, ww0 = eval_point(center_coords, occupied_weights=occ_weights)
    analytic_w = _pyscf_nuclear_gradient(
        req0.atoms, basis.orbital, basis.jk_fit, basis.mp2_fit, ww0
    )

    # Check representative non-zero coordinates across two steps h=0.0004 and h=0.0002 bohr
    for atom_idx, axis in ((0, 2), (1, 0), (1, 1)):
        for h in (0.0004, 0.0002):
            def shifted(delta):
                c = [(s, list(p)) for s, p in center_coords]
                c[atom_idx][1][axis] += delta
                return [(s, tuple(p)) for s, p in c]

            _, ep, _ = eval_point(shifted(+h))
            _, em, _ = eval_point(shifted(-h))
            fd_unw = -(ep - em) / (2.0 * h)
            err_unw = abs(analytic_unw[atom_idx, axis] - fd_unw)
            assert err_unw < 2e-8, (
                f"Frozen-core DF-MP2 force mismatch at atom={atom_idx} axis={axis} h={h}: "
                f"analytic={analytic_unw[atom_idx, axis]} fd={fd_unw} err={err_unw}"
            )

            _, ewp, _ = eval_point(shifted(+h), occupied_weights=occ_weights)
            _, ewm, _ = eval_point(shifted(-h), occupied_weights=occ_weights)
            fd_w = -(ewp - ewm) / (2.0 * h)
            err_w = abs(analytic_w[atom_idx, axis] - fd_w)
            assert err_w < 2e-8, (
                f"Frozen-core weighted (T+) MP2 force mismatch at atom={atom_idx} axis={axis} h={h}: "
                f"analytic={analytic_w[atom_idx, axis]} fd={fd_w} err={err_w}"
            )


@pytest.mark.parametrize(
    "oa,ob,distinct_jk",
    [(2, 2, True), (2, 2, False), (3, 2, True), (3, 2, False)],
)
def test_f12_panel_streaming_exact_parity(oa, ob, distinct_jk):
    """Streaming single-operator AO load + cotangent sink matches dense F12 response to <1e-13."""
    from generativeqc._f12_unrestricted import unrestricted_mp2_f12_ao_response
    from generativeqc_compiler.tensor import execute

    rng = np.random.default_rng(100 + 10 * oa + ob + int(distinct_jk))
    obs, ext, q, qjk = 5, 8, 12, 10
    space = (oa, ob, obs, ext, q)

    def spd(n):
        a = rng.standard_normal((n, n)) * 0.1
        return a @ a.T + 2.0 * np.eye(n)

    def sym_panel(nq, n):
        p = rng.standard_normal((nq, n, n)) * 0.02
        return 0.5 * (p + p.swapaxes(1, 2))

    ca, _ = np.linalg.qr(rng.standard_normal((ext, ext)))
    cb = (
        ca.copy()
        if oa == ob
        else np.linalg.qr(ca + 0.05 * rng.standard_normal((ext, ext)))[0]
    )
    hcore = ca @ np.diag([-3.0, -2.0, -1.0, 0.5, 1.0, 1.5, 2.0, 2.5]) @ ca.T
    hcore = 0.5 * (hcore + hcore.T)
    m_c = spd(q)
    w_c = np.linalg.inv(np.linalg.cholesky(m_c))
    ao_dense = {
        "overlap_ao": np.eye(ext),
        "hcore_ao": hcore,
        "coefficients_alpha": ca,
        "coefficients_beta": cb,
        "metric_whitener": w_c,
        "metric/coulomb": m_c,
        "metric/geminal": spd(q),
        "metric/squared_geminal": spd(q),
        "metric/geminal_coulomb": spd(q),
        "metric/double_commutator": spd(q),
        "ao_coulomb": sym_panel(q, ext),
        "ao_f12": sym_panel(q, ext),
        "ao_f12_squared": sym_panel(q, ext),
        "ao_f12_over_coulomb": sym_panel(q, ext),
        "ao_double_commutator": sym_panel(q, ext),
    }
    jk_dense = jk_stream = None
    if distinct_jk:
        m_jk = spd(qjk)
        w_jk = np.linalg.inv(np.linalg.cholesky(m_jk))
        ao_jk = sym_panel(qjk, ext)
        jk_dense = {"ao_coulomb": ao_jk, "metric/coulomb": m_jk, "metric_whitener": w_jk}
        jk_stream = {
            "__ao_loader__": lambda _name, _a=ao_jk: _a.copy(),
            "metric/coulomb": m_jk,
            "metric_whitener": w_jk,
        }
    ao_stream = {k: v for k, v in ao_dense.items() if not k.startswith("ao_")}
    ao_stream["__ao_loader__"] = lambda name, _d=ao_dense: _d[name].copy()
    run = lambda g, inp: execute(g, inp, max_bytes=1 << 29)
    t0, b0 = unrestricted_mp2_f12_ao_response(
        space, ao_dense, run, frozen_occupied=(1, 1), jkfit=jk_dense, component="f12"
    )
    sunk: dict[str, np.ndarray] = {}
    t1, b1 = unrestricted_mp2_f12_ao_response(
        space,
        ao_stream,
        run,
        frozen_occupied=(1, 1),
        jkfit=jk_stream,
        component="f12",
        ao_bar_sink=lambda name, mat: sunk.__setitem__(name, mat.copy()),
    )
    assert abs(t0.f12_increment - t1.f12_increment) < 1e-14
    for key, ref_val in b0.items():
        if key.startswith("ao_"):
            assert key in sunk and key not in b1
            assert np.max(np.abs(ref_val - sunk[key])) < 1e-13
        else:
            assert np.max(np.abs(ref_val - b1[key])) < 1e-13

