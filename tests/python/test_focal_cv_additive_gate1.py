"""Gate 1 local CPU + PySCF 2.14.0 verification for cv-active & additive Delta-CV.

Verifies:
1. cv_active_frozen_orbitals_for_atom and resolve_frozen_occupied('cv-active') across
   Z=1..54 including Na..Cl (freeze 1 = deep 1s, correlate 2s2p + valence),
   Br ECP10MDF (freeze 0, releasing 3s3p relative to FC=4), and
   I ECP28MDF (freeze 0, releasing 4s4p relative to FC=4).
2. cv-active & frozen-core split-fit DF-MP2 energies and additive Delta-CV vs
   independent PySCF 2.14.0 (< 1e-11 Eh) on H2O (RHF), OH (UHF), and HCl (RHF,
   explicitly asserting Cl freezes 1 orbital = 1s and correlates 2s2p + valence).
3. cv-active DF-MP2 & additive Delta-CV full Cartesian analytic forces vs two-step
   central finite differences (< 2e-8 Eh/bohr) and net-force/torque invariants
   (< 1e-12) on H2O, OH, and HCl.
4. Exact parity of single-pass cv-active/frozen-core F12 pair accumulation and
   shared-reference Delta-CV validation helpers.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import f12_host_libraries

from generativeqc._f12_focal_cv import _paired_cv_fc_requests
from generativeqc._f12_focal_reference import df_reference
from generativeqc._f12_focal_ump2 import FocalCoulombSource, reference_df_ump2
from generativeqc._f12_frozen_core import (
    active_occupied_counts,
    cv_active_frozen_orbitals_for_atom,
    frozen_core_orbitals_for_atom,
    frozen_core_spd_orbitals_for_atom,
    resolve_frozen_occupied,
)
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalRequest,
    FocalSeries,
    plan_singlepoint,
)

_SYM_BY_Z = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 14: "Si", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def _pyscf_basis_dict(basis: BasisSet, atoms) -> dict[str, list]:
    """Convert a GenerativeQC BasisSet to PySCF internal gto.basis format."""
    out: dict[str, list] = {}
    for atom in atoms:
        sym = _SYM_BY_Z[atom.atomic_number]
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


def _make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
    prov = BasisProvenance(source=name, version="1", license="MIT", checksum="0" * 64)
    return BasisSet(name, elems, prov, representation="spherical")


def _shell(ell: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
    return BasisShell(ell, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),))


def test_delta_cv_forward_and_analytic_use_same_ordinary_denominators():
    from generativeqc.experimental.f12_focal import MolecularFocalProvider

    def lih_basis(name, exponents):
        shells = tuple(_shell(0, (exponent,), (1.0,)) for exponent in exponents)
        return _make_basis(name, (ElementBasis(1, shells), ElementBasis(3, shells)))

    # Two close JK exponents amplify the difference between the reconstructed
    # extended Fock diagonal and the canonical reference orbital energies.
    jk = lih_basis("JK", (5.0, 1.1, 0.28, 0.28 * 1.00097, 0.07))
    ri = lih_basis("RI", (6.0, 1.2, 0.30, 0.08))
    basis = FocalBasis(
        2, "LiH", lih_basis("OBS", (1.5, 0.32)), jk_fit=jk,
        mp2_fit=ri, f12_fit=ri, cabs=lih_basis("CABS", (0.09,)), beta=0.9,
    )
    series = FocalSeries(basis)
    plan = plan_singlepoint(
        [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 4.0))],
        series, series, None, root_id="rhf", correlation_space_id="cv-active",
        backend="cpu_reference",
    )
    request = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    shell_library, one_electron_library = f12_host_libraries()
    provider = MolecularFocalProvider(
        shell_library=shell_library, one_electron_library=one_electron_library,
    )
    center = provider.delta_cv_energy_and_forces(request)
    forward = provider.delta_cv_energy(request, expected_cabs_rank=center.cabs_rank)
    assert center.cv_frozen == forward.cv_frozen == (0, 0)
    assert center.fc_frozen == forward.fc_frozen == (1, 1)
    for field in (
        "cv_ordinary_mp2_Eh", "fc_ordinary_mp2_Eh", "delta_ordinary_mp2_Eh",
        "delta_f12_increment_Eh", "delta_mp2_f12_Eh",
    ):
        assert getattr(forward, field) == pytest.approx(
            getattr(center, field), abs=2e-13, rel=0,
        )


def _build_split_gate1_bases() -> FocalBasis:
    """Compact spherical orbital, JKFit and distinct MP2Fit bases for H, O, and Cl."""
    h_orb = ElementBasis(
        1,
        (
            _shell(0, (3.42525091, 0.62391373), (0.15432897, 0.53532814)),
            _shell(0, (0.16885540,), (1.0,)),
        ),
    )
    o_orb = ElementBasis(
        8,
        (
            _shell(0, (130.70932, 23.808861, 6.4436083), (0.15432897, 0.53532814, 0.44463454)),
            _shell(0, (5.0331513, 1.1695961), (-0.09996723, 0.39951283)),
            _shell(0, (0.3803890,), (1.0,)),
            _shell(1, (5.0331513, 1.1695961), (0.15591627, 0.60768372)),
            _shell(1, (0.3803890,), (1.0,)),
        ),
    )
    # 3-21G-like spherical basis for Cl (4s3p + 1 tight d for 2s2p CV correlation = 18 AO)
    cl_orb = ElementBasis(
        17,
        (
            _shell(0, (662.424, 100.123, 21.8495), (0.0609777, 0.362709, 0.690138)),
            _shell(0, (43.3948, 12.1924), (-0.237435, 1.12361)),
            _shell(0, (2.29747, 0.720496), (-0.431068, 1.24682)),
            _shell(0, (0.209640,), (1.0,)),
            _shell(1, (43.3948, 12.1924), (0.157802, 0.909804)),
            _shell(1, (2.29747, 0.720496), (0.292731, 0.785240)),
            _shell(1, (0.209640,), (1.0,)),
            _shell(2, (1.85,), (1.0,)),
        ),
    )
    h_jk = ElementBasis(
        1,
        (
            _shell(0, (6.0,), (1.0,)),
            _shell(0, (1.5,), (1.0,)),
            _shell(0, (0.4,), (1.0,)),
            _shell(1, (1.8,), (1.0,)),
            _shell(1, (0.5,), (1.0,)),
        ),
    )
    o_jk = ElementBasis(
        8,
        (
            _shell(0, (60.0,), (1.0,)),
            _shell(0, (15.0,), (1.0,)),
            _shell(0, (4.0,), (1.0,)),
            _shell(0, (1.1,), (1.0,)),
            _shell(0, (0.35,), (1.0,)),
            _shell(1, (8.0,), (1.0,)),
            _shell(1, (2.2,), (1.0,)),
            _shell(1, (0.6,), (1.0,)),
            _shell(2, (3.0,), (1.0,)),
            _shell(2, (0.8,), (1.0,)),
        ),
    )
    cl_jk = ElementBasis(
        17,
        (
            _shell(0, (180.0,), (1.0,)),
            _shell(0, (42.0,), (1.0,)),
            _shell(0, (11.0,), (1.0,)),
            _shell(0, (3.0,), (1.0,)),
            _shell(0, (0.85,), (1.0,)),
            _shell(0, (0.25,), (1.0,)),
            _shell(1, (24.0,), (1.0,)),
            _shell(1, (6.5,), (1.0,)),
            _shell(1, (1.8,), (1.0,)),
            _shell(1, (0.5,), (1.0,)),
            _shell(2, (9.0,), (1.0,)),
            _shell(2, (2.4,), (1.0,)),
            _shell(2, (0.65,), (1.0,)),
        ),
    )
    h_mp2 = ElementBasis(
        1,
        (
            _shell(0, (5.2,), (1.0,)),
            _shell(0, (1.2,), (1.0,)),
            _shell(0, (0.32,), (1.0,)),
            _shell(1, (1.4,), (1.0,)),
        ),
    )
    o_mp2 = ElementBasis(
        8,
        (
            _shell(0, (48.0,), (1.0,)),
            _shell(0, (11.0,), (1.0,)),
            _shell(0, (2.8,), (1.0,)),
            _shell(0, (0.75,), (1.0,)),
            _shell(1, (6.5,), (1.0,)),
            _shell(1, (1.7,), (1.0,)),
            _shell(1, (0.45,), (1.0,)),
            _shell(2, (2.4,), (1.0,)),
            _shell(2, (0.65,), (1.0,)),
        ),
    )
    cl_mp2 = ElementBasis(
        17,
        (
            _shell(0, (140.0,), (1.0,)),
            _shell(0, (32.0,), (1.0,)),
            _shell(0, (8.5,), (1.0,)),
            _shell(0, (2.3,), (1.0,)),
            _shell(0, (0.65,), (1.0,)),
            _shell(1, (18.0,), (1.0,)),
            _shell(1, (4.8,), (1.0,)),
            _shell(1, (1.3,), (1.0,)),
            _shell(1, (0.38,), (1.0,)),
            _shell(2, (7.0,), (1.0,)),
            _shell(2, (1.9,), (1.0,)),
            _shell(2, (0.52,), (1.0,)),
            _shell(3, (3.2,), (1.0,)),
        ),
    )
    orb = _make_basis("gate1-cv-orb", (h_orb, o_orb, cl_orb))
    jk = _make_basis("gate1-cv-jk", (h_jk, o_jk, cl_jk))
    mp2 = _make_basis("gate1-cv-mp2", (h_mp2, o_mp2, cl_mp2))
    return FocalBasis(2, "gate1-cv-split", orb, jk_fit=jk, mp2_fit=mp2, f12_fit=mp2, cabs=jk, beta=1.4)


def _pyscf_coulomb_source(request: FocalRequest, fit: BasisSet, atoms) -> FocalCoulombSource:
    from pyscf import df as pyscf_df, gto
    from scipy.linalg import solve_triangular

    atom_spec = [(_SYM_BY_Z[a.atomic_number], tuple(a.position)) for a in atoms]
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
    j3c = pyscf_df.incore.aux_e2(mol, auxmol, intor="int3c2e")
    three_center = np.ascontiguousarray(np.transpose(j3c, (2, 0, 1)))
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
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> np.ndarray:
    """Contract overlap, hcore, JKFit and MP2Fit integral weights with exact PySCF derivatives."""
    from pyscf import df as pyscf_df, gto

    atom_spec = [(_SYM_BY_Z[a.atomic_number], tuple(a.position)) for a in atoms]
    mol = gto.M(
        atom=atom_spec,
        basis=_pyscf_basis_dict(basis_orb, atoms),
        unit="Bohr",
        cart=False,
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    jkmol = pyscf_df.addons.make_auxmol(mol, auxbasis=_pyscf_basis_dict(basis_jk, atoms))
    mpmol = pyscf_df.addons.make_auxmol(mol, auxbasis=_pyscf_basis_dict(basis_mp2, atoms))

    nat = len(atoms)
    grad = np.zeros((nat, 3))
    aoslices = mol.aoslice_by_atom()
    s1 = -mol.intor("int1e_ipovlp")
    t1 = -mol.intor("int1e_ipkin")
    v1_bra = -mol.intor("int1e_ipnuc")
    w_s = weights["overlap"]
    w_h = weights["hcore"]
    for a in range(nat):
        p0, p1 = aoslices[a, 2], aoslices[a, 3]
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
        ip1 = -pyscf_df.incore.aux_e2(mol, auxmol, intor="int3c2e_ip1")
        ip2 = -pyscf_df.incore.aux_e2(mol, auxmol, intor="int3c2e_ip2")
        m_ip1 = -auxmol.intor("int2c2e_ip1")
        w3_nnq = np.transpose(w_3c, (1, 2, 0))
        for a in range(nat):
            p0, p1 = aoslices[a, 2], aoslices[a, 3]
            q0, q1 = auxslices[a, 2], auxslices[a, 3]
            for xyz in range(3):
                grad[a, xyz] += np.sum(w3_nnq[p0:p1, :, :] * ip1[xyz, p0:p1, :, :])
                grad[a, xyz] += np.sum(w3_nnq[:, p0:p1, :] * np.transpose(ip1[xyz, p0:p1, :, :], (1, 0, 2)))
                grad[a, xyz] += np.sum(w3_nnq[:, :, q0:q1] * ip2[xyz, :, :, q0:q1])
                grad[a, xyz] += np.sum(w_2c[q0:q1, :] * m_ip1[xyz, q0:q1, :])
                grad[a, xyz] += np.sum(w_2c[:, q0:q1] * m_ip1[xyz, q0:q1, :].T)

    add_df_grad(jkmol, weights["three_center"], weights["metric"])
    add_df_grad(mpmol, weights["mp2_three_center"], weights["mp2_metric"])
    return -grad


def test_cv_active_atom_table_and_resolution():
    """Verify exact cv-active deep-core frozen counts across Z=1..54 and ECP halogens."""
    # Z <= 10: correlate all electrons incl. 1s
    for z in (1, 2, 3, 5, 6, 7, 8, 9, 10):
        assert cv_active_frozen_orbitals_for_atom(z) == 0

    # 11 <= Z <= 18 (Na..Ar): keep deep 1s frozen (1 orbital), correlate 2s2p + valence!
    for z in (11, 12, 13, 14, 15, 16, 17, 18):
        assert cv_active_frozen_orbitals_for_atom(z) == 1
        assert frozen_core_orbitals_for_atom(z) == (1 if z == 11 else 5)

    # 19 <= Z <= 36 (Br): deep 1s2s2p = 5 pairs -> 0 for ECP10MDF (releasing 3s3p = 4 orbitals vs FC)
    assert cv_active_frozen_orbitals_for_atom(35, ecp_core_electrons=0) == 5
    assert cv_active_frozen_orbitals_for_atom(35, ecp_core_electrons=10) == 0
    assert frozen_core_orbitals_for_atom(35, ecp_core_electrons=10) - cv_active_frozen_orbitals_for_atom(35, ecp_core_electrons=10) == 4

    # 37 <= Z <= 54 (I): deep 1s..3d = 14 pairs -> 0 for ECP28MDF (releasing 4s4p = 4 orbitals vs FC)
    assert cv_active_frozen_orbitals_for_atom(53, ecp_core_electrons=0) == 14
    assert cv_active_frozen_orbitals_for_atom(53, ecp_core_electrons=28) == 0
    assert frozen_core_orbitals_for_atom(53, ecp_core_electrons=28) - cv_active_frozen_orbitals_for_atom(53, ecp_core_electrons=28) == 4

    with pytest.raises(ValueError):
        cv_active_frozen_orbitals_for_atom(0)
    with pytest.raises(ValueError):
        cv_active_frozen_orbitals_for_atom(55)
    with pytest.raises(ValueError):
        cv_active_frozen_orbitals_for_atom(17, ecp_core_electrons=3)
    with pytest.raises(ValueError):
        cv_active_frozen_orbitals_for_atom(17, ecp_core_electrons=-2)

    # Verify resolve_frozen_occupied on HCl (Cl freezes 1 = 1s, correlates 2s2p + valence)
    basis = _build_split_gate1_bases()
    hcl_atoms = (("Cl", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.41)))
    plan_cv = plan_singlepoint(
        hcl_atoms, FocalSeries(basis), FocalSeries(basis), None,
        charge=0, multiplicity=1, root_id="rhf", correlation_space_id="cv-active", backend="cpu_reference",
    )
    req_cv = next(r for r, _ in plan_cv.steps if r.role == "mp2_f12_correlation")
    req_cv_paired, req_fc_paired = _paired_cv_fc_requests(req_cv, None)
    assert req_cv_paired.correlation_space_id == "cv-active"
    assert req_fc_paired.correlation_space_id == "frozen-core"
    # HCl has 18 electrons -> (9, 9) occupied orbitals
    assert resolve_frozen_occupied(req_cv_paired, (9, 9)) == (1, 1)
    assert active_occupied_counts((9, 9), (1, 1)) == (8, 8)
    assert resolve_frozen_occupied(req_fc_paired, (9, 9)) == (5, 5)
    assert active_occupied_counts((9, 9), (5, 5)) == (4, 4)

    # Verify Br ECP10MDF and I ECP28MDF resolution via resolve_frozen_occupied
    import json
    dummy_ecp = json.dumps(
        [{"ecp_type": "scalar_ecp", "angular_momentum": [0], "r_exponents": [0], "gaussian_exponents": ["1.0"], "coefficients": [["1.0"]]}]
    )
    h_eb = basis.orbital.by_element[1]
    br_eb = ElementBasis(35, (_shell(0, (1.0,), (1.0,)),), ecp_core_electrons=10, ecp_data=dummy_ecp)
    i_eb = ElementBasis(53, (_shell(0, (1.0,), (1.0,)),), ecp_core_electrons=28, ecp_data=dummy_ecp)
    orb_ecp = _make_basis("ecp-test-orb", (h_eb, br_eb, i_eb))
    fb_ecp = FocalBasis(2, "ecp-test", orb_ecp, jk_fit=orb_ecp, mp2_fit=orb_ecp, f12_fit=orb_ecp, cabs=orb_ecp, beta=1.4)
    for sym in ("Br", "I"):
        atoms_hx = ((sym, (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.7)))
        p_cv = plan_singlepoint(
            atoms_hx, FocalSeries(fb_ecp), FocalSeries(fb_ecp), None,
            charge=0, multiplicity=1, root_id="rhf", correlation_space_id="cv-active", backend="cpu_reference",
        )
        r_cv = next(r for r, _ in p_cv.steps if r.role == "mp2_f12_correlation")
        _, r_fc_default = _paired_cv_fc_requests(r_cv, None)
        # HBr/HI with ECP10MDF/ECP28MDF have 25+1 = 26 explicit electrons -> (13, 13) occupied.
        # Production default valence space for Br/I ECP10MDF/ECP28MDF is frozen-core-spd (FC9).
        assert resolve_frozen_occupied(r_cv, (13, 13)) == (0, 0)
        assert r_fc_default.correlation_space_id == "frozen-core-spd"
        assert resolve_frozen_occupied(r_fc_default, (13, 13)) == (9, 9)
        assert active_occupied_counts((13, 13), (9, 9)) == (4, 4)
        p_fc4 = plan_singlepoint(
            atoms_hx, FocalSeries(fb_ecp), FocalSeries(fb_ecp), None,
            charge=0, multiplicity=1, root_id="rhf", correlation_space_id="frozen-core", backend="cpu_reference",
        )
        r_fc4_req = next(r for r, _ in p_fc4.steps if r.role == "mp2_f12_correlation")
        _, r_fc4 = _paired_cv_fc_requests(r_cv, r_fc4_req)
        assert resolve_frozen_occupied(r_fc4, (13, 13)) == (4, 4)
        p_spd = plan_singlepoint(
            atoms_hx, FocalSeries(fb_ecp), FocalSeries(fb_ecp), None,
            charge=0, multiplicity=1, root_id="rhf", correlation_space_id="frozen-core-spd", backend="cpu_reference",
        )
        r_spd = next(r for r, _ in p_spd.steps if r.role == "mp2_f12_correlation")
        _, r_fc_spd = _paired_cv_fc_requests(r_cv, r_spd)
        assert resolve_frozen_occupied(r_fc_spd, (13, 13)) == (9, 9)
        assert active_occupied_counts((13, 13), (9, 9)) == (4, 4)
    assert frozen_core_spd_orbitals_for_atom(35, 10) == 9
    assert frozen_core_spd_orbitals_for_atom(53, 28) == 9
    assert frozen_core_spd_orbitals_for_atom(35, 0) == 14
    assert frozen_core_spd_orbitals_for_atom(53, 0) == 23
    assert frozen_core_spd_orbitals_for_atom(8, 0) == 1
    assert frozen_core_spd_orbitals_for_atom(11, 0) == 1
    assert frozen_core_spd_orbitals_for_atom(17, 0) == 5


@pytest.mark.parametrize(
    "label,root,atoms_spec,mult,expected_cv_frozen,expected_fc_frozen",
    [
        ("H2O", "rhf", (("O", (0.0, 0.0, 0.0)), ("H", (0.0, -1.43, 1.11)), ("H", (0.0, 1.43, 1.11))), 1, 0, 1),
        ("OH", "uhf", (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.84))), 2, 0, 1),
        ("HCl", "rhf", (("Cl", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.41))), 1, 1, 5),
    ],
)
def test_cv_active_df_mp2_matches_pyscf_energy(
    label, root, atoms_spec, mult, expected_cv_frozen, expected_fc_frozen
):
    """Verify cv-active, frozen-core, and Delta-CV DF-MP2 energies vs PySCF 2.14.0 (< 1e-11 Eh)."""
    pyscf_gto = pytest.importorskip("pyscf.gto")
    pyscf_scf = pytest.importorskip("pyscf.scf")
    pyscf_mp = pytest.importorskip("pyscf.mp")

    basis = _build_split_gate1_bases()
    plan_cv = plan_singlepoint(
        atoms_spec, FocalSeries(basis), FocalSeries(basis), None,
        charge=0, multiplicity=mult, root_id=root, correlation_space_id="cv-active", backend="cpu_reference",
    )
    req_cv = next(r for r, _ in plan_cv.steps if r.role == "mp2_f12_correlation")
    req_cv, req_fc = _paired_cv_fc_requests(req_cv, None)

    ref_cv = df_reference(req_cv, forces=False, budget_mib=512)
    occs = tuple(ref_cv.occupied[k] for k in ref_cv._pair_spin_indices(req_cv))
    assert resolve_frozen_occupied(req_cv, occs) == (expected_cv_frozen, expected_cv_frozen)
    assert resolve_frozen_occupied(req_fc, occs) == (expected_fc_frozen, expected_fc_frozen)

    jk_cv = _pyscf_coulomb_source(req_cv, basis.jk_fit, req_cv.atoms)
    mp2_cv = _pyscf_coulomb_source(req_cv, basis.mp2_fit, req_cv.atoms)
    res_cv = reference_df_ump2(req_cv, ref_cv, jk_cv, mp2_cv, budget_mib=512)

    # Share the EXACT same DF-HF reference with frozen-core!
    ref_fc = replace(ref_cv, surface_id=req_fc.surface_id)
    jk_fc = replace(jk_cv, surface_id=req_fc.surface_id)
    mp2_fc = replace(mp2_cv, surface_id=req_fc.surface_id)
    res_fc = reference_df_ump2(req_fc, ref_fc, jk_fc, mp2_fc, budget_mib=512)

    native_cv = sum(res_cv.spin_energies)
    native_fc = sum(res_fc.spin_energies)
    native_dcv = native_cv - native_fc

    # Independent PySCF 2.14.0 reference
    mol = pyscf_gto.M(
        atom=[(s, c) for s, c in atoms_spec],
        basis=_pyscf_basis_dict(basis.orbital, req_cv.atoms),
        unit="Bohr",
        cart=False,
        charge=0,
        spin=mult - 1,
        verbose=0,
    )
    jk_aux = _pyscf_basis_dict(basis.jk_fit, req_cv.atoms)
    mp2_aux = _pyscf_basis_dict(basis.mp2_fit, req_cv.atoms)
    mf = (pyscf_scf.RHF(mol) if root == "rhf" else pyscf_scf.UHF(mol)).density_fit(auxbasis=jk_aux)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 100
    mf.kernel()
    assert mf.converged
    assert abs(ref_cv.energy - mf.e_tot) < 1e-11

    mp_cls = pyscf_mp.MP2 if root == "rhf" else pyscf_mp.UMP2
    e_pyscf_cv, _ = mp_cls(mf, frozen=expected_cv_frozen).density_fit(auxbasis=mp2_aux).kernel()
    e_pyscf_fc, _ = mp_cls(mf, frozen=expected_fc_frozen).density_fit(auxbasis=mp2_aux).kernel()
    e_pyscf_dcv = e_pyscf_cv - e_pyscf_fc

    assert abs(native_cv - e_pyscf_cv) < 1e-11, f"{label} cv-active mismatch: {native_cv} vs {e_pyscf_cv}"
    assert abs(native_fc - e_pyscf_fc) < 1e-11, f"{label} frozen-core mismatch: {native_fc} vs {e_pyscf_fc}"
    assert abs(native_dcv - e_pyscf_dcv) < 1e-11, f"{label} Delta-CV mismatch: {native_dcv} vs {e_pyscf_dcv}"
    # Core-valence correlation must be strictly negative (stabilizing)
    assert native_dcv < -1e-4, f"{label} expected negative Delta-CV, got {native_dcv}"


@pytest.mark.parametrize(
    "label,root,atoms_spec,mult,axes",
    [
        ("H2O", "rhf", (("O", (0.0, 0.0, 0.0)), ("H", (0.12, -1.43, 1.11)), ("H", (-0.05, 1.38, 1.16))), 1, ((0, 2), (1, 1))),
        ("OH", "uhf", (("O", (0.0, 0.0, 0.0)), ("H", (0.15, -0.08, 1.82))), 2, ((0, 2), (1, 0))),
        ("HCl", "rhf", (("Cl", (0.0, 0.0, 0.0)), ("H", (0.11, -0.07, 2.40))), 1, ((0, 2), (1, 0))),
    ],
)
def test_cv_active_and_delta_cv_df_mp2_forces_two_step_fd(label, root, atoms_spec, mult, axes):
    """Verify cv-active DF-MP2 and additive Delta-CV analytic forces vs two-step FD (< 2e-8 Eh/bohr)."""
    pytest.importorskip("pyscf.gto")
    basis = _build_split_gate1_bases()

    def eval_pair(coords, *, seed=None):
        plan_cv = plan_singlepoint(
            coords, FocalSeries(basis), FocalSeries(basis), None,
            charge=0, multiplicity=mult, root_id=root, correlation_space_id="cv-active", backend="cpu_reference",
        )
        req_cv = next(r for r, _ in plan_cv.steps if r.role == "mp2_f12_correlation")
        req_cv, req_fc = _paired_cv_fc_requests(req_cv, None)
        ref_cv = df_reference(req_cv, forces=False, budget_mib=512, seed=seed)
        jk_cv = _pyscf_coulomb_source(req_cv, basis.jk_fit, req_cv.atoms)
        mp2_cv = _pyscf_coulomb_source(req_cv, basis.mp2_fit, req_cv.atoms)
        out_cv = reference_df_ump2(req_cv, ref_cv, jk_cv, mp2_cv, budget_mib=512)
        out_fc = reference_df_ump2(
            req_fc,
            replace(ref_cv, surface_id=req_fc.surface_id),
            replace(jk_cv, surface_id=req_fc.surface_id),
            replace(mp2_cv, surface_id=req_fc.surface_id),
            budget_mib=512,
        )
        return req_cv, ref_cv, sum(out_cv.spin_energies), sum(out_fc.spin_energies), out_cv.weights, out_fc.weights

    from generativeqc._f12_focal_reference import FocalReferenceSeed

    req0, ref0, e_cv0, e_fc0, w_cv0, w_fc0 = eval_pair(atoms_spec)
    seed0 = FocalReferenceSeed(req0, ref0, minimum_overlap=0.999)
    f_cv = _pyscf_nuclear_gradient(req0.atoms, basis.orbital, basis.jk_fit, basis.mp2_fit, w_cv0, charge=0, multiplicity=mult)
    f_fc = _pyscf_nuclear_gradient(req0.atoms, basis.orbital, basis.jk_fit, basis.mp2_fit, w_fc0, charge=0, multiplicity=mult)
    f_dcv = f_cv - f_fc

    # Translational & rotational force invariants
    xyz = np.asarray([p for _, p in atoms_spec], dtype=np.float64)
    for tag, f_arr in (("cv", f_cv), ("dcv", f_dcv)):
        assert float(np.max(np.abs(np.sum(f_arr, axis=0)))) < 1e-12, f"{label} {tag} net force violation"
        assert float(np.max(np.abs(np.sum(np.cross(xyz, f_arr), axis=0)))) < 1e-12, f"{label} {tag} torque violation"

    for atom_idx, axis in axes:
        for h in (0.0004, 0.0002):
            def shifted(delta):
                c = [(s, list(p)) for s, p in atoms_spec]
                c[atom_idx][1][axis] += delta
                return [(s, tuple(p)) for s, p in c]

            _, _, e_cv_p, e_fc_p, _, _ = eval_pair(shifted(+h), seed=seed0)
            _, _, e_cv_m, e_fc_m, _, _ = eval_pair(shifted(-h), seed=seed0)
            fd_cv = -(e_cv_p - e_cv_m) / (2.0 * h)
            fd_dcv = -((e_cv_p - e_fc_p) - (e_cv_m - e_fc_m)) / (2.0 * h)
            err_cv = abs(f_cv[atom_idx, axis] - fd_cv)
            err_dcv = abs(f_dcv[atom_idx, axis] - fd_dcv)
            assert err_cv < 2e-8, f"{label} cv-active force err={err_cv:.3e} at ({atom_idx},{axis}) h={h}"
            assert err_dcv < 2e-8, f"{label} Delta-CV force err={err_dcv:.3e} at ({atom_idx},{axis}) h={h}"


@pytest.mark.parametrize(
    "oa,ob,cv_frozen,fc_frozen",
    [(5, 5, (0, 0), (1, 1)), (5, 4, (0, 0), (1, 1)), (6, 6, (1, 1), (3, 3))],
)
def test_single_pass_cv_fc_f12_pair_accumulation_parity(oa, ob, cv_frozen, fc_frozen):
    """Single-pass cv-active/frozen-core pair accumulation matches separate F12 evaluations to <1e-14."""
    from generativeqc._f12_spin_pairs import molecular_pair_energies
    from generativeqc._f12_unrestricted import unrestricted_mp2_f12_ao_response
    from generativeqc_compiler.tensor import execute

    rng = np.random.default_rng(777 + 13 * oa + ob + cv_frozen[0] + fc_frozen[0])
    obs, ext, q = oa + 3, oa + 6, 12
    space = (oa, ob, obs, ext, q)

    def spd(n):
        a = rng.standard_normal((n, n)) * 0.08
        return a @ a.T + 2.0 * np.eye(n)

    def sym_panel(nq, n):
        p = rng.standard_normal((nq, n, n)) * 0.02
        return 0.5 * (p + p.swapaxes(1, 2))

    ca, _ = np.linalg.qr(rng.standard_normal((ext, ext)))
    cb = ca.copy() if oa == ob else np.linalg.qr(ca + 0.05 * rng.standard_normal((ext, ext)))[0]
    eigs = np.linspace(-3.0, 2.5, ext)
    hcore = ca @ np.diag(eigs) @ ca.T
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
    run = lambda g, inp: execute(g, inp, max_bytes=1 << 29)
    t_cv, _ = unrestricted_mp2_f12_ao_response(space, ao_dense, run, frozen_occupied=cv_frozen, component="f12")
    t_fc, _ = unrestricted_mp2_f12_ao_response(space, ao_dense, run, frozen_occupied=fc_frozen, component="f12")

    # Compare against single-pass molecular_pair_energies subset accumulation
    from generativeqc._f12_spin_ao import _orbital_sources
    from generativeqc._f12_spin_fock import spin_fock

    data, _ = _orbital_sources(space, ao_dense, run, None, projected=None, role="shared")
    src = {k.removeprefix("source/"): v for k, v in data.items() if k.startswith("source/")}
    fock = spin_fock((oa, ob, ext, q), src, run)
    fc_by_spin = {"alpha": fc_frozen[0], "beta": fc_frozen[1]}
    cv_sum = 0.0
    fc_sum = 0.0
    for item in molecular_pair_energies(space, data, src, fock, run, frozen_occupied=cv_frozen, include_ordinary=True):
        sl, sr = item.spins
        ii, jj = item.occupied_indices
        cv_sum += item.f12
        if ii >= fc_by_spin[sl] and jj >= fc_by_spin[sr]:
            fc_sum += item.f12
    assert abs(cv_sum - t_cv.f12_increment) < 1e-13
    assert abs(fc_sum - t_fc.f12_increment) < 1e-13
    assert abs((cv_sum - fc_sum) - (t_cv.f12_increment - t_fc.f12_increment)) < 1e-13


def test_chunked_shell_projection_exact_parity():
    """Chunked 3-center AO block packing and scattering matches unchunked bit-for-bit (supports int64 offsets)."""
    from generativeqc._f12_focal_integral_sink import ShellProjection

    rng = np.random.default_rng(20261123)
    # 4 orbital/CABS shells (s, p, d, p) and 3 auxiliary shells (s, p, d)
    angular = np.array([0, 1, 2, 1, 0, 1, 2], dtype=np.int32)
    ncomb = 4
    naux_sh = 3
    carts = (angular + 1) * (angular + 2) // 2
    sphs = 2 * angular + 1
    norb = int(sphs[:ncomb].sum())
    naux = int(sphs[ncomb:].sum())

    rows = np.full((len(angular), int(sphs.max())), -1, dtype=np.int32)
    for start, end in ((0, ncomb), (ncomb, len(angular))):
        off = 0
        for i in range(start, end):
            w = int(sphs[i])
            rows[i, :w] = np.arange(off, off + w)
            off += w

    orb_pairs = np.stack(np.tril_indices(ncomb), axis=1).astype(np.int32)
    ao_blocks = np.empty((naux_sh, len(orb_pairs), 3), dtype=np.int32)
    ao_blocks[:, :, :2] = orb_pairs
    ao_blocks[:, :, 2] = np.arange(ncomb, len(angular), dtype=np.int32)[:, None]
    blocks = ao_blocks.reshape(-1, 3)
    counts = np.prod(carts[blocks], axis=1).astype(np.int64)
    offsets_i64 = np.r_[0, np.cumsum(counts)].astype(np.int64)

    layout = {
        "angular": angular,
        "ao_blocks": blocks,
        "ao_offsets": offsets_i64,
        "spherical_rows": rows,
        "orbital_shell_count": np.int32(ncomb),
        "ao_sizes": np.array([norb, naux], dtype=np.int32),
        "transform_0": np.eye(1),
        "transform_1": rng.standard_normal((3, 3)),
        "transform_2": rng.standard_normal((5, 6)),
    }
    proj = ShellProjection(layout, "ao")
    ranges = proj.block_chunks(max_cart=60)
    assert len(ranges) > 3
    assert ranges[0][0] == 0 and ranges[-1][1] == len(blocks)

    # 1. Pack parity (spherical -> Cartesian)
    w_sph = rng.standard_normal((naux, norb, norb))
    packed_full = proj.pack(w_sph)
    packed_chunks = np.concatenate([proj.pack_slice(w_sph, s, e) for s, e in ranges])
    assert np.allclose(packed_full, packed_chunks, rtol=1e-14, atol=1e-13)

    # 2. Scatter parity (Cartesian -> spherical)
    raw_cart = rng.standard_normal(int(offsets_i64[-1]))
    sph_full = proj.spherical_values(raw_cart)
    sph_chunked = np.empty_like(sph_full)
    for s, e in ranges:
        proj.spherical_values_into(raw_cart[int(offsets_i64[s]) : int(offsets_i64[e])], sph_chunked, s, e)
    assert np.allclose(sph_full, sph_chunked, rtol=1e-14, atol=1e-13)


def test_q_chunked_orbital_sources_forward_and_adjoint_parity(monkeypatch):
    """Q-chunked AO->MO projection and reverse adjoint (with tail tile) match unchunked < 1e-13."""
    import generativeqc._f12_spin_ao as spin_ao
    from generativeqc._f12_spin_ao import _orbital_sources
    from generativeqc_compiler.tensor import execute

    rng = np.random.default_rng(20261124)
    oa, ob, obs, ext, aux = 3, 2, 5, 8, 11  # aux=11 not divisible by q_chunk=3
    space = (oa, ob, obs, ext, aux)

    def spd(n):
        a = rng.standard_normal((n, n)) * 0.08
        return a @ a.T + 2.0 * np.eye(n)

    def sym_panel(nq, n):
        p = rng.standard_normal((nq, n, n)) * 0.03
        return 0.5 * (p + p.swapaxes(1, 2))

    ca, _ = np.linalg.qr(rng.standard_normal((ext, ext)))
    cb, _ = np.linalg.qr(ca + 0.05 * rng.standard_normal((ext, ext)))
    m_c = spd(aux)
    w_c = np.linalg.inv(np.linalg.cholesky(m_c))
    ao = {
        "overlap_ao": np.eye(ext),
        "hcore_ao": sym_panel(1, ext)[0],
        "coefficients_alpha": ca,
        "coefficients_beta": cb,
        "metric_whitener": w_c,
        "metric/coulomb": m_c,
        "metric/geminal": spd(aux),
        "metric/squared_geminal": spd(aux),
        "metric/geminal_coulomb": spd(aux),
        "metric/double_commutator": spd(aux),
        "ao_coulomb": sym_panel(aux, ext),
        "ao_f12": sym_panel(aux, ext),
        "ao_f12_squared": sym_panel(aux, ext),
        "ao_f12_over_coulomb": sym_panel(aux, ext),
        "ao_double_commutator": sym_panel(aux, ext),
    }
    run = lambda g, inp: execute(g, inp, max_bytes=1 << 29)

    # Forward parity
    monkeypatch.setattr(spin_ao, "default_auxiliary_batch_size", lambda naux, n: naux)
    d_full, _ = _orbital_sources(space, ao, run, None, projected=None, role="shared")
    monkeypatch.setattr(spin_ao, "default_auxiliary_batch_size", lambda naux, n: 3)
    d_chunk, _ = _orbital_sources(space, ao, run, None, projected=None, role="shared")
    assert set(d_full.keys()) == set(d_chunk.keys())
    for k in d_full:
        assert np.allclose(d_full[k], d_chunk[k], rtol=1e-13, atol=1e-13), f"forward mismatch on {k}"

    # Reverse adjoint parity (including ao_bar_sink streaming mode!)
    orb_w = {k: rng.standard_normal(v.shape) for k, v in d_full.items() if k != "space"}
    monkeypatch.setattr(spin_ao, "default_auxiliary_batch_size", lambda naux, n: naux)
    _, b_full = _orbital_sources(space, ao, run, orb_w, projected=d_full, role="shared")
    sunk: dict[str, np.ndarray] = {}
    monkeypatch.setattr(spin_ao, "default_auxiliary_batch_size", lambda naux, n: 3)
    _, b_chunk = _orbital_sources(
        space, ao, run, orb_w, projected=d_chunk, role="shared",
        ao_bar_sink=lambda name, mat: sunk.__setitem__(name, mat.copy()),
    )
    assert b_full is not None and b_chunk is not None
    for k, ref_v in b_full.items():
        if k.startswith("ao_"):
            assert np.allclose(ref_v, sunk[k], rtol=1e-13, atol=1e-13), f"sunk adjoint mismatch on {k}"
        else:
            assert np.allclose(ref_v, b_chunk[k], rtol=1e-13, atol=1e-13), f"adjoint mismatch on {k}"


def _bse_element_basis(bse_name: str, sym: str) -> ElementBasis:
    import json
    import basis_set_exchange as bse
    from generativeqc.elements import atomic_number

    data = json.loads(bse.get_basis(bse_name, elements=[sym], fmt="json"))
    z = atomic_number(sym)
    elem = data["elements"][str(z)]
    shells = []
    for sh in elem["electron_shells"]:
        ang = sh["angular_momentum"]
        exps = tuple(str(x) for x in sh["exponents"])
        if len(ang) == 1:
            coeffs = tuple(tuple(str(c) for c in row) for row in sh["coefficients"])
            shells.append(BasisShell(int(ang[0]), exps, coeffs))
        else:
            for ell, row in zip(ang, sh["coefficients"], strict=True):
                shells.append(BasisShell(int(ell), exps, (tuple(str(c) for c in row),)))
    return ElementBasis(z, tuple(shells))


def _basis_to_gbs_text(basis: BasisSet, syms: list[str]) -> str:
    from generativeqc.elements import atomic_number

    l_char = "SPDFGHI"
    lines = ["spherical", "****"]
    for sym in sorted(set(syms), key=atomic_number):
        eb = basis.by_element[atomic_number(sym)]
        lines.append(f"{sym} 0")
        for sh in eb.shells:
            ell = l_char[sh.angular_momentum]
            for crow in sh.coefficients:
                lines.append(f"{ell} {len(sh.exponents)} 1.00")
                for exp, coeff in zip(sh.exponents, crow, strict=True):
                    lines.append(f"  {float(exp):.14e}  {float(coeff):.14e}")
        lines.append("****")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    "label,atoms_spec,expected_cv_frozen,expected_fc_frozen",
    [
        (
            # GSCDB W4-17_h2o (CC BY 4.0, see THIRD_PARTY_NOTICES.md), in Bohr.
            "H2O",
            (
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, 1.8101686547790252)),
                ("H", (1.7554778816245764, 0.0, -0.4415970623550433)),
            ),
            0,
            1,
        ),
        (
            "HCl",
            (
                ("Cl", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, 2.41)),
            ),
            1,
            5,
        ),
    ],
)
def test_df_mp2_f12_3c_fix_delta_cv_matches_psi4_blackbox_oracle(
    label, atoms_spec, expected_cv_frozen, expected_fc_frozen, tmp_path, monkeypatch
):
    """Independent black-box Psi4 1.11 DF-MP2-F12 3C(FIX) oracle check on H2O (0 vs 1) and HCl (1 vs 5) in cc-pCVDZ-F12 (beta=1.4)."""
    import os

    monkeypatch.chdir(tmp_path)
    psi4 = pytest.importorskip("psi4")
    pytest.importorskip("basis_set_exchange")

    from generativeqc.experimental.f12_focal import MolecularFocalProvider

    shell_lib, one_lib = f12_host_libraries()

    syms = [s for s, _ in atoms_spec]
    uniq = sorted(set(syms))
    orb_elems, cabs_elems, ri_elems = [], [], []
    for sym in uniq:
        if sym == "H":
            orb_elems.append(_bse_element_basis("cc-pVDZ-F12", "H"))
            cabs_elems.append(_bse_element_basis("cc-pVDZ-F12-OPTRI", "H"))
            ri_elems.append(_bse_element_basis("aug-cc-pVTZ-RIFIT", "H"))
        else:
            orb_elems.append(_bse_element_basis("cc-pCVDZ-F12", sym))
            cabs_elems.append(_bse_element_basis("cc-pCVDZ-F12-OPTRI", sym))
            ri_elems.append(_bse_element_basis("cc-pCVDZ-F12-RIFIT", sym))

    orb_bs = _make_basis(f"cv-orb-{label}", tuple(orb_elems))
    cabs_bs = _make_basis(f"cv-cabs-{label}", tuple(cabs_elems))
    ri_bs = _make_basis(f"cv-ri-{label}", tuple(ri_elems))

    # Psi4 1.11 DF-MP2-F12 uses one common DF_BASIS_F12 fitting basis across SCF, MP2, and F12
    beta = 1.4
    fb = FocalBasis(
        2,
        f"psi4-oracle/{label}/cc-pCVDZ-F12",
        orb_bs,
        jk_fit=ri_bs,
        mp2_fit=ri_bs,
        f12_fit=ri_bs,
        cabs=cabs_bs,
        beta=beta,
    )
    series = FocalSeries(fb)
    plan_cv = plan_singlepoint(
        atoms_spec,
        series,
        series,
        None,
        charge=0,
        multiplicity=1,
        root_id="rhf",
        correlation_space_id="cv-active",
        backend="cpu_reference",
        slater_normalization="negative_exponential",
    )
    req_cv = next(r for r, _ in plan_cv.steps if r.role == "mp2_f12_correlation")
    # Psi4 1.11 evaluates unnormalized f12 = -exp(-beta*r) ("negative_exponential");
    # select that mode explicitly to verify exact parity against Psi4 1.11 at beta=1.4.
    provider = MolecularFocalProvider(
        shell_library=shell_lib,
        one_electron_library=one_lib,
        budget_mib=4096,
        slater_normalization="negative_exponential",
    )
    gqc = provider.delta_cv_energy(req_cv)
    assert gqc.cv_frozen == (expected_cv_frozen, expected_cv_frozen)
    assert gqc.fc_frozen == (expected_fc_frozen, expected_fc_frozen)

    # Independent black-box Psi4 1.11 DF-MP2-F12 3C(FIX) on the identical sealed .gbs files
    orb_name = f"gqc_oracle_orb_{label.lower()}"
    ri_name = f"gqc_oracle_ri_{label.lower()}"
    cabs_name = f"gqc_oracle_cabs_{label.lower()}"
    (tmp_path / f"{orb_name}.gbs").write_text(_basis_to_gbs_text(orb_bs, syms))
    (tmp_path / f"{ri_name}.gbs").write_text(_basis_to_gbs_text(ri_bs, syms))
    (tmp_path / f"{cabs_name}.gbs").write_text(_basis_to_gbs_text(cabs_bs, syms))
    monkeypatch.setenv("PSIPATH", str(tmp_path) + ":" + os.environ.get("PSIPATH", ""))

    psi4.core.be_quiet()
    psi4.set_memory("4 GB")
    geom_str = (
        "0 1\n"
        + "\n".join(f"{s} {x:.16f} {y:.16f} {z:.16f}" for s, (x, y, z) in atoms_spec)
        + "\nunits bohr\nsymmetry c1\nno_reorient\nno_com\n"
    )
    p4_res = {}
    for tag, nf in (("cv", expected_cv_frozen), ("fc", expected_fc_frozen)):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.geometry(geom_str)
        psi4.set_options(
            {
                "basis": orb_name,
                "df_basis_f12": ri_name,
                "cabs_basis": cabs_name,
                "f12_beta": beta,
                "num_frozen_docc": nf,
                "scf_type": "df",
                "mp2_type": "df",
                "cabs_singles": True,
                "e_convergence": 1e-12,
                "d_convergence": 1e-12,
                "ints_tolerance": 1e-14,
            }
        )
        psi4.energy("mp2-f12")
        hf = float(psi4.variable("SCF TOTAL ENERGY"))
        mp2_c = float(psi4.variable("MP2 CORRELATION ENERGY"))
        f12_c = float(psi4.variable("MP2-F12 CORRELATION ENERGY"))
        p4_res[tag] = (hf, mp2_c, f12_c - mp2_c, f12_c)

    p4_d_ord = p4_res["cv"][1] - p4_res["fc"][1]
    p4_d_f12 = p4_res["cv"][2] - p4_res["fc"][2]
    p4_d_tot = p4_res["cv"][3] - p4_res["fc"][3]

    assert abs(gqc.reference.energy - p4_res["cv"][0]) < 1e-8
    assert abs(gqc.cv_ordinary_mp2_Eh - p4_res["cv"][1]) < 1e-10
    assert abs(gqc.fc_ordinary_mp2_Eh - p4_res["fc"][1]) < 1e-10
    assert abs(gqc.cv_f12_increment_Eh - p4_res["cv"][2]) < 5e-9
    assert abs(gqc.fc_f12_increment_Eh - p4_res["fc"][2]) < 5e-9
    assert abs(gqc.cv_mp2_f12_Eh - p4_res["cv"][3]) < 5e-9
    assert abs(gqc.fc_mp2_f12_Eh - p4_res["fc"][3]) < 5e-9
    assert abs(gqc.delta_ordinary_mp2_Eh - p4_d_ord) < 1e-10
    assert abs(gqc.delta_f12_increment_Eh - p4_d_f12) < 5e-9
    assert abs(gqc.delta_mp2_f12_Eh - p4_d_tot) < 5e-9


@pytest.mark.parametrize(
    "label,atoms_spec,charge,expected_cv_frozen,expected_fc_frozen",
    [
        (
            # GSCDB W4-17_h2o (CC BY 4.0, see THIRD_PARTY_NOTICES.md), in Bohr.
            "H2O",
            (
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, 1.8101686547790252)),
                ("H", (1.7554778816245764, 0.0, -0.4415970623550433)),
            ),
            0,
            0,
            1,
        ),
        (
            "OH-",
            (
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, 1.825)),
            ),
            -1,
            0,
            1,
        ),
    ],
)
def test_delta_cv_mp2_ha_pwcvtz_matches_pyscf_h2o_and_oh_anion(
    label, atoms_spec, charge, expected_cv_frozen, expected_fc_frozen
):
    """Verify pure conventional DF-MP2 ha-pwCVTZ Delta-CV (zero CABS/F12) vs PySCF 2.14.0 (< 1e-11 Eh)."""

    pyscf_gto = pytest.importorskip("pyscf.gto")
    pyscf_scf = pytest.importorskip("pyscf.scf")
    pyscf_mp = pytest.importorskip("pyscf.mp")
    pytest.importorskip("basis_set_exchange")

    from generativeqc.experimental.f12_focal import (
        MolecularFocalProvider,
        conventional_mp2_request,
    )

    shell_lib, one_lib = f12_host_libraries()

    orb_bs = _make_basis(
        f"ha-pwcvtz-orb-{label}",
        (_bse_element_basis("cc-pVTZ", "H"), _bse_element_basis("aug-cc-pwCVTZ", "O")),
    )
    ri_bs = _make_basis(
        f"ha-pwcvtz-ri-{label}",
        (_bse_element_basis("cc-pVTZ-RIFIT", "H"), _bse_element_basis("aug-cc-pwCVTZ-RIFIT", "O")),
    )
    jk_bs = _make_basis(
        f"ha-pwcvtz-jk-{label}",
        (_bse_element_basis("def2-universal-JKFit", "H"), _bse_element_basis("def2-universal-JKFit", "O")),
    )
    # Zero CABS, zero F12, beta=None
    fb = FocalBasis(
        3,
        f"gate1/ha-pwcvtz/{label}",
        orb_bs,
        jk_fit=jk_bs,
        mp2_fit=ri_bs,
        f12_fit=None,
        cabs=None,
        beta=None,
    )
    req_cv = conventional_mp2_request(
        atoms_spec,
        fb,
        root_id="rhf",
        correlation_space_id="cv-active",
        charge=charge,
        multiplicity=1,
        backend="cpu_reference",
    )
    provider = MolecularFocalProvider(
        shell_library=shell_lib,
        one_electron_library=one_lib,
        budget_mib=4096,
    )
    gqc = provider.delta_cv_mp2_energy(req_cv)
    assert gqc.cabs_rank == 0
    assert gqc.cv_f12_increment_Eh == 0.0
    assert gqc.fc_f12_increment_Eh == 0.0
    assert gqc.delta_f12_increment_Eh == 0.0
    assert gqc.cv_frozen == (expected_cv_frozen, expected_cv_frozen)
    assert gqc.fc_frozen == (expected_fc_frozen, expected_fc_frozen)

    mol = pyscf_gto.M(
        atom=list(atoms_spec),
        basis=_pyscf_basis_dict(orb_bs, req_cv.atoms),
        unit="Bohr",
        cart=False,
        charge=charge,
        spin=0,
        verbose=0,
    )
    mf = pyscf_scf.RHF(mol).density_fit(auxbasis=_pyscf_basis_dict(jk_bs, req_cv.atoms))
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    assert mf.converged
    aux_ri = _pyscf_basis_dict(ri_bs, req_cv.atoms)
    py_cv = float(pyscf_mp.MP2(mf, frozen=expected_cv_frozen).density_fit(auxbasis=aux_ri).kernel()[0])
    py_fc = float(pyscf_mp.MP2(mf, frozen=expected_fc_frozen).density_fit(auxbasis=aux_ri).kernel()[0])
    py_dcv = py_cv - py_fc

    assert abs(gqc.reference.energy - float(mf.e_tot)) < 1e-11
    assert abs(gqc.cv_ordinary_mp2_Eh - py_cv) < 1e-11
    assert abs(gqc.fc_ordinary_mp2_Eh - py_fc) < 1e-11
    assert abs(gqc.delta_ordinary_mp2_Eh - py_dcv) < 1e-11
    assert abs(gqc.delta_mp2_f12_Eh - py_dcv) < 1e-11


@pytest.mark.parametrize(
    "label,root,atoms_spec,charge,mult,axes",
    [
        ("H2O", "rhf", (("O", (0.0, 0.0, 0.0)), ("H", (0.0, -1.43, 1.11)), ("H", (0.0, 1.43, 1.11))), 0, 1, ((0, 2), (1, 1))),
        ("OH-", "rhf", (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.825))), -1, 1, ((0, 2), (1, 2))),
        ("OH", "uhf", (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.84))), 0, 2, ((0, 2), (1, 2))),
        ("HCl", "rhf", (("Cl", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.41))), 0, 1, ((0, 2), (1, 2))),
    ],
)
def test_delta_cv_mp2_single_pass_forces_and_fallback_df_hf(
    label, root, atoms_spec, charge, mult, axes, monkeypatch
):
    """Verify single-pass tiled DF-MP2 Delta-CV E+F (zero CABS/F12) matches reference_df_ump2, 2-step FD, and native DF-HF fallback."""

    from generativeqc import _f12_focal_cv as cv_mod
    from generativeqc.experimental.f12_focal import (
        MolecularFocalProvider,
        conventional_mp2_request,
    )

    shell_lib, one_lib = f12_host_libraries()

    full_basis = _build_split_gate1_bases()
    mp2_only_basis = FocalBasis(
        2,
        f"gate1-mp2-only/{label}",
        full_basis.orbital,
        jk_fit=full_basis.jk_fit,
        mp2_fit=full_basis.mp2_fit,
        f12_fit=None,
        cabs=None,
        beta=None,
    )
    provider = MolecularFocalProvider(
        shell_library=shell_lib,
        one_electron_library=one_lib,
        budget_mib=1024,
    )
    req_cv = conventional_mp2_request(
        atoms_spec,
        mp2_only_basis,
        root_id=root,
        correlation_space_id="cv-active",
        charge=charge,
        multiplicity=mult,
        backend="cpu_reference",
    )

    # Force multi-tile j-tiling & multi-chunk Q-chunking on small molecules to exercise padding & chunk boundaries!
    orig_pairs = cv_mod._tiled_mp2_delta_cv_pairs
    orig_fac = cv_mod._mp2_factors_q_chunked
    monkeypatch.setattr(
        cv_mod,
        "_tiled_mp2_delta_cv_pairs",
        lambda *a, **kw: orig_pairs(*a, **{**kw, "max_tile_elements": 600}),
    )
    monkeypatch.setattr(
        cv_mod,
        "_mp2_factors_q_chunked",
        lambda *a, **kw: orig_fac(*a, **{**kw, "max_chunk_elements": 800}),
    )

    res_ef = provider.delta_cv_mp2_energy_and_forces(req_cv)
    res_e = provider.delta_cv_mp2_energy(req_cv)
    assert abs(res_ef.delta_ordinary_mp2_Eh - res_e.delta_ordinary_mp2_Eh) < 1e-13
    assert res_ef.delta_forces is not None

    # Verify translational & rotational invariants < 1e-12
    xyz = np.asarray([pos for _, pos in atoms_spec], dtype=np.float64)
    net_f = np.max(np.abs(np.sum(res_ef.delta_forces, axis=0)))
    torque = np.max(np.abs(np.sum(np.cross(xyz, res_ef.delta_forces), axis=0)))
    assert net_f < 1e-12, f"{label} net force {net_f:.3e}"
    assert torque < 1e-12, f"{label} torque {torque:.3e}"

    # Verify native focal-source DF-HF fallback reproduces exact df_reference energy & Delta-CV (< 1e-11 Eh)
    from generativeqc._f12_focal_one_electron import one_electron_sources

    one_chk = one_electron_sources(req_cv, library=one_lib, derivatives=False, orbital_only=True, budget_mib=1024)
    jk_chk = provider._panel(req_cv, "jk_fit", orbital_only=True)
    run_cpu = cv_mod._checked_runner(provider, False, 1024)
    ref_fb = cv_mod._solve_df_reference_from_focal_sources(provider, req_cv, one_chk, jk_chk, run=run_cpu)
    assert abs(ref_fb.energy - res_ef.reference.energy) < 1e-11

    # Verify analytic Delta-F against two-step central finite differences (< 2e-8 Eh/bohr)
    for atom_idx, axis in axes:
        errs = []
        for h in (4e-4, 2e-4):
            pts = []
            for sign in (+1.0, -1.0):
                disp = [list(pos) for _, pos in atoms_spec]
                disp[atom_idx][axis] += sign * h
                d_atoms = tuple((s, tuple(p)) for (s, _), p in zip(atoms_spec, disp, strict=True))
                r_d = conventional_mp2_request(
                    d_atoms,
                    mp2_only_basis,
                    root_id=root,
                    correlation_space_id="cv-active",
                    charge=charge,
                    multiplicity=mult,
                    backend="cpu_reference",
                )
                pts.append(provider.delta_cv_mp2_energy(r_d).delta_ordinary_mp2_Eh)
            fd = -(pts[0] - pts[1]) / (2.0 * h)
            errs.append(abs(float(res_ef.delta_forces[atom_idx, axis]) - fd))
        assert errs[1] < 2e-8, f"{label} axis=({atom_idx},{axis}) FD errs={errs}"
        assert errs[1] < errs[0] * 0.4 or min(errs) < 2e-9


def test_small_core_halide_atomic_orbital_g_admission() -> None:
    """Verify bare small-core Br/I atoms admit orbital g-shells across ecp.py and Calculator."""
    import json
    from generativeqc import Atom, BasisProvenance, BasisSet, BasisShell, Calculator, ElementBasis, _native
    from generativeqc.ecp import fitted_hf_ecp_domain, g_shell_small_core_halide, resolve_ecp

    dummy_ecp = json.dumps(
        [
            {
                "ecp_type": "scalar_ecp",
                "angular_momentum": [0],
                "r_exponents": [2],
                "gaussian_exponents": ["1.0"],
                "coefficients": [["1.0"]],
            }
        ]
    )
    prov = BasisProvenance("unit", "unit", "unit", "0" * 64)
    for z, core in ((35, 10), (53, 28)):
        shells = tuple(BasisShell(ell, ("1.0",), (("1.0",),)) for ell in range(5))
        orb = BasisSet(
            f"test-g-{z}",
            (ElementBasis(z, shells, ecp_core_electrons=core, ecp_data=dummy_ecp),),
            prov,
            "spherical",
        )
        jk = BasisSet(
            f"test-jk-{z}",
            (ElementBasis(z, (BasisShell(0, ("1.0",), (("1.0",),)),)),),
            prov,
            "spherical",
        )
        atoms = (Atom(z, (0.0, 0.0, 0.0)),)
        assert g_shell_small_core_halide(orb, atoms)
        assert fitted_hf_ecp_domain("uhf", orb, atoms, 0, 2)
        assert resolve_ecp(orb, atoms) is not None
        calc = object.__new__(Calculator)
        calc._method = _native.METHOD_UHF
        calc._method_name = "uhf"
        calc._device_name = "cuda"
        calc._density_fitting_mode = _native.DENSITY_FITTING_CUDA
        calc._precision_mode = _native.PRECISION_FP64
        calc._basis = orb
        calc._auxiliary_basis = jk
        for op in ("overlap", "kinetic", "nuclear_attraction", "df_three_center"):
            assert calc._full_df_basis_consumer(op, 0, atoms=atoms) == "full_df"
            assert calc._full_df_basis_consumer(op, 1, atoms=atoms) == "full_df"
        for m_code in (_native.METHOD_UCCSD, _native.METHOD_UCCSD_T):
            cc_calc = object.__new__(Calculator)
            cc_calc._method = m_code
            cc_calc._device_name = "cuda"
            cc_calc._density_fitting_mode = _native.DENSITY_FITTING_FULL_CUDA
            cc_calc._basis = orb
            cc_calc._auxiliary_basis = jk
            assert cc_calc._full_df_basis_consumer("df_three_center", 0, atoms=atoms) == "full_df"
            assert cc_calc._full_df_basis_consumer("df_metric", 1, atoms=atoms) == "full_df"

        # Orbital h-shell (l=5) must still be rejected
        orb_h = BasisSet(
            f"test-h-{z}",
            (
                ElementBasis(
                    z,
                    shells + (BasisShell(5, ("1.0",), (("1.0",),)),),
                    ecp_core_electrons=core,
                    ecp_data=dummy_ecp,
                ),
            ),
            prov,
            "spherical",
        )
        assert not fitted_hf_ecp_domain("uhf", orb_h, atoms, 0, 2)


def test_naf_df_mp2_uses_independent_pyscf_frozen_lists():
    from generativeqc import Atom
    from generativeqc.experimental.f12_focal import MolecularFocalProvider, conventional_mp2_request

    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, mp, scf

    basis = FocalBasis(2, "NaF", "sto-3g", jk_fit="def2-svp", mp2_fit="def2-svp")
    atoms = (Atom(11, (0., 0., 0.)), Atom(9, (0., 0., 3.60)))
    req = conventional_mp2_request(
        atoms, basis, root_id="rhf", correlation_space_id="cv-active", backend="cpu_reference",
    )
    shell_library, one_library = f12_host_libraries()
    provider = MolecularFocalProvider(shell_library=shell_library, one_electron_library=one_library)
    actual = provider.delta_cv_mp2_energy(req, replace(req, correlation_space_id="frozen-core-spd"))
    assert actual.frozen_core_definition_version == 2
    assert actual.cv_frozen == (1, 1)
    assert actual.fc_frozen == (2, 2)
    assert actual.reference.frozen_frame is not None
    assert actual.reference.frozen_frame.channels("frozen-core-spd")[0].indices == (0, 1)

    mol = gto.M(atom=[("Na", (0., 0., 0.)), ("F", (0., 0., 3.60))], unit="Bohr",
                basis=_pyscf_basis_dict(basis.orbital, atoms), verbose=0)
    fit = _pyscf_basis_dict(basis.jk_fit, atoms)
    mf = scf.RHF(mol).density_fit(auxbasis=fit)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    mf.kernel()
    assert mf.converged, pyscf.__version__
    assert actual.reference.energy == pytest.approx(mf.e_tot, rel=0, abs=1e-10)
    for frozen, energy in (([0], actual.cv_ordinary_mp2_Eh), ([0, 1], actual.fc_ordinary_mp2_Eh)):
        oracle, _ = mp.MP2(mf, frozen=frozen).density_fit(auxbasis=fit).kernel()
        assert energy == pytest.approx(oracle, rel=0, abs=1e-10)
