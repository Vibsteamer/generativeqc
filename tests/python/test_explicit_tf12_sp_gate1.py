"""Gate 1 local CPU verification for Kohn extended-SP explicit (T)-F12 (spatial RHF).

Verifies:
1. External zero-coupling degeneration: with F12 couplings (C and A/Z2) disabled,
   spatial RHF (T)-F12 matches independent external PySCF 2.14.0 DF-RCCSD(T) to < 1e-11 Eh
   on H2O (AE & FC), CH4 (FC), and HBr (ECP10MDF + FC) with split JK/MP2 fits.
2. GeCCo 33-term BH contract parity:
   - Verifies gecco_physical_triples_action.py physical triples action & sign conventions.
   - Reproduces GeCCo's exact CCPT-R12 BH energy (-0.002312260360 Eh for z2_mode="ffg"
     and -0.002280453001 Eh for z2_mode="direct") and all 18 term groups to < 1e-11 Eh
     via native C++ spatial contractions (no spin-orbital expansion).
   - Verifies DF C++ path matches exact-4-index C++ path to < 1e-11 Eh on exact Cholesky factors.
3. Quantitative 1/beta Slater normalization, Z2 mode (ffg vs direct vs zero), and C2
   decomposition on molecular DF-F12 BH and H2O (AE & FC), plus rigid-translation invariance.
"""

from __future__ import annotations

import hashlib
import os
import importlib.util
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from conftest import f12_host_libraries

from generativeqc._f12_focal_explicit_tf12 import (
    explicit_tf12_df_cpu,
    explicit_tf12_exact4_cpu,
    spatial_rhf_explicit_tf12_energy,
)
from generativeqc._f12_focal_one_electron import one_electron_sources
from generativeqc._f12_focal_reference import FocalReference, df_reference
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalSeries,
    MolecularFocalProvider,
    plan_singlepoint,
)
from _bse_basis_support import br_ecp10mdf


def _explicit_tf12_disabled_detail() -> str | None:
    import ctypes as ct

    from generativeqc import _native

    lib = _native.load_library(device="cpu")
    fn = getattr(lib, "generativeqc_rcc_explicit_tf12_disabled_detail_v1", None)
    if fn is None:
        return None
    fn.argtypes = []
    fn.restype = ct.c_char_p
    raw = fn()
    return None if raw is None else raw.decode("utf-8")


def _require_explicit_tf12_enabled() -> None:
    detail = _explicit_tf12_disabled_detail()
    if detail is not None:
        pytest.skip(detail)


def test_explicit_tf12_disabled_stub_contract():
    """Verify default-OFF stub returns GENERATIVEQC_STATUS_NOT_IMPLEMENTED cleanly without mutating outputs."""
    import ctypes as ct

    from generativeqc import _native

    lib = _native.load_library(device="cpu")
    detail = _explicit_tf12_disabled_detail()
    if detail is None:
        # Built with -DGENERATIVEQC_ENABLE_EXPLICIT_TF12=ON: engine is enabled.
        return
    assert "-DGENERATIVEQC_ENABLE_EXPLICIT_TF12=ON" in detail

    ptr = ct.POINTER(ct.c_double)
    zeros = (ct.c_double * 1)(0.0)
    energy = ct.c_double(123.0)
    breakdown = (ct.c_double * 30)(*([456.0] * 30))

    df_cpu = lib.generativeqc_rcc_explicit_tf12_df_cpu_v1
    df_cpu.argtypes = [ct.c_size_t] * 6 + [ptr] * 18 + [ct.c_uint32, ct.c_double, ptr, ptr, ct.c_size_t]
    df_cpu.restype = ct.c_int
    assert df_cpu(
        1, 0, 2, 3, 1, 0, *([zeros] * 18), 2, 1.0, ct.byref(energy), breakdown, 30
    ) == _native.STATUS_NOT_IMPLEMENTED

    exact4_cpu = lib.generativeqc_rcc_explicit_tf12_exact4_cpu_v1
    exact4_cpu.argtypes = [ct.c_size_t] * 4 + [ptr] * 19 + [ct.c_uint32, ct.c_double, ptr, ptr, ct.c_size_t]
    exact4_cpu.restype = ct.c_int
    assert exact4_cpu(
        1, 0, 2, 3, *([zeros] * 19), 2, 1.0, ct.byref(energy), breakdown, 30
    ) == _native.STATUS_NOT_IMPLEMENTED

    df_cuda = lib.generativeqc_rcc_explicit_tf12_df_cuda_v1
    df_cuda.argtypes = (
        [ct.c_int, ct.c_size_t, ct.c_int]
        + [ct.c_size_t] * 6
        + [ptr] * 18
        + [ct.c_uint32, ct.c_double, ptr, ptr, ct.c_size_t]
    )
    df_cuda.restype = ct.c_int
    assert df_cuda(
        0, 64 << 20, 0, 1, 0, 2, 3, 1, 0, *([zeros] * 18), 2, 1.0, ct.byref(energy), breakdown, 30
    ) == _native.STATUS_NOT_IMPLEMENTED
    assert (energy.value, breakdown[0]) == (123.0, 456.0)


def test_native_df_triples_rejects_short_breakdown_before_publishing_energy():
    _require_explicit_tf12_enabled()
    import ctypes as ct

    from generativeqc import _native

    fn = _native.load_library(device="cpu").generativeqc_rcc_explicit_tf12_df_cpu_v1
    ptr = ct.POINTER(ct.c_double)
    fn.argtypes = [ct.c_size_t] * 6 + [ptr] * 18 + [ct.c_uint32, ct.c_double, ptr, ptr, ct.c_size_t]
    fn.restype = ct.c_int
    zeros = (ct.c_double * 1)(0)
    occupied = (ct.c_double * 1)(-1)
    virtual = (ct.c_double * 1)(1)
    energy = ct.c_double(123.0)
    breakdown = (ct.c_double * 1)(456.0)
    status = fn(1, 0, 2, 2, 1, 0, *([zeros] * 7), occupied, virtual, *([None] * 9),
                2, 1.0, ct.byref(energy), breakdown, len(breakdown))
    assert status == 1
    assert (energy.value, breakdown[0]) == (123.0, 456.0)

def _oracle_dir(variable, checks):
    path = os.environ.get(variable)
    if not path:
        pytest.skip(f"Set {variable} to the GeCCo fixtures for this independent oracle")
    directory = Path(path)
    for name, expected in checks.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    return directory


def _make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
    prov = BasisProvenance(source=name, version="1", license="MIT", checksum="0" * 64)
    return BasisSet(name, elems, prov, representation="spherical")


def _sh(l: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
    return BasisShell(l, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),))


def _compact_split_basis(elements: tuple[int, ...], *, beta: float = 0.9) -> FocalBasis:
    """Compact spherical split-fit (JKFit != MP2Fit == F12Fit, OptRI) basis for H, B, C, O, Br(ECP10MDF)."""
    br_ecp = br_ecp10mdf()
    orb_map = {
        1: ElementBasis(
            1,
            (
                _sh(0, (3.42525091, 0.62391373), (0.15432897, 0.53532814)),
                _sh(0, (0.16885540,), (1.0,)),
            ),
        ),
        5: ElementBasis(
            5,
            (
                _sh(0, (48.791113, 8.8873622, 2.405267), (0.15432897, 0.53532814, 0.44463454)),
                _sh(0, (2.2369561, 0.5198205), (-0.09996723, 0.39951283)),
                _sh(0, (0.1690618,), (1.0,)),
                _sh(1, (2.2369561, 0.5198205), (0.15591627, 0.60768372)),
                _sh(1, (0.1690618,), (1.0,)),
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
        5: ElementBasis(5, (_sh(0, (24.0,), (1.0,)), _sh(0, (6.0,), (1.0,)), _sh(0, (1.5,), (1.0,)), _sh(0, (0.38,), (1.0,)), _sh(1, (3.2,), (1.0,)), _sh(1, (0.8,), (1.0,)), _sh(2, (1.2,), (1.0,)))),
        6: ElementBasis(6, (_sh(0, (36.0,), (1.0,)), _sh(0, (9.0,), (1.0,)), _sh(0, (2.2,), (1.0,)), _sh(0, (0.55,), (1.0,)), _sh(1, (4.5,), (1.0,)), _sh(1, (1.1,), (1.0,)), _sh(2, (1.6,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (60.0,), (1.0,)), _sh(0, (15.0,), (1.0,)), _sh(0, (4.0,), (1.0,)), _sh(0, (1.1,), (1.0,)), _sh(1, (7.0,), (1.0,)), _sh(1, (1.8,), (1.0,)), _sh(2, (2.2,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (32.0,), (1.0,)), _sh(0, (9.0,), (1.0,)), _sh(0, (2.5,), (1.0,)), _sh(0, (0.7,), (1.0,)), _sh(0, (0.22,), (1.0,)), _sh(1, (12.0,), (1.0,)), _sh(1, (3.2,), (1.0,)), _sh(1, (0.85,), (1.0,)), _sh(2, (4.5,), (1.0,)), _sh(2, (1.2,), (1.0,)))),
    }
    mp2_map = {
        1: ElementBasis(1, (_sh(0, (5.2,), (1.0,)), _sh(0, (1.2,), (1.0,)), _sh(0, (0.32,), (1.0,)), _sh(1, (1.2,), (1.0,)))),
        5: ElementBasis(5, (_sh(0, (19.0,), (1.0,)), _sh(0, (4.8,), (1.0,)), _sh(0, (1.2,), (1.0,)), _sh(0, (0.32,), (1.0,)), _sh(1, (2.7,), (1.0,)), _sh(1, (0.68,), (1.0,)), _sh(2, (1.0,), (1.0,)))),
        6: ElementBasis(6, (_sh(0, (28.0,), (1.0,)), _sh(0, (7.2,), (1.0,)), _sh(0, (1.8,), (1.0,)), _sh(0, (0.45,), (1.0,)), _sh(1, (3.8,), (1.0,)), _sh(1, (0.95,), (1.0,)), _sh(2, (1.3,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (48.0,), (1.0,)), _sh(0, (11.0,), (1.0,)), _sh(0, (2.8,), (1.0,)), _sh(0, (0.75,), (1.0,)), _sh(1, (5.8,), (1.0,)), _sh(1, (1.4,), (1.0,)), _sh(2, (1.9,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (26.0,), (1.0,)), _sh(0, (7.5,), (1.0,)), _sh(0, (2.1,), (1.0,)), _sh(0, (0.58,), (1.0,)), _sh(0, (0.18,), (1.0,)), _sh(1, (9.5,), (1.0,)), _sh(1, (2.6,), (1.0,)), _sh(1, (0.70,), (1.0,)), _sh(2, (3.8,), (1.0,)), _sh(2, (1.0,), (1.0,)))),
    }
    cabs_map = {
        1: ElementBasis(1, (_sh(0, (0.8,), (1.0,)), _sh(1, (0.9,), (1.0,)))),
        5: ElementBasis(5, (_sh(0, (1.1,), (1.0,)), _sh(1, (1.1,), (1.0,)), _sh(2, (0.65,), (1.0,)))),
        6: ElementBasis(6, (_sh(0, (1.4,), (1.0,)), _sh(1, (1.4,), (1.0,)), _sh(2, (0.8,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (2.0,), (1.0,)), _sh(1, (2.0,), (1.0,)), _sh(2, (1.2,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (1.1,), (1.0,)), _sh(1, (1.1,), (1.0,)), _sh(2, (1.4,), (1.0,)))),
    }
    uniq = tuple(sorted(set(elements)))
    orb = _make_basis("gate1-tf12-orb", tuple(orb_map[z] for z in uniq))
    jk = _make_basis("gate1-tf12-jk", tuple(jk_map[z] for z in uniq))
    mp2 = _make_basis("gate1-tf12-mp2", tuple(mp2_map[z] for z in uniq))
    cabs = _make_basis("gate1-tf12-cabs", tuple(cabs_map[z] for z in uniq))
    return FocalBasis(2, "gate1-tf12-split", orb, jk_fit=jk, mp2_fit=mp2, f12_fit=mp2, cabs=cabs, beta=beta)


def _basis_set_to_pyscf(basis: BasisSet) -> tuple[dict, dict]:
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


def _solve_rhf_ref(provider: MolecularFocalProvider, req) -> FocalReference:
    has_ecp = any(
        req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons for a in req.atoms
    )
    if not has_ecp:
        return df_reference(req, forces=False, budget_mib=provider.budget_mib)

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
    ao3 = np.asarray(jk.obs_coulomb(one.obs_ao_count), dtype=np.float64)
    B = np.einsum("PQ,Qmn->Pmn", jk.whitener, ao3, optimize=True)
    n_elec = (
        sum(
            a.atomic_number - req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons
            for a in req.atoms
        )
        - req.charge
    )
    nocc = n_elec // 2
    enuc = 0.0
    for i, a in enumerate(req.atoms):
        zi = a.atomic_number - req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons
        for j in range(i + 1, len(req.atoms)):
            aj = req.atoms[j]
            zj = aj.atomic_number - req.basis.orbital.by_element[aj.atomic_number].ecp_core_electrons
            r = float(np.linalg.norm(np.array(a.position) - np.array(aj.position)))
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
        F = 0.5 * ((H + J - K) + (H + J - K).T)
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
    e_tot = enuc + 0.5 * float(np.sum(D * (H + F)))
    return FocalReference(
        surface_id=req.surface_id,
        basis_id=req.basis.basis_id,
        orbital_id=req.basis.orbital.identity,
        jk_fit_id=req.basis.jk_fit.identity,
        executed_backend=req.backend,
        energy=e_tot,
        iterations=it,
        occupied=(nocc,),
        overlap=S,
        hcore=H,
        coefficients=(C,),
        density=(D,),
        fock=(F,),
        orbital_energies=(eps,),
        forces=None,
    )


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
    ],
)
def test_zero_coupling_matches_pyscf_df_rccsdt(
    label: str,
    atoms_spec: tuple[tuple[str, tuple[float, float, float]], ...],
    elements: tuple[int, ...],
    corr_space: str,
) -> None:
    """Gate 1: Zero-coupling (T)-F12 matches external PySCF 2.14.0 DF-RCCSD(T) < 1e-11 Eh."""
    _require_explicit_tf12_enabled()
    pyscf = pytest.importorskip("pyscf")
    from pyscf import cc, gto, scf

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
    ref = _solve_rhf_ref(provider, req)
    res = spatial_rhf_explicit_tf12_energy(provider, req, ref, zero_coupling=True)
    frozen = int(res.telemetry["frozen_occupied"])

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
    assert abs(float(mf.e_tot) - float(ref.energy)) < 1e-11

    mycc = cc.RCCSD(mf, frozen=frozen if frozen > 0 else None).density_fit(auxbasis=pyscf_mp2)
    mycc.conv_tol = 1e-13
    mycc.conv_tol_normt = 1e-12
    mycc.max_cycle = 200
    mycc.kernel()
    assert mycc.converged
    e_t_pyscf = float(mycc.ccsd_t())

    assert res.ccsd_f12b_energy is not None
    ccsd_diff = abs(res.ccsd_f12b_energy - float(mycc.e_corr))
    t_diff = abs(res.energy - e_t_pyscf)
    print(
        f"[PYSCF {pyscf.__version__} ZERO-COUPLING] {label} ({corr_space}): "
        f"E_(T)={res.energy:+.12f} vs PySCF={e_t_pyscf:+.12f}, "
        f"t_diff={t_diff:.3e} Eh, ccsd_diff={ccsd_diff:.3e} Eh"
    )
    assert ccsd_diff < 1e-11
    assert t_diff < 1e-11
    assert abs(res.breakdown["a_sum"]) == 0.0
    assert abs(res.breakdown["delta_c2"]) == 0.0


def test_gecco_physical_triples_action_contract() -> None:
    """Gate 1: Verify the pinned GeCCo physical triples action subgraph (terms 31, 32)."""
    directory = _oracle_dir("GENERATIVEQC_F12_GECCO_CONTRACT", {
        "gecco_physical_triples_action.py": "509f0da2c7b24841f9a790de92d8e2bdb45bbf291451b20e146d51601c95a3c7",
        "gecco_ccpt_graphs.json": "28a64ac7b0bd9e776ac11c50256526e0399c98935a956eff8179cb3e1e17e285",
    })
    graph_path = directory / "gecco_ccpt_graphs.json"
    action_py = directory / "gecco_physical_triples_action.py"
    spec = importlib.util.spec_from_file_location("gecco_physical_triples_action", action_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    evidence = mod.test_action(graph_path)
    assert evidence["state"] == "physical_triples_one_body_action_verified"
    assert evidence["terms"] == [31, 32]
    assert evidence["full_noncanonical_action_max_difference"] < 2e-13
    assert evidence["canonical_diagonal_action_max_difference"] < 2e-13
    assert evidence["canonical_signed_linear_solve_residual"] < 2e-13


def _load_gecco_bh_arrays(directory):
    """Read archived GeCCo BH DALTON MO integrals & T0 parent amplitudes (read-only)."""
    N, O, OBS = 76, 3, 19
    V = OBS - O

    def read_1e(path: Path) -> np.ndarray:
        data = path.read_bytes()
        pos = 8 + struct.unpack("<i", data[:4])[0]
        M = np.zeros((N, N), dtype=np.float64)
        while pos < len(data):
            n = struct.unpack("<i", data[pos : pos + 4])[0]
            rec = data[pos + 4 : pos + 4 + n]
            pos += 8 + n
            length = struct.unpack("<q", rec[:8])[0]
            if length <= 0:
                break
            idx = np.frombuffer(rec[8 : 8 + 4 * length], dtype="<i2").reshape(length, 2) - 1
            vals = np.frombuffer(rec[8 + 4 * length : 8 + 12 * length], dtype="<f8")
            for (p, q), val in zip(idx, vals, strict=True):
                M[p, q] = M[q, p] = val
        return M

    def read_2e_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
        data = path.read_bytes()
        pos = 8 + struct.unpack("<i", data[:4])[0]
        all_idx, all_val = [], []
        while pos < len(data):
            n = struct.unpack("<i", data[pos : pos + 4])[0]
            rec = data[pos + 4 : pos + 4 + n]
            pos += 8 + n
            length = struct.unpack("<q", rec[:8])[0]
            if length <= 0:
                break
            idx = np.frombuffer(rec[8 : 8 + 8 * length], dtype="<i2").reshape(length, 4) - 1
            vals = np.frombuffer(rec[8 + 8 * length : 8 + 16 * length], dtype="<f8")
            all_idx.append(idx)
            all_val.append(vals)
        return np.vstack(all_idx), np.concatenate(all_val)

    def unpack_sym(path: Path, shape: tuple[int, ...]) -> np.ndarray:
        idx, vals = read_2e_raw(path)
        T = np.zeros(shape, dtype=np.float64)
        p, q, r, s = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        for a, b, c, d_ in (
            (p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
            (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p),
        ):
            m = (a < shape[0]) & (b < shape[1]) & (c < shape[2]) & (d_ < shape[3])
            T[a[m], b[m], c[m], d_[m]] = vals[m]
        return T

    F = read_1e(directory / "MO_F")
    K = read_1e(directory / "MO_K")
    eps = np.diag(F)[:OBS]
    eps_o = np.ascontiguousarray(eps[:O])
    eps_v = np.ascontiguousarray(eps[O:OBS])
    fov = np.ascontiguousarray(F[:O, O:OBS])

    g_chem = unpack_sym(directory / "MO_G", (N, N, N, OBS))
    g_obs = g_chem[:OBS, :OBS, :OBS, :OBS]
    f12 = np.ascontiguousarray(unpack_sym(directory / "MO_F12", (N, OBS, N, OBS)).transpose(0, 2, 1, 3))
    fg = np.ascontiguousarray(unpack_sym(directory / "MO_FG", (OBS, OBS, OBS, OBS)).transpose(0, 2, 1, 3))
    ff = np.ascontiguousarray(unpack_sym(directory / "MO_FF", (OBS, OBS, OBS, OBS)).transpose(0, 2, 1, 3))
    ftf = np.ascontiguousarray(unpack_sym(directory / "MO_FTF", (OBS, OBS, OBS, OBS)).transpose(0, 2, 1, 3))
    idx_fb, val_fb = read_2e_raw(directory / "MO_FFBAR")
    ffbar_c = np.zeros((OBS, OBS, OBS, OBS), dtype=np.float64)
    for (p, q, r, s), val in zip(idx_fb, val_fb, strict=True):
        ffbar_c[p, q, r, s] = ffbar_c[r, s, p, q] = val
    ffbar = np.ascontiguousarray(ffbar_c.transpose(0, 2, 1, 3))

    # Unpack GeCCo parent spatial T1, T2 from T0_list.da
    irrep_v = [list(range(0, 7)), list(range(7, 11)), list(range(11, 15)), [15]]

    def gen_ms0_pairs(grps):
        out = []
        for grp in grps:
            so_list = [2 * p + s for p in grp for s in (0, 1)]
            for q_idx, Q in enumerate(so_list):
                for p_idx in range(q_idx):
                    P = so_list[p_idx]
                    if (P & 1) != (Q & 1):
                        out.append((P, Q))
        return out

    ab_v, ab_o = gen_ms0_pairs(irrep_v), gen_ms0_pairs([list(range(O))])
    t0_da = np.frombuffer((directory / "T0_list.da").read_bytes(), dtype="<f8")[:978]
    t1 = np.zeros((O, V), dtype=np.float64)
    idx = 0
    for i in range(O):
        for a in range(7):
            t1[i, a] = t0_da[idx]
            idx += 1
    t2_so = np.zeros((2 * O, 2 * O, 2 * V, 2 * V), dtype=np.float64)
    idx = 42 + 99
    for i, j in ab_o:
        for a, b in ab_v:
            val = t0_da[idx]
            idx += 1
            t2_so[i, j, a, b] = t2_so[j, i, b, a] = val
            t2_so[j, i, a, b] = t2_so[i, j, b, a] = -val
    t2 = np.ascontiguousarray(t2_so[0::2, 1::2, 0::2, 1::2])

    f_d = 0.375 * f12 + 0.125 * np.swapaxes(f12, -2, -1)
    c2_ab = np.ascontiguousarray(
        np.einsum("ax,xbkd->abkd", F[O:OBS, OBS:], f_d[OBS:, O:OBS, :O, O:OBS])
        + np.einsum("axkd,bx->abkd", f_d[O:OBS, OBS:, :O, O:OBS], F[O:OBS, OBS:])
    )
    g_obs_obs_ext_ext = np.ascontiguousarray(g_chem[:, :OBS, :, :OBS].transpose(1, 3, 0, 2))
    g_chem_ext_ext_o_v = np.ascontiguousarray(g_chem[:, :, :O, O:OBS])
    g_chem_ext_o_ext_v = np.ascontiguousarray(g_chem[:, :O, :, O:OBS])

    return {
        "o_full": O, "frozen": 0, "obs": OBS, "ext": N,
        "ovvv": np.ascontiguousarray(g_obs[:O, O:, O:, O:]),
        "ovoo": np.ascontiguousarray(g_obs[:O, O:, :O, :O]),
        "ovov": np.ascontiguousarray(g_obs[:O, O:, :O, O:]),
        "fov": fov, "t1": t1, "t2": t2, "eps_o": eps_o, "eps_v": eps_v,
        "c2_ab": c2_ab,
        "g_obs_obs_ext_ext": g_obs_obs_ext_ext,
        "g_chem_ext_ext_o_v": g_chem_ext_ext_o_v,
        "g_chem_ext_o_ext_v": g_chem_ext_o_ext_v,
        "f12": f12, "fg": fg, "ff": ff, "ftf": ftf, "ffbar": ffbar,
        "f_ext": F, "k_ext": K,
    }


def test_gecco_bh_33term_exact4_parity() -> None:
    """Gate 1: Native C++ spatial contractions match GeCCo BH CCPT-R12 oracle to < 1e-11 Eh."""
    _require_explicit_tf12_enabled()
    directory = _oracle_dir("GENERATIVEQC_F12_GECCO_BH", {
        "MO_F": "b3abfb67b24aac731b664f31ab02fb5568065d07d06e950ccfe2df39a0a45b5d",
        "MO_K": "0a8b42c62303c7121eb35c8a1fe951e19d6ce9be39e334d3593f7d7b03d5b678",
        "MO_G": "a80e07709492bdbfeb5a8b8bebcf0eedec25197175b786b948910242d8dab5e0",
        "MO_F12": "0df2aede94af28cc7ecf20b0f7660582e5ad31f2c2345cdf094d5158b7863c88",
        "MO_FG": "2e4ef4117d58ef1e80d9c711f75a3948d7f70658860dcf4db2135d7c03c36c63",
        "MO_FF": "7c63e9141c2136b80e051fa47369e6870abe71e9bcce4819eb6696c553ec3773",
        "MO_FTF": "f4cdde1c5bb38e8544b3181a68077edf32382b28fdc61410c53a09b345995600",
        "MO_FFBAR": "188370397e2e9cef8653c1672b7071fb1d69263d80e4db9f565cc616ea660134",
        "T0_list.da": "d224589fd38a2a11badb2bd4cb75d052ce5cb6345ce49feb6fcf6cb61b7fd7c2",
    })
    oracle_path = _oracle_dir("GENERATIVEQC_F12_GECCO_CONTRACT", {
        "gecco_bh_oracle.json": "5f8135f02a6c59877c9e6a56f524037e40d314cdef922f63b4e731e291b760cd",
    }) / "gecco_bh_oracle.json"
    oracle = json.loads(oracle_path.read_text())
    expected_full = float(oracle["results"]["CCPT-R12"]["full_noniterative_triples_Eh"])
    arrays = _load_gecco_bh_arrays(directory)

    e_ffg, bk_ffg = explicit_tf12_exact4_cpu(**arrays, z2_mode="ffg", geminal_scale=1.0)
    e_dir, bk_dir = explicit_tf12_exact4_cpu(**arrays, z2_mode="direct", geminal_scale=1.0)

    print(
        f"[GECCO BH 33-TERM C++ SPATIAL] ffg={e_ffg:+.12f} (oracle={expected_full:+.12f}, "
        f"diff={e_ffg - expected_full:+.3e} Eh), direct={e_dir:+.12f} (diff_vs_gecco_dir={e_dir - (-0.002280453001):+.3e} Eh)"
    )
    assert abs(e_ffg - expected_full) < 1e-11
    assert abs(e_dir - (-0.002280453001)) < 1e-11

    # Assert exact individual GeCCo term-group values (all printed to 10+ digits in bh_gecco_replay.log.gz)
    assert abs(bk_ffg["t4_x2_h7"] - 0.1631141451e-04) < 1e-11
    assert abs(bk_ffg["t15_x3_h9"] - 0.1631141451e-04) < 1e-11
    assert abs(bk_ffg["t3_xh_h7"] - 0.8275107515e-05) < 1e-11
    assert abs(bk_ffg["t7_t1_v7"] - 0.3303016573e-04) < 1e-11
    assert abs(bk_ffg["t8_v8"] - (-0.3682399918e-02)) < 1e-11
    assert abs(bk_ffg["t16_bh4"] - 0.2876381705e-03) < 1e-11
    assert abs(bk_ffg["t17_fo_x7"] - (-0.3549591019e-04)) < 1e-11
    assert abs(bk_ffg["t18_x7_fo"] - (-0.3549591019e-04)) < 1e-11
    assert abs(bk_ffg["t19_fo_x4"] - 0.1291690897e-03) < 1e-11
    assert abs(bk_ffg["t20_fo_x7_ex"] - (-0.3437548897e-04)) < 1e-11
    assert abs(bk_ffg["t21_b4"] - 0.1008046676e-02) < 1e-11
    assert abs(bk_ffg["t23_fv_x4"] - 0.4807986214e-04) < 1e-11
    assert abs(bk_ffg["t24_fv_x7"] - (-0.2639069835e-04)) < 1e-11
    assert abs((bk_ffg["t9_v10"] + bk_ffg["t22_b7"] + bk_ffg["t5_z2"]) - 0.1204533303e-02) < 1e-11
    assert abs(bk_ffg["e_pt_with_c2"] - (-0.001249743972)) < 1e-11
    assert abs(bk_dir["t5_z2"] - (-0.000135309926)) < 1e-11


def test_df_vs_exact4_algebraic_equivalence_with_frozen_core() -> None:
    """Gate 1: 3-index DF C++ engine matches exact-4-index C++ engine < 1e-12 Eh (with frozen core & 1/beta)."""
    _require_explicit_tf12_enabled()
    rng = np.random.default_rng(20261123)
    o_full, frozen, obs, ext, q_mp2, q_f12 = 3, 1, 7, 12, 14, 15
    o, v = o_full - frozen, obs - o_full

    # Build symmetric 3-index DF factors (ext x ext symmetric for Coulomb, obs x obs symmetric for OBS slices)
    raw_g_ee = rng.normal(scale=0.25, size=(q_f12, ext, ext))
    b_g_ext_ext = 0.5 * (raw_g_ee + raw_g_ee.swapaxes(1, 2))
    b_g_ext_obs = np.ascontiguousarray(b_g_ext_ext[:, :, :obs])
    # Random invertible whitener -> fit = W.T @ b_g (for robust DF)
    fit_ee = rng.normal(scale=0.2, size=(q_f12, ext, ext))
    fit_ee = 0.5 * (fit_ee + fit_ee.swapaxes(1, 2))
    fit_ext_obs = np.ascontiguousarray(fit_ee[:, :, :obs])

    def sym_ext_obs():
        m = rng.normal(scale=0.15, size=(q_f12, ext, ext))
        m = 0.5 * (m + m.swapaxes(1, 2))
        return np.ascontiguousarray(m[:, :, :obs])

    r_f_ext_obs = sym_ext_obs()
    r_ff_ext_obs = sym_ext_obs()
    r_fg_obs_obs = np.ascontiguousarray(sym_ext_obs()[:, :obs, :])
    r_u_obs_obs = np.ascontiguousarray(sym_ext_obs()[:, :obs, :])

    # Distinct MP2Fit factors for split-fit check
    b_mp2_obs = rng.normal(scale=0.25, size=(q_mp2, obs, obs))
    b_mp2_obs = 0.5 * (b_mp2_obs + b_mp2_obs.swapaxes(1, 2))
    boo = np.ascontiguousarray(b_mp2_obs[:, frozen:o_full, frozen:o_full])
    bov = np.ascontiguousarray(b_mp2_obs[:, frozen:o_full, o_full:obs])
    bvv = np.ascontiguousarray(b_mp2_obs[:, o_full:obs, o_full:obs])
    ovoo = np.einsum("qia,qjk->iajk", bov, boo)
    ovov = np.einsum("qia,qjb->iajb", bov, bov)
    ovvv = np.einsum("qia,qbc->iabc", bov, bvv)

    f_ext = rng.normal(scale=0.1, size=(ext, ext))
    f_ext = 0.5 * (f_ext + f_ext.T)
    eps_o = np.array([-0.85, -0.45], dtype=np.float64)
    eps_v = np.array([0.25, 0.55, 0.90, 1.35], dtype=np.float64)
    f_ext[:obs, :obs] = 0.0
    f_ext[0, 0] = -4.5
    for i in range(o):
        f_ext[frozen + i, frozen + i] = eps_o[i]
    for a in range(v):
        f_ext[o_full + a, o_full + a] = eps_v[a]
    fov = np.zeros((o, v), dtype=np.float64)
    k_ext = rng.normal(scale=0.1, size=(ext, ext))
    k_ext = 0.5 * (k_ext + k_ext.T)

    t1 = rng.normal(scale=0.05, size=(o, v))
    t2 = rng.normal(scale=0.05, size=(o, o, v, v))
    t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))

    # Reconstruct exact 4-index tensors from the 3-index DF factors!
    def robust_phys(fL, rL, fR, rR):
        chem = np.einsum("qpr,qst->prst", fL, rR) + np.einsum("qpr,qst->prst", rL, fR)
        return np.ascontiguousarray(chem.transpose(0, 2, 1, 3))

    f12 = robust_phys(fit_ext_obs, r_f_ext_obs, fit_ext_obs, r_f_ext_obs)  # [ext, ext, obs, obs]
    fit_oo = fit_ext_obs[:, :obs, :]
    r_ff_oo = r_ff_ext_obs[:, :obs, :]
    fg = robust_phys(fit_oo, r_fg_obs_obs, fit_oo, r_fg_obs_obs)
    ff = robust_phys(fit_oo, r_ff_oo, fit_oo, r_ff_oo)
    ftf = robust_phys(fit_oo, r_u_obs_obs, fit_oo, r_u_obs_obs)
    # ffbar = <p q | (F+K)_1 f^2 + (F+K)_2 f^2 | r s> with (F+K) acting through full ext!
    fpk_oe = f_ext[:obs, :] + k_ext[:obs, :]
    fk_f = np.einsum("pt,qtr->qpr", fpk_oe, fit_ext_obs)
    fk_rff = np.einsum("pt,qtr->qpr", fpk_oe, r_ff_ext_obs)
    ffbar = np.ascontiguousarray(
        (
            np.einsum("qpr,qst->pqrt" if False else "qpr,qst->prst", fk_f, r_ff_oo)
            + np.einsum("qpr,qst->prst", fk_rff, fit_oo)
            + np.einsum("qpr,qst->prst", fit_oo, fk_rff)
            + np.einsum("qpr,qst->prst", r_ff_oo, fk_f)
        ).transpose(0, 2, 1, 3)
    )
    g_obs_obs_ext_ext = np.ascontiguousarray(
        np.einsum("qap,qbq->pqab", b_g_ext_obs, b_g_ext_obs)
        if False
        else np.einsum("qap,qbr->prab", b_g_ext_obs, b_g_ext_obs)
    )
    g_chem_ext_ext_o_v = np.ascontiguousarray(
        np.einsum("qab,qkc->abkc", b_g_ext_ext, b_g_ext_obs[:, frozen:o_full, o_full:obs])
    )
    g_chem_ext_o_ext_v = np.ascontiguousarray(
        np.einsum("qbk,qac->bkac", b_g_ext_obs[:, :, frozen:o_full], b_g_ext_obs[:, :, o_full:obs])
    )
    f_d = 0.375 * f12 + 0.125 * np.swapaxes(f12, -2, -1)
    c2_ab = np.ascontiguousarray(
        np.einsum("ax,xbkd->abkd", f_ext[o_full:obs, obs:], f_d[obs:, o_full:obs, frozen:o_full, o_full:obs])
        + np.einsum("axkd,bx->abkd", f_d[o_full:obs, obs:, frozen:o_full, o_full:obs], f_ext[o_full:obs, obs:])
    )

    for z2_mode in ("direct", "ffg", "zero"):
        for gscale in (1.0, 1.0 / 0.9):
            e_df, bk_df = explicit_tf12_df_cpu(
                o_full=o_full, frozen=frozen, obs=obs, ext=ext, q_mp2=q_mp2, q_f12=q_f12,
                bov=bov, bvv=bvv, ovoo=ovoo, ovov=ovov, fov=fov,
                t1=t1, t2=t2, eps_o=eps_o, eps_v=eps_v,
                b_g_ext_obs=b_g_ext_obs, b_g_ext_ext=b_g_ext_ext, fit_ext_obs=fit_ext_obs,
                r_f_ext_obs=r_f_ext_obs, r_fg_obs_obs=r_fg_obs_obs, r_ff_ext_obs=r_ff_ext_obs,
                r_u_obs_obs=r_u_obs_obs, f_ext=f_ext, k_ext=k_ext,
                z2_mode=z2_mode, geminal_scale=gscale,
            )
            e_ex, bk_ex = explicit_tf12_exact4_cpu(
                o_full=o_full, frozen=frozen, obs=obs, ext=ext,
                ovvv=ovvv, ovoo=ovoo, ovov=ovov, fov=fov,
                t1=t1, t2=t2, eps_o=eps_o, eps_v=eps_v, c2_ab=c2_ab,
                g_obs_obs_ext_ext=g_obs_obs_ext_ext,
                g_chem_ext_ext_o_v=g_chem_ext_ext_o_v,
                g_chem_ext_o_ext_v=g_chem_ext_o_ext_v,
                f12=f12, fg=fg, ff=ff, ftf=ftf, ffbar=ffbar,
                f_ext=f_ext, k_ext=k_ext, z2_mode=z2_mode, geminal_scale=gscale,
            )
            assert abs(e_df - e_ex) < 1e-12, f"DF vs exact4 mismatch ({z2_mode}, s={gscale}): {e_df} vs {e_ex}"
            for key in (
                "e_pt_with_c2", "e_pt_conv", "t3_xh_h7", "t4_x2_h7", "t5_z2",
                "t6_t1_v9", "t7_t1_v7", "t8_v8", "t9_v10", "t16_bh4",
                "t17_fo_x7", "t19_fo_x4", "t20_fo_x7_ex", "t21_b4", "t22_b7",
                "t23_fv_x4", "t24_fv_x7", "a_sum",
            ):
                assert abs(bk_df[key] - bk_ex[key]) < 1e-12, f"term {key} mismatch: {bk_df[key]} vs {bk_ex[key]}"
    # Verify exact s vs s^2 scaling law between gscale=1.0 and gscale=1/0.9 at fixed (t1, t2)!
    s = 1.0 / 0.9
    _, b1 = explicit_tf12_df_cpu(
        o_full=o_full, frozen=frozen, obs=obs, ext=ext, q_mp2=q_mp2, q_f12=q_f12,
        bov=bov, bvv=bvv, ovoo=ovoo, ovov=ovov, fov=fov, t1=t1, t2=t2, eps_o=eps_o, eps_v=eps_v,
        b_g_ext_obs=b_g_ext_obs, b_g_ext_ext=b_g_ext_ext, fit_ext_obs=fit_ext_obs,
        r_f_ext_obs=r_f_ext_obs, r_fg_obs_obs=r_fg_obs_obs, r_ff_ext_obs=r_ff_ext_obs,
        r_u_obs_obs=r_u_obs_obs, f_ext=f_ext, k_ext=k_ext, z2_mode="ffg", geminal_scale=1.0,
    )
    _, bs = explicit_tf12_df_cpu(
        o_full=o_full, frozen=frozen, obs=obs, ext=ext, q_mp2=q_mp2, q_f12=q_f12,
        bov=bov, bvv=bvv, ovoo=ovoo, ovov=ovov, fov=fov, t1=t1, t2=t2, eps_o=eps_o, eps_v=eps_v,
        b_g_ext_obs=b_g_ext_obs, b_g_ext_ext=b_g_ext_ext, fit_ext_obs=fit_ext_obs,
        r_f_ext_obs=r_f_ext_obs, r_fg_obs_obs=r_fg_obs_obs, r_ff_ext_obs=r_ff_ext_obs,
        r_u_obs_obs=r_u_obs_obs, f_ext=f_ext, k_ext=k_ext, z2_mode="ffg", geminal_scale=s,
    )
    for k_lin in ("t6_t1_v9", "t7_t1_v7", "t8_v8", "t9_v10"):
        assert abs(bs[k_lin] - s * b1[k_lin]) < 1e-13
    for k_quad in ("t3_xh_h7", "t4_x2_h7", "t5_z2", "t16_bh4", "t17_fo_x7", "t19_fo_x4", "t21_b4", "t22_b7", "t23_fv_x4", "t24_fv_x7"):
        assert abs(bs[k_quad] - (s * s) * b1[k_quad]) < 1e-13
