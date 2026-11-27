"""Gate 1 verification for dynamic-dimension AOT ordinary DF-MP2 (and MP2-F12 pairs).

Step 12.1 coverage (ordinary_df_mp2):
- Closed-shell RHF spatial spin-symmetric fast path (2*t - t^T + i <= j) vs
  general UHF AOT path (aa/bb/ab) vs TensorIR oracle on H2O, CH4, IHB93 (OH- * H2O).
- Open-shell UHF (oa != ob, fa != fb, unequal occupied weights) on OH and bare Br atom (ECP10MDF).
- ECP HBr (ECP10MDF) under all-active, FC4 (frozen-core), and FC9 (frozen-core-spd).
- Edge cases: o_act=1 (H2), oa_act=1/ob_act=0 (H atom), compute_response=False,
  bounded-memory fallback (tiny budget_bytes), finiteness/missing-symbol error guards,
  and strict 0 TensorIR calls in ordinary_df_ump2_unrelaxed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import f12_host_shell

from generativeqc import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc._f12_focal_one_electron import one_electron_sources
from generativeqc._f12_focal_reference import FocalReference
from generativeqc._f12_focal_ump2 import (
    FocalCoulombSource,
    ordinary_df_ump2_unrelaxed,
    reference_df_ump2,
)
from generativeqc._f12_frozen_core import resolve_frozen_occupied
from generativeqc._f12_mp2_pair_native import (
    bind_ordinary_df_mp2,
    run_ordinary_df_mp2_native,
)
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalSeries,
    MolecularFocalProvider,
    df_reference,
    plan_singlepoint,
)
from _bse_basis_support import br_ecp10mdf

# H2O, IHB93, OH and HBr are the GSCDB records W4-17_h2o,
# IHB100x10_05.012_OH-Oa__water--hydroxide_080, W4-17_oh and P34_P06_HBr
# (J. Liang and M. Head-Gordon, J. Chem. Theory Comput. 21, 12601 (2025);
# CC BY 4.0, see THIRD_PARTY_NOTICES.md), converted from Å to Bohr.
GEOM_H2O = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 1.8101686547790252)),
    ("H", (1.7554778816245764, 0.0, -0.4415970623550433)),
)
# Project methane: the Td layout of the other F12 tests displaced by round
# offsets (Bohr), so that no force component vanishes by symmetry.
GEOM_CH4 = (
    ("C", (0.02, -0.01, 0.01)),
    ("H", (1.25, 1.19, 1.13)),
    ("H", (-1.13, -1.27, 1.19)),
    ("H", (-1.21, 1.08, -1.24)),
    ("H", (1.17, -1.21, -1.10)),
)
GEOM_IHB93 = (
    ("O", (-0.14875916494149566, -1.7463483403282758, -1.5325739171875632)),
    ("H", (0.059884585630443576, -0.08313623318156121, -0.15465977807385586)),
    ("H", (-0.5589300330890784, -3.127114233011312, -0.43176062629401624)),
    ("O", (0.22212408920524368, 1.4074915673882387, 1.1577380419586902)),
    ("H", (-1.349583684417033, 2.3011601556349195, 1.0000618358015534)),
)
GEOM_OH = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 1.834281919177214)),
)
GEOM_HBR = (
    ("H", (0.0, 0.0, -2.5991784446895245)),
    ("Br", (0.0, 0.0, 0.07426245724554352)),
)
GEOM_BR = (("Br", (0.0, 0.0, 0.0)),)
GEOM_H2 = (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4)))
GEOM_H = (("H", (0.0, 0.0, 0.0)),)


@pytest.fixture(scope="module")
def host_sink() -> Path:
    return f12_host_shell()


def _sh(l: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
    return BasisShell(l, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),))


def _make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
    prov = BasisProvenance(source=name, version="1", license="MIT", checksum="0" * 64)
    return BasisSet(name, elems, prov, representation="spherical")


def _compact_split_basis(elements: tuple[int, ...]) -> FocalBasis:
    """Compact spherical split-fit (JKFit != MP2Fit == F12Fit, OptRI) basis for H, C, O, Br(ECP10MDF)."""
    br_ecp = br_ecp10mdf()
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
                _sh(1, (0.11,), (1.0,)),
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
        6: ElementBasis(6, (_sh(0, (36.0,), (1.0,)), _sh(0, (9.0,), (1.0,)), _sh(0, (2.2,), (1.0,)), _sh(0, (0.55,), (1.0,)), _sh(1, (4.5,), (1.0,)), _sh(1, (1.1,), (1.0,)), _sh(2, (1.6,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (60.0,), (1.0,)), _sh(0, (15.0,), (1.0,)), _sh(0, (4.0,), (1.0,)), _sh(0, (1.1,), (1.0,)), _sh(0, (0.28,), (1.0,)), _sh(1, (7.0,), (1.0,)), _sh(1, (1.8,), (1.0,)), _sh(1, (0.42,), (1.0,)), _sh(2, (2.2,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (32.0,), (1.0,)), _sh(0, (9.0,), (1.0,)), _sh(0, (2.5,), (1.0,)), _sh(0, (0.7,), (1.0,)), _sh(0, (0.22,), (1.0,)), _sh(1, (12.0,), (1.0,)), _sh(1, (3.2,), (1.0,)), _sh(1, (0.85,), (1.0,)), _sh(2, (4.5,), (1.0,)), _sh(2, (1.2,), (1.0,)))),
    }
    mp2_map = {
        1: ElementBasis(1, (_sh(0, (5.2,), (1.0,)), _sh(0, (1.2,), (1.0,)), _sh(0, (0.32,), (1.0,)), _sh(1, (1.2,), (1.0,)))),
        6: ElementBasis(6, (_sh(0, (28.0,), (1.0,)), _sh(0, (7.2,), (1.0,)), _sh(0, (1.8,), (1.0,)), _sh(0, (0.45,), (1.0,)), _sh(1, (3.8,), (1.0,)), _sh(1, (0.95,), (1.0,)), _sh(2, (1.3,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (48.0,), (1.0,)), _sh(0, (11.0,), (1.0,)), _sh(0, (2.8,), (1.0,)), _sh(0, (0.75,), (1.0,)), _sh(0, (0.22,), (1.0,)), _sh(1, (5.8,), (1.0,)), _sh(1, (1.4,), (1.0,)), _sh(1, (0.35,), (1.0,)), _sh(2, (1.9,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (26.0,), (1.0,)), _sh(0, (7.5,), (1.0,)), _sh(0, (2.1,), (1.0,)), _sh(0, (0.58,), (1.0,)), _sh(0, (0.18,), (1.0,)), _sh(1, (9.5,), (1.0,)), _sh(1, (2.6,), (1.0,)), _sh(1, (0.70,), (1.0,)), _sh(2, (3.8,), (1.0,)), _sh(2, (1.0,), (1.0,)))),
    }
    cabs_map = {
        1: ElementBasis(1, (_sh(0, (0.8,), (1.0,)), _sh(1, (0.9,), (1.0,)))),
        6: ElementBasis(6, (_sh(0, (1.4,), (1.0,)), _sh(1, (1.4,), (1.0,)), _sh(2, (0.8,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (2.0,), (1.0,)), _sh(1, (2.0,), (1.0,)), _sh(2, (1.2,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (1.1,), (1.0,)), _sh(1, (1.1,), (1.0,)), _sh(2, (1.4,), (1.0,)))),
    }
    uniq = tuple(sorted(set(elements)))
    orb = _make_basis("gate1-aot-orb", tuple(orb_map[z] for z in uniq))
    jk = _make_basis("gate1-aot-jk", tuple(jk_map[z] for z in uniq))
    mp2 = _make_basis("gate1-aot-mp2", tuple(mp2_map[z] for z in uniq))
    cabs = _make_basis("gate1-aot-cabs", tuple(cabs_map[z] for z in uniq))
    return FocalBasis(2, "gate1-aot-split", orb, jk_fit=jk, mp2_fit=mp2, f12_fit=mp2, cabs=cabs, beta=1.0)


def _canonical_reference_for_test(prov: MolecularFocalProvider, req) -> FocalReference:
    """Use native df_reference when AE on CPU, or solve exact DF-RHF/UHF on native ECP+JKFit integrals."""
    has_ecp = any(
        req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons for a in req.atoms
    )
    if not has_ecp:
        return df_reference(req, forces=False, budget_mib=prov.budget_mib)

    from scipy.linalg import eigh

    one = one_electron_sources(
        req,
        library=prov.one_electron_library or prov.shell_library,
        orbital_only=True,
        derivatives=False,
        budget_mib=prov.budget_mib,
    )
    jk = prov._panel(req, "jk_fit", orbital_only=True, allow_streaming=False)
    S = np.asarray(one.overlap, dtype=np.float64)
    H = np.asarray(one.hcore, dtype=np.float64)
    ao3 = np.asarray(jk.data["ao_coulomb"], dtype=np.float64)
    B = np.einsum("PQ,Qmn->Pmn", jk.whitener, ao3, optimize=True)
    n_elec = sum(
        a.atomic_number - req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons
        for a in req.atoms
    ) - req.charge
    n_unpaired = req.multiplicity - 1
    oa = (n_elec + n_unpaired) // 2
    ob = (n_elec - n_unpaired) // 2
    enuc = 0.0
    for i, a in enumerate(req.atoms):
        zi = a.atomic_number - req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons
        for j in range(i + 1, len(req.atoms)):
            zj = req.atoms[j].atomic_number - req.basis.orbital.by_element[req.atoms[j].atomic_number].ecp_core_electrons
            r = np.linalg.norm(np.array(a.position) - np.array(req.atoms[j].position))
            enuc += zi * zj / r

    if req.root_id == "rhf":
        eps, C = eigh(H, S)
        D = 2.0 * (C[:, :oa] @ C[:, :oa].T)
        f_hist, e_hist = [], []
        F = H.copy()
        for it in range(1, 150):
            c_occ = C[:, :oa]
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
                for ai in range(m):
                    for bi in range(m):
                        A[ai, bi] = float(np.dot(e_hist[ai], e_hist[bi]))
                    A[ai, m] = A[m, ai] = -1.0
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
            D = 2.0 * (C[:, :oa] @ C[:, :oa].T)
        eps, C = eigh(F, S)
        D = 2.0 * (C[:, :oa] @ C[:, :oa].T)
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
            occupied=(oa,),
            iterations=it,
            executed_backend=req.backend,
            surface_id=req.surface_id,
            basis_id=req.basis.basis_id,
            orbital_id=req.basis.orbital.identity,
            jk_fit_id=req.basis.jk_fit.identity,
        )

    # UHF DF-SCF (e.g., bare Br doublet) with deterministic slight orbital-splitting seed
    n = len(H)
    seed_a = np.diag(np.linspace(-1e-3, 1e-3, n))
    ea, Ca = eigh(H + seed_a, S)
    eb, Cb = eigh(H - seed_a, S)
    Da = Ca[:, :oa] @ Ca[:, :oa].T
    Db = Cb[:, :ob] @ Cb[:, :ob].T
    f_hist, e_hist = [], []
    Fa, Fb = H.copy(), H.copy()
    for it in range(1, 200):
        q_vec = np.einsum("Pmn,mn->P", B, Da + Db, optimize=True)
        J = np.einsum("P,Pmn->mn", q_vec, B, optimize=True)
        Ba = np.einsum("Pmn,ni->Pmi", B, Ca[:, :oa], optimize=True)
        Ka = np.einsum("Pmi,Pni->mn", Ba, Ba, optimize=True)
        Bb = np.einsum("Pmn,ni->Pmi", B, Cb[:, :ob], optimize=True)
        Kb = np.einsum("Pmi,Pni->mn", Bb, Bb, optimize=True)
        Fa = 0.5 * ((H + J - Ka) + (H + J - Ka).T)
        Fb = 0.5 * ((H + J - Kb) + (H + J - Kb).T)
        err_a = Fa @ Da @ S - S @ Da @ Fa
        err_b = Fb @ Db @ S - S @ Db @ Fb
        if max(np.max(np.abs(err_a)), np.max(np.abs(err_b))) < 1e-12:
            break
        f_hist.append((Fa.copy(), Fb.copy()))
        e_hist.append(np.concatenate([err_a.ravel(), err_b.ravel()]))
        if len(f_hist) > 8:
            f_hist.pop(0)
            e_hist.pop(0)
        if len(f_hist) >= 2:
            m = len(f_hist)
            A = np.zeros((m + 1, m + 1))
            for ai in range(m):
                for bi in range(m):
                    A[ai, bi] = float(np.dot(e_hist[ai], e_hist[bi]))
                A[ai, m] = A[m, ai] = -1.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                coeff = np.linalg.solve(A, rhs)[:m]
                Fa_use = sum(c * fm[0] for c, fm in zip(coeff, f_hist, strict=True))
                Fb_use = sum(c * fm[1] for c, fm in zip(coeff, f_hist, strict=True))
            except np.linalg.LinAlgError:
                Fa_use, Fb_use = Fa, Fb
        else:
            Fa_use, Fb_use = Fa, Fb
        ea, Ca = eigh(Fa_use, S)
        eb, Cb = eigh(Fb_use, S)
        Da = Ca[:, :oa] @ Ca[:, :oa].T
        Db = Cb[:, :ob] @ Cb[:, :ob].T
    ea, Ca = eigh(Fa, S)
    eb, Cb = eigh(Fb, S)
    Da = Ca[:, :oa] @ Ca[:, :oa].T
    Db = Cb[:, :ob] @ Cb[:, :ob].T
    e_elec = 0.5 * float(np.sum(Da * (H + Fa)) + np.sum(Db * (H + Fb)))
    return FocalReference(
        energy=enuc + e_elec,
        forces=None,
        overlap=S,
        hcore=H,
        coefficients=np.stack([Ca, Cb], axis=0),
        orbital_energies=np.stack([ea, eb], axis=0),
        density=np.stack([Da, Db], axis=0),
        fock=np.stack([Fa, Fb], axis=0),
        occupied=(oa, ob),
        iterations=it,
        executed_backend=req.backend,
        surface_id=req.surface_id,
        basis_id=req.basis.basis_id,
        orbital_id=req.basis.orbital.identity,
        jk_fit_id=req.basis.jk_fit.identity,
    )


def _setup_mp2_problem(
    atoms_spec: tuple[tuple[str, tuple[float, float, float]], ...],
    elements: tuple[int, ...],
    *,
    charge: int,
    multiplicity: int,
    root_id: str,
    corr_space: str,
    host_sink: Path,
):
    basis = _compact_split_basis(elements)
    plan = plan_singlepoint(
        atoms_spec,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=charge,
        multiplicity=multiplicity,
        root_id=root_id,
        correlation_space_id=corr_space,
        backend="cpu_reference",
    )
    req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    prov = MolecularFocalProvider(shell_library=host_sink, budget_mib=512)
    ref = _canonical_reference_for_test(prov, req)
    n = len(ref.hcore)
    jk_panel = prov._panel(req, "jk_fit", orbital_only=True, allow_streaming=False)
    mp2_panel = prov._panel(req, "mp2_fit", orbital_only=True, allow_streaming=False)
    jk_src = jk_panel.obs(req, n)
    mp2_src = mp2_panel.obs(req, n)
    return prov, req, ref, jk_src, mp2_src


class _ForbidTensorIRRunner:
    """Sentinel runner that tracks calls and fails immediately if TensorIR is invoked."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, program, inputs):
        self.calls += 1
        raise AssertionError(
            f"TensorIR runner must NOT be called in AOT ordinary_df_mp2 (called {self.calls} times)"
        )


def _assert_unrelaxed_close(
    label: str,
    got,
    ref,
    *,
    e_tol: float = 1e-12,
    grad_tol: float = 1e-11,
) -> None:
    for spin_name, gv, rv in zip(("aa", "ab", "bb"), got.spin_energies, ref.spin_energies, strict=True):
        diff = abs(gv - rv)
        assert diff <= e_tol, f"{label} E_{spin_name} diff {diff:.3e} > {e_tol} ({gv} vs {rv})"
    tot_diff = abs(sum(got.spin_energies) - sum(ref.spin_energies))
    assert tot_diff <= e_tol, f"{label} E_tot diff {tot_diff:.3e} > {e_tol}"
    for key in ("fock_alpha", "fock_beta"):
        diff = float(np.max(np.abs(got.fock_seeds[key] - ref.fock_seeds[key])))
        assert diff <= grad_tol, f"{label} {key} max diff {diff:.3e} > {grad_tol}"
    for name, a, b in (
        ("obs_coefficients", got.obs_coefficients, ref.obs_coefficients),
        ("mp2_three_center", got.mp2_three_center, ref.mp2_three_center),
        ("mp2_metric", got.mp2_metric, ref.mp2_metric),
    ):
        diff = float(np.max(np.abs(a - b)))
        assert diff <= grad_tol, f"{label} {name} max diff {diff:.3e} > {grad_tol}"


@pytest.mark.parametrize(
    "label,atoms_spec,elements,charge,corr_space,weighted",
    [
        ("H2O-AE", GEOM_H2O, (1, 8), 0, "all-active", False),
        ("H2O-FC", GEOM_H2O, (1, 8), 0, "frozen-core", False),
        ("H2O-FC-weighted", GEOM_H2O, (1, 8), 0, "frozen-core", True),
        ("CH4-AE", GEOM_CH4, (1, 6), 0, "all-active", False),
        ("CH4-FC", GEOM_CH4, (1, 6), 0, "frozen-core", False),
        ("IHB93-AE", GEOM_IHB93, (1, 8), -1, "all-active", False),
        ("IHB93-FC", GEOM_IHB93, (1, 8), -1, "frozen-core", False),
        ("IHB93-FC-weighted", GEOM_IHB93, (1, 8), -1, "frozen-core", True),
    ],
)
def test_ordinary_df_mp2_rhf_three_way_parity(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    atoms_spec,
    elements: tuple[int, ...],
    charge: int,
    corr_space: str,
    weighted: bool,
) -> None:
    """Prove RHF spatial fast path == general UHF AOT path == TensorIR oracle with 0 TensorIR calls."""
    _, req, ref, _, mp2_src = _setup_mp2_problem(
        atoms_spec,
        elements,
        charge=charge,
        multiplicity=1,
        root_id="rhf",
        corr_space=corr_space,
        host_sink=host_sink,
    )
    o = ref.occupied[0]
    f = resolve_frozen_occupied(req, (o, o))[0]
    o_act = o - f
    occ_w = None
    if weighted:
        rng = np.random.default_rng(1201 + o_act)
        w = rng.uniform(0.7, 1.4, size=o_act)
        occ_w = (w, w.copy())

    # 1. RHF spatial fast path (default AOT, sentinel forbids any TensorIR call)
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    sentinel = _ForbidTensorIRRunner()
    res_rhf_aot = ordinary_df_ump2_unrelaxed(
        req,
        ref,
        mp2_src,
        run=sentinel,
        budget_mib=512,
        occupied_weights=occ_w,
        shell_library=host_sink,
    )
    assert sentinel.calls == 0

    # 2. General UHF AOT path (same_spin=False on the exact same RHF inputs)
    native_fn = bind_ordinary_df_mp2(host_reference=True, shell_library=host_sink)
    assert native_fn is not None
    c0 = ref.coefficients[0]
    e0 = ref.orbital_energies[0]
    se_uhf, fa_uhf, fb_uhf, bca_uhf, bcb_uhf, bao_uhf, bm_uhf = run_ordinary_df_mp2_native(
        native_fn,
        same_spin=False,
        compute_response=True,
        oa=o,
        fa=f,
        ob=o,
        fb=f,
        budget_bytes=512 << 20,
        coeff_a=c0,
        coeff_b=c0,
        eps_a=e0,
        eps_b=e0,
        ao_coul=mp2_src.three_center,
        whitener=mp2_src.whitener,
        occ_w_a=None if occ_w is None else occ_w[0],
        occ_w_b=None if occ_w is None else occ_w[1],
    )
    assert fa_uhf is not None and fb_uhf is not None
    assert bca_uhf is not None and bcb_uhf is not None
    assert bao_uhf is not None and bm_uhf is not None

    from generativeqc._f12_focal_ump2 import FocalUMP2Unrelaxed

    res_uhf_aot = FocalUMP2Unrelaxed(
        se_uhf,
        {"fock_alpha": fa_uhf, "fock_beta": fb_uhf},
        (bca_uhf + bcb_uhf)[None, :, :],
        bao_uhf,
        bm_uhf,
    )

    # Also verify compute_response=False energy-only mode matches
    se_eonly, *_ = run_ordinary_df_mp2_native(
        native_fn,
        same_spin=True,
        compute_response=False,
        oa=o,
        fa=f,
        ob=o,
        fb=f,
        budget_bytes=512 << 20,
        coeff_a=c0,
        coeff_b=c0,
        eps_a=e0,
        eps_b=e0,
        ao_coul=mp2_src.three_center,
        whitener=mp2_src.whitener,
        occ_w_a=None if occ_w is None else occ_w[0],
        occ_w_b=None if occ_w is None else occ_w[1],
    )
    assert max(abs(a - b) for a, b in zip(se_eonly, res_rhf_aot.spin_energies, strict=True)) <= 1e-13

    # 3. TensorIR oracle
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", "1")
    res_tir = ordinary_df_ump2_unrelaxed(
        req,
        ref,
        mp2_src,
        budget_mib=512,
        occupied_weights=occ_w,
        shell_library=host_sink,
    )

    _assert_unrelaxed_close(f"{label}[RHF-AOT vs TensorIR]", res_rhf_aot, res_tir)
    _assert_unrelaxed_close(f"{label}[UHF-AOT vs TensorIR]", res_uhf_aot, res_tir)
    _assert_unrelaxed_close(f"{label}[RHF-AOT vs UHF-AOT]", res_rhf_aot, res_uhf_aot)


@pytest.mark.parametrize(
    "label,atoms_spec,elements,corr_space,weighted",
    [
        ("OH-AE", GEOM_OH, (1, 8), "all-active", False),
        ("OH-FC", GEOM_OH, (1, 8), "frozen-core", False),
        ("OH-FC-weighted", GEOM_OH, (1, 8), "frozen-core", True),
        ("OH-asymmetric-frozen", GEOM_OH, (1, 8), "frozen:1,0", False),
        ("Br-atom-AE", GEOM_BR, (35,), "all-active", False),
        ("Br-atom-FC4", GEOM_BR, (35,), "frozen-core", False),
        ("Br-atom-FC9", GEOM_BR, (35,), "frozen-core-spd", False),
        ("Br-atom-FC9-weighted", GEOM_BR, (35,), "frozen-core-spd", True),
    ],
)
def test_ordinary_df_mp2_uhf_parity(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    atoms_spec,
    elements: tuple[int, ...],
    corr_space: str,
    weighted: bool,
) -> None:
    """Verify general UHF AOT (oa != ob, fa != fb, bare Br ECP10MDF) vs TensorIR oracle."""
    _, req, ref, _, mp2_src = _setup_mp2_problem(
        atoms_spec,
        elements,
        charge=0,
        multiplicity=2,
        root_id="uhf",
        corr_space=corr_space,
        host_sink=host_sink,
    )
    oa, ob = ref.occupied[0], ref.occupied[1]
    assert oa != ob
    fa, fb = resolve_frozen_occupied(req, (oa, ob))
    if label == "Br-atom-FC4":
        assert (fa, fb) == (4, 4)
    elif label == "Br-atom-FC9":
        assert (fa, fb) == (9, 9)

    occ_w = None
    if weighted:
        rng = np.random.default_rng(2201 + oa + fa)
        occ_w = (rng.uniform(0.6, 1.5, size=oa - fa), rng.uniform(0.6, 1.5, size=ob - fb))

    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    sentinel = _ForbidTensorIRRunner()
    res_aot = ordinary_df_ump2_unrelaxed(
        req,
        ref,
        mp2_src,
        run=sentinel,
        budget_mib=512,
        occupied_weights=occ_w,
        shell_library=host_sink,
    )
    assert sentinel.calls == 0

    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", "1")
    res_tir = ordinary_df_ump2_unrelaxed(
        req,
        ref,
        mp2_src,
        budget_mib=512,
        occupied_weights=occ_w,
        shell_library=host_sink,
    )
    _assert_unrelaxed_close(f"{label}[UHF-AOT vs TensorIR]", res_aot, res_tir)


@pytest.mark.parametrize(
    "corr_space,expected_frozen",
    [
        ("all-active", (0, 0)),
        ("frozen-core", (4, 4)),
        ("frozen-core-spd", (9, 9)),
    ],
)
def test_ordinary_df_mp2_hbr_ecp_fc4_fc9_parity(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
    corr_space: str,
    expected_frozen: tuple[int, int],
) -> None:
    """Verify ECP HBr (all-active, FC4, FC9) 3-way unrelaxed parity + relaxed weights parity."""
    _, req, ref, jk_src, mp2_src = _setup_mp2_problem(
        GEOM_HBR,
        (1, 35),
        charge=0,
        multiplicity=1,
        root_id="rhf",
        corr_space=corr_space,
        host_sink=host_sink,
    )
    o = ref.occupied[0]
    frozen = resolve_frozen_occupied(req, (o, o))
    assert frozen == expected_frozen

    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    sentinel = _ForbidTensorIRRunner()
    res_rhf = ordinary_df_ump2_unrelaxed(
        req, ref, mp2_src, run=sentinel, budget_mib=512, shell_library=host_sink
    )
    assert sentinel.calls == 0

    native_fn = bind_ordinary_df_mp2(host_reference=True, shell_library=host_sink)
    assert native_fn is not None
    c0, e0 = ref.coefficients[0], ref.orbital_energies[0]
    se_u, fa_u, fb_u, bca_u, bcb_u, bao_u, bm_u = run_ordinary_df_mp2_native(
        native_fn,
        same_spin=False,
        compute_response=True,
        oa=o,
        fa=frozen[0],
        ob=o,
        fb=frozen[1],
        budget_bytes=512 << 20,
        coeff_a=c0,
        coeff_b=c0,
        eps_a=e0,
        eps_b=e0,
        ao_coul=mp2_src.three_center,
        whitener=mp2_src.whitener,
    )
    from generativeqc._f12_focal_ump2 import FocalUMP2Unrelaxed

    assert fa_u is not None and fb_u is not None and bca_u is not None and bcb_u is not None
    assert bao_u is not None and bm_u is not None
    res_uhf = FocalUMP2Unrelaxed(
        se_u, {"fock_alpha": fa_u, "fock_beta": fb_u}, (bca_u + bcb_u)[None, :, :], bao_u, bm_u
    )

    # Relaxed reference_df_ump2 weights AOT vs TensorIR
    rel_aot = reference_df_ump2(req, ref, jk_src, mp2_src, budget_mib=512, shell_library=host_sink)

    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", "1")
    res_tir = ordinary_df_ump2_unrelaxed(
        req, ref, mp2_src, budget_mib=512, shell_library=host_sink
    )
    rel_tir = reference_df_ump2(req, ref, jk_src, mp2_src, budget_mib=512, shell_library=host_sink)

    _assert_unrelaxed_close(f"HBr-{corr_space}[RHF-AOT vs TensorIR]", res_rhf, res_tir)
    _assert_unrelaxed_close(f"HBr-{corr_space}[RHF-AOT vs UHF-AOT]", res_rhf, res_uhf)

    assert abs(sum(rel_aot.spin_energies) - sum(rel_tir.spin_energies)) <= 1e-12
    for key in rel_tir.weights:
        diff = float(np.max(np.abs(rel_aot.weights[key] - rel_tir.weights[key])))
        assert diff <= 1e-11, f"HBr-{corr_space} relaxed weight {key} diff {diff:.3e}"


def test_ordinary_df_mp2_edge_cases_and_bounded_fallback(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify o_act=1 (H2), single-electron H, tiny-budget fallback, and error guards."""
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)

    # 1. H2 (o_act = 1 -> E_aa = E_bb = 0, E_ab != 0)
    _, req_h2, ref_h2, _, mp2_h2 = _setup_mp2_problem(
        GEOM_H2, (1,), charge=0, multiplicity=1, root_id="rhf", corr_space="all-active", host_sink=host_sink
    )
    sentinel = _ForbidTensorIRRunner()
    res_h2_aot = ordinary_df_ump2_unrelaxed(
        req_h2, ref_h2, mp2_h2, run=sentinel, budget_mib=512, shell_library=host_sink
    )
    assert sentinel.calls == 0
    assert res_h2_aot.spin_energies[0] == 0.0 and res_h2_aot.spin_energies[2] == 0.0
    assert res_h2_aot.spin_energies[1] < -1e-4

    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", "1")
    res_h2_tir = ordinary_df_ump2_unrelaxed(
        req_h2, ref_h2, mp2_h2, budget_mib=512, shell_library=host_sink
    )
    _assert_unrelaxed_close("H2[o_act=1]", res_h2_aot, res_h2_tir)

    # 2. H atom (oa=1, ob=0 -> zero correlation energy and zero cotangents)
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    _, req_h, ref_h, _, mp2_h = _setup_mp2_problem(
        GEOM_H, (1,), charge=0, multiplicity=2, root_id="uhf", corr_space="all-active", host_sink=host_sink
    )
    res_h_aot = ordinary_df_ump2_unrelaxed(
        req_h, ref_h, mp2_h, run=_ForbidTensorIRRunner(), budget_mib=512, shell_library=host_sink
    )
    assert res_h_aot.spin_energies == (0.0, 0.0, 0.0)
    assert np.all(res_h_aot.fock_seeds["fock_alpha"] == 0.0)
    assert np.all(res_h_aot.fock_seeds["fock_beta"] == 0.0)
    assert np.all(res_h_aot.obs_coefficients == 0.0)
    assert np.all(res_h_aot.mp2_three_center == 0.0)
    assert np.all(res_h_aot.mp2_metric == 0.0)

    # 3. Tiny budget_bytes forcing bounded-memory bar_Fo fallback matches full triangle storage
    native_fn = bind_ordinary_df_mp2(host_reference=True, shell_library=host_sink)
    assert native_fn is not None
    for label, geom, mult, root in (
        ("H2O-fallback", GEOM_H2O, 1, "rhf"),
        ("OH-fallback", GEOM_OH, 2, "uhf"),
    ):
        _, req_b, ref_b, _, mp2_b = _setup_mp2_problem(
            geom, (1, 8), charge=0, multiplicity=mult, root_id=root, corr_space="all-active", host_sink=host_sink
        )
        same = root == "rhf"
        oa = ref_b.occupied[0]
        ob = oa if same else ref_b.occupied[1]
        ca = ref_b.coefficients[0]
        cb = ca if same else ref_b.coefficients[1]
        ea = ref_b.orbital_energies[0]
        eb = ea if same else ref_b.orbital_energies[1]

        out_hi = run_ordinary_df_mp2_native(
            native_fn, same_spin=same, compute_response=True, oa=oa, fa=0, ob=ob, fb=0,
            budget_bytes=512 << 20, coeff_a=ca, coeff_b=cb, eps_a=ea, eps_b=eb,
            ao_coul=mp2_b.three_center, whitener=mp2_b.whitener,
        )
        # budget_bytes = 64 bytes triggers non-triangular bounded fallback (clamped internally to >=1MiB?
        # pass budget_bytes = 1 and check small buffer path directly when store_tri is false).
        out_lo = run_ordinary_df_mp2_native(
            native_fn, same_spin=same, compute_response=True, oa=oa, fa=0, ob=ob, fb=0,
            budget_bytes=64, coeff_a=ca, coeff_b=cb, eps_a=ea, eps_b=eb,
            ao_coul=mp2_b.three_center, whitener=mp2_b.whitener,
        )
        assert max(abs(a - b) for a, b in zip(out_hi[0], out_lo[0], strict=True)) <= 1e-13
        for idx in range(1, 7):
            assert np.max(np.abs(out_hi[idx] - out_lo[idx])) <= 1e-12, f"{label} field {idx}"

    # 4. Non-finite input guard
    bad_eps = ref_h2.orbital_energies[0].copy()
    bad_eps[0] = np.nan
    with pytest.raises(RuntimeError, match="status -2"):
        run_ordinary_df_mp2_native(
            native_fn, same_spin=True, compute_response=True, oa=1, fa=0, ob=1, fb=0,
            budget_bytes=512 << 20, coeff_a=ref_h2.coefficients[0], coeff_b=ref_h2.coefficients[0],
            eps_a=bad_eps, eps_b=bad_eps, ao_coul=mp2_h2.three_center, whitener=mp2_h2.whitener,
        )

    # 5. Missing symbol on explicit shell_library raises (zero silent fallback)
    with pytest.raises(RuntimeError, match="does not export"):
        ordinary_df_ump2_unrelaxed(
            req_h2,
            ref_h2,
            mp2_h2,
            budget_mib=512,
            shell_library=Path("/nonexistent/fake_shell_library.so"),
        )


# =============================================================================
# Step 12.2 Gate 1: AOT MP2-F12 forward pair energies (molecular_pair_energies)
# =============================================================================


def _setup_f12_pair_inputs(
    atoms_spec: tuple[tuple[str, tuple[float, float, float]], ...],
    elements: tuple[int, ...],
    *,
    charge: int,
    multiplicity: int,
    root_id: str,
    corr_space: str,
    host_sink: Path,
):
    from functools import partial

    from generativeqc._f12_focal_frame import canonical_pair_frame
    from generativeqc._f12_spin_ao import _orbital_sources
    from generativeqc._f12_spin_fock import spin_fock
    from generativeqc_compiler.tensor import execute

    prov, req, ref, _, _ = _setup_mp2_problem(
        atoms_spec,
        elements,
        charge=charge,
        multiplicity=multiplicity,
        root_id=root_id,
        corr_space=corr_space,
        host_sink=host_sink,
    )
    one, (jk, _mp2, f12) = prov._cached_pair_sources(req, derivatives=False, budget=512)
    space, ao, jkfit, _proj = canonical_pair_frame(
        req,
        ref,
        one,
        jk.joint(req, explicit=False),
        f12.joint(req, explicit=True),
        expected_cabs_rank=None,
        relative_cutoff=prov.relative_cutoff,
        library=prov.response_library,
    )
    oa, ob, obs, ext, _aux = space
    run_cpu = partial(execute, max_bytes=512 << 20)
    shared = ("overlap_ao", "hcore_ao", "coefficients_alpha", "coefficients_beta")
    fock_ao = {**{k: ao[k] for k in shared}, **jkfit}
    fock_data, _ = _orbital_sources(
        (oa, ob, obs, ext, len(jk.whitener)),
        fock_ao,
        run_cpu,
        None,
        projected=None,
        role="fock",
    )
    data, _ = _orbital_sources(space, ao, run_cpu, None, projected=None)
    src = {k.removeprefix("source/"): v for k, v in data.items() if k.startswith("source/")}
    fsrc = {k.removeprefix("source/"): v for k, v in fock_data.items() if k.startswith("source/")}
    fock = spin_fock((oa, ob, ext, len(jk.whitener)), fsrc, run_cpu)
    frozen = resolve_frozen_occupied(req, (oa, ob))
    return prov, req, space, data, src, fock, frozen, run_cpu


def _assert_pair_energies_close(
    label: str,
    got: list,
    ref: list,
    *,
    include_ordinary: bool,
    tol: float = 1e-12,
) -> None:
    assert len(got) == len(ref), f"{label} pair count {len(got)} != {len(ref)}"
    max_f12_pair = 0.0
    max_ord_pair = 0.0
    for g, r in zip(got, ref, strict=True):
        assert g.spins == r.spins and g.occupied_indices == r.occupied_indices
        df = abs(g.f12 - r.f12)
        if df > max_f12_pair:
            max_f12_pair = df
        if include_ordinary:
            assert g.mp2 is not None and r.mp2 is not None
            do = abs(g.mp2 - r.mp2)
            if do > max_ord_pair:
                max_ord_pair = do
        else:
            assert g.mp2 is None and r.mp2 is None
    tot_f12_diff = abs(sum(p.f12 for p in got) - sum(p.f12 for p in ref))
    assert max_f12_pair <= tol, f"{label} max pair F12 diff {max_f12_pair:.3e} > {tol}"
    assert tot_f12_diff <= tol, f"{label} total F12 diff {tot_f12_diff:.3e} > {tol}"
    if include_ordinary:
        tot_ord_diff = abs(sum(p.mp2 for p in got) - sum(p.mp2 for p in ref))
        assert max_ord_pair <= tol, f"{label} max pair MP2 diff {max_ord_pair:.3e} > {tol}"
        assert tot_ord_diff <= tol, f"{label} total MP2 diff {tot_ord_diff:.3e} > {tol}"


def _weighted_spin_totals(pairs: list, frozen: tuple[int, int], occ_w: tuple[np.ndarray, np.ndarray]):
    fmap = {"alpha": (frozen[0], occ_w[0]), "beta": (frozen[1], occ_w[1])}
    f12_s = {("alpha", "alpha"): 0.0, ("alpha", "beta"): 0.0, ("beta", "beta"): 0.0}
    ord_s = {("alpha", "alpha"): 0.0, ("alpha", "beta"): 0.0, ("beta", "beta"): 0.0}
    for p in pairs:
        sl, sr = p.spins
        fl, wl = fmap[sl]
        fr, wr = fmap[sr]
        w = 0.5 * (float(wl[p.occupied_indices[0] - fl]) + float(wr[p.occupied_indices[1] - fr]))
        f12_s[p.spins] += w * p.f12
        if p.mp2 is not None:
            ord_s[p.spins] += w * p.mp2
    return (
        (f12_s[("alpha", "alpha")], f12_s[("alpha", "beta")], f12_s[("beta", "beta")]),
        (ord_s[("alpha", "alpha")], ord_s[("alpha", "beta")], ord_s[("beta", "beta")]),
    )


@pytest.mark.parametrize(
    "label,atoms_spec,elements,charge,corr_space,include_ordinary",
    [
        ("H2O-AE-ord", GEOM_H2O, (1, 8), 0, "all-active", True),
        ("H2O-FC-ord", GEOM_H2O, (1, 8), 0, "frozen-core", True),
        ("H2O-FC-f12only", GEOM_H2O, (1, 8), 0, "frozen-core", False),
        ("CH4-AE-ord", GEOM_CH4, (1, 6), 0, "all-active", True),
        ("CH4-FC-ord", GEOM_CH4, (1, 6), 0, "frozen-core", True),
        ("IHB93-AE-ord", GEOM_IHB93, (1, 8), -1, "all-active", True),
        ("IHB93-FC-ord", GEOM_IHB93, (1, 8), -1, "frozen-core", True),
    ],
)
def test_mp2_f12_pair_energies_rhf_three_way_parity(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    atoms_spec,
    elements: tuple[int, ...],
    charge: int,
    corr_space: str,
    include_ordinary: bool,
) -> None:
    """Prove RHF spatial fast path == general UHF AOT == TensorIR oracle to <=1e-12 with 0 TensorIR calls."""
    from generativeqc._f12_mp2_pair_native import (
        bind_mp2_f12_pair_energies,
        run_mp2_f12_pair_energies_native,
    )
    from generativeqc._f12_spin_pairs import (
        _is_rhf_spatial_identity_pair_data,
        molecular_pair_energies,
    )

    _, _, space, data, src, fock, frozen, run_cpu = _setup_f12_pair_inputs(
        atoms_spec,
        elements,
        charge=charge,
        multiplicity=1,
        root_id="rhf",
        corr_space=corr_space,
        host_sink=host_sink,
    )
    assert _is_rhf_spatial_identity_pair_data(space, data, src, fock, frozen)

    # 1. Default AOT RHF spatial fast path via molecular_pair_energies (0 TensorIR calls)
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    sentinel = _ForbidTensorIRRunner()
    pairs_rhf_aot = list(
        molecular_pair_energies(
            space,
            data,
            src,
            fock,
            sentinel,
            frozen_occupied=frozen,
            include_ordinary=include_ordinary,
            shell_library=host_sink,
        )
    )
    assert sentinel.calls == 0

    # 2. General UHF AOT path forced (rhf_spatial=False) on identical RHF inputs + occupied_weights
    native_fn = bind_mp2_f12_pair_energies(host_reference=True, shell_library=host_sink)
    assert native_fn is not None
    o_act = space[0] - frozen[0]
    rng = np.random.default_rng(12201 + o_act)
    w = rng.uniform(0.6, 1.5, size=o_act)

    f12_rhf_w, ord_rhf_w, pf_rhf, po_rhf = run_mp2_f12_pair_energies_native(
        native_fn,
        space=space,
        frozen_occupied=frozen,
        rhf_spatial=True,
        include_ordinary=include_ordinary,
        budget_bytes=512 << 20,
        data=data,
        sources=src,
        fock=fock,
        occ_w_a=w,
        occ_w_b=w,
    )
    f12_uhf_w, ord_uhf_w, pf_uhf, po_uhf = run_mp2_f12_pair_energies_native(
        native_fn,
        space=space,
        frozen_occupied=frozen,
        rhf_spatial=False,
        include_ordinary=include_ordinary,
        budget_bytes=512 << 20,
        data=data,
        sources=src,
        fock=fock,
        occ_w_a=w,
        occ_w_b=w,
    )

    # 3. TensorIR oracle
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", "1")
    pairs_tir = list(
        molecular_pair_energies(
            space,
            data,
            src,
            fock,
            run_cpu,
            frozen_occupied=frozen,
            include_ordinary=include_ordinary,
            shell_library=host_sink,
        )
    )

    _assert_pair_energies_close(
        f"{label}[RHF-AOT vs TensorIR]",
        pairs_rhf_aot,
        pairs_tir,
        include_ordinary=include_ordinary,
    )

    # Check general UHF AOT per-pair values against both RHF AOT and TensorIR
    tir_f12 = np.array([p.f12 for p in pairs_tir], dtype=np.float64)
    assert float(np.max(np.abs(pf_uhf - tir_f12))) <= 1e-12
    assert float(np.max(np.abs(pf_rhf - pf_uhf))) <= 1e-12
    if include_ordinary:
        assert po_rhf is not None and po_uhf is not None
        tir_ord = np.array([p.mp2 for p in pairs_tir], dtype=np.float64)
        assert float(np.max(np.abs(po_uhf - tir_ord))) <= 1e-12
        assert float(np.max(np.abs(po_rhf - po_uhf))) <= 1e-12

    # Verify occupied_weights spin totals match exact weighted TensorIR sums
    exp_f12_w, exp_ord_w = _weighted_spin_totals(pairs_tir, frozen, (w, w))
    for k in range(3):
        assert abs(f12_rhf_w[k] - exp_f12_w[k]) <= 1e-12
        assert abs(f12_uhf_w[k] - exp_f12_w[k]) <= 1e-12
        if include_ordinary:
            assert abs(ord_rhf_w[k] - exp_ord_w[k]) <= 1e-12
            assert abs(ord_uhf_w[k] - exp_ord_w[k]) <= 1e-12


@pytest.mark.parametrize(
    "label,atoms_spec,elements,corr_space,include_ordinary",
    [
        ("OH-AE", GEOM_OH, (1, 8), "all-active", True),
        ("OH-FC", GEOM_OH, (1, 8), "frozen-core", True),
        ("OH-FC-f12only", GEOM_OH, (1, 8), "frozen-core", False),
        ("OH-asymmetric-frozen", GEOM_OH, (1, 8), "frozen:1,0", True),
        ("Br-atom-AE", GEOM_BR, (35,), "all-active", True),
        ("Br-atom-FC4", GEOM_BR, (35,), "frozen-core", True),
        ("Br-atom-FC9", GEOM_BR, (35,), "frozen-core-spd", True),
    ],
)
def test_mp2_f12_pair_energies_uhf_parity(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    atoms_spec,
    elements: tuple[int, ...],
    corr_space: str,
    include_ordinary: bool,
) -> None:
    """Verify general UHF dual-spin AOT (oa != ob, fa != fb, bare Br FC4/FC9, occupied_weights)."""
    from generativeqc._f12_mp2_pair_native import (
        bind_mp2_f12_pair_energies,
        run_mp2_f12_pair_energies_native,
    )
    from generativeqc._f12_spin_pairs import molecular_pair_energies

    _, _, space, data, src, fock, frozen, run_cpu = _setup_f12_pair_inputs(
        atoms_spec,
        elements,
        charge=0,
        multiplicity=2,
        root_id="uhf",
        corr_space=corr_space,
        host_sink=host_sink,
    )
    oa, ob = space[0], space[1]
    assert oa != ob
    if label == "Br-atom-FC4":
        assert frozen == (4, 4)
    elif label == "Br-atom-FC9":
        assert frozen == (9, 9)

    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    sentinel = _ForbidTensorIRRunner()
    pairs_aot = list(
        molecular_pair_energies(
            space,
            data,
            src,
            fock,
            sentinel,
            frozen_occupied=frozen,
            include_ordinary=include_ordinary,
            shell_library=host_sink,
        )
    )
    assert sentinel.calls == 0

    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", "1")
    pairs_tir = list(
        molecular_pair_energies(
            space,
            data,
            src,
            fock,
            run_cpu,
            frozen_occupied=frozen,
            include_ordinary=include_ordinary,
            shell_library=host_sink,
        )
    )
    _assert_pair_energies_close(
        f"{label}[UHF-AOT vs TensorIR]",
        pairs_aot,
        pairs_tir,
        include_ordinary=include_ordinary,
    )

    # Also verify unequal occupied_weights in native kernel
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    native_fn = bind_mp2_f12_pair_energies(host_reference=True, shell_library=host_sink)
    assert native_fn is not None
    rng = np.random.default_rng(12251 + oa + frozen[0])
    wa = rng.uniform(0.5, 1.6, size=oa - frozen[0])
    wb = rng.uniform(0.5, 1.6, size=ob - frozen[1])
    f12_w, ord_w, _, _ = run_mp2_f12_pair_energies_native(
        native_fn,
        space=space,
        frozen_occupied=frozen,
        rhf_spatial=False,
        include_ordinary=include_ordinary,
        budget_bytes=512 << 20,
        data=data,
        sources=src,
        fock=fock,
        occ_w_a=wa,
        occ_w_b=wb,
    )
    exp_f12_w, exp_ord_w = _weighted_spin_totals(pairs_tir, frozen, (wa, wb))
    for k in range(3):
        assert abs(f12_w[k] - exp_f12_w[k]) <= 1e-12
        if include_ordinary:
            assert abs(ord_w[k] - exp_ord_w[k]) <= 1e-12


@pytest.mark.parametrize(
    "corr_space,expected_frozen",
    [
        ("all-active", (0, 0)),
        ("frozen-core", (4, 4)),
        ("frozen-core-spd", (9, 9)),
    ],
)
def test_mp2_f12_pair_energies_hbr_ecp_fc4_fc9_parity(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
    corr_space: str,
    expected_frozen: tuple[int, int],
) -> None:
    """Verify ECP HBr (all-active, FC4, FC9) RHF spatial == UHF AOT == TensorIR to <=1e-12."""
    from generativeqc._f12_mp2_pair_native import (
        bind_mp2_f12_pair_energies,
        run_mp2_f12_pair_energies_native,
    )
    from generativeqc._f12_spin_pairs import (
        _is_rhf_spatial_identity_pair_data,
        molecular_pair_energies,
    )

    _, _, space, data, src, fock, frozen, run_cpu = _setup_f12_pair_inputs(
        GEOM_HBR,
        (1, 35),
        charge=0,
        multiplicity=1,
        root_id="rhf",
        corr_space=corr_space,
        host_sink=host_sink,
    )
    assert frozen == expected_frozen
    assert _is_rhf_spatial_identity_pair_data(space, data, src, fock, frozen)

    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    sentinel = _ForbidTensorIRRunner()
    pairs_rhf = list(
        molecular_pair_energies(
            space,
            data,
            src,
            fock,
            sentinel,
            frozen_occupied=frozen,
            include_ordinary=True,
            shell_library=host_sink,
        )
    )
    assert sentinel.calls == 0

    native_fn = bind_mp2_f12_pair_energies(host_reference=True, shell_library=host_sink)
    assert native_fn is not None
    _, _, pf_uhf, po_uhf = run_mp2_f12_pair_energies_native(
        native_fn,
        space=space,
        frozen_occupied=frozen,
        rhf_spatial=False,
        include_ordinary=True,
        budget_bytes=512 << 20,
        data=data,
        sources=src,
        fock=fock,
    )

    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", "1")
    pairs_tir = list(
        molecular_pair_energies(
            space,
            data,
            src,
            fock,
            run_cpu,
            frozen_occupied=frozen,
            include_ordinary=True,
            shell_library=host_sink,
        )
    )
    _assert_pair_energies_close(
        f"HBr-{corr_space}[RHF-AOT vs TensorIR]",
        pairs_rhf,
        pairs_tir,
        include_ordinary=True,
    )
    assert po_uhf is not None
    assert float(np.max(np.abs(pf_uhf - np.array([p.f12 for p in pairs_tir])))) <= 1e-12
    assert float(np.max(np.abs(po_uhf - np.array([p.mp2 for p in pairs_tir])))) <= 1e-12


def test_mp2_f12_pair_energies_edge_cases_nonsym_and_guards(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify H2 (o=1), H atom (ob=0), non-symmetric F/K/S_ab, low_order_energy, and error guards."""
    from generativeqc._f12_mp2_pair_native import (
        bind_mp2_f12_pair_energies,
        run_mp2_f12_pair_energies_native,
    )
    from generativeqc._f12_spin_pairs import (
        _is_rhf_spatial_identity_pair_data,
        molecular_pair_energies,
    )
    from generativeqc_compiler.tensor import execute

    # 1. H2 (o_act = 1 -> single diagonal ab pair (0, 0))
    _, _, space_h2, data_h2, src_h2, fock_h2, frozen_h2, run_h2 = _setup_f12_pair_inputs(
        GEOM_H2, (1,), charge=0, multiplicity=1, root_id="rhf", corr_space="all-active", host_sink=host_sink
    )
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    sentinel = _ForbidTensorIRRunner()
    pairs_h2_aot = list(
        molecular_pair_energies(
            space_h2, data_h2, src_h2, fock_h2, sentinel, frozen_occupied=frozen_h2, shell_library=host_sink
        )
    )
    assert sentinel.calls == 0 and len(pairs_h2_aot) == 1
    assert pairs_h2_aot[0].spins == ("alpha", "beta") and pairs_h2_aot[0].occupied_indices == (0, 0)

    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", "1")
    pairs_h2_tir = list(
        molecular_pair_energies(
            space_h2, data_h2, src_h2, fock_h2, run_h2, frozen_occupied=frozen_h2, shell_library=host_sink
        )
    )
    _assert_pair_energies_close("H2[o=1]", pairs_h2_aot, pairs_h2_tir, include_ordinary=True)

    # 2. H atom (oa=1, ob=0 -> 0 active pairs)
    _, _, space_h, data_h, src_h, fock_h, frozen_h, _ = _setup_f12_pair_inputs(
        GEOM_H, (1,), charge=0, multiplicity=2, root_id="uhf", corr_space="all-active", host_sink=host_sink
    )
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    pairs_h_aot = list(
        molecular_pair_energies(
            space_h, data_h, src_h, fock_h, _ForbidTensorIRRunner(), frozen_occupied=frozen_h, shell_library=host_sink
        )
    )
    assert pairs_h_aot == []

    # 3. Synthetic general non-symmetric Fock/Exchange/cross_overlap (ensuring general UHF path is bit-exact)
    rng = np.random.default_rng(12299)
    space_syn = (2, 2, 5, 7, 4)
    oa, ob, _obs, ext, aux = space_syn
    data_syn: dict[str, np.ndarray] = {
        "cross_overlap": np.eye(ext) + 0.08 * rng.normal(size=(ext, ext)),
    }
    src_syn: dict[str, np.ndarray] = {}
    fock_syn: dict[str, np.ndarray] = {}
    for spin, occ in (("alpha", oa), ("beta", ob)):
        fock_syn["fock_" + spin] = np.diag(np.linspace(-1.5, 2.5, ext)) + 0.05 * rng.normal(size=(ext, ext))
        fock_syn["exchange_" + spin] = 0.04 * rng.normal(size=(ext, ext))
        src_syn["fit_" + spin] = 0.1 * rng.normal(size=(aux, occ, ext))
        src_syn["raw_coulomb_" + spin] = 0.1 * rng.normal(size=(aux, ext, ext))
        for op in ("geminal", "squared_geminal", "geminal_coulomb", "double_commutator"):
            data_syn[f"raw/{op}/{spin}"] = 0.1 * rng.normal(size=(aux, occ, ext))
    for op in ("geminal", "squared_geminal", "geminal_coulomb", "double_commutator"):
        data_syn["metric/" + op] = 0.1 * rng.normal(size=(aux, aux))

    assert not _is_rhf_spatial_identity_pair_data(space_syn, data_syn, src_syn, fock_syn, (0, 0))
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    syn_aot = list(
        molecular_pair_energies(
            space_syn, data_syn, src_syn, fock_syn, _ForbidTensorIRRunner(), shell_library=host_sink
        )
    )
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", "1")
    syn_tir = list(
        molecular_pair_energies(
            space_syn, data_syn, src_syn, fock_syn, execute, shell_library=host_sink
        )
    )
    _assert_pair_energies_close("Synthetic-NonSym", syn_aot, syn_tir, include_ordinary=True)

    # 4. End-to-end MolecularFocalProvider.low_order_energy parity (H2O + OH)
    for name, geom, elems, mult, root in (
        ("H2O-low-order", GEOM_H2O, (1, 8), 1, "rhf"),
        ("OH-low-order", GEOM_OH, (1, 8), 2, "uhf"),
    ):
        basis = _compact_split_basis(elems)
        plan = plan_singlepoint(
            geom, FocalSeries(basis), FocalSeries(basis), None,
            charge=0, multiplicity=mult, root_id=root,
            correlation_space_id="frozen-core", backend="cpu_reference",
        )
        req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
        monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
        e_aot = MolecularFocalProvider(shell_library=host_sink, budget_mib=512).low_order_energy(
            req, include_hf_cabs=False, include_occupied_partitions=True
        )
        monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", "1")
        e_tir = MolecularFocalProvider(shell_library=host_sink, budget_mib=512).low_order_energy(
            req, include_hf_cabs=False, include_occupied_partitions=True
        )
        assert abs(e_aot.f12_increment - e_tir.f12_increment) <= 1e-12, f"{name} F12 diff"
        assert abs(e_aot.mp2_f12_correlation_energy - e_tir.mp2_f12_correlation_energy) <= 1e-12
        assert e_aot.occupied_mp2_f12 is not None and e_tir.occupied_mp2_f12 is not None
        for sa, sb in zip(
            e_aot.occupied_mp2_f12.f12_increment,
            e_tir.occupied_mp2_f12.f12_increment,
            strict=True,
        ):
            assert float(np.max(np.abs(sa - sb))) <= 1e-12

    # 5. Non-finite input guard
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_MP2F12", raising=False)
    native_fn = bind_mp2_f12_pair_energies(host_reference=True, shell_library=host_sink)
    assert native_fn is not None
    bad_fock = dict(fock_h2)
    bad_fa = fock_h2["fock_alpha"].copy()
    bad_fa[0, 0] = np.nan
    bad_fock["fock_alpha"] = bad_fa
    with pytest.raises(RuntimeError, match="status -2"):
        run_mp2_f12_pair_energies_native(
            native_fn,
            space=space_h2,
            frozen_occupied=frozen_h2,
            rhf_spatial=True,
            include_ordinary=True,
            budget_bytes=512 << 20,
            data=data_h2,
            sources=src_h2,
            fock=bad_fock,
        )

    # 6. Missing symbol on explicit shell_library raises (zero silent fallback)
    with pytest.raises(RuntimeError, match="does not export"):
        list(
            molecular_pair_energies(
                space_h2,
                data_h2,
                src_h2,
                fock_h2,
                execute,
                frozen_occupied=frozen_h2,
                shell_library=Path("/nonexistent/fake_shell_library.so"),
            )
        )
