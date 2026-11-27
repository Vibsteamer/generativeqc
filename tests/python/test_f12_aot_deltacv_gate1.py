"""Gate 1 double-oracle verification for dynamic-dimension AOT DF-MP2 Delta-CV.

Verifies all 4 eliminated TensorIR JIT sites in `generativeqc._f12_focal_cv`:
  1. Site 1 (`_solve_df_reference_from_focal_sources`): DF-HF J/K Fock build
  2. Site 2 (`_mp2_factors_q_chunked`): Q-chunked MP2Fit factor forward & reverse
  3. Site 3 (`_tiled_mp2_delta_cv_pairs`): exact cv-active vs FC4/FC9/frozen:<n>
     pair energies & cotangents (RHF spatial 2*t - t^T fast path + general UHF)
  4. Site 4 (`molecular_delta_cv_mp2_energy_and_forces`): JK MO Fock forward,
     Fock pullback, orbital rotation, matrix-free CPHF, nuclear source weights,
     and 1e nuclear contraction.

Asserts against BOTH PySCF 2.14.0 and the original TensorIR oracle
(`GENERATIVEQC_F12_DISABLE_AOT_CV=1`) on:
  - H2O (RHF)
  - IHB93 (OH- . H2O anion, RHF)
  - OH (UHF doublet)
  - HBr (ECP10MDF, BOTH FC4 `frozen-core` AND FC9 `frozen-core-spd`, plus `frozen:3`;
    `frozen:2` would split the compact basis' Br 3p shell and is refused by the gap guard)
with:
  - |Delta E_CV diff| <= 1e-11 Eh
  - |Delta F_CV diff| <= 1e-10 Eh/bohr
  - PySCF 2.14.0 central finite-difference force verification (< 2e-8 Eh/bohr)
  - Explicit `jit_compile_count == 0` via a forbidding sentinel runner
  - Explicit RHF spatial fast path == general UHF AOT path to machine precision.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import numpy as np
import pytest

from conftest import f12_host_shell

from generativeqc import _f12_focal_cv as cv_mod
from generativeqc._f12_focal_one_electron import one_electron_sources
from generativeqc._f12_frozen_core import resolve_frozen_occupied
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.elements import SYMBOLS
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    MolecularFocalProvider,
    conventional_mp2_request,
)
from _bse_basis_support import br_ecp10mdf

# H2O, IHB93, OH and HBr are the GSCDB records W4-17_h2o,
# IHB100x10_05.012_OH-Oa__water--hydroxide_080, W4-17_oh and P34_P06_HBr
# (J. Liang and M. Head-Gordon, J. Chem. Theory Comput. 21, 12601 (2025);
# CC BY 4.0, see THIRD_PARTY_NOTICES.md), converted from Å to Bohr.
_GEOM_H2O = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 1.8101686547790252)),
    ("H", (1.7554778816245764, 0.0, -0.4415970623550433)),
)
_GEOM_IHB93 = (
    ("O", (-0.14875916494149566, -1.7463483403282758, -1.5325739171875632)),
    ("H", (0.059884585630443576, -0.08313623318156121, -0.15465977807385586)),
    ("H", (-0.5589300330890784, -3.127114233011312, -0.43176062629401624)),
    ("O", (0.22212408920524368, 1.4074915673882387, 1.1577380419586902)),
    ("H", (-1.349583684417033, 2.3011601556349195, 1.0000618358015534)),
)
_GEOM_OH = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 1.834281919177214)),
)
_GEOM_HBR = (
    ("H", (0.0, 0.0, -2.5991784446895245)),
    ("Br", (0.0, 0.0, 0.07426245724554352)),
)


class ForbiddingCountingRunner:
    """Sentinel TensorIR runner that counts and forbids any JIT/TensorIR execution."""

    def __init__(self) -> None:
        self.jit_compile_count = 0
        self.invocations: list[str] = []

    def __call__(self, graph, inputs):
        self.jit_compile_count += 1
        self.invocations.append(repr(tuple(graph.outputs)))
        raise AssertionError(
            f"Unexpected TensorIR execution on AOT Delta-CV path: outputs={tuple(graph.outputs)}"
        )

    @contextmanager
    def batch(self, graph):
        self.jit_compile_count += 1
        self.invocations.append("batch:" + repr(tuple(graph.outputs)))
        raise AssertionError(
            f"Unexpected TensorIR batch compilation on AOT Delta-CV path: outputs={tuple(graph.outputs)}"
        )
        yield  # pragma: no cover


class CountingTensorIRRunner:
    """Wrapper around CPU TensorIR interpreter verifying the oracle actually runs TensorIR."""

    def __init__(self, max_bytes: int = 1 << 30) -> None:
        self.max_bytes = max_bytes
        self.call_count = 0

    def __call__(self, graph, inputs):
        from generativeqc_compiler.tensor import execute

        self.call_count += 1
        return execute(graph, inputs, max_bytes=self.max_bytes)

    @contextmanager
    def batch(self, graph):
        from generativeqc_compiler.tensor import execute

        def exec_one(inputs):
            self.call_count += 1
            return execute(graph, inputs, max_bytes=self.max_bytes)

        yield exec_one


def _sh(ell: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
    return BasisShell(ell, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),))


def _make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
    prov = BasisProvenance(source=name, version="1", license="MIT", checksum="0" * 64)
    return BasisSet(name, elems, prov, representation="spherical")


def _build_split_cv_basis(elements: tuple[int, ...], tag: str) -> FocalBasis:
    """Compact spherical split-fit (JKFit != MP2Fit, zero CABS/F12) basis for H, O, Br(ECP10MDF)."""
    br_ecp = br_ecp10mdf()
    orb_map = {
        1: ElementBasis(
            1,
            (
                _sh(0, (3.42525091, 0.62391373), (0.15432897, 0.53532814)),
                _sh(0, (0.16885540,), (1.0,)),
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
        1: ElementBasis(
            1,
            (
                _sh(0, (6.0,), (1.0,)),
                _sh(0, (1.5,), (1.0,)),
                _sh(0, (0.4,), (1.0,)),
                _sh(1, (1.4,), (1.0,)),
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
                _sh(0, (32.0,), (1.0,)),
                _sh(0, (9.0,), (1.0,)),
                _sh(0, (2.5,), (1.0,)),
                _sh(0, (0.7,), (1.0,)),
                _sh(0, (0.22,), (1.0,)),
                _sh(1, (12.0,), (1.0,)),
                _sh(1, (3.2,), (1.0,)),
                _sh(1, (0.85,), (1.0,)),
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
                _sh(0, (26.0,), (1.0,)),
                _sh(0, (7.5,), (1.0,)),
                _sh(0, (2.1,), (1.0,)),
                _sh(0, (0.58,), (1.0,)),
                _sh(0, (0.18,), (1.0,)),
                _sh(1, (9.5,), (1.0,)),
                _sh(1, (2.6,), (1.0,)),
                _sh(1, (0.70,), (1.0,)),
                _sh(2, (3.8,), (1.0,)),
                _sh(2, (1.0,), (1.0,)),
            ),
        ),
    }
    uniq = tuple(sorted(set(elements)))
    orb = _make_basis(f"aot-cv-orb-{tag}", tuple(orb_map[z] for z in uniq))
    jk = _make_basis(f"aot-cv-jk-{tag}", tuple(jk_map[z] for z in uniq))
    mp2 = _make_basis(f"aot-cv-mp2-{tag}", tuple(mp2_map[z] for z in uniq))
    return FocalBasis(
        2,
        f"aot-cv-split/{tag}",
        orb,
        jk_fit=jk,
        mp2_fit=mp2,
        f12_fit=None,
        cabs=None,
        beta=None,
    )


def _basis_to_pyscf(basis: BasisSet) -> tuple[dict, dict]:
    pyscf_basis: dict[str, list] = {}
    pyscf_ecp: dict[str, list] = {}
    for z, eb in basis.by_element.items():
        sym = SYMBOLS[z - 1]
        shells = []
        for sh in eb.shells:
            for crow in sh.coefficients:
                row = [sh.angular_momentum]
                for exp, coeff in zip(sh.exponents, crow, strict=True):
                    row.append([float(exp), float(coeff)])
                shells.append(row)
        pyscf_basis[sym] = shells
        if eb.ecp_core_electrons > 0 and eb.ecp_data:
            pots = json.loads(eb.ecp_data) if isinstance(eb.ecp_data, str) else eb.ecp_data
            max_l = max(int(p["angular_momentum"][0]) for p in pots)
            by_l: dict[int, list[tuple[int, float, float]]] = {}
            for p in pots:
                l_val = int(p["angular_momentum"][0])
                ang = -1 if l_val == max_l else l_val
                for r_exp, g_exp, coeff in zip(
                    p["r_exponents"],
                    p["gaussian_exponents"],
                    p["coefficients"][0],
                    strict=True,
                ):
                    by_l.setdefault(ang, []).append((int(r_exp), float(g_exp), float(coeff)))
            ecp_shells = []
            for ang in sorted(by_l):
                max_n = max(t[0] for t in by_l[ang])
                buckets: list[list[list[float]]] = [[] for _ in range(max_n + 1)]
                for r_exp, g_exp, coeff in by_l[ang]:
                    buckets[r_exp].append([g_exp, coeff])
                ecp_shells.append([ang, buckets])
            pyscf_ecp[sym] = [int(eb.ecp_core_electrons), ecp_shells]
    return pyscf_basis, pyscf_ecp


def _pyscf_delta_cv_energy(
    atoms_spec,
    basis: FocalBasis,
    *,
    root: str,
    charge: int,
    mult: int,
    cv_frozen: int,
    fc_frozen: int,
    dm0=None,
) -> tuple[float, float, float, float]:
    from pyscf import gto, mp, scf

    orb_b, ecp_b = _basis_to_pyscf(basis.orbital)
    jk_b, _ = _basis_to_pyscf(basis.jk_fit)
    mp2_b, _ = _basis_to_pyscf(basis.mp2_fit)
    mol = gto.M(
        atom=[(s, tuple(xyz)) for s, xyz in atoms_spec],
        basis=orb_b,
        ecp=ecp_b if ecp_b else None,
        unit="Bohr",
        cart=False,
        charge=charge,
        spin=mult - 1,
        verbose=0,
    )
    mf = (scf.RHF(mol) if root == "rhf" else scf.UHF(mol)).density_fit(auxbasis=jk_b)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 120
    mf.kernel(dm0=dm0)
    assert mf.converged
    mp_cls = mp.MP2 if root == "rhf" else mp.UMP2
    e_cv = float(mp_cls(mf, frozen=cv_frozen).density_fit(auxbasis=mp2_b).kernel()[0])
    e_fc = float(mp_cls(mf, frozen=fc_frozen).density_fit(auxbasis=mp2_b).kernel()[0])
    return float(mf.e_tot), e_cv, e_fc, e_cv - e_fc


@pytest.mark.parametrize(
    "label,atoms_spec,elems,root,charge,mult,fc_mode,exp_cv,exp_fc,fd_axis",
    [
        ("H2O", _GEOM_H2O, (1, 8), "rhf", 0, 1, "frozen-core", 0, 1, (1, 2)),
        ("IHB93", _GEOM_IHB93, (1, 8), "rhf", -1, 1, "frozen-core", 0, 2, (1, 1)),
        ("OH", _GEOM_OH, (1, 8), "uhf", 0, 2, "frozen-core", 0, 1, (1, 2)),
        ("HBr-FC4", _GEOM_HBR, (1, 35), "rhf", 0, 1, "frozen-core", 0, 4, (0, 2)),
        ("HBr-FC9", _GEOM_HBR, (1, 35), "rhf", 0, 1, "frozen-core-spd", 0, 9, (0, 2)),
        # Compact Br basis: 3p (0-2) lies below 3s (3), so frozen:3 closes the 3p shell.
        ("HBr-frozen3", _GEOM_HBR, (1, 35), "rhf", 0, 1, "frozen:3", 0, 3, (0, 2)),
    ],
)
def test_aot_deltacv_double_oracle_pyscf_and_tensorir_zero_jit(
    label,
    atoms_spec,
    elems,
    root,
    charge,
    mult,
    fc_mode,
    exp_cv,
    exp_fc,
    fd_axis,
    monkeypatch,
):
    """Double-oracle Gate 1 check: AOT vs PySCF 2.14.0 & TensorIR with jit_compile_count == 0."""
    pytest.importorskip("pyscf.gto")
    host_lib = f12_host_shell()
    basis = _build_split_cv_basis(elems, label)

    req_cv = conventional_mp2_request(
        atoms_spec,
        basis,
        root_id=root,
        correlation_space_id="cv-active",
        charge=charge,
        multiplicity=mult,
        backend="cpu_reference",
    )
    req_fc = conventional_mp2_request(
        atoms_spec,
        basis,
        root_id=root,
        correlation_space_id=fc_mode,
        charge=charge,
        multiplicity=mult,
        backend="cpu_reference",
    )

    # 1) AOT native path with forbidding sentinel runner -> MUST have jit_compile_count == 0!
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_CV", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_CV", "1")
    sentinel = ForbiddingCountingRunner()
    prov_aot = MolecularFocalProvider(
        shell_library=host_lib,
        one_electron_library=host_lib,
        run=sentinel,
        budget_mib=2048,
    )
    res_e_aot = prov_aot.delta_cv_mp2_energy(req_cv, req_fc)
    res_ef_aot = prov_aot.delta_cv_mp2_energy_and_forces(req_cv, req_fc)
    assert sentinel.jit_compile_count == 0, f"{label}: unexpected JIT calls {sentinel.invocations}"
    assert res_ef_aot.cv_frozen == (exp_cv, exp_cv)
    assert res_ef_aot.fc_frozen == (exp_fc, exp_fc)
    assert res_ef_aot.delta_forces is not None
    assert abs(res_e_aot.delta_ordinary_mp2_Eh - res_ef_aot.delta_ordinary_mp2_Eh) < 1e-14

    # Net force and torque invariants
    xyz = np.asarray([pos for _, pos in atoms_spec], dtype=np.float64)
    net_f = float(np.max(np.abs(np.sum(res_ef_aot.delta_forces, axis=0))))
    torque = float(np.max(np.abs(np.sum(np.cross(xyz, res_ef_aot.delta_forces), axis=0))))
    assert net_f < 1e-12, f"{label}: net force invariant {net_f:.3e}"
    assert torque < 1e-12, f"{label}: torque invariant {torque:.3e}"

    # 2) Original TensorIR oracle (GENERATIVEQC_F12_DISABLE_AOT_CV=1)
    monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_CV", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_CV", "1")
    tir_runner = CountingTensorIRRunner()
    prov_tir = MolecularFocalProvider(
        shell_library=host_lib,
        one_electron_library=host_lib,
        run=tir_runner,
        budget_mib=2048,
    )
    res_ef_tir = prov_tir.delta_cv_mp2_energy_and_forces(req_cv, req_fc)
    assert tir_runner.call_count > 0, "TensorIR oracle failed to execute TensorIR path"
    assert res_ef_tir.delta_forces is not None

    de_tir = abs(res_ef_aot.delta_ordinary_mp2_Eh - res_ef_tir.delta_ordinary_mp2_Eh)
    de_cv_tir = abs(res_ef_aot.cv_ordinary_mp2_Eh - res_ef_tir.cv_ordinary_mp2_Eh)
    de_fc_tir = abs(res_ef_aot.fc_ordinary_mp2_Eh - res_ef_tir.fc_ordinary_mp2_Eh)
    df_tir = float(np.max(np.abs(res_ef_aot.delta_forces - res_ef_tir.delta_forces)))
    assert de_tir <= 1e-11, f"{label}: AOT vs TensorIR Delta-E diff {de_tir:.3e}"
    assert de_cv_tir <= 1e-11, f"{label}: AOT vs TensorIR cv-E diff {de_cv_tir:.3e}"
    assert de_fc_tir <= 1e-11, f"{label}: AOT vs TensorIR fc-E diff {de_fc_tir:.3e}"
    assert df_tir <= 1e-10, f"{label}: AOT vs TensorIR Delta-F diff {df_tir:.3e}"

    # 3) Independent PySCF 2.14.0 oracle (energy + central FD force!)
    py_hf, py_cv, py_fc, py_dcv = _pyscf_delta_cv_energy(
        atoms_spec,
        basis,
        root=root,
        charge=charge,
        mult=mult,
        cv_frozen=exp_cv,
        fc_frozen=exp_fc,
    )
    assert abs(res_ef_aot.reference.energy - py_hf) <= 1e-10, f"{label}: HF mismatch vs PySCF"
    assert abs(res_ef_aot.cv_ordinary_mp2_Eh - py_cv) <= 1e-11, f"{label}: E_cv mismatch vs PySCF"
    assert abs(res_ef_aot.fc_ordinary_mp2_Eh - py_fc) <= 1e-11, f"{label}: E_fc mismatch vs PySCF"
    de_pyscf = abs(res_ef_aot.delta_ordinary_mp2_Eh - py_dcv)
    assert de_pyscf <= 1e-11, f"{label}: Delta-E mismatch vs PySCF {de_pyscf:.3e}"

    # Independent PySCF 2.14.0 two-step central FD check on analytic AOT Delta-F
    atom_idx, axis = fd_axis
    h = 2e-4

    def _shifted(sign: float):
        disp = [list(pos) for _, pos in atoms_spec]
        disp[atom_idx][axis] += sign * h
        return tuple((s, tuple(p)) for (s, _), p in zip(atoms_spec, disp, strict=True))

    _, _, _, dcv_p = _pyscf_delta_cv_energy(
        _shifted(+1.0), basis, root=root, charge=charge, mult=mult, cv_frozen=exp_cv, fc_frozen=exp_fc
    )
    _, _, _, dcv_m = _pyscf_delta_cv_energy(
        _shifted(-1.0), basis, root=root, charge=charge, mult=mult, cv_frozen=exp_cv, fc_frozen=exp_fc
    )
    fd_pyscf = -(dcv_p - dcv_m) / (2.0 * h)
    df_pyscf_fd = abs(float(res_ef_aot.delta_forces[atom_idx, axis]) - fd_pyscf)
    assert df_pyscf_fd < 2e-8, f"{label}: AOT Delta-F vs PySCF FD err={df_pyscf_fd:.3e}"

    print(
        f"[GATE1] {label:12s} dE_PySCF={de_pyscf:.2e} dE_TIR={de_tir:.2e} "
        f"dF_TIR={df_tir:.2e} dF_PySCF_FD={df_pyscf_fd:.2e} jit={sentinel.jit_compile_count}"
    )


def test_aot_deltacv_refuses_frozen_count_inside_degenerate_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """frozen:2 freezes two of the three degenerate Br 3p orbitals of the compact basis."""
    host_lib = f12_host_shell()
    basis = _build_split_cv_basis((1, 35), "HBr-frozen2")
    req_cv = conventional_mp2_request(
        _GEOM_HBR,
        basis,
        root_id="rhf",
        correlation_space_id="cv-active",
        backend="cpu_reference",
    )
    req_fc = conventional_mp2_request(
        _GEOM_HBR,
        basis,
        root_id="rhf",
        correlation_space_id="frozen:2",
        backend="cpu_reference",
    )
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_CV", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_CV", "1")
    provider = MolecularFocalProvider(
        shell_library=host_lib,
        one_electron_library=host_lib,
        run=ForbiddingCountingRunner(),
        budget_mib=2048,
    )
    with pytest.raises(
        ValueError, match=r"n_frozen=2 .*splits a near-degenerate shell"
    ):
        provider.delta_cv_mp2_energy_and_forces(req_cv, req_fc)


@pytest.mark.parametrize(
    "label,atoms_spec,elems,root,charge,mult",
    [
        ("H2O", _GEOM_H2O, (1, 8), "rhf", 0, 1),
        ("IHB93", _GEOM_IHB93, (1, 8), "rhf", -1, 1),
        ("OH", _GEOM_OH, (1, 8), "uhf", 0, 2),
        ("HBr", _GEOM_HBR, (1, 35), "rhf", 0, 1),
    ],
)
def test_aot_site1_fallback_df_hf_parity_and_zero_jit(
    label, atoms_spec, elems, root, charge, mult, monkeypatch
):
    """Explicitly exercise Site 1 (_solve_df_reference_from_focal_sources) with jit_compile_count == 0."""
    pytest.importorskip("pyscf.gto")
    host_lib = f12_host_shell()
    basis = _build_split_cv_basis(elems, f"site1-{label}")
    req = conventional_mp2_request(
        atoms_spec,
        basis,
        root_id=root,
        correlation_space_id="cv-active",
        charge=charge,
        multiplicity=mult,
        backend="cpu_reference",
    )

    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_CV", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_CV", "1")
    sentinel = ForbiddingCountingRunner()
    prov = MolecularFocalProvider(
        shell_library=host_lib,
        one_electron_library=host_lib,
        run=sentinel,
        budget_mib=2048,
    )
    one = one_electron_sources(req, library=host_lib, derivatives=False, orbital_only=True, budget_mib=2048)
    jk = prov._panel(req, "jk_fit", orbital_only=True)
    ref_aot = cv_mod._solve_df_reference_from_focal_sources(prov, req, one, jk, run=sentinel)
    assert sentinel.jit_compile_count == 0

    monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_CV", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_CV", "1")
    tir_runner = CountingTensorIRRunner()
    ref_tir = cv_mod._solve_df_reference_from_focal_sources(prov, req, one, jk, run=tir_runner)
    assert tir_runner.call_count > 0

    py_hf, _, _, _ = _pyscf_delta_cv_energy(
        atoms_spec, basis, root=root, charge=charge, mult=mult, cv_frozen=0, fc_frozen=1
    )
    assert abs(ref_aot.energy - ref_tir.energy) < 1e-12
    assert abs(ref_aot.energy - py_hf) < 1e-11


@pytest.mark.parametrize(
    "label,atoms_spec,elems,charge,fc_mode",
    [
        ("H2O", _GEOM_H2O, (1, 8), 0, "frozen-core"),
        ("IHB93", _GEOM_IHB93, (1, 8), -1, "frozen-core"),
        ("HBr-FC4", _GEOM_HBR, (1, 35), 0, "frozen-core"),
        ("HBr-FC9", _GEOM_HBR, (1, 35), 0, "frozen-core-spd"),
    ],
)
def test_aot_rhf_spatial_fast_path_equals_uhf_general_path(
    label, atoms_spec, elems, charge, fc_mode, monkeypatch
):
    """RHF spatial 2*t - t^T fast path MUST match general UHF AOT path to machine precision."""
    host_lib = f12_host_shell()
    basis = _build_split_cv_basis(elems, f"rhf-vs-uhf-{label}")

    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_CV", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_CV", "1")
    sentinel = ForbiddingCountingRunner()
    prov = MolecularFocalProvider(
        shell_library=host_lib,
        one_electron_library=host_lib,
        run=sentinel,
        budget_mib=2048,
    )

    req_cv_r = conventional_mp2_request(
        atoms_spec, basis, root_id="rhf", correlation_space_id="cv-active",
        charge=charge, multiplicity=1, backend="cpu_reference",
    )
    req_fc_r = conventional_mp2_request(
        atoms_spec, basis, root_id="rhf", correlation_space_id=fc_mode,
        charge=charge, multiplicity=1, backend="cpu_reference",
    )
    req_cv_u = conventional_mp2_request(
        atoms_spec, basis, root_id="uhf", correlation_space_id="cv-active",
        charge=charge, multiplicity=1, backend="cpu_reference",
    )
    req_fc_u = conventional_mp2_request(
        atoms_spec, basis, root_id="uhf", correlation_space_id=fc_mode,
        charge=charge, multiplicity=1, backend="cpu_reference",
    )

    # 1) Full end-to-end E+F comparison: root_id="rhf" vs root_id="uhf" (mult=1)
    res_r = prov.delta_cv_mp2_energy_and_forces(req_cv_r, req_fc_r)
    res_u = prov.delta_cv_mp2_energy_and_forces(req_cv_u, req_fc_u)
    assert sentinel.jit_compile_count == 0
    assert res_r.delta_forces is not None and res_u.delta_forces is not None
    assert abs(res_r.delta_ordinary_mp2_Eh - res_u.delta_ordinary_mp2_Eh) < 1e-12
    assert float(np.max(np.abs(res_r.delta_forces - res_u.delta_forces))) < 1e-11

    # 2) Direct pair-kernel comparison (restricted=True vs force_uhf_path=True on exact same RHF B)
    from generativeqc._f12_cv_dfmp2_native import bind_delta_cv_native

    aot = bind_delta_cv_native(host_lib, host_reference=True)
    assert aot is not None
    ref_r = res_r.reference
    indices = ref_r._pair_spin_indices(req_cv_r)
    occs = tuple(ref_r.occupied[k] for k in indices)
    cv_frz = resolve_frozen_occupied(req_cv_r, occs)
    fc_frz = resolve_frozen_occupied(req_fc_r, occs)
    mp2_src = prov._panel(req_cv_r, "mp2_fit", orbital_only=True).obs(req_cv_r, len(ref_r.hcore))
    B_a, _ = cv_mod._mp2_factors_q_chunked(
        mp2_src.three_center, mp2_src.whitener, ref_r.coefficients[0], occs[0], cv_frz[0], sentinel, aot=aot
    )
    B_map = {"alpha": B_a, "beta": B_a}

    cv_r, fc_r, barB_r, fock_r = cv_mod._tiled_mp2_delta_cv_pairs(
        ref_r, indices, cv_frz, fc_frz, B_map, sentinel, need_cotangents=True, aot=aot, force_uhf_path=False
    )
    cv_u, fc_u, barB_u, fock_u = cv_mod._tiled_mp2_delta_cv_pairs(
        ref_r, indices, cv_frz, fc_frz, B_map, sentinel, need_cotangents=True, aot=aot, force_uhf_path=True
    )
    assert np.allclose(cv_r, cv_u, rtol=0, atol=1e-13), f"{label}: cv_spins mismatch {cv_r} vs {cv_u}"
    assert np.allclose(fc_r, fc_u, rtol=0, atol=1e-13), f"{label}: fc_spins mismatch {fc_r} vs {fc_u}"
    assert np.allclose(barB_r["alpha"], barB_u["alpha"], rtol=0, atol=1e-13), f"{label}: bar_B mismatch"
    assert np.allclose(fock_r["fock_alpha"], fock_u["fock_alpha"], rtol=0, atol=1e-13), f"{label}: fock_alpha mismatch"
    assert np.allclose(fock_r["fock_beta"], fock_u["fock_beta"], rtol=0, atol=1e-13), f"{label}: fock_beta mismatch"
    assert sentinel.jit_compile_count == 0


def test_aot_multichunk_multitile_and_boundary_cases(monkeypatch):
    """Verify forced multi-chunk/multi-tile boundaries, UHF ECP cation (oa!=ob), and zero-shift H2."""
    host_lib = f12_host_shell()
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_CV", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_CV", "1")
    sentinel = ForbiddingCountingRunner()
    prov = MolecularFocalProvider(
        shell_library=host_lib,
        one_electron_library=host_lib,
        run=sentinel,
        budget_mib=2048,
    )

    # 1) Multi-chunk / multi-tile parity on HBr (FC9) and OH (UHF)
    for label, atoms_spec, elems, root, charge, mult, fc_mode in (
        ("OH", _GEOM_OH, (1, 8), "uhf", 0, 2, "frozen-core"),
        ("HBr-FC9", _GEOM_HBR, (1, 35), "rhf", 0, 1, "frozen-core-spd"),
        ("HBr+-UHF-FC9", _GEOM_HBR, (1, 35), "uhf", 1, 2, "frozen-core-spd"),
    ):
        basis = _build_split_cv_basis(elems, f"tile-{label}")
        req_cv = conventional_mp2_request(
            atoms_spec, basis, root_id=root, correlation_space_id="cv-active",
            charge=charge, multiplicity=mult, backend="cpu_reference",
        )
        req_fc = conventional_mp2_request(
            atoms_spec, basis, root_id=root, correlation_space_id=fc_mode,
            charge=charge, multiplicity=mult, backend="cpu_reference",
        )
        res_full = prov.delta_cv_mp2_energy_and_forces(req_cv, req_fc)

        orig_pairs = cv_mod._tiled_mp2_delta_cv_pairs
        orig_fac = cv_mod._mp2_factors_q_chunked
        with monkeypatch.context() as mp:
            mp.setattr(
                cv_mod,
                "_tiled_mp2_delta_cv_pairs",
                lambda *a, **kw: orig_pairs(*a, **{**kw, "max_tile_elements": 350}),
            )
            mp.setattr(
                cv_mod,
                "_mp2_factors_q_chunked",
                lambda *a, **kw: orig_fac(*a, **{**kw, "max_chunk_elements": 500}),
            )
            res_tiled = prov.delta_cv_mp2_energy_and_forces(req_cv, req_fc)

        assert res_full.delta_forces is not None and res_tiled.delta_forces is not None
        assert abs(res_full.delta_ordinary_mp2_Eh - res_tiled.delta_ordinary_mp2_Eh) < 1e-13
        assert float(np.max(np.abs(res_full.delta_forces - res_tiled.delta_forces))) < 1e-12

        # Also compare HBr+ UHF (oa != ob, ECP10MDF, FC9) against BOTH TensorIR and PySCF 2.14.0
        # (seeding PySCF with the same UHF determinant to avoid linear 2Pi/2Sigma state mismatch).
        if label == "HBr+-UHF-FC9":
            with monkeypatch.context() as mp:
                mp.delenv("GENERATIVEQC_F12_REQUIRE_AOT_CV", raising=False)
                mp.setenv("GENERATIVEQC_F12_DISABLE_AOT_CV", "1")
                tir_runner = CountingTensorIRRunner()
                prov_tir = MolecularFocalProvider(
                    shell_library=host_lib,
                    one_electron_library=host_lib,
                    run=tir_runner,
                    budget_mib=2048,
                )
                res_tir = prov_tir.delta_cv_mp2_energy_and_forces(req_cv, req_fc)
                assert tir_runner.call_count > 0
                assert res_tir.delta_forces is not None
                assert abs(res_full.delta_ordinary_mp2_Eh - res_tir.delta_ordinary_mp2_Eh) < 1e-12
                assert float(np.max(np.abs(res_full.delta_forces - res_tir.delta_forces))) < 1e-11

            pytest.importorskip("pyscf.gto")
            py_hf, _, _, py_dcv = _pyscf_delta_cv_energy(
                atoms_spec,
                basis,
                root="uhf",
                charge=1,
                mult=2,
                cv_frozen=0,
                fc_frozen=9,
                dm0=res_full.reference.density,
            )
            assert abs(res_full.reference.energy - py_hf) < 1e-11
            assert abs(res_full.delta_ordinary_mp2_Eh - py_dcv) < 1e-11

    # 2) Zero-shift early return (H2: cv_frozen == fc_frozen == (0, 0))
    basis_h = _build_split_cv_basis((1,), "h2-zero")
    h2_atoms = (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4)))
    req_h2 = conventional_mp2_request(
        h2_atoms, basis_h, root_id="rhf", correlation_space_id="cv-active",
        charge=0, multiplicity=1, backend="cpu_reference",
    )
    res_h2 = prov.delta_cv_mp2_energy_and_forces(req_h2)
    assert res_h2.delta_ordinary_mp2_Eh == 0.0
    assert res_h2.delta_forces is not None and np.all(res_h2.delta_forces == 0.0)
    assert sentinel.jit_compile_count == 0
