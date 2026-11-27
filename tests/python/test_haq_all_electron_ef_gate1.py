"""Gate 1 qualification for all-electron non-ECP orbital g-shell DF-HF E+F admission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import (
    Atom,
    BasisProvenance,
    BasisSet,
    BasisShell,
    Calculator,
    ElementBasis,
    _native,
)
from generativeqc._f12_focal_reference import FocalReferenceSeed, df_reference
from generativeqc.experimental.f12_focal import FocalBasis, FocalRequest


def _prov(tag: str) -> BasisProvenance:
    return BasisProvenance(tag, "1", "MIT", hashlib.sha256(tag.encode()).hexdigest())


def _synthetic_basis(
    name: str,
    elements: tuple[int, ...],
    *,
    heavy_max_l: int,
    h_max_l: int = 3,
    representation: str = "spherical",
    ecp_elements: dict[int, tuple[int, str]] | None = None,
) -> BasisSet:
    """Build a minimal valid all-electron (or ECP) BasisSet spanning s..max_l."""
    ecp_elements = ecp_elements or {}
    exps = ("6.0", "1.8", "0.55", "0.22", "0.40", "0.35", "0.25")
    records = []
    for z in sorted(set(elements)):
        max_l = h_max_l if z == 1 else heavy_max_l
        shells = [BasisShell(0, ("12.0", "2.4"), (("0.4", "0.7"),))]
        if z > 2:
            shells.append(BasisShell(0, ("0.55",), (("1.0",),)))
            shells.append(BasisShell(1, ("1.4",), (("1.0",),)))
        if z > 10:
            shells.append(BasisShell(0, ("0.18",), (("1.0",),)))
            shells.append(BasisShell(1, ("0.35",), (("1.0",),)))
        for ell in range(1, max_l + 1):
            shells.append(BasisShell(ell, (exps[min(ell, len(exps) - 1)],), (("1.0",),)))
        core, ecp_data = ecp_elements.get(z, (0, None))
        records.append(
            ElementBasis(
                z,
                tuple(shells),
                ecp_core_electrons=core,
                ecp_data=ecp_data,
            )
        )
    return BasisSet(name, tuple(records), _prov(name), representation=representation)


def _metadata_calculator(
    orbital: BasisSet,
    auxiliary: BasisSet | None,
    *,
    method: str = "rhf",
    density_fitting: str = "cuda",
    device: str = "cuda",
    precision: str = "fp64",
) -> Calculator:
    """Create a metadata-only Calculator matching Calculator.__init__ state."""
    calc = Calculator.__new__(Calculator)
    calc._method = _native.METHOD_RHF if method == "rhf" else _native.METHOD_UHF
    calc._method_name = method
    calc._device_name = device
    df_map = {
        "none": _native.DENSITY_FITTING_NONE,
        "cuda": _native.DENSITY_FITTING_CUDA,
        "cpu_reference": _native.DENSITY_FITTING_CPU_REFERENCE,
    }
    calc._density_fitting_mode = df_map[density_fitting]
    calc._precision_mode = (
        _native.PRECISION_FP64 if precision == "fp64" else _native.PRECISION_AUTO
    )
    calc._basis = orbital
    calc._auxiliary_basis = auxiliary
    calc._representation_name = orbital.representation
    calc._ks_options = None
    calc._dispersion_method_ir = None
    calc._singlepoint_context = None
    calc._capabilities = SimpleNamespace(
        family="hartree_fock", supported_properties=frozenset(("energy", "forces"))
    )
    calc._library = SimpleNamespace(
        generativeqc_scf_auxiliary_system_create_v1=object()
    )
    return calc


SYSTEM_SPECS = [
    (
        "H2O",
        "rhf",
        0,
        1,
        (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.43, 1.11)), ("H", (0.0, -1.43, 1.11))),
    ),
    (
        "OH",
        "uhf",
        0,
        2,
        # O-H length of GSCDB W4-17_oh (CC BY 4.0, see THIRD_PARTY_NOTICES.md).
        (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.834281919177214))),
    ),
    (
        "IL16_NEU",
        "rhf",
        0,
        1,
        (
            ("C", (0.0, 0.0, 0.0)),
            ("N", (0.0, 0.0, 2.8)),
            ("Cl", (4.5, 0.0, 0.0)),
            ("H", (1.8, 0.0, 0.0)),
        ),
    ),
    (
        "CNa3O3",
        "rhf",
        1,
        1,
        (
            ("C", (0.0, 0.0, 0.0)),
            ("O", (2.4, 0.0, 0.0)),
            ("Na", (0.0, 4.2, 0.0)),
        ),
    ),
]


@pytest.mark.parametrize(
    "label,method,charge,multiplicity,atoms_raw",
    SYSTEM_SPECS,
    ids=[row[0] for row in SYSTEM_SPECS],
)
def test_cuda_df_all_electron_haq_ef_preflight_admitted(
    label: str,
    method: str,
    charge: int,
    multiplicity: int,
    atoms_raw: tuple[tuple[str, tuple[float, float, float]], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atoms = tuple(Atom.from_value(a) for a in atoms_raw)
    zs = tuple(a.atomic_number for a in atoms)
    orb = _synthetic_basis(f"haq-orb-{label}", zs, heavy_max_l=4, h_max_l=3)
    jk = _synthetic_basis(f"haq-jk-{label}", zs, heavy_max_l=5, h_max_l=4)

    calc = _metadata_calculator(orb, jk, method=method)
    report = calc.basis_capability_report(atoms, compute_forces=True)
    assert report["eligible"], report["reasons"]

    orbital_checks = [c for c in report["checks"] if c["role"] == "orbital"]
    aux_checks = [c for c in report["checks"] if c["role"] == "auxiliary"]
    assert {c["operator"] for c in orbital_checks} == {
        "overlap",
        "kinetic",
        "nuclear_attraction",
        "df_three_center",
    }
    assert all(c["consumer"] == "full_df" for c in orbital_checks)
    assert all(c["max_supported_angular_momentum"] == 4 for c in orbital_checks)
    assert {c["derivative_order"] for c in orbital_checks} == {0, 1}

    # Auxiliary handle role must remain scf_df so native focal_df_reference and
    # valid_system_roles receive scf_auxiliary (not correlation_auxiliary).
    assert all(c["consumer"] == "scf_df" for c in aux_checks)
    assert all(c["max_supported_angular_momentum"] == 5 for c in aux_checks)
    assert calc._auxiliary_basis_consumer("df_metric") == "scf_df"

    # df_reference(forces=True) must also pass preflight and reach context creation.
    focal_basis = FocalBasis(4, f"haq-{label}", orb, jk_fit=jk)
    request = FocalRequest(
        atoms=atoms,
        charge=charge,
        multiplicity=multiplicity,
        root_id=method,
        correlation_space_id="frozen-core",
        surface_id=f"preflight-{label}",
        basis=focal_basis,
        role="mp2_f12_correlation",
        backend="cuda",
    )

    class ReachedNativeContext(Exception):
        pass

    lib = _native.load_library(device="cpu")
    monkeypatch.setattr(_native, "load_library", lambda **_: lib)
    monkeypatch.setattr(
        lib,
        "generativeqc_context_create",
        lambda *_: (_ for _ in ()).throw(ReachedNativeContext()),
    )
    with pytest.raises(ReachedNativeContext):
        df_reference(request, forces=True)


def test_non_df_cartesian_mixed_precision_non_halide_ecp_and_h_shell_guards_preserved() -> None:
    atoms = (Atom(8, (0.0, 0.0, 0.0)), Atom(1, (0.0, 0.0, 1.8)))
    orb_g = _synthetic_basis("orb-g", (8, 1), heavy_max_l=4, h_max_l=3)
    jk_h = _synthetic_basis("jk-h", (8, 1), heavy_max_l=5, h_max_l=4)

    # 1. Direct ERI (non-DF) must keep orbital l <= 3 guard.
    direct_calc = _metadata_calculator(
        orb_g, None, method="rhf", density_fitting="none"
    )
    direct_report = direct_calc.basis_capability_report(atoms, compute_forces=True)
    assert not direct_report["eligible"]
    assert any("eri" in r and "l<=3" in r for r in direct_report["reasons"])

    # 2. Mixed precision (precision="auto") must keep orbital l <= 3 guard.
    auto_calc = _metadata_calculator(
        orb_g, jk_h, method="rhf", density_fitting="cuda", precision="auto"
    )
    auto_report = auto_calc.basis_capability_report(atoms, compute_forces=True)
    assert not auto_report["eligible"]
    assert any("l=4" in r and "l<=3" in r for r in auto_report["reasons"])

    # 3. Cartesian representation must keep orbital l <= 3 guard.
    cart_orb = replace(orb_g, representation="cartesian")
    cart_jk = replace(jk_h, representation="cartesian")
    cart_calc = _metadata_calculator(cart_orb, cart_jk, method="uhf")
    cart_report = cart_calc.basis_capability_report(atoms, compute_forces=True)
    assert not cart_report["eligible"]
    assert any("l=4" in r and "l<=3" in r for r in cart_report["reasons"])

    # 4. Non-halide ECP (e.g., Na with 10 core electrons) must reject orbital l=4.
    ecp_payload = json.dumps(
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
    na_atoms = (Atom(11, (0.0, 0.0, 0.0)), Atom(1, (0.0, 0.0, 3.2)))
    na_ecp_orb = _synthetic_basis(
        "na-ecp-g",
        (11, 1),
        heavy_max_l=4,
        h_max_l=2,
        ecp_elements={11: (10, ecp_payload)},
    )
    na_jk = _synthetic_basis("na-jk", (11, 1), heavy_max_l=4, h_max_l=2)
    ecp_calc = _metadata_calculator(na_ecp_orb, na_jk, method="rhf")
    ecp_report = ecp_calc.basis_capability_report(na_atoms, compute_forces=True)
    assert not ecp_report["eligible"]

    # 5. Orbital h-shell (l=5) and auxiliary i-shell (l=6) must still be rejected.
    orb_h = _synthetic_basis("orb-h", (8, 1), heavy_max_l=5, h_max_l=3)
    h_report = _metadata_calculator(orb_h, jk_h, method="rhf").basis_capability_report(
        atoms, compute_forces=True
    )
    assert not h_report["eligible"]
    assert any("l=5" in r and "l<=4" in r for r in h_report["reasons"])

    jk_i = _synthetic_basis("jk-i", (8, 1), heavy_max_l=6, h_max_l=4)
    i_report = _metadata_calculator(orb_g, jk_i, method="uhf").basis_capability_report(
        atoms, compute_forces=True
    )
    assert not i_report["eligible"]
    assert any("l=6" in r and "l<=5" in r for r in i_report["reasons"])


def test_resources_hf_admits_all_electron_cuda_df_g_shell() -> None:
    from generativeqc.resources_hf import hf_resource_request

    # Closed-shell RHF with orbital l=4, auxiliary l=5 and explicit DF budget
    atoms = (Atom(8, (0.0, 0.0, 0.0)), Atom(1, (0.0, 0.0, 1.8)))
    orb_g = _synthetic_basis("res-orb-g", (8, 1), heavy_max_l=4, h_max_l=3)
    jk_h = _synthetic_basis("res-jk-h", (8, 1), heavy_max_l=5, h_max_l=4)
    req_rhf = hf_resource_request(
        [atoms],
        method="rhf",
        basis=orb_g,
        basis_representation="spherical",
        backend="cuda",
        charges=[-1],
        multiplicities=[1],
        density_fitting="cuda",
        auxiliary_basis=jk_h,
        precision="fp64",
        density_fitting_memory_budget_bytes=256 << 20,
    )
    assert req_rhf.unsupported_reason is None
    assert len(req_rhf.candidates) >= 1

    # Open-shell UHF with orbital l=4 and auxiliary l=5 also passes basis preflight
    req_uhf = hf_resource_request(
        [atoms],
        method="uhf",
        basis=orb_g,
        basis_representation="spherical",
        backend="cuda",
        charges=[0],
        multiplicities=[2],
        density_fitting="cuda",
        auxiliary_basis=jk_h,
        precision="fp64",
        density_fitting_memory_budget_bytes=256 << 20,
    )
    assert '"maximum_angular": 4' in req_uhf.identity.topology
    assert '"maximum_angular": 5' in req_uhf.identity.topology


def test_cpu_df_reference_orbital_g_central_fd_rhf_and_uhf() -> None:
    """Verify analytic DF-RHF and DF-UHF forces with orbital g-shells on CPU against central FD."""
    # Compact all-electron spherical orbital basis containing an explicit O g-shell (l=4)
    # and matching spherical JKFit (through g on O, d on H; CPU dynamic Hermite covers l <= 4).
    o_orb = ElementBasis(
        8,
        (
            BasisShell(0, ("130.7", "23.80", "6.443"), (("0.1543", "0.5353", "0.4446"),)),
            BasisShell(0, ("5.033", "1.169", "0.3804"), (("-0.0999", "0.3995", "0.7001"),)),
            BasisShell(1, ("5.033", "1.169", "0.3804"), (("0.1559", "0.6077", "0.3919"),)),
            BasisShell(2, ("1.2",), (("1.0",),)),
            BasisShell(4, ("0.95",), (("1.0",),)),
        ),
    )
    h_orb = ElementBasis(
        1,
        (
            BasisShell(0, ("3.425", "0.6239", "0.1689"), (("0.1543", "0.5353", "0.4446"),)),
            BasisShell(1, ("0.85",), (("1.0",),)),
        ),
    )
    o_jk = ElementBasis(
        8,
        (
            BasisShell(0, ("80.0",), (("1.0",),)),
            BasisShell(0, ("18.0",), (("1.0",),)),
            BasisShell(0, ("4.5",), (("1.0",),)),
            BasisShell(0, ("1.1",), (("1.0",),)),
            BasisShell(1, ("12.0",), (("1.0",),)),
            BasisShell(1, ("2.8",), (("1.0",),)),
            BasisShell(1, ("0.7",), (("1.0",),)),
            BasisShell(2, ("5.0",), (("1.0",),)),
            BasisShell(2, ("1.3",), (("1.0",),)),
            BasisShell(3, ("2.4",), (("1.0",),)),
            BasisShell(4, ("1.9",), (("1.0",),)),
        ),
    )
    h_jk = ElementBasis(
        1,
        (
            BasisShell(0, ("12.0",), (("1.0",),)),
            BasisShell(0, ("2.5",), (("1.0",),)),
            BasisShell(0, ("0.55",), (("1.0",),)),
            BasisShell(1, ("2.0",), (("1.0",),)),
            BasisShell(2, ("1.4",), (("1.0",),)),
        ),
    )
    orb = BasisSet("g-test-orb", (h_orb, o_orb), _prov("g-test-orb"), representation="spherical")
    jk = BasisSet("g-test-jk", (h_jk, o_jk), _prov("g-test-jk"), representation="spherical")
    fb = FocalBasis(4, "cpu-g-fd", orb, jk_fit=jk)

    # 1. Closed-shell H2O (RHF)
    h2o_xyz = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.430, 1.107], [0.0, -1.430, 1.107]], dtype=np.float64
    )

    def h2o_req(xyz: np.ndarray) -> FocalRequest:
        return FocalRequest(
            atoms=tuple(
                Atom.from_value((sym, tuple(pos)))
                for sym, pos in zip(("O", "H", "H"), xyz, strict=True)
            ),
            charge=0,
            multiplicity=1,
            root_id="rhf",
            correlation_space_id="all-active",
            surface_id="h2o-g-fd",
            basis=fb,
            role="hf_cabs",
            backend="cpu_reference",
        )

    center_req = h2o_req(h2o_xyz)
    center = df_reference(center_req, forces=True, budget_mib=512)
    assert center.forces is not None
    # Net force and torque invariants
    assert float(np.max(np.abs(center.forces.sum(axis=0)))) < 1e-12
    torque = np.sum(np.cross(h2o_xyz, center.forces), axis=0)
    assert float(np.max(np.abs(torque))) < 1e-11
    seed = FocalReferenceSeed(center_req, center)
    h = 2e-4
    for atom_idx, axis in ((0, 2), (1, 1)):
        shift = np.zeros_like(h2o_xyz)
        shift[atom_idx, axis] = h
        e_plus = df_reference(h2o_req(h2o_xyz + shift), forces=False, budget_mib=512, seed=seed).energy
        e_minus = df_reference(h2o_req(h2o_xyz - shift), forces=False, budget_mib=512, seed=seed).energy
        f_fd = -(e_plus - e_minus) / (2.0 * h)
        assert abs(center.forces[atom_idx, axis] - f_fd) < 1e-8

    # 2. Open-shell OH (UHF, bond-axis FD with center seed to preserve pi branch)
    oh_xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.834]], dtype=np.float64)

    def oh_req(xyz: np.ndarray) -> FocalRequest:
        return FocalRequest(
            atoms=tuple(
                Atom.from_value((sym, tuple(pos)))
                for sym, pos in zip(("O", "H"), xyz, strict=True)
            ),
            charge=0,
            multiplicity=2,
            root_id="uhf",
            correlation_space_id="all-active",
            surface_id="oh-g-fd",
            basis=fb,
            role="hf_cabs",
            backend="cpu_reference",
        )

    oh_center_req = oh_req(oh_xyz)
    oh_center = df_reference(oh_center_req, forces=True, budget_mib=512)
    assert oh_center.forces is not None
    assert float(np.max(np.abs(oh_center.forces.sum(axis=0)))) < 1e-12
    oh_seed = FocalReferenceSeed(oh_center_req, oh_center)
    fd_by_step = {}
    for step in (2e-4, 1e-4):
        shift = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, step]], dtype=np.float64)
        e_p = df_reference(oh_req(oh_xyz + shift), forces=False, budget_mib=512, seed=oh_seed).energy
        e_m = df_reference(oh_req(oh_xyz - shift), forces=False, budget_mib=512, seed=oh_seed).energy
        fd_by_step[step] = -(e_p - e_m) / (2.0 * step)
    # 2-point error scales as O(h^2); h=1e-4 and Richardson extrapolation must be well below 1e-8.
    assert abs(oh_center.forces[1, 2] - fd_by_step[1e-4]) < 5e-9
    f_rich = (4.0 * fd_by_step[1e-4] - fd_by_step[2e-4]) / 3.0
    assert abs(oh_center.forces[1, 2] - f_rich) < 2e-9


def test_cabs_singles_streaming_loader_matches_dense_pullback() -> None:
    """Verify in-place streaming three_center_loader + obs_three_center matches dense CABS singles."""
    from generativeqc._f12_cabs_fock import obs_sources, physical_fock
    from generativeqc_compiler.tensor import execute

    rng = np.random.default_rng(20261125)
    obs, ext, q, occ = 4, 7, 5, 2
    a = rng.normal(size=(ext, ext))
    h = 0.5 * (a + a.T)
    raw = rng.normal(size=(q, ext, ext))
    raw = 0.5 * (raw + raw.swapaxes(1, 2))
    m_half = rng.normal(size=(q, q))
    metric = m_half @ m_half.T + np.eye(q)
    w = np.linalg.inv(np.linalg.cholesky(metric))
    q_mat, _ = np.linalg.qr(rng.normal(size=(ext, ext)))
    frames = [q_mat, q_mat.copy()]

    def run(prog, inputs):
        return execute(prog, inputs, max_bytes=64 << 20)

    f_dense, pb_dense = physical_fock(h, raw, w, frames, (occ, occ), run, False, None)
    f_stream, pb_stream = physical_fock(
        h, None, w, frames, (occ, occ), run, False, None, raw_loader=lambda: raw.copy()
    )
    for a_mat, b_mat in zip(f_dense, f_stream, strict=True):
        np.testing.assert_allclose(a_mat, b_mat, atol=1e-13, rtol=0)

    seed = rng.normal(size=(ext, ext))
    seed = 0.5 * (seed + seed.T)
    ga_d, hc_d, tc_d, m_d = pb_dense([seed, seed])
    ga_s, hc_s, tc_s, m_s = pb_stream([seed, seed])
    for d_arr, s_arr in ((ga_d[0], ga_s[0]), (ga_d[1], ga_s[1]), (hc_d, hc_s), (tc_d, tc_s), (m_d, m_s)):
        np.testing.assert_allclose(d_arr, s_arr, atol=1e-12, rtol=0)

    # Verify obs_sources shares alpha/beta cleanly for identical frames and matches full panel slice.
    src_full = obs_sources(raw, frames, obs, run)
    src_obs = obs_sources(raw[:, :obs, :obs].copy(), frames, obs, run)
    assert src_obs["raw_coulomb_alpha"] is src_obs["raw_coulomb_beta"]
    np.testing.assert_allclose(src_full["raw_coulomb_alpha"], src_obs["raw_coulomb_alpha"], atol=1e-13, rtol=0)

    # Verify in-place spin_fock_pullback matches out-of-place bit-for-bit on closed-shell spin-symmetric sources.
    from generativeqc._f12_spin_fock import spin_fock_pullback

    raw_c = rng.normal(size=(q, ext, ext))
    fit_c = rng.normal(size=(q, occ, ext))
    h_c = 0.5 * (a + a.T)
    sources_ref = {
        "raw_coulomb_alpha": raw_c.copy(),
        "raw_coulomb_beta": raw_c.copy(),
        "fit_alpha": fit_c.copy(),
        "fit_beta": fit_c.copy(),
        "hcore_alpha": h_c.copy(),
        "hcore_beta": h_c.copy(),
    }
    raw_shared = raw_c.copy()
    sources_ip = {
        "raw_coulomb_alpha": raw_shared,
        "raw_coulomb_beta": raw_shared,
        "fit_alpha": fit_c.copy(),
        "fit_beta": fit_c.copy(),
        "hcore_alpha": h_c.copy(),
        "hcore_beta": h_c.copy(),
    }
    f_seed = rng.normal(size=(ext, ext))
    k_seed = rng.normal(size=(ext, ext))
    seeds = {
        "fock_alpha": f_seed,
        "fock_beta": f_seed.copy(),
        "exchange_alpha": k_seed,
        "exchange_beta": k_seed.copy(),
    }
    out_ref = spin_fock_pullback((occ, occ, ext, q), sources_ref, seeds, run, batch_size=2)
    out_ip = spin_fock_pullback(
        (occ, occ, ext, q), sources_ip, seeds, run, batch_size=2, inplace_raw_coulomb=True
    )
    assert out_ip["raw_coulomb_alpha"] is raw_shared
    assert out_ip["raw_coulomb_beta"] is raw_shared
    for key in out_ref:
        np.testing.assert_allclose(out_ip[key], out_ref[key], atol=1e-13, rtol=0)
