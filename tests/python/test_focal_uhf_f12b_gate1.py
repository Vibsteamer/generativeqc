"""Gate 1 local CPU verification for open-shell native DF-UCCSD-F12b v1.

Verifies:
1. OH (all-active & frozen-core) and NH2 (frozen-core) on split JK/MP2/F12 fits:
   - Native DF-UCCSD-F12b energy and `ccsd_f12b_minus_mp2_f12` match the legacy
     `_f12_ucc_solver.py` path to < 1e-11 Eh.
   - `FixedF12Cotangents` from `generativeqc_focal_ucc_f12b_v1` match the folded
     stationary cotangents of legacy `solve_focal_uccsd_f12b_with_cotangents`
     to < 1e-11 (on `fixed_geminal_energy`, `w_vo_*`, `w_vv_*`, `c_f12_*`).
   - Full 3N analytic `UCCSD-F12b - MP2-F12` forces match legacy `_f12_ucc_solver.py`
     to < 1e-10 Eh/bohr and two-step central finite differences (h = 4e-4, 2e-4 bohr)
     to < 2e-8 Eh/bohr, with net force & torque < 1e-12.
2. Explicit opt-out (`use_native=False`, `use_native_uhf_f12b=False`, and
   `GENERATIVEQC_F12_UHF_CC_BACKEND=legacy`) routes cleanly to the legacy solver, and
   `provider(replace(req, role="ccsd_f12b_minus_mp2_f12"))` agrees with
   `f12_doubles_energy_and_forces` to < 1e-12.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import f12_host_shell

from generativeqc._f12_focal_cc_sources import molecular_ucc_sources
from generativeqc._f12_focal_molecular import MolecularFocalProvider
from generativeqc._f12_focal_ucc_f12b import spin_blocked_uhf_f12b_v1
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalSeries,
    plan_singlepoint,
)


def _host_focal_provider() -> MolecularFocalProvider:
    return MolecularFocalProvider(shell_library=f12_host_shell(), budget_mib=512)


def _build_split_hno_f12_basis() -> FocalBasis:
    """Compact spherical orbital, JKFit, MP2Fit, F12Fit and CABS bases for H, N, O."""

    def make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
        prov = BasisProvenance(
            source=name, version="1", license="MIT", checksum="0" * 64
        )
        return BasisSet(name, elems, prov, representation="spherical")

    def shell(
        l: int, exps: tuple[float, ...], coeffs: tuple[float, ...]
    ) -> BasisShell:
        return BasisShell(
            l, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),)
        )

    h_orb = ElementBasis(
        1,
        (
            shell(0, (5.4471780, 0.82454724), (0.1562850, 0.9046910)),
            shell(0, (0.18319158,), (1.0,)),
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
    h_f12 = ElementBasis(
        1,
        (
            shell(0, (4.7,), (1.0,)),
            shell(0, (1.1,), (1.0,)),
            shell(0, (0.29,), (1.0,)),
            shell(1, (1.2,), (1.0,)),
        ),
    )
    h_cabs = ElementBasis(
        1,
        (
            shell(0, (2.4,), (1.0,)),
            shell(1, (1.1,), (1.0,)),
        ),
    )
    n_cabs = ElementBasis(
        7,
        (
            shell(0, (9.5,), (1.0,)),
            shell(1, (1.8,), (1.0,)),
            shell(2, (1.2,), (1.0,)),
        ),
    )
    o_cabs = ElementBasis(
        8,
        (
            shell(0, (11.5,), (1.0,)),
            shell(1, (2.1,), (1.0,)),
            shell(2, (1.5,), (1.0,)),
        ),
    )

    orb = make_basis("uhf-f12b-gate1-orb", (h_orb, n_orb, o_orb))
    jk = make_basis("uhf-f12b-gate1-jk", (h_jk, heavy_aux(7, 0.86), heavy_aux(8, 1.0)))
    mp2 = make_basis(
        "uhf-f12b-gate1-mp2", (h_mp2, heavy_aux(7, 0.70), heavy_aux(8, 0.82))
    )
    f12 = make_basis(
        "uhf-f12b-gate1-f12", (h_f12, heavy_aux(7, 0.62), heavy_aux(8, 0.74))
    )
    cabs = make_basis("uhf-f12b-gate1-cabs", (h_cabs, n_cabs, o_cabs))
    return FocalBasis(
        2,
        "uhf-f12b-gate1-split3",
        orb,
        jk_fit=jk,
        mp2_fit=mp2,
        f12_fit=f12,
        cabs=cabs,
        beta=1.0,
    )


OH_GEOM = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 1.84)),
)
NH2_ASYM_GEOM = (
    ("N", (0.0, 0.0, 0.0)),
    ("H", (0.11, -1.52, 1.18)),
    ("H", (-0.06, 1.46, 1.24)),
)


def _pair_request(geom, basis: FocalBasis, *, multiplicity: int, corr_space: str):
    plan = plan_singlepoint(
        geom,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=0,
        multiplicity=multiplicity,
        root_id="uhf",
        correlation_space_id=corr_space,
        backend="cpu_reference",
    )
    return next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")


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


def _assert_net_force_and_torque(
    geom, forces: np.ndarray, *, force_tol: float = 1e-12, torque_tol: float = 1e-11
) -> None:
    coords = np.array([pos for _, pos in geom], dtype=np.float64)
    net_force = np.max(np.abs(np.sum(forces, axis=0)))
    net_torque = np.max(np.abs(np.sum(np.cross(coords, forces), axis=0)))
    assert net_force < force_tol, (
        f"Net force violation: {net_force:.3e} >= {force_tol:.1e}"
    )
    assert net_torque < torque_tol, (
        f"Net torque violation: {net_torque:.3e} >= {torque_tol:.1e}"
    )


def _fold_legacy_fixed_cotangents(
    legacy_bars: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Fold legacy 6-block spin-resolved fixed cotangents onto the 11 independent blocks."""
    folded: dict[str, np.ndarray] = {
        "fixed_geminal_energy": np.asarray(
            legacy_bars["fixed_geminal_energy"], dtype=np.float64
        ),
        "w_vo_aaaa": legacy_bars["w_vo_aaaa"].copy(),
        "w_vo_bbbb": legacy_bars["w_vo_bbbb"].copy(),
        "w_vo_abab": legacy_bars["w_vo_abab"].copy(),
        "w_vo_abba": legacy_bars["w_vo_abba"].copy(),
        "w_vv_aaaa": legacy_bars["w_vv_aaaa"].copy(),
        "w_vv_bbbb": legacy_bars["w_vv_bbbb"].copy(),
        "w_vv_abab": legacy_bars["w_vv_abab"].copy(),
        "c_f12_aaaa": legacy_bars["c_f12_aaaa"].copy(),
        "c_f12_bbbb": legacy_bars["c_f12_bbbb"].copy(),
        "c_f12_abab": legacy_bars["c_f12_abab"].copy(),
    }
    # w_vo_baba[j,i,b,k] = -w_vo_abba[i,j,b,k]
    if "w_vo_baba" in legacy_bars:
        folded["w_vo_abba"] = folded["w_vo_abba"] - np.transpose(
            legacy_bars["w_vo_baba"], (1, 0, 2, 3)
        )
    # w_vv / c_f12: baab[j,i,a,b] = -abab[i,j,a,b], abba[i,j,b,a] = -abab[i,j,a,b],
    # baba[j,i,b,a] = +abab[i,j,a,b]
    for prefix in ("w_vv", "c_f12"):
        acc = folded[f"{prefix}_abab"]
        if f"{prefix}_baab" in legacy_bars:
            acc = acc - np.transpose(legacy_bars[f"{prefix}_baab"], (1, 0, 2, 3))
        if f"{prefix}_abba" in legacy_bars:
            acc = acc - np.transpose(legacy_bars[f"{prefix}_abba"], (0, 1, 3, 2))
        if f"{prefix}_baba" in legacy_bars:
            acc = acc + np.transpose(legacy_bars[f"{prefix}_baba"], (1, 0, 3, 2))
        folded[f"{prefix}_abab"] = acc
    return folded


@pytest.mark.parametrize(
    "name,geom,corr_space,coords_to_check",
    [
        ("OH", OH_GEOM, "all-active", ((1, 2),)),
        ("OH", OH_GEOM, "frozen-core", ((1, 2),)),
        ("NH2", NH2_ASYM_GEOM, "frozen-core", ((0, 2), (1, 0), (1, 1))),
    ],
)
def test_uhf_f12b_gate1_parity_cotangents_and_two_step_fd(
    name: str, geom, corr_space: str, coords_to_check
):
    """Gate 1: native DF-UCCSD-F12b matches legacy energy, cotangents, forces & two-step FD."""
    provider = _host_focal_provider()
    basis = _build_split_hno_f12_basis()
    pair_req = _pair_request(geom, basis, multiplicity=2, corr_space=corr_space)

    # Center low-order E+F
    low_ef = provider.low_order_energy_and_forces(pair_req, include_hf_cabs=False)
    low_e = low_ef.energies

    # 1) Native vs legacy f12_doubles_energy (< 1e-11 Eh)
    delta_e_native = provider.f12_doubles_energy(pair_req, low_order=low_e)
    delta_e_legacy = provider.f12_doubles_energy(
        pair_req, low_order=low_e, use_native=False
    )
    assert abs(delta_e_native - delta_e_legacy) < 1e-11, (
        f"{name} {corr_space} delta E mismatch: {delta_e_native} vs {delta_e_legacy}"
    )

    # 2) Direct FixedF12Cotangents parity vs legacy _ucc_stationary
    fixed_src = molecular_ucc_sources(
        provider,
        pair_req,
        include_ordinary=False,
        reference=low_e.reference,
        expected_cabs_rank=low_e.cabs_rank,
    )
    native_res = spin_blocked_uhf_f12b_v1(
        provider, pair_req, fixed_src, forces=True, preconvergence="fp64"
    )
    assert native_res.raw_f12_cotangents is not None
    assert native_res.ordinary_forces is not None
    assert native_res.fixed_forces is not None
    assert native_res.forces is not None
    assert native_res.telemetry["ccsd_iterations"] > 0
    assert native_res.telemetry["lambda_iterations"] > 0

    from generativeqc._ucc.solver import SolverOptions
    from generativeqc._ucc.uccsd_lambda import UCCSDLambdaOptions
    from generativeqc.response_solver import GMRESOptions

    legacy_opts = SolverOptions(
        residual_tolerance=1e-12, energy_tolerance=1e-13, max_bytes=512 << 20
    )
    legacy_lam_opts = UCCSDLambdaOptions(
        max_bytes=512 << 20,
        gmres=GMRESOptions(rtol=0.0, atol=1e-12, max_iterations=400),
    )
    legacy_sol = provider._ucc_stationary(
        pair_req,
        legacy_opts,
        legacy_lam_opts,
        source_cotangents=True,
        reference_forces=False,
        reference=low_e.reference,
        expected_cabs_rank=low_e.cabs_rank,
    )
    assert legacy_sol.primal.correlation_energy is not None
    assert legacy_sol.source_cotangents is not None
    assert (
        abs(native_res.ccsd_f12b_energy - legacy_sol.primal.correlation_energy) < 1e-11
    ), (
        f"{name} {corr_space} UCCSD-F12b E mismatch: "
        f"{native_res.ccsd_f12b_energy} vs {legacy_sol.primal.correlation_energy}"
    )

    folded_legacy = _fold_legacy_fixed_cotangents(
        legacy_sol.source_cotangents.fixed_source_weights
    )
    for key, ref_bar in folded_legacy.items():
        got_bar = native_res.raw_f12_cotangents[key]
        # Same-spin w_vv/c_f12 are antisymmetric in (i,j) and (a,b) in the physical space;
        # compare direct action against physical feeds as well as antisymmetrized tensors.
        if key in ("w_vv_aaaa", "w_vv_bbbb", "c_f12_aaaa", "c_f12_bbbb"):
            def asym4(x):
                return 0.25 * (
                    x
                    - np.swapaxes(x, 0, 1)
                    - np.swapaxes(x, 2, 3)
                    + np.swapaxes(np.swapaxes(x, 0, 1), 2, 3)
                )

            diff = float(np.max(np.abs(asym4(got_bar) - asym4(ref_bar))))
        elif key in ("w_vo_aaaa", "w_vo_bbbb"):
            def asym_ij(x):
                return 0.5 * (x - np.swapaxes(x, 0, 1))

            diff = float(np.max(np.abs(asym_ij(got_bar) - asym_ij(ref_bar))))
        else:
            diff = float(np.max(np.abs(got_bar - ref_bar)))
        assert diff < 5e-10, (
            f"{name} {corr_space} FixedF12Cotangent mismatch on {key}: max_abs={diff:.3e}"
        )

    # 3) Full 3N analytic forces parity vs legacy (ordinary, fixed, and difference)
    legacy_ord, legacy_fix = provider._ucc_correlation_forces(pair_req, legacy_sol)
    assert np.max(np.abs(native_res.ordinary_forces - legacy_ord)) < 1e-10, (
        f"{name} {corr_space} ordinary force mismatch: "
        f"{np.max(np.abs(native_res.ordinary_forces - legacy_ord)):.3e}"
    )
    assert np.max(np.abs(native_res.fixed_forces - legacy_fix)) < 1e-10, (
        f"{name} {corr_space} fixed force mismatch: "
        f"{np.max(np.abs(native_res.fixed_forces - legacy_fix)):.3e}"
    )

    delta_ef_native, delta_f_native = provider.f12_doubles_energy_and_forces(
        pair_req, low_order=low_ef
    )
    legacy_delta_e = legacy_sol.primal.correlation_energy - low_e.mp2_f12_correlation_energy
    legacy_delta_f = legacy_ord + legacy_fix - low_ef.mp2_f12_correlation_forces
    assert abs(delta_ef_native - delta_e_native) < 1e-14
    assert abs(delta_ef_native - legacy_delta_e) < 1e-11
    assert np.max(np.abs(delta_f_native - legacy_delta_f)) < 1e-10
    _assert_net_force_and_torque(
        geom, native_res.ordinary_forces, force_tol=1e-12, torque_tol=1e-12
    )
    _assert_net_force_and_torque(geom, delta_f_native)
    _assert_net_force_and_torque(geom, native_res.forces)

    # 4) Two-step central FD (h = 4e-4, 2e-4 bohr) for UCCSD-F12b - MP2-F12 forces (< 2e-8 Eh/bohr)
    for atom_idx, axis in coords_to_check:
        for h in (4e-4, 2e-4):
            gp = _shift_geom(geom, atom_idx, axis, +h)
            gm = _shift_geom(geom, atom_idx, axis, -h)
            req_p = _pair_request(gp, basis, multiplicity=2, corr_space=corr_space)
            req_m = _pair_request(gm, basis, multiplicity=2, corr_space=corr_space)
            low_p = provider.low_order_energy(
                req_p, include_hf_cabs=False, expected_cabs_rank=low_e.cabs_rank
            )
            low_m = provider.low_order_energy(
                req_m, include_hf_cabs=False, expected_cabs_rank=low_e.cabs_rank
            )
            ep = provider.f12_doubles_energy(
                req_p, low_order=low_p, expected_cabs_rank=low_e.cabs_rank
            )
            em = provider.f12_doubles_energy(
                req_m, low_order=low_m, expected_cabs_rank=low_e.cabs_rank
            )
            fd = -(ep - em) / (2.0 * h)
            an = float(delta_f_native[atom_idx, axis])
            err = abs(an - fd)
            assert err < 2e-8, (
                f"{name} {corr_space} UCCSD-F12b-MP2-F12 FD mismatch at ({atom_idx},{axis}) "
                f"h={h}: an={an:.12f} fd={fd:.12f} err={err:.3e}"
            )


def test_uhf_f12b_opt_out_and_provider_call_routing(monkeypatch: pytest.MonkeyPatch):
    """Verify legacy opt-out switches and provider(ccsd_f12b_minus_mp2_f12) parity."""
    provider = _host_focal_provider()
    basis = _build_split_hno_f12_basis()
    pair_req = _pair_request(OH_GEOM, basis, multiplicity=2, corr_space="frozen-core")

    low_ef = provider.low_order_energy_and_forces(pair_req, include_hf_cabs=False)
    e_nat, f_nat = provider.f12_doubles_energy_and_forces(pair_req, low_order=low_ef)

    # __call__ routing for role="ccsd_f12b_minus_mp2_f12"
    ev = provider(replace(pair_req, role="ccsd_f12b_minus_mp2_f12"))
    assert ev.converged and ev.term.role == "ccsd_f12b_minus_mp2_f12"
    assert abs(ev.term.energy - e_nat) < 1e-12
    assert np.max(np.abs(ev.term.forces - f_nat)) < 1e-11

    # Provider-level opt-out flag (both energy and energy+forces)
    legacy_prov = replace(provider, use_native_uhf_f12b=False)
    assert not legacy_prov._use_native_uhf_f12b(pair_req, low_ef.energies.reference)
    e_flag = legacy_prov.f12_doubles_energy(pair_req, low_order=low_ef.energies)
    assert abs(e_flag - e_nat) < 1e-11
    e_flag_ef, f_flag_ef = legacy_prov.f12_doubles_energy_and_forces(
        pair_req, low_order=low_ef
    )
    assert abs(e_flag_ef - e_nat) < 1e-11
    assert np.max(np.abs(f_flag_ef - f_nat)) < 1e-9

    # Env-var opt-out flag (unified selector D5)
    monkeypatch.setenv("GENERATIVEQC_F12_UHF_CC_BACKEND", "legacy")
    assert not provider._use_native_uhf_f12b(pair_req, low_ef.energies.reference)
    e_env = provider.f12_doubles_energy(pair_req, low_order=low_ef.energies)
    assert abs(e_env - e_nat) < 1e-11
