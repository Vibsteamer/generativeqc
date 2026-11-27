"""Native scalar-relativistic 1-electron X2C (DLU-X2C-1e & Global X2C-1e) and pure additive Delta-SR."""

from __future__ import annotations

import ctypes as ct
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.linalg

from . import _native
from ._api_types import Atom
if TYPE_CHECKING:
    from collections.abc import Sequence

    from .basis import BasisSet

# A speed_of_light argument of 0 selects this value; the native ABI rejects
# negative or non-finite values.
SPEED_OF_LIGHT_AU = 137.03599967994
X2c1eModeName = Literal["none", "dlu1e", "global1e", "dlu_local_r"]
X2C1E_MODES: dict[str, int] = {
    "none": 0,
    "dlu1e": 1,
    "global1e": 2,
    "dlu_local_r": 3,
}


def _mode_code(mode: str) -> int:
    if mode not in X2C1E_MODES:
        raise ValueError(f"Unsupported X2C-1e mode {mode!r}; expected one of {tuple(X2C1E_MODES)}")
    return X2C1E_MODES[mode]


def _ao_count(atoms: Sequence[Atom], basis: BasisSet) -> int:
    spherical = basis.representation == "spherical"
    total = 0
    for atom in atoms:
        eb = basis.by_element[atom.atomic_number]
        for sh in eb.shells:
            l = sh.angular_momentum
            ncomp = (2 * l + 1) if spherical else ((l + 1) * (l + 2) // 2)
            total += ncomp * len(sh.coefficients)
    return total


def _double_ptr(arr: np.ndarray | None):
    if arr is None:
        return ct.POINTER(ct.c_double)()
    return arr.ctypes.data_as(ct.POINTER(ct.c_double))


def _create_aux_system_direct(
    lib: ct.CDLL,
    ctx: ct.c_void_p,
    atoms: Sequence[Atom],
    aux_basis: BasisSet,
    charge: int,
    multiplicity: int,
) -> ct.c_void_p:
    """Create a native auxiliary system handle directly (admitting l<=5 h-shells on CPU)."""
    atom_array = (_native.AtomDescriptor * len(atoms))(
        *(
            _native.AtomDescriptor(
                a.atomic_number,
                float(a.position[0]),
                float(a.position[1]),
                float(a.position[2]),
            )
            for a in atoms
        )
    )
    primitives = []
    shells = []
    max_l = 0
    for idx, atom in enumerate(atoms):
        eb = aux_basis.by_element[atom.atomic_number]
        for sh in eb.shells:
            max_l = max(max_l, sh.angular_momentum)
            exps = [float(x) for x in sh.exponents]
            for crow in sh.coefficients:
                offset = len(primitives)
                for e, c in zip(exps, crow, strict=True):
                    primitives.append(_native.PrimitiveDescriptor(e, float(c)))
                shells.append(_native.ShellDescriptor(idx, sh.angular_momentum, offset, len(exps)))
    prim_array = (_native.PrimitiveDescriptor * len(primitives))(*primitives)
    shell_array = (_native.ShellDescriptor * len(shells))(*shells)
    rep = (
        _native.BASIS_SPHERICAL
        if aux_basis.representation == "spherical"
        else _native.BASIS_CARTESIAN
    )
    desc = _native.SystemDescriptor(
        ct.sizeof(_native.SystemDescriptor),
        _native.ABI_VERSION,
        atom_array,
        len(atoms),
        shell_array,
        len(shells),
        prim_array,
        len(primitives),
        charge,
        multiplicity,
        rep,
    )
    sys_h = ct.c_void_p()
    creator = (
        lib.generativeqc_scf_auxiliary_system_create_v1
        if max_l >= 5
        else lib.generativeqc_system_create
    )
    _native.check(lib, creator(ctx, ct.byref(desc), ct.byref(sys_h)), context=ctx)
    return sys_h


def _create_cpu_context_and_system(
    owners: ExitStack,
    atoms: Sequence[Atom],
    basis: BasisSet,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    x2c1e_mode: str = "none",
    speed_of_light: float = 0.0,
    auxiliary_basis: BasisSet | None = None,
):
    from .calculator import Calculator

    method = "uhf" if multiplicity != 1 else "rhf"
    calc = Calculator(
        method=method,
        basis=basis,
        auxiliary_basis=None,
        density_fitting="none",
        device="cpu",
        precision="fp64",
    )
    owners.callback(calc.clear_cache)
    lib = calc._library
    ctx = ct.c_void_p()
    _native.check(
        lib,
        lib.generativeqc_context_create(ct.byref(calc._context_descriptor()), ct.byref(ctx)),
    )
    owners.callback(lib.generativeqc_context_destroy, ctx)
    system = calc._create_native_system(ctx, atoms, charge, multiplicity)
    owners.callback(lib.generativeqc_system_destroy, system)
    if x2c1e_mode != "none":
        _native.check(
            lib,
            lib.generativeqc_system_set_x2c1e_v1(
                ctx,
                system,
                ct.c_uint32(_mode_code(x2c1e_mode)),
                ct.c_double(float(speed_of_light)),
            ),
            context=ctx,
        )
    aux_sys = None
    if auxiliary_basis is not None:
        aux_sys = _create_aux_system_direct(lib, ctx, atoms, auxiliary_basis, charge, multiplicity)
        owners.callback(lib.generativeqc_system_destroy, aux_sys)
    return lib, ctx, system, aux_sys


def stvw_integrals(
    atoms: Sequence[Atom],
    basis: BasisSet,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate native public-AO S, T, V (ionic charges), and W = <nabla|V_nuc|nabla>."""
    n = _ao_count(atoms, basis)
    s = np.empty((n, n), dtype=np.float64)
    t = np.empty((n, n), dtype=np.float64)
    v = np.empty((n, n), dtype=np.float64)
    w = np.empty((n, n), dtype=np.float64)
    with ExitStack() as owners:
        lib, ctx, sys_h, _ = _create_cpu_context_and_system(
            owners, atoms, basis, charge=charge, multiplicity=multiplicity
        )
        _native.check(
            lib,
            lib.generativeqc_system_stvw_integrals_v1(
                ctx,
                sys_h,
                _double_ptr(s),
                _double_ptr(t),
                _double_ptr(v),
                _double_ptr(w),
                ct.c_size_t(n * n),
            ),
            context=ctx,
        )
    return s, t, v, w


def x2c1e_hcore(
    atoms: Sequence[Atom],
    basis: BasisSet,
    *,
    mode: X2c1eModeName = "dlu1e",
    speed_of_light: float = 0.0,
    delta_only: bool = False,
    charge: int = 0,
    multiplicity: int = 1,
) -> np.ndarray:
    """Evaluate native scalar-relativistic X2C-1e hcore (excluding V_ECP) or delta hcore."""
    n = _ao_count(atoms, basis)
    out = np.empty((n, n), dtype=np.float64)
    with ExitStack() as owners:
        lib, ctx, sys_h, _ = _create_cpu_context_and_system(
            owners, atoms, basis, charge=charge, multiplicity=multiplicity
        )
        _native.check(
            lib,
            lib.generativeqc_system_x2c1e_hcore_v1(
                ctx,
                sys_h,
                ct.c_uint32(_mode_code(mode)),
                ct.c_double(float(speed_of_light)),
                ct.c_int(int(delta_only)),
                _double_ptr(out),
                ct.c_size_t(n * n),
            ),
            context=ctx,
        )
    return out


def x2c1e_hcore_derivatives(
    atoms: Sequence[Atom],
    basis: BasisSet,
    *,
    mode: X2c1eModeName = "dlu1e",
    speed_of_light: float = 0.0,
    delta_only: bool = False,
    charge: int = 0,
    multiplicity: int = 1,
) -> np.ndarray:
    """Evaluate exact native Jacobian d(h_X2C)/dR or d(Delta h_SR)/dR with shape (natom, 3, n, n)."""
    n = _ao_count(atoms, basis)
    natom = len(atoms)
    out = np.empty((natom, 3, n, n), dtype=np.float64)
    with ExitStack() as owners:
        lib, ctx, sys_h, _ = _create_cpu_context_and_system(
            owners, atoms, basis, charge=charge, multiplicity=multiplicity
        )
        _native.check(
            lib,
            lib.generativeqc_system_x2c1e_hcore_derivatives_v1(
                ctx,
                sys_h,
                ct.c_uint32(_mode_code(mode)),
                ct.c_double(float(speed_of_light)),
                ct.c_int(int(delta_only)),
                _double_ptr(out),
                ct.c_size_t(out.size),
            ),
            context=ctx,
        )
    return out


def x2c1e_weighted_gradient(
    atoms: Sequence[Atom],
    basis: BasisSet,
    hcore_weights: np.ndarray,
    *,
    mode: X2c1eModeName = "dlu1e",
    speed_of_light: float = 0.0,
    delta_only: bool = False,
    charge: int = 0,
    multiplicity: int = 1,
) -> np.ndarray:
    """Evaluate O(1)-Sylvester/adjoint +d/dR Tr(W_h * h_X2C) or +d/dR Tr(W_h * Delta h_SR)."""
    n = _ao_count(atoms, basis)
    w = np.ascontiguousarray(hcore_weights, dtype=np.float64)
    if w.shape != (n, n):
        raise ValueError(f"Expected hcore_weights shape {(n, n)}, got {w.shape}")
    out = np.empty((len(atoms), 3), dtype=np.float64)
    with ExitStack() as owners:
        lib, ctx, sys_h, _ = _create_cpu_context_and_system(
            owners, atoms, basis, charge=charge, multiplicity=multiplicity
        )
        _native.check(
            lib,
            lib.generativeqc_system_x2c1e_weighted_gradient_v1(
                ctx,
                sys_h,
                ct.c_uint32(_mode_code(mode)),
                ct.c_double(float(speed_of_light)),
                ct.c_int(int(delta_only)),
                _double_ptr(w),
                ct.c_size_t(n * n),
                _double_ptr(out),
                ct.c_size_t(out.size),
            ),
            context=ctx,
        )
    return out


def one_electron_integrals_cpu(
    atoms: Sequence[Atom],
    basis: BasisSet,
    *,
    x2c1e_mode: X2c1eModeName = "none",
    speed_of_light: float = 0.0,
    charge: int = 0,
    multiplicity: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate native CPU overlap, hcore (including checked ECP + optional X2C-1e), and E_nuc."""
    n = _ao_count(atoms, basis)
    overlap = np.empty((n, n), dtype=np.float64)
    hcore = np.empty((n, n), dtype=np.float64)
    e_nuc = ct.c_double(0.0)
    with ExitStack() as owners:
        lib, ctx, sys_h, _ = _create_cpu_context_and_system(
            owners,
            atoms,
            basis,
            charge=charge,
            multiplicity=multiplicity,
            x2c1e_mode=x2c1e_mode,
            speed_of_light=speed_of_light,
        )
        _native.check(
            lib,
            lib.generativeqc_system_one_electron_integrals_cpu_v1(
                ctx,
                sys_h,
                _double_ptr(overlap),
                _double_ptr(hcore),
                ct.c_size_t(n * n),
                ct.byref(e_nuc),
            ),
            context=ctx,
        )
    return overlap, hcore, float(e_nuc.value)


def one_electron_gradient_cpu(
    atoms: Sequence[Atom],
    basis: BasisSet,
    overlap_weights: np.ndarray | None,
    hcore_weights: np.ndarray | None,
    *,
    x2c1e_mode: X2c1eModeName = "none",
    speed_of_light: float = 0.0,
    include_nuclear_repulsion: bool = True,
    charge: int = 0,
    multiplicity: int = 1,
) -> np.ndarray:
    """Contract overlap and hcore weights against native CPU 1e + ECP + X2C-1e derivatives."""
    n = _ao_count(atoms, basis)
    wo = None if overlap_weights is None else np.ascontiguousarray(overlap_weights, dtype=np.float64)
    wh = None if hcore_weights is None else np.ascontiguousarray(hcore_weights, dtype=np.float64)
    out = np.empty((len(atoms), 3), dtype=np.float64)
    with ExitStack() as owners:
        lib, ctx, sys_h, _ = _create_cpu_context_and_system(
            owners,
            atoms,
            basis,
            charge=charge,
            multiplicity=multiplicity,
            x2c1e_mode=x2c1e_mode,
            speed_of_light=speed_of_light,
        )
        _native.check(
            lib,
            lib.generativeqc_system_one_electron_gradient_cpu_v1(
                ctx,
                sys_h,
                _double_ptr(wo),
                _double_ptr(wh),
                ct.c_size_t(n * n),
                ct.c_int(int(include_nuclear_repulsion)),
                _double_ptr(out),
                ct.c_size_t(out.size),
            ),
            context=ctx,
        )
    return out


def df_integrals_cpu(
    atoms: Sequence[Atom],
    orbital_basis: BasisSet,
    aux_basis: BasisSet,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate native CPU DF metric (naux, naux) and three-center (nbf, nbf, naux)."""
    n = _ao_count(atoms, orbital_basis)
    a = _ao_count(atoms, aux_basis)
    metric = np.empty((a, a), dtype=np.float64)
    three_center = np.empty((n, n, a), dtype=np.float64)
    with ExitStack() as owners:
        lib, ctx, orb_h, aux_h = _create_cpu_context_and_system(
            owners,
            atoms,
            orbital_basis,
            charge=charge,
            multiplicity=multiplicity,
            auxiliary_basis=aux_basis,
        )
        _native.check(
            lib,
            lib.generativeqc_system_df_integrals_cpu_v1(
                ctx,
                orb_h,
                aux_h,
                _double_ptr(metric),
                ct.c_size_t(a * a),
                _double_ptr(three_center),
                ct.c_size_t(n * n * a),
            ),
            context=ctx,
        )
    return metric, three_center


def df_gradient_cpu(
    atoms: Sequence[Atom],
    orbital_basis: BasisSet,
    aux_basis: BasisSet,
    metric_weights: np.ndarray,
    three_center_weights: np.ndarray,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> np.ndarray:
    """Contract metric (naux, naux) and three-center (nbf, nbf, naux) weights with native CPU DF derivatives."""
    n = _ao_count(atoms, orbital_basis)
    a = _ao_count(atoms, aux_basis)
    wm = np.ascontiguousarray(metric_weights, dtype=np.float64)
    wa = np.ascontiguousarray(three_center_weights, dtype=np.float64)
    if wm.shape != (a, a) or wa.shape != (n, n, a):
        raise ValueError(f"DF weight shapes mismatch: got {wm.shape} and {wa.shape}")
    out = np.empty((len(atoms), 3), dtype=np.float64)
    with ExitStack() as owners:
        lib, ctx, orb_h, aux_h = _create_cpu_context_and_system(
            owners,
            atoms,
            orbital_basis,
            charge=charge,
            multiplicity=multiplicity,
            auxiliary_basis=aux_basis,
        )
        _native.check(
            lib,
            lib.generativeqc_system_df_gradient_cpu_v1(
                ctx,
                orb_h,
                aux_h,
                _double_ptr(wm),
                ct.c_size_t(a * a),
                _double_ptr(wa),
                ct.c_size_t(n * n * a),
                _double_ptr(out),
                ct.c_size_t(out.size),
            ),
            context=ctx,
        )
    return out


# ==================== Shared-Integral Native DF-HF & DF-MP2 Engine ====================


def _require_basis_set_exchange() -> Any:
    """Import optional `basis_set_exchange` with actionable install guidance."""
    try:
        import basis_set_exchange as bse
    except ImportError as exc:
        raise ImportError(
            "basis_set_exchange is required to build X2C/auxiliary bases from BSE; "
            "install the optional extra `pip install 'generativeqc[bse]'` "
            "(or `pip install 'generativeqc[reference-test]'` / `pip install basis-set-exchange`)."
        ) from exc
    return bse


def build_unc_x2c_basis_from_bse(
    elements: Sequence[str],
    *,
    ae_basis_name: str = "x2c-tzvpall",
    ecp_basis_name: str = "cc-pvdz-pp",
) -> BasisSet:
    """Build a strictly decontracted AE + contracted ECP spherical BasisSet from BSE.

    Decontracting all AE shells (s, p, d, f) eliminates all contraction error (< 0.001 kcal/mol)
    when taking the pure additive difference E(DLU-X2C-1e) - E(NR), while leaving ECP atoms
    (Br ECP10MDF, I ECP28MDF) in their contracted relativistic pseudopotential basis.
    """
    import hashlib
    import json

    bse = _require_basis_set_exchange()

    from ._delta_sr_bse import load_pinned_bse_element
    from .basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
    from .elements import atomic_number

    uniq = sorted(set(elements), key=atomic_number)
    elem_records: list[ElementBasis] = []
    hasher = hashlib.sha256()
    for sym in uniq:
        z = atomic_number(sym)
        is_ecp = sym in ("Br", "I")
        bname = ecp_basis_name if is_ecp else ae_basis_name
        raw, entry = load_pinned_bse_element(bse, bname, z)
        hasher.update(raw.encode("utf-8"))
        if is_ecp:
            shells: list[BasisShell] = []
            for sh in entry["electron_shells"]:
                ang = sh["angular_momentum"]
                exps = tuple(str(x) for x in sh["exponents"])
                if len(ang) == 1:
                    coeffs = tuple(tuple(str(c) for c in row) for row in sh["coefficients"])
                    shells.append(BasisShell(int(ang[0]), exps, coeffs))
                else:
                    for ell, row in zip(ang, sh["coefficients"], strict=True):
                        shells.append(BasisShell(int(ell), exps, (tuple(str(c) for c in row),)))
            ecp_core = int(entry.get("ecp_electrons", 0))
            ecp_json = (
                json.dumps(entry["ecp_potentials"], sort_keys=True) if ecp_core > 0 else None
            )
            elem_records.append(
                ElementBasis(z, tuple(shells), ecp_core_electrons=ecp_core, ecp_data=ecp_json)
            )
        else:
            by_l: dict[int, list[float]] = {}
            for sh in entry["electron_shells"]:
                for ell in sh["angular_momentum"]:
                    for exp_str in sh["exponents"]:
                        by_l.setdefault(int(ell), []).append(float(exp_str))
            shells = []
            for ell in sorted(by_l):
                exps_sorted = sorted(by_l[ell], reverse=True)
                uniq_exps: list[float] = []
                for e in exps_sorted:
                    if not uniq_exps or abs(e - uniq_exps[-1]) > 1e-9 * max(1.0, uniq_exps[-1]):
                        uniq_exps.append(e)
                for e in uniq_exps:
                    shells.append(BasisShell(ell, (f"{e:.12g}",), (("1.0",),)))
            elem_records.append(ElementBasis(z, tuple(shells)))
    prov = BasisProvenance(
        source=f"BSE unc-{ae_basis_name} (AE) + {ecp_basis_name} (ECP)",
        version=f"basis_set_exchange {bse.get_version()}",
        license="BSD-3-Clause",
        checksum=hasher.hexdigest(),
    )
    return BasisSet(f"unc-{ae_basis_name}+{ecp_basis_name}", tuple(elem_records), prov, "spherical")


def build_aux_basis_from_bse(elements: Sequence[str], role: Literal["jk_fit", "mp2_fit"]) -> BasisSet:
    """Build spherical JKFit or MP2Fit auxiliary BasisSet for all benchmark/OMol elements."""
    import hashlib

    bse = _require_basis_set_exchange()

    from ._delta_sr_bse import load_pinned_bse_element
    from .basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
    from .elements import atomic_number

    uniq = sorted(set(elements), key=atomic_number)
    elem_records: list[ElementBasis] = []
    hasher = hashlib.sha256()
    for sym in uniq:
        z = atomic_number(sym)
        if role == "jk_fit":
            bname = "def2-universal-jkfit"
        else:
            bname = "def2-tzvp-rifit"
        raw, entry = load_pinned_bse_element(bse, bname, z)
        hasher.update(raw.encode("utf-8"))
        shells: list[BasisShell] = []
        for sh in entry["electron_shells"]:
            ang = sh["angular_momentum"]
            exps = tuple(str(x) for x in sh["exponents"])
            if len(ang) == 1:
                coeffs = tuple(tuple(str(c) for c in row) for row in sh["coefficients"])
                shells.append(BasisShell(int(ang[0]), exps, coeffs))
            else:
                for ell, row in zip(ang, sh["coefficients"], strict=True):
                    shells.append(BasisShell(int(ell), exps, (tuple(str(c) for c in row),)))
        elem_records.append(ElementBasis(z, tuple(shells)))
    prov = BasisProvenance(
        source=f"BSE {role}",
        version=f"basis_set_exchange {bse.get_version()}",
        license="BSD-3-Clause",
        checksum=hasher.hexdigest(),
    )
    return BasisSet(f"bse-{role}", tuple(elem_records), prov, "spherical")
