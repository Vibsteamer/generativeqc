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
from ._f12_frozen_core import (
    FROZEN_CORE_DEFINITION_VERSION,
    FrozenOccupiedFrame,
    resolve_frozen_occupied_for_atoms,
    select_frozen_occupied_frame,
)

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


@dataclass(frozen=True)
class _DfFactor:
    metric: np.ndarray        # (naux, naux)
    whitener: np.ndarray      # (nrank, naux): W @ M @ W.T = I
    b_ao: np.ndarray          # (nrank, nbf, nbf): B_Q = sum_P W[Q, P] A[:, :, P]


def _factor_df(metric: np.ndarray, three_center: np.ndarray, rel_tol: float = 1e-12) -> _DfFactor:
    naux = metric.shape[0]
    nbf = three_center.shape[0]
    m_sym = 0.5 * (metric + metric.T)
    try:
        l_chol = scipy.linalg.cholesky(m_sym, lower=True)
        # Check relative condition number via diagonal ratio before trusting Cholesky
        d_diag = np.diag(l_chol)
        if np.min(d_diag) ** 2 <= rel_tol * (np.max(d_diag) ** 2):
            raise scipy.linalg.LinAlgError("near-dependent metric")
        whitener = scipy.linalg.solve_triangular(l_chol, np.eye(naux), lower=True)
    except scipy.linalg.LinAlgError:
        evals, evecs = scipy.linalg.eigh(m_sym)
        keep = evals > rel_tol * max(float(evals[-1]), 1.0)
        whitener = (evecs[:, keep] / np.sqrt(evals[keep])).T
    whitener = np.ascontiguousarray(whitener, dtype=np.float64)
    # three_center is (nbf, nbf, naux) -> b_ao (nrank, nbf, nbf)
    b_flat = whitener @ three_center.reshape(nbf * nbf, naux).T
    b_ao = np.ascontiguousarray(b_flat.reshape(whitener.shape[0], nbf, nbf))
    return _DfFactor(metric=m_sym, whitener=whitener, b_ao=b_ao)


def _df_bar_b_to_am(df: _DfFactor, bar_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert dE/dB_ao (nrank, nbf, nbf) into raw (bar_metric, bar_three_center)."""
    nrank, nbf, _ = bar_b.shape
    naux = df.metric.shape[0]
    bar_b_2d = bar_b.reshape(nrank, nbf * nbf)
    b_2d = df.b_ao.reshape(nrank, nbf * nbf)
    bar_a = (bar_b_2d.T @ df.whitener).reshape(nbf, nbf, naux)
    g_qq = bar_b_2d @ b_2d.T
    g_qq = 0.5 * (g_qq + g_qq.T)
    bar_m = -0.5 * (df.whitener.T @ g_qq @ df.whitener)
    bar_m = 0.5 * (bar_m + bar_m.T)
    return bar_m, bar_a


def _df_finalize_am(df: _DfFactor, g_qq: np.ndarray, bar_a_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nbf = df.b_ao.shape[1]
    naux = df.metric.shape[0]
    g_sym = 0.5 * (g_qq + g_qq.T)
    bar_m = -0.5 * (df.whitener.T @ g_sym @ df.whitener)
    bar_m = 0.5 * (bar_m + bar_m.T)
    return bar_m, bar_a_2d.reshape(nbf, nbf, naux)


@dataclass(frozen=True)
class _HfState:
    energy: float
    unrestricted: bool
    nocc: tuple[int, int]
    coefficients: tuple[np.ndarray, np.ndarray]      # (C_a, C_b) each (n, n)
    orbital_energies: tuple[np.ndarray, np.ndarray]  # (eps_a, eps_b) each (n,)
    densities: tuple[np.ndarray, np.ndarray]         # (D_a, D_b) each (n, n) (1e per occ)
    focks: tuple[np.ndarray, np.ndarray]             # (F_a, F_b) each (n, n)
    iterations: int


def _df_jk_from_factors(
    jk: _DfFactor,
    c_occ_a: np.ndarray,
    c_occ_b: np.ndarray | None,
    unrestricted: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute J[D_tot], K[D_a], K[D_b] from orthonormal DF tensor B_Q using pure BLAS GEMM."""
    nrank, nbf, _ = jk.b_ao.shape
    d_a = c_occ_a @ c_occ_a.T
    if not unrestricted:
        d_tot = 2.0 * d_a
    else:
        assert c_occ_b is not None
        d_b = c_occ_b @ c_occ_b.T
        d_tot = d_a + d_b

    b_2d = jk.b_ao.reshape(nrank, nbf * nbf)
    v_q = b_2d @ d_tot.reshape(-1)
    j_mat = (v_q @ b_2d).reshape(nbf, nbf)

    def build_k(c_occ: np.ndarray) -> np.ndarray:
        no = c_occ.shape[1]
        if no == 0:
            return np.zeros((nbf, nbf), dtype=np.float64)
        k_out = np.zeros((nbf, nbf), dtype=np.float64)
        blk = 64
        for q0 in range(0, nrank, blk):
            q1 = min(q0 + blk, nrank)
            b_cnt = q1 - q0
            bq = jk.b_ao[q0:q1]
            # (b_cnt * nbf, nbf) @ (nbf, no) -> (b_cnt, nbf, no)
            x = (bq.reshape(b_cnt * nbf, nbf) @ c_occ).reshape(b_cnt, nbf, no)
            x_2d = np.ascontiguousarray(x.transpose(1, 0, 2)).reshape(nbf, b_cnt * no)
            k_out += x_2d @ x_2d.T
        return k_out

    k_a = build_k(c_occ_a)
    k_b = build_k(c_occ_b) if unrestricted and c_occ_b is not None else k_a
    return j_mat, k_a, k_b


def _df_jk_general_density(
    jk: _DfFactor,
    p_a: np.ndarray,
    p_b: np.ndarray,
    unrestricted: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute J[P_a + P_b], K[P_a], K[P_b] for arbitrary symmetric AO matrices P_a, P_b via BLAS."""
    nrank, nbf, _ = jk.b_ao.shape
    p_tot = 2.0 * p_a if not unrestricted else (p_a + p_b)
    b_2d = jk.b_ao.reshape(nrank, nbf * nbf)
    v_q = b_2d @ p_tot.reshape(-1)
    j_mat = (v_q @ b_2d).reshape(nbf, nbf)

    def build_k_sym(p_mat: np.ndarray) -> np.ndarray:
        k_out = np.zeros((nbf, nbf), dtype=np.float64)
        blk = 32
        for q0 in range(0, nrank, blk):
            q1 = min(q0 + blk, nrank)
            b_cnt = q1 - q0
            bq = jk.b_ao[q0:q1]
            bp = (bq.reshape(b_cnt * nbf, nbf) @ p_mat).reshape(b_cnt, nbf, nbf)
            bp_2d = np.ascontiguousarray(bp.transpose(1, 0, 2)).reshape(nbf, b_cnt * nbf)
            bq_2d = np.ascontiguousarray(bq.transpose(1, 0, 2)).reshape(nbf, b_cnt * nbf)
            k_out += bp_2d @ bq_2d.T
        return 0.5 * (k_out + k_out.T)

    k_a = build_k_sym(p_a)
    k_b = build_k_sym(p_b) if unrestricted else k_a
    return j_mat, k_a, k_b


def _build_sad_density(
    atoms: Sequence[Atom],
    basis: BasisSet,
    overlap: np.ndarray,
    hcore: np.ndarray,
    jk: _DfFactor,
    nocc: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build native Superposition of Atomic Densities (SAD) guess from molecular blocks."""
    nbf = overlap.shape[0]
    spherical = basis.representation == "spherical"
    slices: list[tuple[int, int, int, int]] = []  # (p0, p1, z, ecp_core)
    cur = 0
    for atom in atoms:
        eb = basis.by_element[atom.atomic_number]
        cnt = sum(
            ((2 * sh.angular_momentum + 1) if spherical else ((sh.angular_momentum + 1) * (sh.angular_momentum + 2) // 2))
            * len(sh.coefficients)
            for sh in eb.shells
        )
        slices.append((cur, cur + cnt, atom.atomic_number, eb.ecp_core_electrons))
        cur += cnt

    d_tot = np.zeros((nbf, nbf), dtype=np.float64)
    nrank = jk.b_ao.shape[0]
    atom_cache: dict[int, np.ndarray] = {}
    for p0, p1, z, ecp_core in slices:
        na_e = z - ecp_core
        nb_a = p1 - p0
        if na_e <= 0 or nb_a == 0:
            continue
        if z in atom_cache:
            d_tot[p0:p1, p0:p1] = atom_cache[z]
            continue
        # Isolated neutral atom 1e Hamiltonian (including ECP if present, excluding other nuclei!)
        s_a, h_a, _ = one_electron_integrals_cpu(
            [Atom(z, (0.0, 0.0, 0.0))], basis, x2c1e_mode="none", charge=0, multiplicity=1 + (na_e % 2)
        )
        b_a = jk.b_ao[:, p0:p1, p0:p1]  # (nrank, nb_a, nb_a)
        b_a_2d = b_a.reshape(nrank, nb_a * nb_a)
        es, vs = scipy.linalg.eigh(s_a)
        keep = es > 1e-12
        xa = vs[:, keep] / np.sqrt(es[keep])

        # Fractional occupation (neutral atom, spherical-average over near-degenerate valence)
        occ = np.zeros(xa.shape[1], dtype=np.float64)
        d_a = np.zeros((nb_a, nb_a), dtype=np.float64)
        for it in range(30):
            if it == 0:
                f_a = h_a
            else:
                vq = b_a_2d @ d_a.reshape(-1)
                j_a = (vq @ b_a_2d).reshape(nb_a, nb_a)
                w_occ = np.sqrt(np.maximum(occ, 0.0) * 0.5)
                c_w = c_a * w_occ[None, :]
                x_k = (b_a.reshape(nrank * nb_a, nb_a) @ c_w).reshape(nrank, nb_a, -1)
                xk_2d = np.ascontiguousarray(x_k.transpose(1, 0, 2)).reshape(nb_a, -1)
                k_a = xk_2d @ xk_2d.T
                f_a = h_a + j_a - k_a
            f_o = xa.T @ f_a @ xa
            f_o = 0.5 * (f_o + f_o.T)
            eps_a, co_a = scipy.linalg.eigh(f_o)
            c_a = xa @ co_a
            occ[:] = 0.0
            rem = float(na_e)
            idx = 0
            nmo = len(eps_a)
            while rem > 1e-12 and idx < nmo:
                j = idx + 1
                while j < nmo and abs(eps_a[j] - eps_a[idx]) < 1e-3:
                    j += 1
                deg = j - idx
                cap = 2.0 * deg
                if rem >= cap:
                    occ[idx:j] = 2.0
                    rem -= cap
                else:
                    occ[idx:j] = rem / deg
                    rem = 0.0
                idx = j
            d_new = (c_a * occ[None, :]) @ c_a.T
            if it > 0 and float(np.max(np.abs(d_new - d_a))) < 1e-5:
                d_a = d_new
                break
            d_a = 0.5 * d_new + 0.5 * d_a if it > 0 else d_new
        atom_cache[z] = d_a
        d_tot[p0:p1, p0:p1] = d_a

    # Split total SAD density into alpha/beta spin channels matching nocc
    na, nb = nocc
    ntot = na + nb
    if ntot == 0:
        return np.zeros_like(d_tot), np.zeros_like(d_tot)
    fa = float(na) / float(ntot)
    fb = float(nb) / float(ntot)
    return fa * d_tot, fb * d_tot


def _solve_df_hf(
    overlap: np.ndarray,
    hcore: np.ndarray,
    e_nuc: float,
    jk: _DfFactor,
    nocc: tuple[int, int],
    *,
    unrestricted: bool,
    initial_densities: tuple[np.ndarray, np.ndarray] | None = None,
    e_tol: float = 1e-13,
    g_tol: float = 1e-11,
    max_iter: int = 160,
) -> _HfState:
    """Solve canonical DF-RHF or DF-UHF on shared orthonormal JKFit tensor."""
    nbf = overlap.shape[0]
    evals_s, evecs_s = scipy.linalg.eigh(overlap)
    keep = evals_s > 1e-12
    x_orth = evecs_s[:, keep] / np.sqrt(evals_s[keep])
    na, nb = nocc

    def diag_fock(f_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f_orth = x_orth.T @ f_mat @ x_orth
        f_orth = 0.5 * (f_orth + f_orth.T)
        eps, c_orth = scipy.linalg.eigh(f_orth)
        return eps, x_orth @ c_orth

    def run_scf(
        seed: tuple[np.ndarray, np.ndarray] | None,
        *,
        level_shift: float = 0.0,
        damping: float = 1.0,
        use_diis: bool = True,
        diis_space: int = 8,
        etol: float = e_tol,
        gtol: float = g_tol,
        nmax: int = max_iter,
    ) -> tuple[_HfState, bool]:
        if seed is None:
            eps_a, c_a = diag_fock(hcore)
            eps_b, c_b = (eps_a, c_a) if not unrestricted else diag_fock(hcore)
            d_a = c_a[:, :na] @ c_a[:, :na].T
            d_b = d_a if not unrestricted else (c_b[:, :nb] @ c_b[:, :nb].T)
            if unrestricted and na == nb and na > 0:
                # Slight symmetry-breaking guess only if na==nb open-shell (rare)
                pass
        else:
            d_a, d_b = seed[0].copy(), seed[1].copy()
            c_a = np.zeros((nbf, nbf))
            c_b = np.zeros((nbf, nbf))
            eps_a = np.zeros(nbf)
            eps_b = np.zeros(nbf)

        f_hist: list[tuple[np.ndarray, np.ndarray]] = []
        e_hist: list[np.ndarray] = []
        e_prev = 0.0
        converged = False
        f_a = hcore.copy()
        f_b = hcore.copy()

        for it in range(1, nmax + 1):
            if (it == 1 and seed is not None) or damping < 1.0:
                j_mat, k_a, k_b = _df_jk_general_density(jk, d_a, d_b, unrestricted)
            else:
                j_mat, k_a, k_b = _df_jk_from_factors(
                    jk, c_a[:, :na], c_b[:, :nb] if unrestricted else None, unrestricted
                )
            f_a = hcore + j_mat - k_a
            f_b = f_a if not unrestricted else (hcore + j_mat - k_b)
            f_a = 0.5 * (f_a + f_a.T)
            f_b = 0.5 * (f_b + f_b.T)

            if not unrestricted:
                e_elec = float(np.sum(d_a * (hcore + f_a)))
            else:
                e_elec = 0.5 * float(np.sum(d_a * (hcore + f_a)) + np.sum(d_b * (hcore + f_b)))
            e_tot = e_nuc + e_elec

            # Pulay commutator error for DIIS and exact MO-basis orbital gradient for convergence
            err_a = x_orth.T @ (f_a @ d_a @ overlap - overlap @ d_a @ f_a) @ x_orth
            if unrestricted:
                err_b = x_orth.T @ (f_b @ d_b @ overlap - overlap @ d_b @ f_b) @ x_orth
                err_vec = np.concatenate([err_a.ravel(), err_b.ravel()])
            else:
                err_vec = err_a.ravel()
            g_rms = float(np.sqrt(np.mean(err_vec**2)))
            if it > 1 or seed is None:
                ga_mo = c_a[:, na:].T @ f_a @ c_a[:, :na] if na > 0 and na < c_a.shape[1] else np.zeros((1,))
                g_max = float(np.max(np.abs(ga_mo)))
                if unrestricted and nb > 0 and nb < c_b.shape[1]:
                    gb_mo = c_b[:, nb:].T @ f_b @ c_b[:, :nb]
                    g_max = max(g_max, float(np.max(np.abs(gb_mo))))
            else:
                g_max = float(np.max(np.abs(err_vec)))
            de = abs(e_tot - e_prev) if it > 1 else 1.0
            e_prev = e_tot

            if it > 1 and level_shift == 0.0 and de < etol and g_max < gtol:
                converged = True
                break

            # CDIIS extrapolation (skip pushing non-idempotent initial seed on it==1)
            f_eff_a, f_eff_b = f_a, f_b
            if use_diis and not (it == 1 and seed is not None):
                f_hist.append((f_a.copy(), f_b.copy()))
                e_hist.append(err_vec)
                if len(f_hist) > diis_space:
                    f_hist.pop(0)
                    e_hist.pop(0)
                m = len(e_hist)
                if m >= 2:
                    b_mat = np.empty((m + 1, m + 1), dtype=np.float64)
                    b_mat[-1, :] = -1.0
                    b_mat[:, -1] = -1.0
                    b_mat[-1, -1] = 0.0
                    for i in range(m):
                        for j in range(i, m):
                            val = float(np.dot(e_hist[i], e_hist[j]))
                            b_mat[i, j] = val
                            b_mat[j, i] = val
                    rhs = np.zeros(m + 1, dtype=np.float64)
                    rhs[-1] = -1.0
                    try:
                        coeffs = np.linalg.solve(b_mat, rhs)[:m]
                        f_eff_a = sum(w * fh[0] for w, fh in zip(coeffs, f_hist, strict=True))
                        f_eff_b = (
                            f_eff_a
                            if not unrestricted
                            else sum(w * fh[1] for w, fh in zip(coeffs, f_hist, strict=True))
                        )
                    except np.linalg.LinAlgError:
                        f_hist.clear()
                        e_hist.clear()

            if level_shift > 0.0:
                proj_va = overlap - overlap @ d_a @ overlap
                f_eff_a = f_eff_a + level_shift * proj_va
                if unrestricted:
                    proj_vb = overlap - overlap @ d_b @ overlap
                    f_eff_b = f_eff_b + level_shift * proj_vb

            eps_a, c_a = diag_fock(f_eff_a)
            eps_b, c_b = (eps_a, c_a) if not unrestricted else diag_fock(f_eff_b)
            d_a_new = c_a[:, :na] @ c_a[:, :na].T
            d_b_new = d_a_new if not unrestricted else (c_b[:, :nb] @ c_b[:, :nb].T)
            if damping < 1.0:
                d_a = damping * d_a_new + (1.0 - damping) * d_a
                d_b = d_a if not unrestricted else (damping * d_b_new + (1.0 - damping) * d_b)
            else:
                d_a, d_b = d_a_new, d_b_new

            if it > 1 and level_shift > 0.0 and de < etol and g_rms < gtol:
                converged = True
                break

        # Final strict canonical diagonalization & consistent energy on converged density
        j_mat, k_a, k_b = _df_jk_from_factors(
            jk, c_a[:, :na], c_b[:, :nb] if unrestricted else None, unrestricted
        )
        f_a = 0.5 * ((hcore + j_mat - k_a) + (hcore + j_mat - k_a).T)
        f_b = f_a if not unrestricted else 0.5 * ((hcore + j_mat - k_b) + (hcore + j_mat - k_b).T)
        eps_a, c_a = diag_fock(f_a)
        eps_b, c_b = (eps_a, c_a) if not unrestricted else diag_fock(f_b)
        d_a = c_a[:, :na] @ c_a[:, :na].T
        d_b = d_a if not unrestricted else (c_b[:, :nb] @ c_b[:, :nb].T)
        j_mat, k_a, k_b = _df_jk_from_factors(
            jk, c_a[:, :na], c_b[:, :nb] if unrestricted else None, unrestricted
        )
        f_a = 0.5 * ((hcore + j_mat - k_a) + (hcore + j_mat - k_a).T)
        f_b = f_a if not unrestricted else 0.5 * ((hcore + j_mat - k_b) + (hcore + j_mat - k_b).T)
        eps_a, c_a = diag_fock(f_a)
        eps_b, c_b = (eps_a, c_a) if not unrestricted else diag_fock(f_b)
        d_a = c_a[:, :na] @ c_a[:, :na].T
        d_b = d_a if not unrestricted else (c_b[:, :nb] @ c_b[:, :nb].T)
        j_mat, k_a, k_b = _df_jk_from_factors(
            jk, c_a[:, :na], c_b[:, :nb] if unrestricted else None, unrestricted
        )
        f_a = 0.5 * ((hcore + j_mat - k_a) + (hcore + j_mat - k_a).T)
        f_b = f_a if not unrestricted else 0.5 * ((hcore + j_mat - k_b) + (hcore + j_mat - k_b).T)
        if not unrestricted:
            e_tot = e_nuc + float(np.sum(d_a * (hcore + f_a)))
        else:
            e_tot = e_nuc + 0.5 * float(np.sum(d_a * (hcore + f_a)) + np.sum(d_b * (hcore + f_b)))
        return (
            _HfState(
                energy=e_tot,
                unrestricted=unrestricted,
                nocc=nocc,
                coefficients=(c_a, c_b),
                orbital_energies=(eps_a, eps_b),
                densities=(d_a, d_b),
                focks=(f_a, f_b),
                iterations=it,
            ),
            converged,
        )

    state, ok = run_scf(initial_densities)
    if not ok:
        rec, _ = run_scf(
            initial_densities,
            level_shift=0.5,
            damping=0.4,
            use_diis=True,
            etol=1e-7,
            gtol=1e-4,
            nmax=180,
        )
        state, ok = run_scf(rec.densities, level_shift=0.0, damping=1.0, use_diis=True, nmax=200)
        if not ok:
            state, ok = run_scf(
                state.densities, level_shift=0.0, damping=0.5, use_diis=True, nmax=240
            )
    if not ok:
        raise RuntimeError("Shared-integral native DF-HF did not converge")
    return state


@dataclass(frozen=True)
class _RelaxedWeights:
    overlap: np.ndarray           # (n, n)
    hcore: np.ndarray             # (n, n)
    jk_metric: np.ndarray         # (q_jk, q_jk)
    jk_three_center: np.ndarray   # (n, n, q_jk)
    mp2_metric: np.ndarray | None
    mp2_three_center: np.ndarray | None


def _df_hf_weights(
    hf: _HfState,
    jk: _DfFactor,
    *,
    accum_jk_a_2d: np.ndarray | None = None,
    accum_sign: float = 1.0,
) -> _RelaxedWeights:
    """Exact stationary 1e and JKFit derivative weights for DF-RHF or DF-UHF (streamed Q-blocks)."""
    nrank, nbf, _ = jk.b_ao.shape
    naux = jk.metric.shape[0]
    na, nb = hf.nocc
    d_a, d_b = hf.densities
    c_a, c_b = hf.coefficients
    ea, eb = hf.orbital_energies
    d_tot = (2.0 * d_a) if not hf.unrestricted else (d_a + d_b)

    w_a = (c_a[:, :na] * ea[:na]) @ c_a[:, :na].T if na > 0 else np.zeros((nbf, nbf))
    if not hf.unrestricted:
        w_s = -2.0 * w_a
    else:
        w_b = (c_b[:, :nb] * eb[:nb]) @ c_b[:, :nb].T if nb > 0 else np.zeros((nbf, nbf))
        w_s = -(w_a + w_b)

    b_2d = jk.b_ao.reshape(nrank, nbf * nbf)
    v_q = b_2d @ d_tot.reshape(-1)
    d_tot_flat = d_tot.reshape(-1)
    c_occ_a = c_a[:, :na]
    c_occ_b = c_b[:, :nb]

    own_a = accum_jk_a_2d is None
    bar_a_2d = np.zeros((nbf * nbf, naux), dtype=np.float64) if own_a else accum_jk_a_2d
    g_qq = np.zeros((nrank, nrank), dtype=np.float64)
    blk = 32
    for q0 in range(0, nrank, blk):
        q1 = min(q0 + blk, nrank)
        b_cnt = q1 - q0
        bq = jk.b_ao[q0:q1]
        bar_blk = np.outer(v_q[q0:q1], d_tot_flat).reshape(b_cnt, nbf, nbf)
        if na > 0:
            xa = (bq.reshape(b_cnt * nbf, nbf) @ c_occ_a).reshape(b_cnt, nbf, na)
            bija = np.matmul(c_occ_a.T, xa)  # (b_cnt, na, na)
            ex_a = np.matmul(np.matmul(c_occ_a, bija), c_occ_a.T)
            bar_blk -= (2.0 if not hf.unrestricted else 1.0) * ex_a
        if hf.unrestricted and nb > 0:
            xb = (bq.reshape(b_cnt * nbf, nbf) @ c_occ_b).reshape(b_cnt, nbf, nb)
            bijb = np.matmul(c_occ_b.T, xb)
            bar_blk -= np.matmul(np.matmul(c_occ_b, bijb), c_occ_b.T)
        bar_blk_2d = bar_blk.reshape(b_cnt, nbf * nbf)
        if accum_sign == 1.0:
            bar_a_2d += bar_blk_2d.T @ jk.whitener[q0:q1]
        else:
            bar_a_2d += accum_sign * (bar_blk_2d.T @ jk.whitener[q0:q1])
        g_qq[q0:q1] = bar_blk_2d @ b_2d.T

    g_sym = 0.5 * (g_qq + g_qq.T)
    jk_m = -0.5 * (jk.whitener.T @ g_sym @ jk.whitener)
    jk_m = 0.5 * (jk_m + jk_m.T)
    jk_a = bar_a_2d.reshape(nbf, nbf, naux) if own_a else np.empty((0, 0, 0))
    return _RelaxedWeights(
        overlap=0.5 * (w_s + w_s.T),
        hcore=0.5 * (d_tot + d_tot.T),
        jk_metric=jk_m,
        jk_three_center=jk_a,
        mp2_metric=None,
        mp2_three_center=None,
    )


def _df_mp2_energy_and_weights(
    hf: _HfState,
    jk: _DfFactor,
    mp2: _DfFactor,
    frozen: tuple[int, int],
    *,
    need_weights: bool,
    accum_jk_a_2d: np.ndarray | None = None,
    accum_mp2_a_2d: np.ndarray | None = None,
    accum_sign: float = 1.0,
) -> tuple[float, _RelaxedWeights | None]:
    """Fast batched OpenBLAS DF-RMP2 / DF-UMP2 correlation energy and exact Z-vector CPHF weights."""
    nbf = hf.coefficients[0].shape[0]
    na, nb = hf.nocc
    fa, fb = frozen
    n_act_a = na - fa
    n_act_b = nb - fb
    n_vir_a = nbf - na
    n_vir_b = nbf - nb
    nrank_mp2 = mp2.b_ao.shape[0]

    # One-electron / zero-pair case (e.g. H atom): identically zero correlation!
    has_pairs = (
        (n_act_a >= 2 and n_vir_a >= 1)
        or (n_act_b >= 2 and n_vir_b >= 1)
        or (n_act_a >= 1 and n_act_b >= 1 and n_vir_a >= 1 and n_vir_b >= 1)
    )
    if not has_pairs:
        if not need_weights:
            return 0.0, None
        zeros_n = np.zeros((nbf, nbf), dtype=np.float64)
        return 0.0, _RelaxedWeights(
            overlap=zeros_n,
            hcore=zeros_n,
            jk_metric=np.zeros_like(jk.metric),
            jk_three_center=np.zeros((nbf, nbf, jk.metric.shape[0]), dtype=np.float64),
            mp2_metric=np.zeros_like(mp2.metric),
            mp2_three_center=np.zeros((nbf, nbf, mp2.metric.shape[0]), dtype=np.float64),
        )

    c_a, c_b = hf.coefficients
    ea, eb = hf.orbital_energies

    def build_b_ov(c_mat: np.ndarray, f_cnt: int, o_cnt: int) -> np.ndarray:
        # Returns (n_act, n_vir, nrank_mp2) contiguous via streamed BLAS GEMM.
        n_act_loc = o_cnt - f_cnt
        n_vir_loc = nbf - o_cnt
        c_act = c_mat[:, f_cnt:o_cnt]
        c_vir = c_mat[:, o_cnt:]
        out_ov = np.empty((n_act_loc, n_vir_loc, nrank_mp2), dtype=np.float64)
        blk = 64
        for q0 in range(0, nrank_mp2, blk):
            q1 = min(q0 + blk, nrank_mp2)
            b_cnt = q1 - q0
            bq = mp2.b_ao[q0:q1]
            # bq symmetric: (b_cnt*nbf, nbf) @ c_act -> (b_cnt, nbf, n_act)
            t1 = (bq.reshape(b_cnt * nbf, nbf) @ c_act).reshape(b_cnt, nbf, n_act_loc)
            # (b_cnt, n_act, nbf) @ c_vir -> (b_cnt, n_act, n_vir)
            t2 = np.matmul(t1.transpose(0, 2, 1), c_vir)
            out_ov[:, :, q0:q1] = t2.transpose(1, 2, 0)
        return out_ov

    if not hf.unrestricted:
        # Fast closed-shell RHF specialization
        b_ov = build_b_ov(c_a, fa, na)  # (n_act, nv, Q)
        eo = ea[fa:na]
        ev = ea[na:]
        nv = n_vir_a
        n_act = n_act_a
        e_corr = 0.0
        p_oo_act = np.zeros((n_act, n_act), dtype=np.float64) if need_weights else None
        p_vv = np.zeros((nv, nv), dtype=np.float64) if need_weights else None
        # g_b_ia[i, a, Q] = dE_corr / dB_mp2[Q, i, a] (spin-summed = 4 * sum_{j,b} u_{ij}^{ab} B_{jb}^Q)
        g_b = np.zeros((n_act, nv, nrank_mp2), dtype=np.float64) if need_weights else None

        # Precompute ev[:, None] + ev[None, :]
        ev_sum = ev[:, None] + ev[None, :]
        for i in range(n_act):
            bi = b_ov[i]  # (nv, Q)
            for j in range(i, n_act):
                bj = b_ov[j]  # (nv, Q)
                g_ij = bi @ bj.T  # (nv, nv)
                denom = (eo[i] + eo[j]) - ev_sum
                t_ij = g_ij / denom
                u_ij = 2.0 * t_ij - t_ij.T
                e_ij = float(np.sum(u_ij * g_ij))
                e_corr += e_ij if i == j else (2.0 * e_ij)
                if need_weights:
                    assert p_vv is not None and g_b is not None and p_oo_act is not None
                    if i == j:
                        p_vv += 2.0 * (u_ij @ t_ij.T)
                        g_b[i] += 4.0 * (u_ij @ bj)
                    else:
                        # Contribute both (i, j) and (j, i)
                        p_vv += 2.0 * (u_ij @ t_ij.T + u_ij.T @ t_ij)
                        g_b[i] += 4.0 * (u_ij @ bj)
                        g_b[j] += 4.0 * (u_ij.T @ bi)

        if not need_weights:
            return e_corr, None

        assert p_oo_act is not None and p_vv is not None and g_b is not None
        # Compute P_oo_act[i, j] = -2 * sum_{k, a, b} u_{ik}^{ab} t_{jk}^{ab} cleanly
        # Batched over k: form t_k[i, a, b] via GEMM
        for k in range(n_act):
            bk = b_ov[k]  # (nv, Q)
            # g_ik[i, a, b] for all i: reshape b_ov (n_act*nv, Q) @ bk.T -> (n_act, nv, nv)
            g_k = (b_ov.reshape(n_act * nv, nrank_mp2) @ bk.T).reshape(n_act, nv, nv)
            denom_k = (eo[:, None, None] + eo[k]) - ev_sum[None, :, :]
            t_k = g_k / denom_k
            u_k = 2.0 * t_k - t_k.transpose(0, 2, 1)
            u_flat = u_k.reshape(n_act, nv * nv)
            t_flat = t_k.reshape(n_act, nv * nv)
            p_oo_act -= 2.0 * (u_flat @ t_flat.T)

        # Split spin-summed RHF quantities into per-spin (alpha == beta == 0.5 * total)
        p_oo_s = (0.5 * p_oo_act, 0.5 * p_oo_act)
        p_vv_s = (0.5 * p_vv, 0.5 * p_vv)
        g_b_s = (0.5 * g_b, 0.5 * g_b)
    else:
        # General UHF: aa, ab, bb channels
        b_a = build_b_ov(c_a, fa, na) if (n_act_a > 0 and n_vir_a > 0) else np.zeros((n_act_a, n_vir_a, nrank_mp2))
        b_b = build_b_ov(c_b, fb, nb) if (n_act_b > 0 and n_vir_b > 0) else np.zeros((n_act_b, n_vir_b, nrank_mp2))
        eo_a, ev_a = ea[fa:na], ea[na:]
        eo_b, ev_b = eb[fb:nb], eb[nb:]
        e_corr = 0.0
        p_oo_a = np.zeros((n_act_a, n_act_a), dtype=np.float64)
        p_oo_b = np.zeros((n_act_b, n_act_b), dtype=np.float64)
        p_vv_a = np.zeros((n_vir_a, n_vir_a), dtype=np.float64)
        p_vv_b = np.zeros((n_vir_b, n_vir_b), dtype=np.float64)
        g_b_a = np.zeros((n_act_a, n_vir_a, nrank_mp2), dtype=np.float64)
        g_b_b = np.zeros((n_act_b, n_vir_b, nrank_mp2), dtype=np.float64)

        def accum_same_spin(
            b_s: np.ndarray, eo: np.ndarray, ev: np.ndarray, p_oo: np.ndarray, p_vv: np.ndarray, gb: np.ndarray
        ) -> float:
            n_act, nv, _ = b_s.shape
            if n_act < 2 or nv == 0:
                return 0.0
            ev_sum = ev[:, None] + ev[None, :]
            e_ss = 0.0
            for i in range(n_act):
                bi = b_s[i]
                for j in range(i + 1, n_act):
                    bj = b_s[j]
                    g_ij = bi @ bj.T
                    denom = (eo[i] + eo[j]) - ev_sum
                    t_ij = g_ij / denom
                    u_ij = t_ij - t_ij.T
                    e_ss += float(np.sum(u_ij * g_ij))  # 0.5 * 2 (for i<j)
                    if need_weights:
                        p_vv += u_ij @ t_ij.T + u_ij.T @ t_ij
                        gb[i] += 2.0 * (u_ij @ bj)
                        gb[j] += 2.0 * (u_ij.T @ bi)
            if need_weights:
                for k in range(n_act):
                    bk = b_s[k]
                    g_k = (b_s.reshape(n_act * nv, nrank_mp2) @ bk.T).reshape(n_act, nv, nv)
                    denom_k = (eo[:, None, None] + eo[k]) - ev_sum[None, :, :]
                    t_k = g_k / denom_k
                    u_k = t_k - t_k.transpose(0, 2, 1)
                    p_oo -= u_k.reshape(n_act, nv * nv) @ t_k.reshape(n_act, nv * nv).T
            return e_ss

        e_corr += accum_same_spin(b_a, eo_a, ev_a, p_oo_a, p_vv_a, g_b_a)
        e_corr += accum_same_spin(b_b, eo_b, ev_b, p_oo_b, p_vv_b, g_b_b)

        # Opposite-spin (alpha-beta)
        if n_act_a > 0 and n_act_b > 0 and n_vir_a > 0 and n_vir_b > 0:
            ev_ab = ev_a[:, None] + ev_b[None, :]
            for i in range(n_act_a):
                bi = b_a[i]
                for j in range(n_act_b):
                    bj = b_b[j]
                    g_ij = bi @ bj.T
                    t_ij = g_ij / ((eo_a[i] + eo_b[j]) - ev_ab)
                    e_corr += float(np.sum(t_ij * g_ij))
                    if need_weights:
                        p_vv_a += t_ij @ t_ij.T
                        p_vv_b += t_ij.T @ t_ij
                        g_b_a[i] += 2.0 * (t_ij @ bj)
                        g_b_b[j] += 2.0 * (t_ij.T @ bi)
            if need_weights:
                for j in range(n_act_b):
                    bj = b_b[j]
                    g_j = (b_a.reshape(n_act_a * n_vir_a, nrank_mp2) @ bj.T).reshape(
                        n_act_a, n_vir_a, n_vir_b
                    )
                    t_j = g_j / ((eo_a[:, None, None] + eo_b[j]) - ev_ab[None, :, :])
                    tf = t_j.reshape(n_act_a, n_vir_a * n_vir_b)
                    p_oo_a -= tf @ tf.T
                for i in range(n_act_a):
                    bi = b_a[i]
                    g_i = (b_b.reshape(n_act_b * n_vir_b, nrank_mp2) @ bi.T).reshape(
                        n_act_b, n_vir_b, n_vir_a
                    )
                    t_i = g_i / ((eo_b[:, None, None] + eo_a[i]) - ev_ab.T[None, :, :])
                    tf = t_i.reshape(n_act_b, n_vir_b * n_vir_a)
                    p_oo_b -= tf @ tf.T

        if not need_weights:
            return e_corr, None
        p_oo_s = (p_oo_a, p_oo_b)
        p_vv_s = (p_vv_a, p_vv_b)
        g_b_s = (g_b_a, g_b_b)

    # ---- Exact Z-vector CPHF Relaxation & Streamed BLAS Integral Weights ----
    naux_mp2 = mp2.metric.shape[0]
    mp2_b_2d = mp2.b_ao.reshape(nrank_mp2, nbf * nbf)
    own_mp2_a = accum_mp2_a_2d is None
    bar_a_mp2_2d = (
        np.zeros((nbf * nbf, naux_mp2), dtype=np.float64) if own_mp2_a else accum_mp2_a_2d
    )
    g_qq_mp2 = np.zeros((nrank_mp2, nrank_mp2), dtype=np.float64)
    l_mats: list[np.ndarray] = []
    spins_to_process = (0,) if not hf.unrestricted else (0, 1)

    for s in spins_to_process:
        c_s = hf.coefficients[s]
        o_s = hf.nocc[s]
        f_s = frozen[s]
        n_act = o_s - f_s
        nv = nbf - o_s
        gb = g_b_s[s]  # (n_act, nv, Q)
        l_s = np.zeros((nbf, nbf), dtype=np.float64)
        if n_act > 0 and nv > 0:
            c_act = c_s[:, f_s:o_s]
            c_vir = c_s[:, o_s:]
            spin_scale = 2.0 if not hf.unrestricted else 1.0
            blk = 32
            for q0 in range(0, nrank_mp2, blk):
                q1 = min(q0 + blk, nrank_mp2)
                b_cnt = q1 - q0
                bq = mp2.b_ao[q0:q1]  # (b, n, n)
                gb_q = np.ascontiguousarray(gb[:, :, q0:q1].transpose(2, 0, 1))  # (b, n_act, nv)
                # Streamed bar_B_mp2 -> (bar_a_mp2_2d, g_qq_mp2)
                tmp_ao = np.matmul(np.matmul(c_act, gb_q), c_vir.T)
                sym_ao = (0.5 * spin_scale) * (tmp_ao + tmp_ao.transpose(0, 2, 1))
                sym_2d = sym_ao.reshape(b_cnt, nbf * nbf)
                if accum_sign == 1.0:
                    bar_a_mp2_2d += sym_2d.T @ mp2.whitener[q0:q1]
                else:
                    bar_a_mp2_2d += accum_sign * (sym_2d.T @ mp2.whitener[q0:q1])
                g_qq_mp2[q0:q1] += sym_2d @ mp2_b_2d.T
                # L_s[:, f_s:o_s] via pure BLAS GEMM
                bv = (bq.reshape(b_cnt * nbf, nbf) @ c_vir).reshape(b_cnt, nbf, nv)
                bv_2d = np.ascontiguousarray(bv.transpose(1, 0, 2)).reshape(nbf, b_cnt * nv)
                gb_i_2d = np.ascontiguousarray(gb_q.transpose(1, 0, 2)).reshape(n_act, b_cnt * nv)
                l_s[:, f_s:o_s] += c_s.T @ (bv_2d @ gb_i_2d.T)
                # L_s[:, o_s:] via pure BLAS GEMM
                bo = (bq.reshape(b_cnt * nbf, nbf) @ c_act).reshape(b_cnt, nbf, n_act)
                bo_2d = np.ascontiguousarray(bo.transpose(1, 0, 2)).reshape(nbf, b_cnt * n_act)
                gb_a_2d = np.ascontiguousarray(gb_q.transpose(2, 0, 1)).reshape(nv, b_cnt * n_act)
                l_s[:, o_s:] += c_s.T @ (bo_2d @ gb_a_2d.T)
        l_mats.append(l_s)
    if not hf.unrestricted:
        l_mats.append(l_mats[0])
    g_sym_mp2 = 0.5 * (g_qq_mp2 + g_qq_mp2.T)
    mp2_m = -0.5 * (mp2.whitener.T @ g_sym_mp2 @ mp2.whitener)
    mp2_m = 0.5 * (mp2_m + mp2_m.T)
    mp2_a = bar_a_mp2_2d.reshape(nbf, nbf, naux_mp2) if own_mp2_a else None
    del g_qq_mp2, g_sym_mp2, g_b_s

    # Assemble non-vo P^{(0)}_s (active oo, vv, and frozen-active oo blocks)
    p0_mo: list[np.ndarray] = []
    for s in (0, 1):
        o_s = hf.nocc[s]
        f_s = frozen[s]
        e_s = hf.orbital_energies[s]
        p_s = np.zeros((nbf, nbf), dtype=np.float64)
        if o_s > f_s:
            p_s[f_s:o_s, f_s:o_s] = 0.5 * (p_oo_s[s] + p_oo_s[s].T)
        if nbf > o_s:
            p_s[o_s:, o_s:] = 0.5 * (p_vv_s[s] + p_vv_s[s].T)
        if f_s > 0 and o_s > f_s:
            # Frozen-core / active-occ canonical frame relaxation
            diff_l = l_mats[s][:f_s, f_s:o_s] - l_mats[s][f_s:o_s, :f_s].T
            denom_fi = 2.0 * (e_s[f_s:o_s][None, :] - e_s[:f_s][:, None])
            p_fi = diff_l / denom_fi
            p_s[:f_s, f_s:o_s] = p_fi
            p_s[f_s:o_s, :f_s] = p_fi.T
        p0_mo.append(p_s)

    p0_ao_a = c_a @ p0_mo[0] @ c_a.T
    p0_ao_b = p0_ao_a if not hf.unrestricted else (c_b @ p0_mo[1] @ c_b.T)
    j0, k0_a, k0_b = _df_jk_general_density(jk, p0_ao_a, p0_ao_b, hf.unrestricted)

    # CPHF RHS for P_vo:
    # (eps_a - eps_i) P_{s, ai} + (C_s^T (J[P_vo_tot] - K[P_vo_s]) C_s)_{ai}
    #   = -0.5 * (L_{s, ai} - L_{s, ia}) - (C_s^T (J[P0_tot] - K[P0_s]) C_s)_{ai}
    def pack_vo(z_a: np.ndarray, z_b: np.ndarray | None) -> np.ndarray:
        if not hf.unrestricted:
            return z_a.ravel()
        assert z_b is not None
        return np.concatenate([z_a.ravel(), z_b.ravel()])

    def unpack_vo(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sz_a = n_vir_a * na
        za = vec[:sz_a].reshape(n_vir_a, na)
        if not hf.unrestricted:
            return za, za
        zb = vec[sz_a:].reshape(n_vir_b, nb)
        return za, zb

    g0_a_vo = c_a[:, na:].T @ (j0 - k0_a) @ c_a[:, :na]
    rhs_a = -0.5 * (l_mats[0][na:, :na] - l_mats[0][:na, na:].T) - g0_a_vo
    diag_a = ea[na:, None] - ea[None, :na]
    if hf.unrestricted:
        g0_b_vo = c_b[:, nb:].T @ (j0 - k0_b) @ c_b[:, :nb]
        rhs_b = -0.5 * (l_mats[1][nb:, :nb] - l_mats[1][:nb, nb:].T) - g0_b_vo
        diag_b = eb[nb:, None] - eb[None, :nb]
        rhs_vec = pack_vo(rhs_a, rhs_b)
        diag_vec = pack_vo(diag_a, diag_b)
    else:
        rhs_vec = pack_vo(rhs_a, None)
        diag_vec = pack_vo(diag_a, None)

    # Precompute B_Q @ C_occ ONCE before CG iterations for O(1) BLAS exchange response.
    nrank_jk = jk.b_ao.shape[0]
    b_jk_2d = jk.b_ao.reshape(nrank_jk, nbf * nbf)
    bo_cache_a = (
        (jk.b_ao.reshape(nrank_jk * nbf, nbf) @ c_a[:, :na]).reshape(nrank_jk, nbf, na)
        if na > 0
        else None
    )
    bo_cache_b = (
        (jk.b_ao.reshape(nrank_jk * nbf, nbf) @ c_b[:, :nb]).reshape(nrank_jk, nbf, nb)
        if (hf.unrestricted and nb > 0)
        else bo_cache_a
    )

    def apply_hessian(vec: np.ndarray) -> np.ndarray:
        za, zb = unpack_vo(vec)
        pa = c_a[:, na:] @ za @ c_a[:, :na].T
        pa = pa + pa.T
        if hf.unrestricted:
            pb = c_b[:, nb:] @ zb @ c_b[:, :nb].T
            pb = pb + pb.T
        else:
            pb = pa
        p_tot = (2.0 * pa) if not hf.unrestricted else (pa + pb)
        v_q = b_jk_2d @ p_tot.reshape(-1)
        j_mat = (v_q @ b_jk_2d).reshape(nbf, nbf)

        def k_vo_ao(c_s: np.ndarray, o_s: int, z_s: np.ndarray, bo_all: np.ndarray | None) -> np.ndarray:
            if o_s == 0 or nbf == o_s or bo_all is None:
                return np.zeros((nbf, nbf), dtype=np.float64)
            u_mat = c_s[:, o_s:] @ z_s  # (nbf, o_s)
            k_out = np.zeros((nbf, nbf), dtype=np.float64)
            blk = 64
            for q0 in range(0, nrank_jk, blk):
                q1 = min(q0 + blk, nrank_jk)
                b_cnt = q1 - q0
                bq = jk.b_ao[q0:q1]
                bu = (bq.reshape(b_cnt * nbf, nbf) @ u_mat).reshape(b_cnt, nbf, o_s)
                bu_2d = np.ascontiguousarray(bu.transpose(1, 0, 2)).reshape(nbf, b_cnt * o_s)
                bo_2d = np.ascontiguousarray(bo_all[q0:q1].transpose(1, 0, 2)).reshape(nbf, b_cnt * o_s)
                k_out += bu_2d @ bo_2d.T
            return k_out + k_out.T

        ka = k_vo_ao(c_a, na, za, bo_cache_a)
        out_a = diag_a * za + c_a[:, na:].T @ (j_mat - ka) @ c_a[:, :na]
        if not hf.unrestricted:
            return pack_vo(out_a, None)
        kb = k_vo_ao(c_b, nb, zb, bo_cache_b)
        out_b = diag_b * zb + c_b[:, nb:].T @ (j_mat - kb) @ c_b[:, :nb]
        return pack_vo(out_a, out_b)

    # Preconditioned Conjugate Gradient for CPHF Z-vector
    if rhs_vec.size == 0 or float(np.linalg.norm(rhs_vec)) == 0.0:
        z_vec = np.zeros_like(rhs_vec)
    else:
        z_vec = rhs_vec / diag_vec
        r_vec = rhs_vec - apply_hessian(z_vec)
        p_vec = r_vec / diag_vec
        rz = float(np.dot(r_vec, p_vec))
        tol = max(1e-13 * float(np.linalg.norm(rhs_vec)), 1e-15)
        for _ in range(120):
            if float(np.linalg.norm(r_vec)) < tol:
                break
            ap = apply_hessian(p_vec)
            denom = float(np.dot(p_vec, ap))
            if abs(denom) < 1e-30:
                break
            alpha = rz / denom
            z_vec = z_vec + alpha * p_vec
            r_vec = r_vec - alpha * ap
            z_new = r_vec / diag_vec
            rz_new = float(np.dot(r_vec, z_new))
            p_vec = z_new + (rz_new / rz) * p_vec
            rz = rz_new
        if float(np.linalg.norm(r_vec)) > 1e-8 * max(1.0, float(np.linalg.norm(rhs_vec))):
            raise RuntimeError(f"DF-MP2 CPHF did not converge: residual {np.linalg.norm(r_vec):.3e}")

    z_a, z_b = unpack_vo(z_vec)
    p_mo = [p0_mo[0].copy(), p0_mo[1].copy()]
    p_mo[0][na:, :na] = z_a
    p_mo[0][:na, na:] = z_a.T
    if hf.unrestricted:
        p_mo[1][nb:, :nb] = z_b
        p_mo[1][:nb, nb:] = z_b.T
    else:
        p_mo[1] = p_mo[0]

    p_ao_a = c_a @ p_mo[0] @ c_a.T
    p_ao_b = p_ao_a if not hf.unrestricted else (c_b @ p_mo[1] @ c_b.T)
    p_ao_tot = (2.0 * p_ao_a) if not hf.unrestricted else (p_ao_a + p_ao_b)
    j_p, k_pa, k_pb = _df_jk_general_density(jk, p_ao_a, p_ao_b, hf.unrestricted)

    # Energy-weighted density W_s = -0.5 * (sym(L_s) + P_s eps_s + eps_s P_s + I_o G_s + G_s I_o)
    def build_w_ao(s_idx: int, k_ps: np.ndarray) -> np.ndarray:
        c_s = hf.coefficients[s_idx]
        e_s = hf.orbital_energies[s_idx]
        o_s = hf.nocc[s_idx]
        l_sym = 0.5 * (l_mats[s_idx] + l_mats[s_idx].T)
        g_s = c_s.T @ (j_p - k_ps) @ c_s
        io_g = np.zeros((nbf, nbf), dtype=np.float64)
        io_g[:o_s, :] += g_s[:o_s, :]
        io_g[:, :o_s] += g_s[:, :o_s]
        w_mo = -0.5 * (l_sym + p_mo[s_idx] * (e_s[:, None] + e_s[None, :]) + io_g)
        return c_s @ w_mo @ c_s.T

    w_ao_a = build_w_ao(0, k_pa)
    w_s_mp2 = (2.0 * w_ao_a) if not hf.unrestricted else (w_ao_a + build_w_ao(1, k_pb))

    # Streamed JKFit 2e response weights from Tr(P_a F_a + P_b F_b) (zero giant bar_b allocation).
    d_a, d_b = hf.densities
    d_tot = (2.0 * d_a) if not hf.unrestricted else (d_a + d_b)
    naux_jk = jk.metric.shape[0]
    vp_q = b_jk_2d @ p_ao_tot.reshape(-1)
    vd_q = b_jk_2d @ d_tot.reshape(-1)
    d_tot_flat = d_tot.reshape(-1)
    p_tot_flat = p_ao_tot.reshape(-1)

    own_jk_a = accum_jk_a_2d is None
    bar_a_jk_2d = (
        np.zeros((nbf * nbf, naux_jk), dtype=np.float64) if own_jk_a else accum_jk_a_2d
    )
    g_qq_jk = np.zeros((nrank_jk, nrank_jk), dtype=np.float64)
    c_occ_a = c_a[:, :na]
    c_occ_b = c_b[:, :nb]
    blk = 32
    for q0 in range(0, nrank_jk, blk):
        q1 = min(q0 + blk, nrank_jk)
        b_cnt = q1 - q0
        bar_blk = (
            np.outer(vp_q[q0:q1], d_tot_flat) + np.outer(vd_q[q0:q1], p_tot_flat)
        ).reshape(b_cnt, nbf, nbf)
        if na > 0 and bo_cache_a is not None:
            pbc_a = np.matmul(p_ao_a, bo_cache_a[q0:q1])  # (b_cnt, n, na)
            term_a = np.matmul(pbc_a, c_occ_a.T)          # (b_cnt, n, n)
            bar_blk -= (2.0 if not hf.unrestricted else 1.0) * (
                term_a + term_a.transpose(0, 2, 1)
            )
        if hf.unrestricted and nb > 0 and bo_cache_b is not None:
            pbc_b = np.matmul(p_ao_b, bo_cache_b[q0:q1])
            term_b = np.matmul(pbc_b, c_occ_b.T)
            bar_blk -= term_b + term_b.transpose(0, 2, 1)
        bar_blk_2d = bar_blk.reshape(b_cnt, nbf * nbf)
        if accum_sign == 1.0:
            bar_a_jk_2d += bar_blk_2d.T @ jk.whitener[q0:q1]
        else:
            bar_a_jk_2d += accum_sign * (bar_blk_2d.T @ jk.whitener[q0:q1])
        g_qq_jk[q0:q1] = bar_blk_2d @ b_jk_2d.T

    g_sym_jk = 0.5 * (g_qq_jk + g_qq_jk.T)
    jk_m = -0.5 * (jk.whitener.T @ g_sym_jk @ jk.whitener)
    jk_m = 0.5 * (jk_m + jk_m.T)
    jk_a = bar_a_jk_2d.reshape(nbf, nbf, naux_jk) if own_jk_a else np.empty((0, 0, 0))
    del g_qq_jk, g_sym_jk, bo_cache_a, bo_cache_b
    return e_corr, _RelaxedWeights(
        overlap=0.5 * (w_s_mp2 + w_s_mp2.T),
        hcore=0.5 * (p_ao_tot + p_ao_tot.T),
        jk_metric=jk_m,
        jk_three_center=jk_a,
        mp2_metric=mp2_m,
        mp2_three_center=mp2_a,
    )


def _partition_df_hf_core(
    atoms: Sequence[Atom],
    basis: BasisSet,
    overlap: np.ndarray,
    reference: _HfState,
    correlation_space: str,
) -> tuple[_HfState, FrozenOccupiedFrame]:
    nspin = 2 if reference.unrestricted else 1
    frame = select_frozen_occupied_frame(
        atoms, basis, overlap, reference.coefficients[:nspin], reference.orbital_energies[:nspin],
        reference.nocc[:nspin], correlation_space,
    )
    if all(order == tuple(range(occ)) for order, occ in
           zip(frame.occupied_order, reference.nocc[:nspin], strict=True)):
        return reference, frame
    coeff = []
    energies = []
    for k, order in enumerate(frame.occupied_order):
        full = (*order, *range(len(order), reference.coefficients[k].shape[1]))
        coeff.append(reference.coefficients[k][:, full])
        energies.append(reference.orbital_energies[k][list(full)])
    return replace(
        reference, coefficients=(coeff[0], coeff[-1]),
        orbital_energies=(energies[0], energies[-1]),
    ), frame


@dataclass(frozen=True)
class DeltaSrResult:
    """Complete pure additive scalar-relativistic Delta-SR (DF-HF & DF-MP2) E & optional F result."""

    mode: X2c1eModeName
    correlation_space: str
    nbf: int
    nocc: tuple[int, int]
    e_hf_nr: float
    e_hf_sr: float
    delta_sr_hf: float
    e_mp2_corr_nr: float
    e_mp2_corr_sr: float
    e_mp2_nr: float
    e_mp2_sr: float
    delta_sr_mp2: float
    forces_hf_nr: np.ndarray | None
    forces_hf_sr: np.ndarray | None
    delta_forces_hf: np.ndarray | None
    forces_mp2_nr: np.ndarray | None
    forces_mp2_sr: np.ndarray | None
    delta_forces_mp2: np.ndarray | None
    timings_s: dict[str, float]
    frozen_core_definition_version: int | None = None
    frozen_nr: FrozenOccupiedFrame | None = None
    frozen_sr: FrozenOccupiedFrame | None = None


def evaluate_delta_sr(
    atoms: Sequence[Atom],
    orbital_basis: BasisSet,
    jk_fit_basis: BasisSet,
    mp2_fit_basis: BasisSet,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    root_id: Literal["rhf", "uhf"] | None = None,
    mode: X2c1eModeName = "dlu1e",
    speed_of_light: float = 0.0,
    correlation_space: str = "all-active",
    forces: bool = True,
    return_component_forces: bool = False,
    include_hf_forces: bool = True,
) -> DeltaSrResult:
    """Evaluate pure additive Delta-SR at both DF-HF and DF-MP2 levels sharing 2e DF integrals.

    All 1e (S, T, V, W, ECP, X2C-1e), 2e DF ((P|Q), (mu nu|P)), and nuclear derivatives
    are computed strictly by native GenerativeQC C++ kernels (zero PySCF calls).
    Supports 'all-active', 'frozen-core' (FC4), 'frozen-core-spd' (FC9), 'cv-active',
    and explicit 'frozen:<n>' correlation spaces consistent with `_f12_frozen_core`.
    """
    t0 = time.perf_counter()
    unrestricted = (multiplicity != 1) if root_id is None else (root_id == "uhf")
    if not unrestricted and multiplicity != 1:
        raise ValueError("RHF requires multiplicity == 1")
    nelec = sum(
        a.atomic_number - orbital_basis.by_element[a.atomic_number].ecp_core_electrons
        for a in atoms
    ) - charge
    nalpha = (nelec + (multiplicity - 1)) // 2
    nbeta = nelec - nalpha
    nocc = (nalpha, nbeta)
    resolve_frozen_occupied_for_atoms(
        atoms,
        orbital_basis,
        nocc,
        correlation_space,
        root_id="uhf" if unrestricted else "rhf",
    )

    # 1. Native 1e (NR + ECP) + X2C-1e delta hcore
    t_1e = time.perf_counter()
    overlap, hcore_nr, e_nuc = one_electron_integrals_cpu(
        atoms, orbital_basis, x2c1e_mode="none", charge=charge, multiplicity=multiplicity
    )
    delta_h = (
        np.zeros_like(hcore_nr)
        if mode == "none"
        else x2c1e_hcore(
            atoms,
            orbital_basis,
            mode=mode,
            speed_of_light=speed_of_light,
            delta_only=True,
            charge=charge,
            multiplicity=multiplicity,
        )
    )
    hcore_sr = hcore_nr + delta_h
    dt_1e = time.perf_counter() - t_1e

    # 2. Native 2e DF integrals built ONCE and shared across NR & X2C (free raw 3c immediately)!
    t_df = time.perf_counter()
    jk_metric, jk_3c = df_integrals_cpu(
        atoms, orbital_basis, jk_fit_basis, charge=charge, multiplicity=multiplicity
    )
    jk_factor = _factor_df(jk_metric, jk_3c)
    del jk_metric, jk_3c
    same_fit = jk_fit_basis.identity == mp2_fit_basis.identity
    if same_fit:
        mp2_factor = jk_factor
    else:
        mp2_metric, mp2_3c = df_integrals_cpu(
            atoms, orbital_basis, mp2_fit_basis, charge=charge, multiplicity=multiplicity
        )
        mp2_factor = _factor_df(mp2_metric, mp2_3c)
        del mp2_metric, mp2_3c
    dt_df = time.perf_counter() - t_df

    # 3. DF-HF (NR seeded from native SAD density, then warm-started X2C-1e!)
    zero_delta_h = not bool(np.any(delta_h))
    t_hf = time.perf_counter()
    sad_seed = _build_sad_density(atoms, orbital_basis, overlap, hcore_nr, jk_factor, nocc)
    hf_nr = _solve_df_hf(
        overlap,
        hcore_nr,
        e_nuc,
        jk_factor,
        nocc,
        unrestricted=unrestricted,
        initial_densities=sad_seed,
    )
    hf_sr = (
        hf_nr
        if zero_delta_h
        else _solve_df_hf(
            overlap,
            hcore_sr,
            e_nuc,
            jk_factor,
            nocc,
            unrestricted=unrestricted,
            initial_densities=hf_nr.densities,
        )
    )
    hf_nr, frozen_nr = _partition_df_hf_core(
        atoms, orbital_basis, overlap, hf_nr, correlation_space,
    )
    if zero_delta_h:
        hf_sr, frozen_sr = hf_nr, frozen_nr
    else:
        hf_sr, frozen_sr = _partition_df_hf_core(
            atoms, orbital_basis, overlap, hf_sr, correlation_space,
        )
    frozen = frozen_nr.counts(correlation_space)
    if frozen != frozen_sr.counts(correlation_space):
        raise RuntimeError("Delta-SR changed the requested frozen occupied counts")
    dt_hf = time.perf_counter() - t_hf

    # 4. DF-MP2 energies (and optional exact Z-vector CPHF weights)
    t_mp2 = time.perf_counter()
    forces_hf_nr = forces_hf_sr = delta_forces_hf = None
    forces_mp2_nr = forces_mp2_sr = delta_forces_mp2 = None
    dt_grad = 0.0

    if zero_delta_h and not return_component_forces:
        e_corr_nr, _ = _df_mp2_energy_and_weights(
            hf_nr, jk_factor, mp2_factor, frozen, need_weights=False
        )
        e_corr_sr = e_corr_nr
        dt_mp2 = time.perf_counter() - t_mp2
        if forces:
            delta_forces_hf = np.zeros((len(atoms), 3), dtype=np.float64) if include_hf_forces else None
            delta_forces_mp2 = np.zeros((len(atoms), 3), dtype=np.float64)
        dt_total = time.perf_counter() - t0
        return DeltaSrResult(
            mode=mode,
            correlation_space=correlation_space,
            nbf=overlap.shape[0],
            nocc=nocc,
            e_hf_nr=hf_nr.energy,
            e_hf_sr=hf_nr.energy,
            delta_sr_hf=0.0,
            e_mp2_corr_nr=e_corr_nr,
            e_mp2_corr_sr=e_corr_sr,
            e_mp2_nr=hf_nr.energy + e_corr_nr,
            e_mp2_sr=hf_nr.energy + e_corr_sr,
            delta_sr_mp2=0.0,
            forces_hf_nr=None,
            forces_hf_sr=None,
            delta_forces_hf=delta_forces_hf,
            forces_mp2_nr=None,
            forces_mp2_sr=None,
            delta_forces_mp2=delta_forces_mp2,
            timings_s={
                "one_electron_s": dt_1e,
                "df_integrals_s": dt_df,
                "df_hf_s": dt_hf,
                "df_mp2_s": dt_mp2,
                "gradient_s": 0.0,
                "total_s": dt_total,
            },
            frozen_core_definition_version=FROZEN_CORE_DEFINITION_VERSION,
            frozen_nr=frozen_nr,
            frozen_sr=frozen_sr,
        )

    if not forces or return_component_forces:
        e_corr_nr, w_mp2_nr = _df_mp2_energy_and_weights(
            hf_nr, jk_factor, mp2_factor, frozen, need_weights=forces
        )
        e_corr_sr, w_mp2_sr = _df_mp2_energy_and_weights(
            hf_sr, jk_factor, mp2_factor, frozen, need_weights=forces
        )
        dt_mp2 = time.perf_counter() - t_mp2
    else:
        # Memory-bounded in-place streaming accumulation of (SR - NR) 3-center weights!
        nbf = overlap.shape[0]
        naux_jk = jk_factor.metric.shape[0]
        naux_mp2 = mp2_factor.metric.shape[0]
        dw_mp2_jk_a_2d = np.zeros((nbf * nbf, naux_jk), dtype=np.float64)
        dw_mp2_a_2d = dw_mp2_jk_a_2d if same_fit else np.zeros((nbf * nbf, naux_mp2), dtype=np.float64)

        e_corr_nr, w_mp2_nr = _df_mp2_energy_and_weights(
            hf_nr,
            jk_factor,
            mp2_factor,
            frozen,
            need_weights=True,
            accum_jk_a_2d=dw_mp2_jk_a_2d,
            accum_mp2_a_2d=dw_mp2_a_2d,
            accum_sign=-1.0,
        )
        assert w_mp2_nr is not None and w_mp2_nr.mp2_metric is not None
        dw_mp2_s = -w_mp2_nr.overlap
        dw_mp2_h = -w_mp2_nr.hcore
        dw_mp2_jk_m = -w_mp2_nr.jk_metric
        dw_mp2_m = -w_mp2_nr.mp2_metric
        del w_mp2_nr

        e_corr_sr, w_mp2_sr = _df_mp2_energy_and_weights(
            hf_sr,
            jk_factor,
            mp2_factor,
            frozen,
            need_weights=True,
            accum_jk_a_2d=dw_mp2_jk_a_2d,
            accum_mp2_a_2d=dw_mp2_a_2d,
            accum_sign=+1.0,
        )
        assert w_mp2_sr is not None and w_mp2_sr.mp2_metric is not None
        dw_mp2_s += w_mp2_sr.overlap
        dw_mp2_h += w_mp2_sr.hcore
        w_mp2_h_sr = w_mp2_sr.hcore
        dw_mp2_jk_m += w_mp2_sr.jk_metric
        dw_mp2_m += w_mp2_sr.mp2_metric
        del w_mp2_sr
        dw_mp2_jk_a = dw_mp2_jk_a_2d.reshape(nbf, nbf, naux_jk)
        dw_mp2_a = None if same_fit else dw_mp2_a_2d.reshape(nbf, nbf, naux_mp2)
        dt_mp2 = time.perf_counter() - t_mp2

    # 5. Analytic nuclear forces via combined difference-weight pullback!
    if forces:
        t_grad = time.perf_counter()
        if not return_component_forces and not include_hf_forces:
            w_hf_nr = _df_hf_weights(
                hf_nr, jk_factor, accum_jk_a_2d=dw_mp2_jk_a_2d, accum_sign=-1.0
            )
            w_hf_sr = _df_hf_weights(
                hf_sr, jk_factor, accum_jk_a_2d=dw_mp2_jk_a_2d, accum_sign=+1.0
            )
        else:
            w_hf_nr = _df_hf_weights(hf_nr, jk_factor)
            w_hf_sr = _df_hf_weights(hf_sr, jk_factor)
        # Release orthonormal DF tensors before gradient contractions to keep peak RSS < 4 GiB!
        del jk_factor, mp2_factor

        def eval_full_grad(
            w_s: np.ndarray,
            w_h: np.ndarray,
            w_jk_m: np.ndarray,
            w_jk_a: np.ndarray,
            w_mp2_m: np.ndarray | None,
            w_mp2_a: np.ndarray | None,
            x2c_mode: X2c1eModeName,
            inc_nuc: bool,
        ) -> np.ndarray:
            g = one_electron_gradient_cpu(
                atoms,
                orbital_basis,
                w_s,
                w_h,
                x2c1e_mode=x2c_mode,
                speed_of_light=speed_of_light,
                include_nuclear_repulsion=inc_nuc,
                charge=charge,
                multiplicity=multiplicity,
            )
            if same_fit and w_mp2_m is not None and w_mp2_a is not None:
                g = g + df_gradient_cpu(
                    atoms,
                    orbital_basis,
                    jk_fit_basis,
                    w_jk_m + w_mp2_m,
                    w_jk_a + w_mp2_a,
                    charge=charge,
                    multiplicity=multiplicity,
                )
            else:
                g = g + df_gradient_cpu(
                    atoms,
                    orbital_basis,
                    jk_fit_basis,
                    w_jk_m,
                    w_jk_a,
                    charge=charge,
                    multiplicity=multiplicity,
                )
                if w_mp2_m is not None and w_mp2_a is not None:
                    g = g + df_gradient_cpu(
                        atoms,
                        orbital_basis,
                        mp2_fit_basis,
                        w_mp2_m,
                        w_mp2_a,
                        charge=charge,
                        multiplicity=multiplicity,
                    )
            return -g  # force = -dE/dR

        if return_component_forces:
            assert w_mp2_nr is not None and w_mp2_sr is not None
            forces_hf_nr = eval_full_grad(
                w_hf_nr.overlap, w_hf_nr.hcore, w_hf_nr.jk_metric, w_hf_nr.jk_three_center,
                None, None, "none", True,
            )
            forces_hf_sr = eval_full_grad(
                w_hf_sr.overlap, w_hf_sr.hcore, w_hf_sr.jk_metric, w_hf_sr.jk_three_center,
                None, None, mode, True,
            )
            delta_forces_hf = forces_hf_sr - forces_hf_nr

            forces_mp2_nr = eval_full_grad(
                w_hf_nr.overlap + w_mp2_nr.overlap,
                w_hf_nr.hcore + w_mp2_nr.hcore,
                w_hf_nr.jk_metric + w_mp2_nr.jk_metric,
                w_hf_nr.jk_three_center + w_mp2_nr.jk_three_center,
                w_mp2_nr.mp2_metric,
                w_mp2_nr.mp2_three_center,
                "none",
                True,
            )
            forces_mp2_sr = eval_full_grad(
                w_hf_sr.overlap + w_mp2_sr.overlap,
                w_hf_sr.hcore + w_mp2_sr.hcore,
                w_hf_sr.jk_metric + w_mp2_sr.jk_metric,
                w_hf_sr.jk_three_center + w_mp2_sr.jk_three_center,
                w_mp2_sr.mp2_metric,
                w_mp2_sr.mp2_three_center,
                mode,
                True,
            )
            delta_forces_mp2 = forces_mp2_sr - forces_mp2_nr
        else:
            # Production fast path: ONE 1e+ECP contraction on difference weights + ONE X2C adjoint
            # contraction + ONE DF contraction per auxiliary fit on difference weights!
            def eval_delta_force(
                dw_s: np.ndarray,
                dw_h_nr: np.ndarray,
                w_h_sr: np.ndarray,
                dw_jk_m: np.ndarray,
                dw_jk_a: np.ndarray,
                dw_mp2_m_in: np.ndarray | None,
                dw_mp2_a_in: np.ndarray | None,
            ) -> np.ndarray:
                g = one_electron_gradient_cpu(
                    atoms,
                    orbital_basis,
                    dw_s,
                    dw_h_nr,
                    x2c1e_mode="none",
                    include_nuclear_repulsion=False,
                    charge=charge,
                    multiplicity=multiplicity,
                )
                if mode != "none":
                    g = g + x2c1e_weighted_gradient(
                        atoms,
                        orbital_basis,
                        w_h_sr,
                        mode=mode,
                        speed_of_light=speed_of_light,
                        delta_only=True,
                        charge=charge,
                        multiplicity=multiplicity,
                    )
                if same_fit:
                    total_m = dw_jk_m if dw_mp2_m_in is None else (dw_jk_m + dw_mp2_m_in)
                    total_a = dw_jk_a if dw_mp2_a_in is None else (dw_jk_a + dw_mp2_a_in)
                    g = g + df_gradient_cpu(
                        atoms,
                        orbital_basis,
                        jk_fit_basis,
                        total_m,
                        total_a,
                        charge=charge,
                        multiplicity=multiplicity,
                    )
                else:
                    g = g + df_gradient_cpu(
                        atoms,
                        orbital_basis,
                        jk_fit_basis,
                        dw_jk_m,
                        dw_jk_a,
                        charge=charge,
                        multiplicity=multiplicity,
                    )
                    if dw_mp2_m_in is not None and dw_mp2_a_in is not None:
                        g = g + df_gradient_cpu(
                            atoms,
                            orbital_basis,
                            mp2_fit_basis,
                            dw_mp2_m_in,
                            dw_mp2_a_in,
                            charge=charge,
                            multiplicity=multiplicity,
                        )
                return -g

            dw_hf_s = w_hf_sr.overlap - w_hf_nr.overlap
            dw_hf_h = w_hf_sr.hcore - w_hf_nr.hcore
            dw_hf_jk_m = w_hf_sr.jk_metric - w_hf_nr.jk_metric
            w_hf_h_sr = w_hf_sr.hcore
            if include_hf_forces:
                dw_hf_jk_a = w_hf_sr.jk_three_center - w_hf_nr.jk_three_center
                del w_hf_nr, w_hf_sr
                delta_forces_hf = eval_delta_force(
                    dw_hf_s, dw_hf_h, w_hf_h_sr, dw_hf_jk_m, dw_hf_jk_a, None, None
                )
                dw_mp2_jk_a += dw_hf_jk_a
                del dw_hf_jk_a
            else:
                del w_hf_nr, w_hf_sr
            dw_mp2_s += dw_hf_s
            dw_mp2_h += dw_hf_h
            dw_mp2_jk_m += dw_hf_jk_m
            del dw_hf_s, dw_hf_h, dw_hf_jk_m
            delta_forces_mp2 = eval_delta_force(
                dw_mp2_s,
                dw_mp2_h,
                w_hf_h_sr + w_mp2_h_sr,
                dw_mp2_jk_m,
                dw_mp2_jk_a,
                dw_mp2_m,
                dw_mp2_a,
            )
        dt_grad = time.perf_counter() - t_grad

    dt_total = time.perf_counter() - t0
    return DeltaSrResult(
        mode=mode,
        correlation_space=correlation_space,
        frozen_core_definition_version=FROZEN_CORE_DEFINITION_VERSION,
        frozen_nr=frozen_nr,
        frozen_sr=frozen_sr,
        nbf=overlap.shape[0],
        nocc=nocc,
        e_hf_nr=hf_nr.energy,
        e_hf_sr=hf_sr.energy,
        delta_sr_hf=hf_sr.energy - hf_nr.energy,
        e_mp2_corr_nr=e_corr_nr,
        e_mp2_corr_sr=e_corr_sr,
        e_mp2_nr=hf_nr.energy + e_corr_nr,
        e_mp2_sr=hf_sr.energy + e_corr_sr,
        delta_sr_mp2=(hf_sr.energy + e_corr_sr) - (hf_nr.energy + e_corr_nr),
        forces_hf_nr=forces_hf_nr,
        forces_hf_sr=forces_hf_sr,
        delta_forces_hf=delta_forces_hf,
        forces_mp2_nr=forces_mp2_nr,
        forces_mp2_sr=forces_mp2_sr,
        delta_forces_mp2=delta_forces_mp2,
        timings_s={
            "one_electron_s": dt_1e,
            "df_integrals_s": dt_df,
            "df_hf_s": dt_hf,
            "df_mp2_s": dt_mp2,
            "gradient_s": dt_grad,
            "total_s": dt_total,
        },
    )


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
