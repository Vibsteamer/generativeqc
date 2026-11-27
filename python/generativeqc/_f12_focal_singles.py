"""CABS singles and its electronic source response on a physical focal DF-HF reference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .elements import checked_integer

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable, Mapping

    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_ao_mo_native import CabsSinglesNativeOps
    from ._f12_focal_one_electron import FocalOneElectron
    from ._f12_focal_reference import FocalReference
    from .experimental.f12_focal import FocalRequest


@dataclass(frozen=True)
class FocalCabsSingles:
    """The HF and singles energies in Eh and optional independent source weights.

    hcore, three_center and metric weights have the shape of their AO sources.
    obs_coefficients differentiates the canonical reference orbitals, one channel
    for RHF or alpha then beta for UHF. cabs_kernel differentiates C_CABS C_CABS.T
    and can seed focal_cabs_projector. The orbital and integral responses are
    required to turn these weights into nuclear forces. With relax=True the
    four integral weights overlap, hcore, three_center and metric already
    include those coefficient and fixed-rank projector responses.
    """

    hf_cabs_energy: float
    spin_energies: tuple[float, float]
    extended_fock: np.ndarray
    max_obs_fock_error: tuple[float, float]
    weights: dict[str, np.ndarray] | None


def _reference_singles(
    matrix: np.ndarray, occupied: int, reverse: bool
) -> tuple[float, np.ndarray | None]:
    # Solve A T - T B = F_ov, with A/B the occupied/extended-virtual Fock blocks.
    if not occupied:
        return 0.0, np.zeros_like(matrix) if reverse else None
    eo, u = np.linalg.eigh(matrix[:occupied, :occupied])
    ev, v = np.linalg.eigh(matrix[occupied:, occupied:])
    gaps = eo[:, None] - ev
    if np.min(np.abs(gaps)) <= 1e-12:
        raise ValueError(
            "CABS singles has a singular occupied/virtual Sylvester equation"
        )
    coupling = matrix[:occupied, occupied:]
    t = u @ ((u.T @ coupling @ v) / gaps) @ v.T
    energy = float(np.vdot(coupling, t))
    if not reverse:
        return energy, None
    return energy, np.block([[-t @ t.T, t], [t.T, t.T @ t]])


def _singles_from_fock(
    fock: np.ndarray,
    spectrum: np.ndarray,
    occupied: int,
    cuda: bool,
    pullback: bool,
    library: ctypes.CDLL | None,
) -> tuple[np.ndarray, float, float, np.ndarray | None]:
    """Use the physical extended Fock and verify its canonical orbital block."""
    from ._f12_cabs import cabs_singles

    if not np.isfinite(fock).all() or not np.allclose(fock, fock.T, atol=2e-8, rtol=0):
        skew = np.max(np.abs(fock - fock.T))
        raise ValueError(f"physical extended DF Fock is asymmetric: max skew {skew} Eh")
    fock = (fock + fock.T) * 0.5
    n = len(spectrum)
    error = float(np.max(np.abs(fock[:n, :n] - np.diag(spectrum))))
    if not np.isfinite(error) or error > 2e-8:
        raise ValueError(f"extended Fock does not reproduce the SCF reference: {error}")
    energy, bar = (
        cabs_singles(fock, occupied, spin_weight=1, pullback=pullback, library=library)
        if cuda
        else _reference_singles(fock, occupied, pullback)
    )
    return fock, energy, error, bar


def _validate_ao_symmetry(h: np.ndarray, three_center: np.ndarray) -> None:
    if not np.allclose(h, h.T, rtol=1e-12, atol=1e-11):
        raise ValueError("physical AO core Hamiltonian is asymmetric")
    for tile in np.array_split(three_center, max(1, (len(three_center) + 3) // 4)):
        if not np.allclose(tile, tile.swapaxes(1, 2), rtol=1e-12, atol=1e-11):
            raise ValueError("physical AO three-center Coulomb integral is asymmetric")


def _verify_obs_three_center_identity(
    full_three_center: np.ndarray,
    obs_three_center: np.ndarray,
    obs: int,
) -> None:
    q = int(full_three_center.shape[0])
    step = max(1, (16 * 1024 * 1024) // max(1, obs * obs * 8))
    for q0 in range(0, q, step):
        q1 = min(q, q0 + step)
        if not np.array_equal(
            full_three_center[q0:q1, :obs, :obs], obs_three_center[q0:q1]
        ):
            raise ValueError(
                "obs_three_center does not match the leading OBS block of three_center"
            )


def _relaxed_weights_aot(
    ops: CabsSinglesNativeOps,
    request: FocalRequest,
    reference: FocalReference,
    one: FocalOneElectron,
    complement: np.ndarray,
    three_center: np.ndarray,
    whitener: np.ndarray,
    weights: dict[str, np.ndarray],
    library: ctypes.CDLL | None,
    relative_cutoff: float,
    budget_mib: int,
    *,
    cabs_frame_bar: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """AOT zero-JIT CABS singles orbital relaxation and nuclear source weights."""
    from scipy.sparse.linalg import LinearOperator, minres

    from ._f12_ao_mo_native import (
        run_ao_mo_bilin_fwd,
        run_ao_mo_project_fwd,
        run_cabs_cphf_action,
        run_cabs_nuclear_weights,
        run_cabs_whiten_3c,
    )
    from ._f12_focal_projector import focal_cabs_projector, full_rank_cabs_overlap_bar
    from .f12_orbital_response import _solve_orbital_response

    mb = checked_integer(
        budget_mib, "orbital response budget in MiB", low=1, high=(2**63 - 1) >> 20
    )
    use_direct_full_rank = (
        cabs_frame_bar is not None
        and complement.shape[1] == len(one.overlap) - one.obs_ao_count
    )
    projector = focal_cabs_projector(
        request,
        one,
        expected_rank=complement.shape[1],
        relative_cutoff=relative_cutoff,
        cotangent=None if use_direct_full_rank else weights["cabs_kernel"],
        library=library,
    )
    if not np.allclose(complement @ complement.T, projector.kernel, rtol=3e-7, atol=3e-7):
        raise ValueError(
            "CABS response must use the same fixed-rank projector as the energy"
        )
    if use_direct_full_rank:
        assert cabs_frame_bar is not None
        overlap = full_rank_cabs_overlap_bar(
            one.overlap, one.obs_ao_count, complement, cabs_frame_bar
        )
    else:
        assert projector.overlap_bar is not None
        overlap = projector.overlap_bar.copy()

    is_rhf = len(reference.occupied) == 1
    occs = reference.occupied * 2 if is_rhf else reference.occupied
    oa, ob = int(occs[0]), int(occs[1])
    obs = reference.overlap.shape[0]
    aux = len(whitener)
    va, vb = obs - oa, obs - ob

    # Project OBS three-center panel and whiten once for RHF, or per-spin for UHF.
    panel = np.ascontiguousarray(three_center[:, :obs, :obs])
    c_a = np.ascontiguousarray(reference.coefficients[0])
    raw_mo_a = run_ao_mo_project_fwd(ops.ao_mo, obs, obs, obs, aux, c_a, panel)
    df_a = run_cabs_whiten_3c(ops, whitener, raw_mo_a)
    del raw_mo_a
    if is_rhf:
        c_b = c_a
        df_b = df_a
    else:
        c_b = np.ascontiguousarray(reference.coefficients[1])
        if np.array_equal(c_a, c_b):
            df_b = df_a
        else:
            raw_mo_b = run_ao_mo_project_fwd(ops.ao_mo, obs, obs, obs, aux, c_b, panel)
            df_b = run_cabs_whiten_3c(ops, whitener, raw_mo_b)
            del raw_mo_b
    del panel

    spin_indices = (0,) if is_rhf else (0, 1)
    drivers_by_spin: dict[int, np.ndarray] = {}
    gaps_by_spin: dict[int, np.ndarray] = {}
    for k in spin_indices:
        o = int(reference.occupied[k])
        coeff = c_a if k == 0 else c_b
        eps = reference.orbital_energies[k]
        bar = weights["obs_coefficients"][k] * (0.5 if is_rhf else 1.0)
        rotation = coeff.T @ bar
        torque = rotation - rotation.T
        if any(
            b.size and np.max(np.abs(b)) > 1e-8
            for b in (torque[:o, :o], torque[o:, o:])
        ):
            raise ValueError(
                "electronic observable fails occupied or virtual orbital gauge invariance"
            )
        mask = np.full((obs, obs), -0.5, dtype=np.float64)
        mask[:o, o:] = -1.0
        mask[o:, :o] = 0.0
        scaled = mask * rotation
        frame_m = 0.5 * (scaled + scaled.T)
        coeff_t = np.ascontiguousarray(coeff.T)
        weight = run_ao_mo_bilin_fwd(ops.ao_mo, frame_m, coeff_t, coeff_t)
        ov_bar = 0.5 * (weight + weight.T)
        overlap[:obs, :obs] += (2.0 if is_rhf else 1.0) * ov_bar
        if o > 0:
            drivers_by_spin[k] = np.ascontiguousarray(torque[o:, :o].ravel())
            gaps_by_spin[k] = np.ascontiguousarray(eps[o:, None] - eps[None, :o])

    minres_tol_kw = "rtol" if "rtol" in signature(minres).parameters else "tol"
    cuda = reference.executed_backend == "cuda"
    r_total = va * oa + vb * ob

    def _cuda_dense_fallback(driver_full: np.ndarray) -> np.ndarray:
        factors = {}
        for spin_idx, (spin_name, o, df_s) in enumerate(
            (("alpha", oa, df_a), ("beta", ob, df_b))
        ):
            if o <= 0:
                continue
            g_s = gaps_by_spin[0] if is_rhf else gaps_by_spin[spin_idx]
            factors.update(
                {
                    "vo_" + spin_name: df_s[:, o:, :o],
                    "oo_" + spin_name: df_s[:, :o, :o],
                    "vv_" + spin_name: df_s[:, o:, o:],
                    "gap_" + spin_name: g_s,
                }
            )
        return _solve_orbital_response(
            (oa, ob, obs, aux),
            factors,
            driver_full,
            np.empty((0, 3, r_total)),
            np.empty((0, 3)),
            (),
            0,
            mb << 20,
            library,
        ).rotations

    z_a_mat: np.ndarray | None = None
    z_b_mat: np.ndarray | None = None
    if r_total > 0:
        if is_rhf:
            ra = va * oa
            g_rhf = drivers_by_spin[0]
            gap_a = gaps_by_spin[0]
            inv_diag = 1.0 / np.maximum(np.abs(gap_a.ravel()), 1e-2)
            precond = LinearOperator(
                (ra, ra), matvec=lambda x: x * inv_diag, dtype=np.float64
            )

            def rhf_matvec(vector: np.ndarray) -> np.ndarray:
                out_a, _ = run_cabs_cphf_action(
                    ops,
                    oa,
                    oa,
                    obs,
                    aux,
                    is_rhf=True,
                    df_a=df_a,
                    df_b=None,
                    gap_a=gap_a,
                    gap_b=None,
                    trial_a=np.ascontiguousarray(vector.reshape(va, oa)),
                    trial_b=None,
                )
                assert out_a is not None
                return out_a.ravel()

            z_a, status = minres(
                LinearOperator((ra, ra), matvec=rhf_matvec, dtype=np.float64),
                g_rhf,
                M=precond,
                **{minres_tol_kw: 1e-12},
                maxiter=max(128, 4 * ra),
            )
            res_norm = float(np.linalg.norm(rhf_matvec(z_a) - g_rhf))
            tol_rhs = 1e-9 * max(1.0, float(np.linalg.norm(g_rhf)))
            if (status != 0 or res_norm > tol_rhs) and cuda and 8 * r_total * r_total <= (mb << 20):
                z_full = _cuda_dense_fallback(np.concatenate((g_rhf, g_rhf)))
                za_f, zb_f = np.split(z_full, 2)
                z_a = 0.5 * (za_f + zb_f)
                res_norm = float(np.linalg.norm(rhf_matvec(z_a) - g_rhf))
                status = 0
            if status != 0:
                raise RuntimeError(
                    f"AOT RHF DF-HF orbital response did not converge ({status})"
                )
            if res_norm > tol_rhs:
                raise ValueError(
                    f"DF-HF orbital response residual {res_norm:.3e} exceeds its tolerance"
                )
            z_a_mat = z_a.reshape(va, oa)
            z_b_mat = z_a_mat
        else:
            driver_parts = []
            gap_parts = []
            if oa > 0:
                driver_parts.append(drivers_by_spin[0])
                gap_parts.append(gaps_by_spin[0].ravel())
            if ob > 0:
                driver_parts.append(drivers_by_spin[1])
                gap_parts.append(gaps_by_spin[1].ravel())
            driver = np.concatenate(driver_parts)
            inv_diag = 1.0 / np.maximum(np.abs(np.concatenate(gap_parts)), 1e-2)
            precond = LinearOperator(
                (r_total, r_total), matvec=lambda x: x * inv_diag, dtype=np.float64
            )
            na_rot = va * oa

            def uhf_matvec(vector: np.ndarray) -> np.ndarray:
                t_a = (
                    np.ascontiguousarray(vector[:na_rot].reshape(va, oa))
                    if oa > 0
                    else None
                )
                t_b = (
                    np.ascontiguousarray(vector[na_rot:].reshape(vb, ob))
                    if ob > 0
                    else None
                )
                out_a, out_b = run_cabs_cphf_action(
                    ops,
                    oa,
                    ob,
                    obs,
                    aux,
                    is_rhf=False,
                    df_a=df_a if oa > 0 else None,
                    df_b=df_b if ob > 0 else None,
                    gap_a=gaps_by_spin.get(0),
                    gap_b=gaps_by_spin.get(1),
                    trial_a=t_a,
                    trial_b=t_b,
                )
                parts = []
                if out_a is not None:
                    parts.append(out_a.ravel())
                if out_b is not None:
                    parts.append(out_b.ravel())
                return np.concatenate(parts)

            z_vec, status = minres(
                LinearOperator(
                    (r_total, r_total), matvec=uhf_matvec, dtype=np.float64
                ),
                driver,
                M=precond,
                **{minres_tol_kw: 1e-12},
                maxiter=max(128, 4 * r_total),
            )
            res_norm = float(np.linalg.norm(uhf_matvec(z_vec) - driver))
            tol_rhs = 1e-9 * max(1.0, float(np.linalg.norm(driver)))
            if (status != 0 or res_norm > tol_rhs) and cuda and 8 * r_total * r_total <= (mb << 20):
                z_vec = _cuda_dense_fallback(driver)
                res_norm = float(np.linalg.norm(uhf_matvec(z_vec) - driver))
                status = 0
            if status != 0:
                raise RuntimeError(
                    f"AOT UHF DF-HF orbital response did not converge ({status})"
                )
            if res_norm > tol_rhs:
                raise ValueError(
                    f"DF-HF orbital response residual {res_norm:.3e} exceeds its tolerance"
                )
            if oa > 0:
                z_a_mat = z_vec[:na_rot].reshape(va, oa)
            if ob > 0:
                z_b_mat = z_vec[na_rot:].reshape(vb, ob)

    p_a = np.zeros((obs, obs), dtype=np.float64)
    sm_a = np.zeros((obs, obs), dtype=np.float64)
    if oa > 0 and z_a_mat is not None:
        p_a[oa:, :oa] = -z_a_mat
        sm_a[oa:, :oa] = z_a_mat * reference.orbital_energies[0][:oa][None, :]

    if is_rhf:
        p_b, sm_b = p_a, sm_a
        same_spin_nuc = True
    else:
        p_b = np.zeros((obs, obs), dtype=np.float64)
        sm_b = np.zeros((obs, obs), dtype=np.float64)
        if ob > 0 and z_b_mat is not None:
            p_b[ob:, :ob] = -z_b_mat
            sm_b[ob:, :ob] = z_b_mat * reference.orbital_energies[1][:ob][None, :]
        same_spin_nuc = (
            oa == ob
            and (df_a is df_b or np.array_equal(df_a, df_b))
            and np.array_equal(c_a, c_b)
            and np.array_equal(p_a, p_b)
            and np.array_equal(sm_a, sm_b)
        )

    bars = run_cabs_nuclear_weights(
        ops,
        oa,
        ob,
        obs,
        aux,
        obs,
        same_spin=same_spin_nuc,
        whitener=whitener,
        c_a=c_a,
        c_b=c_b,
        df_a=df_a,
        df_b=df_b,
        p_a=p_a,
        p_b=p_b,
        sm_a=sm_a,
        sm_b=sm_b,
    )
    if any(not np.isfinite(val).all() for val in bars.values()):
        raise ValueError("Nonfinite DF-UHF nuclear source weights")

    overlap[:obs, :obs] += bars["overlap_ao"]
    h_out = weights["hcore"].copy()
    h_out[:obs, :obs] += bars["hcore_ao"]
    three_out = (
        weights["three_center"]
        if weights["three_center"].flags.writeable
        else weights["three_center"].copy()
    )
    three_out[:, :obs, :obs] += bars["ao_coulomb"]
    return {
        "overlap": overlap,
        "hcore": h_out,
        "three_center": three_out,
        "metric": weights["metric"] + bars["metric/coulomb"],
    }


def reference_cabs_singles(
    request: FocalRequest,
    reference: FocalReference,
    one: FocalOneElectron,
    *,
    complement: np.ndarray,
    three_center: np.ndarray | None,
    metric: np.ndarray,
    whitener: np.ndarray,
    pullback: bool = True,
    relax: bool = False,
    relative_cutoff: float = 1e-9,
    budget_mib: int = 512,
    run: Callable | None = None,
    library: ctypes.CDLL | None = None,
    shell_library: Path | str | ctypes.CDLL | None = None,
    three_center_loader: Callable[[], np.ndarray] | None = None,
    obs_three_center: np.ndarray | None = None,
) -> FocalCabsSingles:
    """Evaluate singles in the same canonical determinant, OBS/OptRI and full-rank JKFit.

    Supply the real-spherical combined-space three-center tensor [JKFit,AO,AO],
    its Coulomb metric, a full-rank W satisfying W.T W = metric^-1, and the
    OBS-orthogonal CABS columns returned by focal_cabs_projector. CPU reference
    uses FP64 TensorIR, a wider congruence accumulator and NumPy LAPACK. CUDA
    uses two-word FP64 for the congruence and requires a configured ordinary
    FP64 CUDA TensorIR runner and the native CUDA singles library. Each spin's
    extended physical Fock must reproduce the canonical reference in its OBS.
    Set relax=True to solve the coupled occupied/virtual response and return
    four relaxed AO integral weights. Contract each with its positive nuclear
    derivative, then subtract their sum from the reference forces. The CABS
    overlap cutoff must match the supplied complementary frame. budget_mib limits
    the default CPU interpreter's logical retained arrays per TensorIR graph
    and the orbital solver workspace. A supplied runner owns its graph budget.
    """
    from generativeqc_compiler.mp2.f12_frame_response import frame_kernel_cotangent
    from generativeqc_compiler.tensor import execute

    from ._f12_cabs_fock import obs_sources, physical_fock

    if (
        request.role != "hf_cabs"
        or request.root_id not in ("rhf", "uhf")
        or request.backend not in ("cpu_reference", "cuda")
    ):
        raise ValueError("CABS singles requires a canonical RHF/UHF HF+CABS request")
    if type(pullback) is not bool or type(relax) is not bool:
        raise TypeError("pullback and relax must be Boolean")
    if relax and not pullback:
        raise ValueError("orbital relaxation requires the singles pullback")
    if (
        reference.surface_id != request.surface_id
        or one.surface_id != request.surface_id
        or reference.basis_id != request.basis.basis_id
        or one.basis_id != request.basis.basis_id
        or reference.executed_backend != request.backend
        or one.executed_backend != request.backend
        or reference.jk_fit_id != request.basis.jk_fit.identity
        or reference.orbital_id != request.basis.orbital.identity
    ):
        raise ValueError(
            "SCF, one-electron sources and CABS request must share their surface, bases and backend"
        )
    cuda = request.backend == "cuda"
    if cuda and run is None:
        raise ValueError("CABS singles on CUDA requires the FP64 CUDA TensorIR runner")
    budget = (
        checked_integer(
            budget_mib, "CABS tensor budget in MiB", low=1, high=(2**63 - 1) >> 20
        )
        << 20
    )
    runner = run if run is not None else partial(execute, max_bytes=budget)

    def checked_run(program: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        out = runner(program, inputs)
        if out.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError("CABS tensor graph executed on another backend")
        return out

    s, h = one.overlap, one.hcore
    atomic = len(s)
    obs = one.obs_ao_count
    if complement.ndim != 2 or metric.ndim != 2:
        raise ValueError("CABS columns and Coulomb metric must be matrices")
    rank = complement.shape[1]
    q = len(metric)
    if not 0 < rank <= atomic - obs or not 0 < obs < atomic or q < 1:
        raise ValueError(
            "CABS and JKFit require nonempty orbital and complementary spaces"
        )
    if three_center is None and three_center_loader is None:
        raise ValueError("CABS singles requires three_center or three_center_loader")
    checks = [
        (s, (atomic, atomic)),
        (h, (atomic, atomic)),
        (complement, (atomic, rank)),
        (metric, (q, q)),
        (whitener, (q, q)),
    ]
    if three_center is not None:
        checks.append((three_center, (q, atomic, atomic)))
    if obs_three_center is not None:
        checks.append((obs_three_center, (q, obs, obs)))
    for arr, shape in checks:
        if arr.dtype != np.float64 or arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError("CABS singles source shape, FP64 type or values disagree")
    if three_center is not None and obs_three_center is not None:
        _verify_obs_three_center_identity(three_center, obs_three_center, obs)
    if not np.allclose(
        s[:obs, :obs], reference.overlap, rtol=0, atol=2e-8
    ) or not np.allclose(h[:obs, :obs], reference.hcore, rtol=0, atol=2e-8):
        raise ValueError("CABS OBS overlap/core differ from the physical reference")
    spectrum = np.linalg.eigvalsh(metric)
    if spectrum[0] <= 1e-10 * spectrum[-1] or not np.allclose(
        whitener @ metric @ whitener.T, np.eye(q), rtol=0, atol=2e-7
    ):
        raise ValueError(
            "CABS singles requires the full-rank JKFit metric and its matching whitener"
        )
    from ._f12_focal_projector import (
        compensated_cabs_frame_cotangent_checks,
        compensated_cabs_orthonormality,
    )

    dual = s @ complement
    if not np.allclose(
        complement.T @ dual, np.eye(rank), atol=2e-7, rtol=0
    ) or not np.allclose(dual[:obs], 0, atol=2e-7, rtol=0):
        gram_oz, obs_dual_oz = compensated_cabs_orthonormality(s, obs, complement)
        if not np.allclose(
            gram_oz, np.eye(rank), atol=2e-7, rtol=0
        ) or not np.allclose(obs_dual_oz, 0, atol=2e-7, rtol=0):
            raise ValueError(
                "CABS columns must be metric-orthonormal and orthogonal to the whole OBS"
            )
    expected = 1 if request.root_id == "rhf" else 2
    if len(reference.occupied) != expected:
        raise ValueError(
            "reference spin-channel count disagrees with the requested root"
        )
    spins = (0, 0) if expected == 1 else (0, 1)
    oa, ob = (reference.occupied[k] for k in spins)
    frames = []
    for k in spins:
        frame = np.zeros((atomic, obs + rank))
        frame[:obs, :obs] = reference.coefficients[k]
        frame[:, obs:] = complement
        frames.append(frame)

    from ._f12_ao_mo_native import (
        resolve_cabs_singles_ops,
        run_cabs_fock_fwd,
        run_cabs_fock_rev,
        run_cabs_frame_kernel_cotangent,
    )
    from ._f12_cabs_fock import _project

    ops = resolve_cabs_singles_ops(
        run,
        shell_library=shell_library,
        host_reference=(not cuda),
    )

    if ops is not None:
        tc_loaded_here = three_center is None
        tc_local = three_center_loader() if tc_loaded_here else three_center
        assert tc_local is not None
        if tc_loaded_here:
            if (
                not isinstance(tc_local, np.ndarray)
                or tc_local.dtype != np.float64
                or tc_local.shape != (q, atomic, atomic)
                or not np.isfinite(tc_local).all()
            ):
                raise ValueError(
                    "CABS singles streamed three_center shape, FP64 type or values disagree"
                )
            if obs_three_center is not None:
                _verify_obs_three_center_identity(tc_local, obs_three_center, obs)
        if not cuda and run is None:
            retained_bytes = (
                tc_local.nbytes
                + whitener.nbytes
                + (3 * q * max(oa, ob) * atomic + 2 * atomic * atomic) * 8
            )
            if retained_bytes > budget:
                raise ValueError(
                    "tensor interpreter logical retained-byte budget exceeded"
                )
        _validate_ao_symmetry(h, tc_local)
        occ_a = (
            np.ascontiguousarray(reference.coefficients[spins[0]][:, :oa])
            if oa > 0
            else None
        )
        occ_b = (
            np.ascontiguousarray(reference.coefficients[spins[1]][:, :ob])
            if ob > 0
            else None
        )
        same_occ = oa == ob and (
            expected == 1
            or oa == 0
            or (occ_a is not None and occ_b is not None and np.array_equal(occ_a, occ_b))
        )
        same_full_frame = expected == 1 or (
            oa == ob
            and np.array_equal(frames[0], frames[1])
            and np.array_equal(
                reference.orbital_energies[0], reference.orbital_energies[1]
            )
        )
        fock_ao_a, fock_ao_b, rho_val, raw_a, fit_a, raw_b, fit_b = run_cabs_fock_fwd(
            ops,
            h,
            tc_local,
            whitener,
            occ_a,
            occ_b,
            occ_rows=obs,
            same_occ=same_occ,
            save_intermediates=pullback,
        )
        if not pullback and tc_loaded_here:
            del tc_local
            tc_local = None
        if same_full_frame:
            f_ext_a = _project(fock_ao_a, frames[0], cuda, library)
            f_a, e_a, err_a, bar_a = _singles_from_fock(
                f_ext_a, reference.orbital_energies[spins[0]], oa, cuda, pullback, library
            )
            matrices = [f_a, f_a.copy()]
            energies = [e_a, e_a]
            errors = [err_a, err_a]
            seeds = [bar_a, bar_a] if pullback else []
        else:
            focks = [
                _project(fock_ao_a, frames[0], cuda, library),
                _project(fock_ao_b, frames[1], cuda, library),
            ]
            matrices = []
            energies = []
            errors = []
            seeds = []
            for k, occ, raw_fock in zip(spins, (oa, ob), focks, strict=True):
                f, energy, err, bar = _singles_from_fock(
                    raw_fock,
                    reference.orbital_energies[k],
                    occ,
                    cuda,
                    pullback,
                    library,
                )
                matrices.append(f)
                energies.append(energy)
                errors.append(err)
                if pullback:
                    seeds.append(bar)
        weights = None
        if pullback:
            assert rho_val is not None and tc_local is not None
            assert seeds[0] is not None and seeds[1] is not None
            same_spin_rev = same_full_frame and (
                seeds[0] is seeds[1] or np.array_equal(seeds[0], seeds[1])
            )
            obs_panel_aot = (
                obs_three_center
                if obs_three_center is not None
                else np.ascontiguousarray(tc_local[:, :obs, :obs])
            )
            ga, gb, core_bar, three_center_bar, metric_bar = run_cabs_fock_rev(
                ops,
                fock_ao_a,
                fock_ao_b,
                frames[0],
                frames[1],
                seeds[0],
                seeds[1],
                tc_local,
                whitener,
                rho_val,
                raw_a,
                fit_a,
                raw_b,
                fit_b,
                occ_a,
                occ_b,
                oa=oa,
                ob=ob,
                occ_rows=obs,
                same_spin=same_spin_rev,
            )
            del raw_a, fit_a, raw_b, fit_b
            if tc_loaded_here:
                del tc_local
                tc_local = None
            cg = ga[:, obs:] + gb[:, obs:]
            transfer = run_cabs_frame_kernel_cotangent(ops, s, complement, cg)
            if not compensated_cabs_frame_cotangent_checks(
                s, obs, complement, cg, transfer, check_dual_gram=True
            ):
                raise ValueError(
                    "CABS singles frame cotangent failed the orbital gauge or kernel reconstruction check"
                )
            cobs = np.stack([ga[:obs, :obs], gb[:obs, :obs]])
            weights = {
                "hcore": core_bar,
                "metric": metric_bar,
                "three_center": three_center_bar,
                "obs_coefficients": cobs
                if expected == 2
                else np.sum(cobs, axis=0, keepdims=True),
                "cabs_kernel": transfer["kernel_bar"],
            }
            if relax:
                weights = _relaxed_weights_aot(
                    ops,
                    request,
                    reference,
                    one,
                    complement,
                    obs_panel_aot,
                    whitener,
                    weights,
                    library,
                    relative_cutoff,
                    budget_mib,
                    cabs_frame_bar=cg,
                )
            del obs_panel_aot
        return FocalCabsSingles(
            reference.energy + sum(energies),
            tuple(energies),
            np.stack(matrices),
            tuple(errors),
            weights,
        )

    verified_loader = three_center_loader
    obs_panel = obs_three_center
    if (
        three_center is None
        and three_center_loader is not None
        and (obs_three_center is not None or (pullback and relax))
    ):
        def _verified_three_center_loader() -> np.ndarray:
            nonlocal obs_panel
            panel = three_center_loader()
            if (
                not isinstance(panel, np.ndarray)
                or panel.dtype != np.float64
                or panel.shape != (q, atomic, atomic)
                or not np.isfinite(panel).all()
            ):
                raise ValueError(
                    "CABS singles streamed three_center shape, FP64 type or values disagree"
                )
            if obs_three_center is not None:
                _verify_obs_three_center_identity(panel, obs_three_center, obs)
            elif obs_panel is None:
                # The response and forward Fock must use the same OBS integrals.
                obs_panel = panel[:, :obs, :obs].copy()
            return panel

        verified_loader = _verified_three_center_loader

    focks, fock_pullback = physical_fock(
        h,
        three_center,
        whitener,
        frames,
        (oa, ob),
        checked_run,
        cuda,
        library,
        raw_loader=verified_loader,
    )
    matrices = []
    energies = []
    errors = []
    seeds = []
    for k, occ, raw_fock in zip(spins, (oa, ob), focks, strict=True):
        f, energy, err, bar = _singles_from_fock(
            raw_fock, reference.orbital_energies[k], occ, cuda, pullback, library,
        )
        matrices.append(f)
        energies.append(energy)
        errors.append(err)
        if pullback:
            seeds.append(bar)
    weights = None
    if pullback:
        (ga, gb), core_bar, three_center_bar, metric_bar = fock_pullback(seeds)
        cg = ga[:, obs:] + gb[:, obs:]
        transfer = checked_run(
            frame_kernel_cotangent(atomic, rank),
            {"overlap": s, "frame": complement, "frame_bar": cg},
        ).outputs
        if not compensated_cabs_frame_cotangent_checks(
            s, obs, complement, cg, transfer, check_dual_gram=True
        ):
            raise ValueError(
                "CABS singles frame cotangent failed the orbital gauge or kernel reconstruction check"
            )
        cobs = np.stack([ga[:obs, :obs], gb[:obs, :obs]])
        weights = {
            "hcore": core_bar,
            "metric": metric_bar,
            "three_center": three_center_bar,
            "obs_coefficients": cobs
            if expected == 2
            else np.sum(cobs, axis=0, keepdims=True),
            "cabs_kernel": transfer["kernel_bar"],
        }
        if relax:
            from ._f12_focal_singles_response import relaxed_weights

            if obs_panel is None:
                assert three_center is not None
                obs_panel = three_center
            raw_obs_sources = obs_sources(obs_panel, frames, obs, checked_run)
            del obs_panel
            weights = relaxed_weights(
                request,
                reference,
                one,
                complement,
                raw_obs_sources,
                whitener,
                weights,
                checked_run,
                library,
                relative_cutoff,
                budget_mib,
                cabs_frame_bar=cg,
            )
    return FocalCabsSingles(
        reference.energy + sum(energies),
        tuple(energies),
        np.stack(matrices),
        tuple(errors),
        weights,
    )
