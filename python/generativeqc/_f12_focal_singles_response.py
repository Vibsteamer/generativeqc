"""Relax a CABS correction through the canonical DF-HF occupied/virtual response."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from scipy.sparse.linalg import LinearOperator

    from ._f12_focal_one_electron import FocalOneElectron
    from ._f12_focal_reference import FocalReference
    from .experimental.f12_focal import FocalRequest

from .elements import checked_integer


def relaxed_weights(
    request: FocalRequest,
    reference: FocalReference,
    one: FocalOneElectron,
    frame: np.ndarray,
    raw: Mapping[str, np.ndarray],
    whitener: np.ndarray,
    weights: Mapping[str, np.ndarray],
    run: Callable,
    library: ctypes.CDLL | None,
    relative_cutoff: float,
    budget_mib: int,
    *,
    native_library: Path | ctypes.CDLL | None = None,
    host_reference: bool | None = None,
    cabs_frame_bar: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    from ._f12_focal_projector import focal_cabs_projector, full_rank_cabs_overlap_bar

    use_direct_full_rank = (
        cabs_frame_bar is not None
        and frame.shape[1] == len(one.overlap) - one.obs_ao_count
    )
    projector = focal_cabs_projector(
        request,
        one,
        expected_rank=frame.shape[1],
        relative_cutoff=relative_cutoff,
        cotangent=None if use_direct_full_rank else weights["cabs_kernel"],
        library=library,
    )
    if not np.allclose(frame @ frame.T, projector.kernel, rtol=3e-7, atol=3e-7):
        raise ValueError(
            "CABS response must use the same fixed-rank projector as the energy"
        )
    if use_direct_full_rank:
        assert cabs_frame_bar is not None
        overlap_seed = full_rank_cabs_overlap_bar(
            one.overlap, one.obs_ao_count, frame, cabs_frame_bar
        )
    else:
        assert projector.overlap_bar is not None
        overlap_seed = projector.overlap_bar
    return relaxed_reference_weights(
        reference,
        raw,
        whitener,
        weights,
        overlap_seed,
        run,
        library,
        budget_mib,
        native_library=native_library,
        host_reference=host_reference,
    )


def _minres_to_residual(
    operator: LinearOperator, driver: np.ndarray, target: float, maxiter: int
) -> np.ndarray:
    """Unpreconditioned MINRES stopped once its residual norm reaches ``target``.

    The Lanczos and Givens recurrences are those of SciPy's ``minres`` without a
    preconditioner; only the stopping rule differs.
    """
    x = np.zeros_like(driver)
    beta1 = float(np.linalg.norm(driver))
    if beta1 <= target:
        return x
    eps = np.finfo(np.float64).eps
    r1 = r2 = y = driver
    oldb, beta, dbar, epsln, phibar = 0.0, beta1, 0.0, 0.0, beta1
    cs, sn = -1.0, 0.0
    w = np.zeros_like(driver)
    w2 = np.zeros_like(driver)
    for iteration in range(1, maxiter + 1):
        v = y / beta
        y = operator.matvec(v)
        if iteration >= 2:
            y = y - (beta / oldb) * r1
        alfa = float(v @ y)
        y = y - (alfa / beta) * r2
        r1, r2 = r2, y
        oldb, beta = beta, float(np.linalg.norm(y))
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        gamma = max(float(np.hypot(gbar, beta)), eps)
        cs, sn = gbar / gamma, beta / gamma
        phi, phibar = cs * phibar, sn * phibar
        w1, w2 = w2, w
        w = (v - oldeps * w1 - delta * w2) / gamma
        x = x + phi * w
        if phibar <= target:
            return x
    raise RuntimeError(
        f"DF-HF coupled orbital response did not reach residual {target:.3e} "
        f"within {maxiter} MINRES iterations"
    )


def _coupled_response_minres(
    operator: LinearOperator,
    driver: np.ndarray,
    preconditioner: LinearOperator | None,
    matrix_free: bool,
) -> np.ndarray:
    """Solve the coupled DF-HF orbital equations with MINRES.

    Matrix-free solves keep the preconditioned 1e-12 SciPy MINRES criterion. SciPy
    stops on backward-error tests such as ||r|| <= rtol ||T_k||_F ||x|| with its
    Lanczos matrix T_k, which left unpreconditioned host responses with
    ||r|| / ||driver|| up to 44 times above rtol. Those solves instead stop on
    ||driver - H z|| <= 1e-11 ||driver||. The target is absolute in the original
    driver: re-solving a residual to a relative tolerance would also resolve
    rounding-level driver components along symmetry null modes of H and inject
    arbitrary null-mode amplitudes into z.
    """
    maxiter = max(128, 4 * driver.size)
    if matrix_free:
        from inspect import signature

        from scipy.sparse.linalg import minres

        rtol_name = "rtol" if "rtol" in signature(minres).parameters else "tol"
        z, status = minres(
            operator, driver, M=preconditioner, maxiter=maxiter, **{rtol_name: 1e-12}
        )
        if status:
            raise RuntimeError(
                f"DF-HF coupled orbital response did not converge ({status})"
            )
        return z
    target = 1e-11 * float(np.linalg.norm(driver))
    z = _minres_to_residual(operator, driver, target, maxiter)
    residual = driver - operator.matvec(z)
    if np.linalg.norm(residual) > target:
        z = z + _minres_to_residual(operator, residual, target, maxiter)
        residual = driver - operator.matvec(z)
    rnorm = float(np.linalg.norm(residual))
    if rnorm > target:
        raise RuntimeError(
            f"DF-HF coupled orbital response residual {rnorm:.3e} did not reach {target:.3e}"
        )
    return z


def _rhf_df_orbital_action(occupied: int, orbitals: int, auxiliaries: int):
    """Apply the spin-symmetric closed-shell RHF coupled orbital Hessian without forming it."""
    from generativeqc_compiler.tensor import (
        Index,
        IndexSpace,
        Program,
        TensorSpec,
        add,
        einsum,
        input_tensor,
        multiply,
    )

    aux = Index("A", IndexSpace("f12_fitting", "auxiliary", auxiliaries))
    occ_space = IndexSpace("occupied_rhf", "occupied", occupied)
    virt_space = IndexSpace("virtual_rhf", "virtual", orbitals - occupied)
    i, j = Index("i", occ_space), Index("j", occ_space)
    a, b = Index("a", virt_space), Index("b", virt_space)

    vv = input_tensor("vv", TensorSpec((aux, a, b), role="input"))
    oo = input_tensor("oo", TensorSpec((aux, i, j), role="input"))
    vo = input_tensor("vo", TensorSpec((aux, a, i), role="input"))
    trial = input_tensor("trial", TensorSpec((a, i), role="input"))
    gap = input_tensor("gap", TensorSpec((a, i), role="input"))

    charge = einsum("Aai,ai->A", vo, trial)
    kv = einsum("Aab,bj->Aaj", vv, trial)
    ki = einsum("Abi,bj->Aij", vo, trial)
    out = add(
        multiply(gap, trial),
        einsum("Aai,A->ai", vo, charge),
        einsum("Aaj,Aij->ai", kv, oo),
        einsum("Aaj,Aij->ai", vo, ki),
        coefficients=(1, 4, -1, -1),
    )
    return Program({"rhf": out}, provenance={"method": "coupled canonical DF-RHF occupied-virtual response"})


def relaxed_reference_weights(
    reference: FocalReference,
    raw: Mapping[str, np.ndarray],
    whitener: np.ndarray,
    weights: Mapping[str, np.ndarray],
    overlap_seed: np.ndarray,
    run: Callable,
    library: ctypes.CDLL | None,
    budget_mib: int,
    *,
    matrix_free: bool = False,
    native_library: Path | ctypes.CDLL | None = None,
    host_reference: bool | None = None,
) -> dict[str, np.ndarray]:
    """Relax a spin-orbital-invariant observable through its physical DF-HF root."""
    from generativeqc_compiler.method.canonical_frame_response import (
        orbital_metric_cotangent,
        orbital_rotation_cotangent,
    )
    from generativeqc_compiler.mp2.f12_unrestricted_reference import (
        unrestricted_df_orbital_action,
    )
    from generativeqc_compiler.tensor import (
        Index,
        IndexSpace,
        Program,
        TensorSpec,
        einsum,
        input_tensor,
    )

    from ._f12_cphf_relax_native import (
        resolve_cphf_relax,
        run_cphf_action_native,
        run_orbital_rot_metric_native,
        run_whiten_3c_native,
    )
    from .f12_orbital_response import (
        _solve_orbital_response,
        unrestricted_nuclear_source_weights,
    )

    mb = checked_integer(
        budget_mib, "orbital response budget in MiB", low=1, high=(2**63 - 1) >> 20
    )
    aot_budget_bytes = min(mb << 20, 512 << 20)
    is_host = (
        (reference.executed_backend != "cuda")
        if host_reference is None
        else bool(host_reference)
    )
    whiten_fn = resolve_cphf_relax(
        native_library, "whiten_3c", fallback_library=library, host_reference=is_host
    )
    rot_metric_fn = resolve_cphf_relax(
        native_library,
        "orbital_rot_metric",
        fallback_library=library,
        host_reference=is_host,
    )
    action_fn = resolve_cphf_relax(
        native_library, "cphf_action", fallback_library=library, host_reference=is_host
    )
    occs = (
        reference.occupied if len(reference.occupied) == 2 else reference.occupied * 2
    )
    from ._f12_spin_fock import _tile_groups, default_auxiliary_batch_size
    from ._f12_spin_pairs import batch_execute

    obs = reference.overlap.shape[0]
    aux = len(whitener)
    space = (*occs, obs, aux)
    r = sum((obs - o) * o for o in occs)
    dense_inputs = sum(
        aux * ((obs - o) * o + o * o + (obs - o) * (obs - o)) + (obs - o) * o
        for o in occs
        if o
    )
    dense_bytes = (dense_inputs + r * r + r) * 8 + (r + 1) * 4
    # Bounded fallback: large QZ orbital-response ranks exceed dense CUDA Hessian
    # headroom and route to the exact preconditioned matrix-free MINRES solver.
    use_matrix_free = matrix_free or (
        reference.executed_backend == "cuda"
        and dense_bytes > min(mb << 20, 2 << 30)
    )
    whiten_width = default_auxiliary_batch_size(aux, obs)
    whiten_groups = _tile_groups(aux, whiten_width)

    def _whiten_tiled(raw_mo: np.ndarray) -> np.ndarray:
        if whiten_fn is not None:
            return run_whiten_3c_native(
                whiten_fn, whitener, raw_mo, budget_bytes=aot_budget_bytes
            )
        mo = IndexSpace("obs", "orbital", obs)
        i, j = Index("p", mo), Index("q", mo)
        out = np.empty((aux, obs, obs), dtype=np.float64)
        for t_len, t_slices in whiten_groups:
            a_t = Index("A", IndexSpace("jkfit_a", "auxiliary", t_len))
            for b_len, b_slices in whiten_groups:
                b_t = Index("B", IndexSpace("jkfit_b", "auxiliary", b_len))
                prog = Program(
                    {
                        "df": einsum(
                            "AB,Bpq->Apq",
                            input_tensor("W", TensorSpec((a_t, b_t), role="input")),
                            input_tensor("raw", TensorSpec((b_t, i, j), role="input")),
                        )
                    }
                )
                with batch_execute(run, prog) as exec_w:
                    for a_rows in t_slices:
                        acc = np.zeros((t_len, obs, obs), dtype=np.float64)
                        for b_rows in b_slices:
                            acc += exec_w(
                                {
                                    "W": np.ascontiguousarray(whitener[a_rows, b_rows]),
                                    "raw": np.ascontiguousarray(raw_mo[b_rows]),
                                }
                            ).outputs["df"]
                        if b_slices[0].start == 0:
                            out[a_rows] = acc
                        else:
                            out[a_rows] += acc
        return out

    central = {"whitener": whitener}
    factors = {}
    drivers = []
    overlap = overlap_seed.copy()
    is_rhf = len(reference.occupied) == 1
    shared_df = None
    for k, (spin, o) in enumerate(zip(("alpha", "beta"), occs, strict=True)):
        index = k if len(reference.occupied) == 2 else 0
        coeff = reference.coefficients[index]
        eps = reference.orbital_energies[index]
        bar = weights["obs_coefficients"][index]
        if is_rhf:
            bar = bar * 0.5
        if is_rhf and shared_df is not None:
            df = shared_df
        else:
            raw_obs = np.ascontiguousarray(raw["raw_coulomb_" + spin][:, :obs, :obs])
            df = _whiten_tiled(raw_obs)
            del raw_obs
            if is_rhf:
                shared_df = df
        central.update(
            {
                key + "_" + spin: val
                for key, val in (
                    ("coefficients", coeff),
                    ("occupied_energies", eps[:o]),
                    ("df", df),
                    ("fock_seed", np.zeros((obs, obs))),
                    ("overlap_seed", np.zeros((obs, obs))),
                )
            }
        )
        # Occupied C'=C_vir U + C_occ (-S'_oo/2). Virtual C' follows from
        # orthogonality, giving -S'_ov-U.T for its occupied component.
        mask = np.full((obs, obs), -0.5)
        mask[:o, o:] = -1.0
        mask[o:, :o] = 0.0
        if rot_metric_fn is not None:
            rotation, ov_bar = run_orbital_rot_metric_native(
                rot_metric_fn,
                coeff,
                bar,
                mask=mask,
                compute_rotation=True,
                compute_overlap=True,
                budget_bytes=aot_budget_bytes,
            )
            assert rotation is not None and ov_bar is not None
        else:
            inputs = {"coefficients": coeff, "frame_bar": bar}
            rotation = run(orbital_rotation_cotangent(obs, obs), inputs).outputs[
                "rotation_gradient"
            ]
            ov_bar = None
        torque = rotation - rotation.T
        if any(
            b.size and np.max(np.abs(b)) > 1e-8
            for b in (torque[:o, :o], torque[o:, o:])
        ):
            raise ValueError(
                "electronic observable fails occupied or virtual orbital gauge invariance"
            )
        if ov_bar is None:
            ov_bar = run(
                orbital_metric_cotangent(obs, obs), {**inputs, "metric_blocks": mask}
            ).outputs["overlap_bar"]
        overlap[:obs, :obs] += ov_bar
        if o:
            drivers.append(torque[o:, :o].ravel())
            if is_rhf and spin == "beta":
                for key in ("vo", "oo", "vv", "gap"):
                    factors[key + "_beta"] = factors[key + "_alpha"]
            else:
                vo_blk = df[:, o:, :o]
                oo_blk = df[:, :o, :o]
                vv_blk = df[:, o:, o:]
                if action_fn is not None:
                    vo_blk = np.ascontiguousarray(vo_blk)
                    oo_blk = np.ascontiguousarray(oo_blk)
                    vv_blk = np.ascontiguousarray(vv_blk)
                factors.update(
                    {
                        key + "_" + spin: val
                        for key, val in (
                            ("vo", vo_blk),
                            ("oo", oo_blk),
                            ("vv", vv_blk),
                            ("gap", np.ascontiguousarray(eps[o:, None] - eps[None, :o])),
                        )
                    }
                )
    driver = np.concatenate(drivers)

    def matvec(vector: np.ndarray) -> np.ndarray:
        if action_fn is not None:
            oa_s, ob_s = occs
            va_s, vb_s = obs - oa_s, obs - ob_s
            offset = 0
            trial_a = None
            trial_b = None
            if oa_s:
                count = va_s * oa_s
                trial_a = np.ascontiguousarray(
                    vector[offset : offset + count].reshape(va_s, oa_s)
                )
                offset += count
            if ob_s:
                count = vb_s * ob_s
                trial_b = np.ascontiguousarray(
                    vector[offset : offset + count].reshape(vb_s, ob_s)
                )
                offset += count
            out_a, out_b = run_cphf_action_native(
                action_fn,
                is_rhf=False,
                oa=oa_s,
                ob=ob_s,
                obs=obs,
                aux=aux,
                vo_a=factors.get("vo_alpha"),
                oo_a=factors.get("oo_alpha"),
                vv_a=factors.get("vv_alpha"),
                gap_a=factors.get("gap_alpha"),
                trial_a=trial_a,
                vo_b=factors.get("vo_beta"),
                oo_b=factors.get("oo_beta"),
                vv_b=factors.get("vv_beta"),
                gap_b=factors.get("gap_beta"),
                trial_b=trial_b,
                budget_bytes=aot_budget_bytes,
            )
            parts = []
            if out_a is not None:
                parts.append(out_a.ravel())
            if out_b is not None:
                parts.append(out_b.ravel())
            return np.concatenate(parts)

        trials = {}
        offset = 0
        for spin, o in zip(("alpha", "beta"), occs, strict=True):
            if o:
                count = (obs - o) * o
                trials["trial_" + spin] = np.ascontiguousarray(
                    vector[offset : offset + count].reshape(obs - o, o)
                )
                offset += count
        acc = {
            spin: np.zeros((obs - o, o), dtype=np.float64)
            for spin, o in zip(("alpha", "beta"), occs, strict=True)
            if o
        }
        for tile_len, slices in whiten_groups:
            action = unrestricted_df_orbital_action(*occs, obs, tile_len)
            with batch_execute(run, action) as exec_act:
                for rows in slices:
                    inputs = dict(trials)
                    for spin, o in zip(("alpha", "beta"), occs, strict=True):
                        if not o:
                            continue
                        inputs["vo_" + spin] = np.ascontiguousarray(factors["vo_" + spin][rows])
                        inputs["oo_" + spin] = np.ascontiguousarray(factors["oo_" + spin][rows])
                        inputs["vv_" + spin] = np.ascontiguousarray(factors["vv_" + spin][rows])
                        inputs["gap_" + spin] = (
                            factors["gap_" + spin]
                            if rows.start == 0
                            else np.zeros_like(factors["gap_" + spin])
                        )
                    out = exec_act(inputs).outputs
                    for spin in acc:
                        acc[spin] += out[spin]
        return np.concatenate([acc[spin].ravel() for spin, o in zip(("alpha", "beta"), occs, strict=True) if o])

    if reference.executed_backend == "cuda" and not use_matrix_free:
        z = _solve_orbital_response(
            space,
            factors,
            driver,
            np.empty((0, 3, r)),
            np.empty((0, 3)),
            (),
            0,
            mb << 20,
            library,
        ).rotations
        if len(reference.occupied) == 1:
            za, zb = np.split(z, 2)
            shared = (za + zb) * 0.5
            z = np.concatenate((shared, shared))
        if np.linalg.norm(matvec(z) - driver) > 1e-9 * max(1.0, np.linalg.norm(driver)):
            raise ValueError("DF-HF orbital response residual exceeds its tolerance")
    elif use_matrix_free and len(reference.occupied) == 1:
        from contextlib import ExitStack
        from inspect import signature

        from scipy.sparse.linalg import LinearOperator, minres

        oa = occs[0]
        va = obs - oa
        ra = va * oa
        g_rhf = drivers[0]
        if not np.allclose(g_rhf, drivers[1], atol=1e-10, rtol=1e-10):
            raise ValueError("RHF orbital response driver is not spin-symmetric")
        inv_diag = 1.0 / np.maximum(np.abs(factors["gap_alpha"].ravel()), 1e-2)
        precond = LinearOperator((ra, ra), matvec=lambda x: x * inv_diag, dtype=np.float64)
        if action_fn is not None:
            def rhf_matvec(vector: np.ndarray) -> np.ndarray:
                trial = np.ascontiguousarray(vector.reshape(va, oa))
                out_a, _ = run_cphf_action_native(
                    action_fn,
                    is_rhf=True,
                    oa=oa,
                    ob=0,
                    obs=obs,
                    aux=aux,
                    vo_a=factors["vo_alpha"],
                    oo_a=factors["oo_alpha"],
                    vv_a=factors["vv_alpha"],
                    gap_a=factors["gap_alpha"],
                    trial_a=trial,
                    budget_bytes=aot_budget_bytes,
                )
                assert out_a is not None
                return out_a.ravel()

            z_a, status = minres(
                LinearOperator((ra, ra), matvec=rhf_matvec, dtype=np.float64),
                g_rhf,
                M=precond,
                **{"rtol" if "rtol" in signature(minres).parameters else "tol": 1e-12},
                maxiter=max(128, 4 * ra),
            )
            if status:
                raise RuntimeError(
                    f"Matrix-free RHF DF-HF orbital response did not converge ({status})"
                )
            res_norm = np.linalg.norm(rhf_matvec(z_a) - g_rhf)
            if res_norm > 1e-9 * max(1.0, np.linalg.norm(g_rhf)):
                raise ValueError(
                    f"Matrix-free RHF DF-HF orbital response residual {res_norm:.3e} exceeds tolerance"
                )
        else:
            zero_gap = np.zeros_like(factors["gap_alpha"])
            with ExitStack() as stack:
                execs = [
                    (
                        slices,
                        stack.enter_context(
                            batch_execute(run, _rhf_df_orbital_action(oa, obs, tile_len))
                        ),
                    )
                    for tile_len, slices in whiten_groups
                ]

                def rhf_matvec(vector: np.ndarray) -> np.ndarray:
                    trial = np.ascontiguousarray(vector.reshape(va, oa))
                    acc = np.zeros((va, oa), dtype=np.float64)
                    for slices, exec_one in execs:
                        for rows in slices:
                            acc += exec_one(
                                {
                                    "vo": np.ascontiguousarray(factors["vo_alpha"][rows]),
                                    "oo": np.ascontiguousarray(factors["oo_alpha"][rows]),
                                    "vv": np.ascontiguousarray(factors["vv_alpha"][rows]),
                                    "gap": factors["gap_alpha"] if rows.start == 0 else zero_gap,
                                    "trial": trial,
                                }
                            ).outputs["rhf"]
                    return acc.ravel()

                z_a, status = minres(
                    LinearOperator((ra, ra), matvec=rhf_matvec, dtype=np.float64),
                    g_rhf,
                    M=precond,
                    **{"rtol" if "rtol" in signature(minres).parameters else "tol": 1e-12},
                    maxiter=max(128, 4 * ra),
                )
                if status:
                    raise RuntimeError(
                        f"Matrix-free RHF DF-HF orbital response did not converge ({status})"
                    )
                res_norm = np.linalg.norm(rhf_matvec(z_a) - g_rhf)
                if res_norm > 1e-9 * max(1.0, np.linalg.norm(g_rhf)):
                    raise ValueError(
                        f"Matrix-free RHF DF-HF orbital response residual {res_norm:.3e} exceeds tolerance"
                    )
        z = np.concatenate((z_a, z_a))
    else:
        from contextlib import ExitStack

        from scipy.sparse.linalg import LinearOperator

        gap_vec = np.concatenate(
            [
                factors["gap_" + spin].ravel()
                for spin, o in zip(("alpha", "beta"), occs, strict=True)
                if o
            ]
        )
        inv_diag = 1.0 / np.maximum(np.abs(gap_vec), 1e-2)
        precond = LinearOperator((r, r), matvec=lambda x: x * inv_diag, dtype=np.float64)
        if action_fn is not None:
            z = _coupled_response_minres(
                LinearOperator((r, r), matvec=matvec, dtype=np.float64),
                driver,
                precond,
                use_matrix_free,
            )
            if len(reference.occupied) == 1:
                za, zb = np.split(z, 2)
                shared = (za + zb) * 0.5
                z = np.concatenate((shared, shared))
            if np.linalg.norm(matvec(z) - driver) > 1e-9 * max(1.0, np.linalg.norm(driver)):
                raise ValueError("DF-HF orbital response residual exceeds its tolerance")
        else:
            zero_gaps = {
                spin: np.zeros_like(factors["gap_" + spin])
                for spin, o in zip(("alpha", "beta"), occs, strict=True)
                if o
            }
            with ExitStack() as stack:
                execs = [
                    (
                        slices,
                        stack.enter_context(
                            batch_execute(run, unrestricted_df_orbital_action(*occs, obs, tile_len))
                        ),
                    )
                    for tile_len, slices in whiten_groups
                ]

                def batched_matvec(vector: np.ndarray) -> np.ndarray:
                    trials = {}
                    offset = 0
                    for spin, o in zip(("alpha", "beta"), occs, strict=True):
                        if o:
                            count = (obs - o) * o
                            trials["trial_" + spin] = np.ascontiguousarray(
                                vector[offset : offset + count].reshape(obs - o, o)
                            )
                            offset += count
                    acc = {
                        spin: np.zeros((obs - o, o), dtype=np.float64)
                        for spin, o in zip(("alpha", "beta"), occs, strict=True)
                        if o
                    }
                    for slices, exec_one in execs:
                        for rows in slices:
                            inputs = dict(trials)
                            for spin, o in zip(("alpha", "beta"), occs, strict=True):
                                if not o:
                                    continue
                                inputs["vo_" + spin] = np.ascontiguousarray(factors["vo_" + spin][rows])
                                inputs["oo_" + spin] = np.ascontiguousarray(factors["oo_" + spin][rows])
                                inputs["vv_" + spin] = np.ascontiguousarray(factors["vv_" + spin][rows])
                                inputs["gap_" + spin] = (
                                    factors["gap_" + spin] if rows.start == 0 else zero_gaps[spin]
                                )
                            out = exec_one(inputs).outputs
                            for spin in acc:
                                acc[spin] += out[spin]
                    return np.concatenate(
                        [
                            acc[spin].ravel()
                            for spin, o in zip(("alpha", "beta"), occs, strict=True)
                            if o
                        ]
                    )

                z = _coupled_response_minres(
                    LinearOperator((r, r), matvec=batched_matvec, dtype=np.float64),
                    driver,
                    precond,
                    use_matrix_free,
                )
                if len(reference.occupied) == 1:
                    za, zb = np.split(z, 2)
                    shared = (za + zb) * 0.5
                    z = np.concatenate((shared, shared))
                if np.linalg.norm(batched_matvec(z) - driver) > 1e-9 * max(1.0, np.linalg.norm(driver)):
                    raise ValueError("DF-HF orbital response residual exceeds its tolerance")
    # With the direct occupied/virtual torque as driver the electronic observable
    # has zero explicit canonical Fock/energy seeds in the nuclear equation.
    bars = unrestricted_nuclear_source_weights(
        space,
        central,
        z,
        run,
        native_library=native_library if native_library is not None else library,
        host_reference=is_host,
        budget_bytes=aot_budget_bytes,
    )
    del central, factors, shared_df
    overlap[:obs, :obs] += bars["overlap_ao"]
    h = weights["hcore"].copy()
    h[:obs, :obs] += bars["hcore_ao"]
    three = (
        weights["three_center"]
        if weights["three_center"].flags.writeable
        else weights["three_center"].copy()
    )
    three[:, :obs, :obs] += bars["ao_coulomb"]
    return {
        "overlap": overlap,
        "hcore": h,
        "three_center": three,
        "metric": weights["metric"] + bars["metric/coulomb"],
    }
