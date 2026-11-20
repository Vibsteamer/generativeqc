"""Complementary orbital space on the joint overlap of a public focal request."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._f12_cabs import CABSProjector, cabs_projector
from .elements import checked_integer

if TYPE_CHECKING:
    import ctypes

    from ._f12_focal_one_electron import FocalOneElectron
    from .experimental.f12_focal import FocalRequest


def _reference(
    s: np.ndarray,
    obs: int,
    cutoff: float,
    guard: float,
    expected: int | None,
    seed: np.ndarray | None,
) -> CABSProjector:
    from generativeqc_compiler.method.matrix_function import SymmetricMatrixFunctionSpec
    from scipy.linalg import cho_factor, cho_solve

    # Lift each OptRI function out of the full OBS before applying the cutoff.
    a, cross = s[:obs, :obs], s[:obs, obs:]
    values = np.linalg.eigvalsh(a)
    if values[0] <= guard * values[-1]:
        raise ValueError("unresolved OBS overlap rank")
    chol = cho_factor(a, lower=True)
    x = -cho_solve(chol, cross)
    lift = np.vstack((x, np.eye(x.shape[1])))
    gram = s[obs:, obs:] + cross.T @ x
    gram = 0.5 * (gram + gram.T)
    spec = SymmetricMatrixFunctionSpec(
        len(gram), "CABS Schur complement", cutoff, guard, function="pseudoinverse"
    )
    state = spec.prepare(gram, expected_rank=expected)
    values = state.eigenvalues
    retained = np.asarray(state.retained)
    largest = values[-1]
    aux_scale = float(np.max(np.diag(s)[obs:]))
    if largest <= guard * aux_scale or state.relative_gap * largest <= guard * max(
        largest, aux_scale
    ):
        raise ValueError("unresolved CABS projected overlap")
    frame = lift @ (state.vectors[:, retained] / np.sqrt(values[retained]))
    full = state.rank == len(gram)
    if full:
        # A physical Gram correction controls roundoff when lift columns nearly cancel.
        wide = frame.astype(np.longdouble)
        metric = wide.T @ (s.astype(np.longdouble) @ wide)
        frame = frame @ (1.5 * np.eye(state.rank) - 0.5 * metric.astype(np.float64))
    kernel = frame @ frame.T
    bar = None
    if seed is not None:
        u = 0.5 * (seed + seed.T)
        if full:
            # A small U K may result from cancellation between large elements
            # of K. Its leading digits also determine the subsequent K U K.
            # The OBS lift term uses the same product.
            wide_kernel = kernel.astype(np.longdouble)
            uk = u.astype(np.longdouble) @ wide_kernel
            bar = np.asarray(-(wide_kernel @ uk), dtype=np.float64)
            rhs = np.asarray(uk[:obs], dtype=np.float64)
        else:
            # The fixed-rank derivative includes rotations across the cutoff.
            bar = lift @ state.vjp(lift.T @ u @ lift) @ lift.T
            rhs = u[:obs] @ kernel
        cross_bar = cho_solve(chol, rhs)
        bar[:obs] -= cross_bar
        bar[:, :obs] -= cross_bar.T
        bar = 0.5 * (bar + bar.T)
    return CABSProjector(
        kernel, np.ascontiguousarray(frame), bar, state.rank, state.relative_gap
    )


def focal_cabs_projector(
    request: FocalRequest,
    one: FocalOneElectron,
    *,
    expected_rank: int | None = None,
    relative_cutoff: float = 1e-9,
    branch_guard: float = 1e-12,
    cotangent: np.ndarray | None = None,
    library: ctypes.CDLL | None = None,
) -> CABSProjector:
    """Build the metric-orthonormal OptRI complement to the complete orbital basis.

    Reuse the combined overlap from one_electron_sources on the same request.
    The columns follow its OBS-first AO ordering. Pass the central rank on
    displaced geometries. A kernel cotangent U requests the symmetric overlap
    weight for d<U, C_CABS C_CABS.T>. Contract that weight with one.d_overlap
    for Cartesian derivatives, including any explicit geometry dependence of U
    separately. The CUDA backend uses the native projector on device 0.
    """
    if (
        request.role not in ("hf_cabs", "mp2_f12_correlation")
        or request.basis.cabs is None
    ):
        raise ValueError(
            "a focal CABS projector requires a declared complementary basis"
        )
    if (
        one.surface_id != request.surface_id
        or one.basis_id != request.basis.basis_id
        or one.executed_backend != request.backend
    ):
        raise ValueError(
            "CABS overlap and focal request refer to different surfaces, bases or backends"
        )
    s = np.asarray(one.overlap)
    if (
        s.dtype != np.float64
        or s.ndim != 2
        or s.shape[0] != s.shape[1]
        or not np.isfinite(s).all()
    ):
        raise ValueError("CABS requires a finite square FP64 overlap")
    if not np.allclose(s, s.T, rtol=1e-12, atol=1e-12):
        raise ValueError("CABS overlap must be symmetric")
    obs = checked_integer(one.obs_ao_count, "orbital dimension", low=1, high=len(s) - 1)
    if expected_rank is not None:
        expected_rank = checked_integer(
            expected_rank, "expected CABS rank", low=1, high=len(s) - obs
        )
    if (
        isinstance(relative_cutoff, (bool, np.bool_))
        or isinstance(branch_guard, (bool, np.bool_))
        or not 0 < branch_guard < relative_cutoff < 1
    ):
        raise ValueError("CABS requires 0 < branch_guard < relative_cutoff < 1")
    if cotangent is not None:
        cotangent = np.asarray(cotangent)
        if (
            cotangent.dtype != np.float64
            or cotangent.shape != s.shape
            or not np.isfinite(cotangent).all()
        ):
            raise ValueError(
                "CABS kernel cotangent requires a finite FP64 array of the overlap shape"
            )
    if request.backend == "cuda":
        return cabs_projector(
            s,
            obs,
            expected_rank=expected_rank,
            relative_cutoff=relative_cutoff,
            branch_guard=branch_guard,
            cotangent=cotangent,
            library=library,
        )
    if request.backend != "cpu_reference":
        raise ValueError("CABS requires an explicit cuda or cpu_reference backend")
    return _reference(
        s, obs, float(relative_cutoff), float(branch_guard), expected_rank, cotangent
    )
