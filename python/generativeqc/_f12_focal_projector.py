"""Complementary orbital space on the joint overlap of a public focal request."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._f12_cabs import CABSProjector, cabs_projector
from .elements import checked_integer

if TYPE_CHECKING:
    import ctypes

    from ._f12_focal_molecular import MolecularFocalProvider
    from ._f12_focal_one_electron import FocalOneElectron
    from ._f12_focal_reference import FocalReference
    from .experimental.f12_focal import FocalRequest


def checked_reference_cabs_rank(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    reference: FocalReference,
    expected_rank: int | None,
) -> int:
    """Measure the CABS rank for correlation shortcuts that skip pair sources."""
    from ._f12_focal_one_electron import one_electron_sources

    one = one_electron_sources(
        request,
        library=provider.one_electron_library or provider.shell_library,
        derivatives=False,
        budget_mib=provider.budget_mib,
    )
    n = one.obs_ao_count
    if (
        reference.overlap.shape != (n, n)
        or reference.hcore.shape != (n, n)
        or not np.allclose(reference.overlap, one.overlap[:n, :n], atol=2e-8, rtol=0)
        or not np.allclose(reference.hcore, one.hcore[:n, :n], atol=2e-8, rtol=0)
    ):
        raise ValueError("CABS and reference orbital operators disagree")
    return focal_cabs_projector(
        request,
        one,
        expected_rank=expected_rank,
        relative_cutoff=provider.relative_cutoff,
        library=provider.response_library,
    ).rank


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


def _two_sum(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return s, e


def _dd_add(
    a_hi: np.ndarray, a_lo: np.ndarray, b_hi: np.ndarray, b_lo: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    s, e = _two_sum(a_hi, b_hi)
    return _two_sum(s, a_lo + b_lo + e)


EXACT_PRODUCT_MULTIPLY_ADD_LIMIT = 1 << 18


class CompensatedProductWorkLimit(RuntimeError):
    """A compensated product needs exact rational work above its bound."""

    def __init__(self, shape: tuple[int, int, int], trigger: str) -> None:
        """Record the refused product shape, its trigger and multiply-add count."""
        m, k, n = shape
        self.shape = shape
        self.trigger = trigger
        self.multiply_adds = m * k * n
        self.limit = EXACT_PRODUCT_MULTIPLY_ADD_LIMIT
        super().__init__(
            f"compensated {m}x{k} @ {k}x{n} product left the Ozaki slice range "
            f"({trigger}); exact rational accumulation needs {self.multiply_adds} "
            f"multiply-adds, above the bound of {self.limit}"
        )


def _exact_mm_dd(
    a_arr: np.ndarray, b_arr: np.ndarray, trigger: str
) -> tuple[np.ndarray, np.ndarray]:
    """Exact rational matrix product rounded to double-double (hi, lo).

    Each rational multiply-add costs microseconds, so a product above
    EXACT_PRODUCT_MULTIPLY_ADD_LIMIT is refused before any rational work.
    """
    from fractions import Fraction

    m, k = a_arr.shape
    _, n = b_arr.shape
    if m * k * n > EXACT_PRODUCT_MULTIPLY_ADD_LIMIT:
        raise CompensatedProductWorkLimit((m, k, n), trigger)
    a_frac = [[Fraction(float(a_arr[i, t])) if a_arr[i, t] != 0.0 else None for t in range(k)] for i in range(m)]
    b_frac = [[Fraction(float(b_arr[t, j])) if b_arr[t, j] != 0.0 else None for j in range(n)] for t in range(k)]
    hi = np.zeros((m, n), dtype=np.float64)
    lo = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        row = a_frac[i]
        for j in range(n):
            total = Fraction(0)
            for t in range(k):
                af = row[t]
                if af is None:
                    continue
                bf = b_frac[t][j]
                if bf is None:
                    continue
                total += af * bf
            h_val = float(total)
            if not np.isfinite(h_val):
                raise OverflowError("Compensated matrix multiplication overflowed FP64")
            l_val = float(total - Fraction(h_val))
            hi[i, j] = h_val
            lo[i, j] = l_val
    return hi, lo


def _ozaki_mm(
    a: np.ndarray, b: np.ndarray, *, slices: int = 4, max_slices: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Ozaki compensated BLAS matrix multiplication returning double-double (hi, lo).

    Dynamically checks the unsliced remainder bound and extends Ozaki slices (or
    escalates to exact rational accumulation) whenever the omitted tail exceeds
    1e-15 * |result| or intermediate scaling would overflow. An escalation above
    the exact work bound raises CompensatedProductWorkLimit.
    """
    a_arr = np.ascontiguousarray(a, dtype=np.float64)
    b_arr = np.ascontiguousarray(b, dtype=np.float64)
    if a_arr.ndim != 2 or b_arr.ndim != 2:
        raise ValueError(f"Ozaki matrix multiplication requires 2D inputs: {a_arr.shape}, {b_arr.shape}")
    m, k = a_arr.shape
    k_b, n = b_arr.shape
    if k != k_b:
        raise ValueError(f"Incompatible Ozaki matrix shapes: {a_arr.shape} vs {b_arr.shape}")
    if m == 0 or n == 0 or k == 0:
        z = np.zeros((m, n), dtype=np.float64)
        return z, np.zeros_like(z)
    if not np.isfinite(a_arr).all() or not np.isfinite(b_arr).all():
        raise ValueError("Nonfinite input to compensated matrix multiplication")

    k_bits = max(1, (k - 1).bit_length())
    bits = min(21, max(16, (53 - k_bits) // 2))
    shift = 53 - bits
    eff_max_slices = max(int(slices), int(max_slices))

    def _next_slice(rem: np.ndarray, axis_max: int) -> tuple[np.ndarray | None, np.ndarray, bool]:
        mx = np.max(np.abs(rem), axis=axis_max, keepdims=True)
        nz = mx > 0
        if not np.any(nz):
            return None, rem, False
        _, exp = np.frexp(np.where(nz, mx, 1.0))
        if int(np.max(exp)) + shift >= 1023:
            return None, rem, True
        tau = np.where(nz, np.ldexp(1.0, exp + shift), 0.0)
        s = (rem + tau) - tau
        if not np.isfinite(s).all():
            return None, rem, True
        return s, rem - s, False

    def _split_initial(x: np.ndarray, axis_max: int) -> tuple[list[np.ndarray], np.ndarray, bool]:
        rem = x.copy()
        parts: list[np.ndarray] = []
        for _ in range(slices):
            s, rem, overflow = _next_slice(rem, axis_max)
            if overflow:
                return parts, rem, True
            if s is None:
                break
            parts.append(s)
        return parts, rem, False

    ap, rem_a, ov_a = _split_initial(a_arr, 1)
    bp, rem_b, ov_b = _split_initial(b_arr, 0)
    if ov_a or ov_b:
        return _exact_mm_dd(a_arr, b_arr, "scaling_overflow")

    hi = np.zeros((m, n), dtype=np.float64)
    lo = np.zeros((m, n), dtype=np.float64)
    if not ap or not bp:
        return hi, lo

    def _accum(prod: np.ndarray) -> bool:
        nonlocal hi, lo
        if not np.isfinite(prod).all():
            return False
        s, e = _two_sum(hi, prod)
        hi, lo = _two_sum(s, lo + e)
        return bool(np.isfinite(hi).all() and np.isfinite(lo).all())

    na, nb = len(ap), len(bp)
    for diag in range(na + nb - 1):
        for p in range(max(0, diag - nb + 1), min(na, diag + 1)):
            q = diag - p
            if not _accum(ap[p] @ bp[q]):
                return _exact_mm_dd(a_arr, b_arr, "partial_product_overflow")

    abs_a: np.ndarray | None = None
    abs_b: np.ndarray | None = None

    def _remainder_exceeds_tol() -> bool:
        nonlocal abs_a, abs_b
        has_ra = bool(np.any(rem_a != 0.0))
        has_rb = bool(np.any(rem_b != 0.0))
        if not has_ra and not has_rb:
            return False
        res = hi + lo
        bound = np.zeros((m, n), dtype=np.float64)
        if has_ra:
            if abs_b is None:
                abs_b = np.abs(b_arr)
            bound = bound + (np.abs(rem_a) @ abs_b)
        if has_rb:
            if abs_a is None:
                abs_a = np.abs(a_arr)
            bound = bound + (abs_a @ np.abs(rem_b))
        if not np.isfinite(bound).all():
            return True
        return bool(np.any(bound > 1e-15 * np.abs(res)))

    while _remainder_exceeds_tol():
        extended = False
        if np.any(rem_a != 0.0) and len(ap) < eff_max_slices:
            sa, rem_a, ov = _next_slice(rem_a, 1)
            if ov:
                return _exact_mm_dd(a_arr, b_arr, "scaling_overflow")
            if sa is not None:
                ap.append(sa)
                extended = True
                for b_part in bp:
                    if not _accum(sa @ b_part):
                        return _exact_mm_dd(a_arr, b_arr, "partial_product_overflow")
        if np.any(rem_b != 0.0) and len(bp) < eff_max_slices:
            sb, rem_b, ov = _next_slice(rem_b, 0)
            if ov:
                return _exact_mm_dd(a_arr, b_arr, "scaling_overflow")
            if sb is not None:
                bp.append(sb)
                extended = True
                for a_part in ap:
                    if not _accum(a_part @ sb):
                        return _exact_mm_dd(a_arr, b_arr, "partial_product_overflow")
        if not extended:
            return _exact_mm_dd(a_arr, b_arr, "slice_limit")

    return _two_sum(hi, lo)


def _ozaki_mm_left_dd(
    a: np.ndarray, b_hi: np.ndarray, b_lo: np.ndarray, *, slices: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a @ (b_hi + b_lo) in double-double returning (hi, lo)."""
    h1, l1 = _ozaki_mm(a, b_hi, slices=slices)
    h2, l2 = _ozaki_mm(a, b_lo, slices=max(2, slices - 1))
    return _dd_add(h1, l1, h2, l2)


def compensated_cabs_orthonormality(
    overlap: np.ndarray, obs: int, frame: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return compensated (C^T S C, (S C)[:obs]) for ill-conditioned diffuse CABS frames."""
    dual_hi, dual_lo = _ozaki_mm(overlap, frame, slices=4)
    gram_hi, gram_lo = _ozaki_mm_left_dd(frame.T, dual_hi, dual_lo, slices=4)
    return gram_hi + gram_lo, (dual_hi + dual_lo)[:obs]


def compensated_cabs_frame_cotangent_checks(
    overlap: np.ndarray,
    obs: int,
    frame: np.ndarray,
    frame_bar: np.ndarray,
    transfer: dict[str, np.ndarray],
    *,
    check_dual_gram: bool = True,
    atol: float = 2e-7,
) -> bool:
    """Validate gauge_skew, reconstruction_residual, and dual_gram with Ozaki-4 fallback.

    Preserves exact FP64 fast path for normal bases; on diffuse aug-QZ/OptRI frames
    where plain FP64 C^T(SC) loses >9 digits to cancellation, verifies the same
    strict tolerance in double-double before rejecting.
    """
    rank = frame.shape[1]
    skew_ok = (
        np.isfinite(transfer["gauge_skew"]).all()
        and float(np.max(np.abs(transfer["gauge_skew"])))
        <= atol * max(1.0, float(np.linalg.norm(frame.T @ frame_bar)))
    )
    recon_ok = (
        np.isfinite(transfer["reconstruction_residual"]).all()
        and float(np.max(np.abs(transfer["reconstruction_residual"])))
        <= atol * max(1.0, float(np.linalg.norm(frame_bar)))
    )
    dual_ok = (
        True
        if not check_dual_gram
        else (
            np.isfinite(transfer["dual_gram"]).all()
            and bool(np.allclose(transfer["dual_gram"], np.eye(rank), atol=atol, rtol=0))
        )
    )
    if skew_ok and recon_ok and dual_ok:
        return True

    if not skew_ok:
        m_hi, m_lo = _ozaki_mm(frame.T, frame_bar, slices=4)
        m_oz = m_hi + m_lo
        skew_ok = (
            np.isfinite(m_oz).all()
            and float(np.max(np.abs(m_oz - m_oz.T)))
            <= atol * max(1.0, float(np.linalg.norm(m_oz)))
        )
    if not dual_ok:
        gram_oz, obs_dual_oz = compensated_cabs_orthonormality(overlap, obs, frame)
        dual_ok = (
            np.isfinite(gram_oz).all()
            and np.isfinite(obs_dual_oz).all()
            and bool(np.allclose(gram_oz, np.eye(rank), atol=atol, rtol=0))
            and bool(np.allclose(obs_dual_oz, 0.0, atol=atol, rtol=0))
        )
    if not recon_ok:
        kb = transfer.get("kernel_bar")
        if kb is not None and np.isfinite(kb).all():
            rec_hi, rec_lo = _ozaki_mm(kb, frame, slices=4)
            rec = 2.0 * (rec_hi + rec_lo) - frame_bar
            recon_ok = (
                np.isfinite(rec).all()
                and float(np.max(np.abs(rec)))
                <= atol * max(1.0, float(np.linalg.norm(frame_bar)))
            )
    return bool(skew_ok and recon_ok and dual_ok)


def full_rank_cabs_overlap_bar(
    overlap: np.ndarray,
    obs: int,
    frame: np.ndarray,
    frame_bar: np.ndarray,
) -> np.ndarray:
    """Compensated direct full-rank CABS overlap cotangent bar_S.

    For a complete OBS-orthogonal CABS frame (rank == atomic - obs) satisfying
    C^T S C = I, S_{o,:} C = 0 and symmetric gauge moment C^T bar_C:
        bar_S = -sym(O @ bar_C @ C^T) - 0.5 * C @ sym(C^T @ bar_C) @ C^T,
    with O = embed(S_{oo}^{-1}). Evaluating the contractions and their sum in
    double-double (Ozaki-4) avoids the ||K||^2 cancellation loss of the indirect
    bar_C -> kernel_bar -> -K kernel_bar K route on diffuse ha-Q-F12 bases.
    """
    from scipy.linalg import cho_factor, cho_solve

    atomic, rank = frame.shape
    if rank != atomic - obs or frame_bar.shape != (atomic, rank):
        raise ValueError(
            "Direct full-rank CABS overlap cotangent requires rank == atomic - obs"
        )
    m_hi, m_lo = _ozaki_mm(frame.T, frame_bar, slices=4)
    sym_hi, sym_lo = _dd_add(0.5 * m_hi, 0.5 * m_lo, 0.5 * m_hi.T, 0.5 * m_lo.T)

    chol = cho_factor(overlap[:obs, :obs], lower=True)
    o_g_obs = cho_solve(chol, frame_bar[:obs, :])
    cr_obs_hi, cr_obs_lo = _ozaki_mm(o_g_obs, frame.T, slices=4)
    cr_hi = np.zeros((atomic, atomic), dtype=np.float64)
    cr_lo = np.zeros((atomic, atomic), dtype=np.float64)
    cr_hi[:obs, :] = cr_obs_hi
    cr_lo[:obs, :] = cr_obs_lo

    mc_hi, mc_lo = _ozaki_mm(sym_hi, frame.T, slices=4)
    mc2_hi, mc2_lo = _ozaki_mm(sym_lo, frame.T, slices=3)
    mc_sum_hi, mc_sum_lo = _dd_add(mc_hi, mc_lo, mc2_hi, mc2_lo)
    norm_hi, norm_lo = _ozaki_mm_left_dd(frame, mc_sum_hi, mc_sum_lo, slices=4)

    s_cr_hi, s_cr_lo = _dd_add(cr_hi, cr_lo, cr_hi.T, cr_lo.T)
    s_nm_hi, s_nm_lo = _dd_add(norm_hi, norm_lo, norm_hi.T, norm_lo.T)
    bar_hi, bar_lo = _dd_add(
        -0.5 * s_cr_hi, -0.5 * s_cr_lo, -0.25 * s_nm_hi, -0.25 * s_nm_lo
    )
    bar_s = bar_hi + bar_lo
    return 0.5 * (bar_s + bar_s.T)
