"""CABS singles and its electronic source response on a physical focal DF-HF reference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from .elements import checked_integer

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable, Mapping

    from generativeqc_compiler.tensor import Execution, Program

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

    if not np.allclose(fock, fock.T, atol=2e-8, rtol=0):
        raise ValueError("physical extended DF Fock is asymmetric")
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


def reference_cabs_singles(
    request: FocalRequest,
    reference: FocalReference,
    one: FocalOneElectron,
    *,
    complement: np.ndarray,
    three_center: np.ndarray,
    metric: np.ndarray,
    whitener: np.ndarray,
    pullback: bool = True,
    relax: bool = False,
    relative_cutoff: float = 1e-9,
    budget_mib: int = 512,
    run: Callable | None = None,
    library: ctypes.CDLL | None = None,
) -> FocalCabsSingles:
    """Evaluate singles in the same canonical determinant, OBS/OptRI and full-rank JKFit.

    Supply the real-spherical combined-space three-center tensor [JKFit,AO,AO],
    its Coulomb metric, a full-rank W satisfying W.T W = metric^-1, and the
    OBS-orthogonal CABS columns returned by focal_cabs_projector. CPU reference
    uses FP64 TensorIR and NumPy LAPACK. CUDA requires a configured ordinary
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

    from ._f12_spin_ao import _orbital_sources
    from ._f12_spin_fock import spin_fock, spin_fock_pullback

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
    for arr, shape in (
        (s, (atomic, atomic)),
        (h, (atomic, atomic)),
        (complement, (atomic, rank)),
        (metric, (q, q)),
        (whitener, (q, q)),
        (three_center, (q, atomic, atomic)),
    ):
        if arr.dtype != np.float64 or arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError("CABS singles source shape, FP64 type or values disagree")
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
    dual = s @ complement
    if not np.allclose(
        complement.T @ dual, np.eye(rank), atol=2e-7, rtol=0
    ) or not np.allclose(dual[:obs], 0, atol=2e-7, rtol=0):
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
    space = (oa, ob, obs, obs + rank, q)
    ao = {
        "overlap_ao": s,
        "hcore_ao": h,
        "ao_coulomb": three_center,
        "metric/coulomb": metric,
        "metric_whitener": whitener,
    }
    for name, k in zip(("alpha", "beta"), spins, strict=True):
        frame = np.zeros((atomic, obs + rank))
        frame[:obs, :obs] = reference.coefficients[k]
        frame[:, obs:] = complement
        ao["coefficients_" + name] = frame
    projected, _ = _orbital_sources(
        space, ao, checked_run, None, projected=None, role="fock"
    )
    values = {
        k.removeprefix("source/"): v
        for k, v in projected.items()
        if k.startswith("source/")
    }
    fock_space = (oa, ob, obs + rank, q)
    focks = spin_fock(fock_space, values, checked_run)
    matrices = []
    energies = []
    errors = []
    seeds = {}
    for name, k, occ in zip(("alpha", "beta"), spins, (oa, ob), strict=True):
        f, energy, err, bar = _singles_from_fock(
            focks["fock_" + name],
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
            seeds["fock_" + name] = bar
    weights = None
    if pullback:
        seed = spin_fock_pullback(fock_space, values, seeds, checked_run)
        _, bars = _orbital_sources(
            space,
            ao,
            checked_run,
            {"source/" + k: v for k, v in seed.items()},
            projected=projected,
            role="fock",
        )
        assert bars is not None
        ga, gb = bars["coefficients_alpha"], bars["coefficients_beta"]
        cg = ga[:, obs:] + gb[:, obs:]
        transfer = checked_run(
            frame_kernel_cotangent(atomic, rank),
            {"overlap": s, "frame": complement, "frame_bar": cg},
        ).outputs
        if (
            np.max(np.abs(transfer["gauge_skew"]))
            > 2e-7 * max(1.0, np.linalg.norm(complement.T @ cg))
            or np.max(np.abs(transfer["reconstruction_residual"]))
            > 2e-7 * max(1.0, np.linalg.norm(cg))
            or not np.allclose(transfer["dual_gram"], np.eye(rank), atol=2e-7, rtol=0)
        ):
            raise ValueError(
                "CABS singles frame cotangent failed the orbital gauge or kernel reconstruction check"
            )
        cobs = np.stack([ga[:obs, :obs], gb[:obs, :obs]])
        weights = {
            "hcore": bars["hcore_ao"],
            "metric": bars["metric/coulomb"],
            "three_center": bars["ao_coulomb"],
            "obs_coefficients": cobs
            if expected == 2
            else np.sum(cobs, axis=0, keepdims=True),
            "cabs_kernel": transfer["kernel_bar"],
        }
        if relax:
            from ._f12_focal_singles_response import relaxed_weights

            weights = relaxed_weights(
                request,
                reference,
                one,
                complement,
                values,
                whitener,
                weights,
                checked_run,
                library,
                relative_cutoff,
                budget_mib,
            )
    return FocalCabsSingles(
        reference.energy + sum(energies),
        tuple(energies),
        np.stack(matrices),
        tuple(errors),
        weights,
    )
