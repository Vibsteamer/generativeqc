"""Selected retained 3C(FIX) F12 increment on a canonical all-active UHF focal root."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable, Mapping

    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_one_electron import FocalOneElectron
    from ._f12_focal_reference import FocalReference
    from .experimental.f12_focal import FocalRequest


@dataclass(frozen=True)
class FocalF12Sources:
    """Joint OBS/OptRI spherical sources for a physical fit and F12 exponent."""

    surface_id: str
    basis_id: str
    fit_id: str
    executed_backend: str
    beta: float | None
    integrals: Mapping[str, np.ndarray]
    whitener: np.ndarray


@dataclass(frozen=True)
class FocalF12Increment:
    energy: float
    active_pairs: int
    cabs_rank: int
    one_electron_weights: dict[str, np.ndarray]
    jkfit_weights: dict[str, np.ndarray]
    f12fit_weights: dict[str, np.ndarray]


def _canonical_occupied_virtual_blocks(
    m: np.ndarray, eps: np.ndarray, occ: int
) -> np.ndarray:
    """Reference same-space multiplier of the canonical Fock constraint."""
    gap = eps[:, None] - eps
    groups = np.arange(len(eps)) < occ
    same = groups[:, None] == groups
    torque = m - m.T
    unresolved = same & (np.abs(gap) <= 1e-8)
    if np.any(np.abs(torque[unresolved]) > 1e-9):
        raise ValueError("F12 increment depends on unresolved canonical gauge")
    return np.divide(-0.5 * torque, gap, out=np.zeros_like(m), where=same & ~unresolved)


def reference_f12_increment(
    request: FocalRequest,
    reference: FocalReference,
    one: FocalOneElectron,
    jk: FocalF12Sources,
    f12: FocalF12Sources,
    *,
    expected_cabs_rank: int | None = None,
    relative_cutoff: float = 1e-9,
    budget_mib: int = 1536,
    run: Callable | None = None,
    library: ctypes.CDLL | None = None,
    canonical_library: ctypes.CDLL | None = None,
) -> FocalF12Increment:
    """Calculate only the explicit fixed-cusp F12 pair energy and its relaxed AO weights.

    Each fitting source must use the same joint AO order as one_electron_sources.
    The JK panel contains raw Coulomb operators. The F12 panel also contains
    f12, f12_squared, f12_over_coulomb and double_commutator, with their metrics
    and 3-center tensors. All weights are positive energy cotangents. The
    energy excludes HF, ordinary UMP2 and CABS singles.
    """
    from generativeqc_compiler.method.canonical_frame_response import (
        orbital_rotation_cotangent,
    )
    from generativeqc_compiler.mp2.f12_frame_response import frame_kernel_cotangent
    from generativeqc_compiler.tensor import execute

    from ._canonical_frame import canonical_frame_seed
    from ._f12_focal_projector import focal_cabs_projector
    from ._f12_focal_singles_response import relaxed_weights
    from ._f12_spin_ao import _orbital_sources
    from ._f12_spin_fock import spin_fock, spin_fock_pullback
    from ._f12_unrestricted import unrestricted_mp2_f12_ao_response
    from .elements import checked_integer

    cuda = request.backend == "cuda"
    if (
        request.role != "mp2_f12_correlation"
        or request.root_id != "uhf"
        or request.correlation_space_id != "all-active"
        or request.backend not in ("cuda", "cpu_reference")
        or (cuda
        and run is None)
    ):
        raise ValueError(
            "selected F12 requires a canonical all-active UHF pair request with its backend runner"
        )
    budget = checked_integer(
        budget_mib, "F12 increment budget in MiB", low=1, high=(2**63 - 1) >> 20
    )
    executor = run if run is not None else partial(execute, max_bytes=budget << 20)

    def checked(g: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        out = executor(g, inputs)
        if out.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError(
                "F12 electronic graph differs from its requested backend"
            )
        return out

    key = (request.surface_id, request.basis.basis_id, request.backend)
    for obj in (reference, one, jk, f12):
        if (obj.surface_id, obj.basis_id, obj.executed_backend) != key:
            raise ValueError(
                "all F12 sources must share the same surface, basis assignments and backend"
            )
    if (jk.fit_id, jk.beta, f12.fit_id, f12.beta) != (
        request.basis.jk_fit.identity,
        None,
        request.basis.f12_fit.identity,
        request.basis.beta,
    ):
        raise ValueError("F12 pair source fitting bases or Gaussian exponent disagree")
    if (
        reference.jk_fit_id != jk.fit_id
        or reference.orbital_id != request.basis.orbital.identity
    ):
        raise ValueError("F12 reference uses different orbital or JKFit bases")
    n = one.obs_ao_count
    atomic = len(one.overlap)
    occs = reference.occupied
    if (
        len(occs) != 2
        or any(not 0 <= o < n for o in occs)
        or not any(occs)
        or not np.allclose(one.overlap[:n, :n], reference.overlap, atol=2e-8, rtol=0)
        or not np.allclose(one.hcore[:n, :n], reference.hcore, atol=2e-8, rtol=0)
    ):
        raise ValueError("incompatible F12/UHF orbital spaces")
    metrics = {
        "metric/coulomb": "metric",
        "metric/geminal": "metric_f12",
        "metric/squared_geminal": "metric_f12_squared",
        "metric/geminal_coulomb": "metric_f12_over_coulomb",
        "metric/double_commutator": "metric_double_commutator",
    }
    for panel, channels in (
        (jk, ("coulomb",)),
        (
            f12,
            ("coulomb", "f12", "f12_squared", "f12_over_coulomb", "double_commutator"),
        ),
    ):
        q = len(panel.whitener)
        fields = {}
        for name in channels:
            fields["ao_" + name] = (q, atomic, atomic)
            fields["metric" if name == "coulomb" else "metric_" + name] = (q, q)
        if set(panel.integrals) != set(fields) or any(
            a.shape != fields[k] or a.dtype != np.float64 or not np.isfinite(a).all()
            for k, a in panel.integrals.items()
        ):
            raise ValueError(
                "incorrect joint F12 metric/three-center operator inventory or physical dimensions"
            )
        w = panel.whitener
        if (
            not q
            or w.dtype != np.float64
            or w.shape != (q, q)
            or not np.allclose(
                w @ panel.integrals["metric"] @ w.T, np.eye(q), atol=2e-7, rtol=0
            )
        ):
            raise ValueError(
                "F12 sources require their own full-rank fitting metric whiteners"
            )
    proj = focal_cabs_projector(
        request,
        one,
        expected_rank=expected_cabs_rank,
        relative_cutoff=relative_cutoff,
        library=library,
    )
    complement = proj.frame
    ext = n + proj.rank
    space = (*occs, n, ext, len(f12.whitener))
    ao = {
        "overlap_ao": one.overlap,
        "hcore_ao": one.hcore,
        "metric_whitener": f12.whitener,
        **{k: v for k, v in f12.integrals.items() if k.startswith("ao_")},
        **{k: f12.integrals[v] for k, v in metrics.items()},
    }
    spins = ("alpha", "beta")
    for k, spin in enumerate(spins):
        frame = np.zeros((atomic, ext))
        frame[:n, :n] = reference.coefficients[k]
        frame[:, n:] = complement
        ao["coefficients_" + spin] = frame
    panel = {
        "ao_coulomb": jk.integrals["ao_coulomb"],
        "metric/coulomb": jk.integrals["metric"],
        "metric_whitener": jk.whitener,
    }
    terms, bars = unrestricted_mp2_f12_ao_response(
        space, ao, checked, jkfit=panel, component="f12"
    )
    # Apply the same-space canonical multipliers to the physical OBS Fock,
    # then relax the remaining occupied/virtual torque through coupled UHF.
    jkspace = (*occs, n, n, len(jk.whitener))
    fspace = (*occs, n, len(jk.whitener))
    obsao = {
        **panel,
        "ao_coulomb": panel["ao_coulomb"][:, :n, :n],
        "hcore_ao": reference.hcore,
        **{"coefficients_" + s: reference.coefficients[k] for k, s in enumerate(spins)},
    }
    projected, _ = _orbital_sources(
        jkspace, obsao, checked, None, projected=None, role="ordinary_fock"
    )
    values = {
        k.removeprefix("source/"): v
        for k, v in projected.items()
        if k.startswith("source/")
    }
    focks = spin_fock(fspace, values, checked)
    seeds = {}
    for k, spin in enumerate(spins):
        eps = reference.orbital_energies[k]
        if not np.allclose(focks["fock_" + spin], np.diag(eps), atol=2e-8, rtol=0):
            raise ValueError("F12 Fock mismatch with its native UHF reference")
        rotation = checked(
            orbital_rotation_cotangent(n, n),
            {
                "coefficients": reference.coefficients[k],
                "frame_bar": bars["coefficients_" + spin][:n, :n],
            },
        ).outputs["rotation_gradient"]
        seed = (
            canonical_frame_seed(
                rotation,
                eps,
                occs[k],
                maximum_bytes=budget << 20,
                library=canonical_library,
            )[0]
            if cuda
            else _canonical_occupied_virtual_blocks(rotation, eps, occs[k])
        )
        seeds["fock_" + spin] = seed
    seed = spin_fock_pullback(fspace, values, seeds, checked)
    _, extra = _orbital_sources(
        jkspace,
        obsao,
        checked,
        {"source/" + k: v for k, v in seed.items()},
        projected=projected,
        role="ordinary_fock",
    )
    assert extra is not None
    h = bars["hcore_ao"]
    h[:n, :n] += extra["hcore_ao"]
    three = bars["jkfit/ao_coulomb"]
    three[:, :n, :n] += extra["ao_coulomb"]
    cbar = np.stack(
        [bars["coefficients_" + s][:n, :n] + extra["coefficients_" + s] for s in spins]
    )
    cg = sum(bars["coefficients_" + s][:, n:] for s in spins)
    transfer = checked(
        frame_kernel_cotangent(atomic, proj.rank),
        {"overlap": one.overlap, "frame": complement, "frame_bar": cg},
    ).outputs
    if np.max(np.abs(transfer["gauge_skew"])) > 2e-7 * max(
        1.0, np.linalg.norm(complement.T @ cg)
    ) or np.max(np.abs(transfer["reconstruction_residual"])) > 2e-7 * max(
        1.0, np.linalg.norm(cg)
    ):
        raise ValueError("selected F12 energy depends on the CABS frame gauge")
    direct = {
        "hcore": h,
        "metric": bars["jkfit/metric/coulomb"] + extra["metric/coulomb"],
        "three_center": three,
        "obs_coefficients": cbar,
        "cabs_kernel": transfer["kernel_bar"],
    }
    relaxed = relaxed_weights(
        request,
        reference,
        one,
        complement,
        values,
        jk.whitener,
        direct,
        checked,
        library,
        relative_cutoff,
        budget,
    )
    return FocalF12Increment(
        terms.f12_increment,
        terms.active_pairs,
        proj.rank,
        {"overlap": relaxed["overlap"] + bars["overlap_ao"], "hcore": relaxed["hcore"]},
        {"metric": relaxed["metric"], "ao_coulomb": relaxed["three_center"]},
        {
            **{v: bars[k] for k, v in metrics.items()},
            **{k: v for k, v in bars.items() if k.startswith("ao_")},
        },
    )
