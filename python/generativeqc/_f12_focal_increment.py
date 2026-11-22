"""Selected retained 3C(FIX) F12 increment on a canonical all-active RHF or UHF focal root."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable, Mapping

    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_cabs import CABSProjector
    from ._f12_focal_one_electron import FocalOneElectron
    from ._f12_focal_reference import FocalReference
    from ._f12_focal_tplus_sources import SpinOccupied
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
    ao_loader: Callable[[str], np.ndarray] | None = None
    obs_coulomb: np.ndarray | None = None


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
    occupied_weights: SpinOccupied | None = None,
    ao_bar_sink: Callable[[str, np.ndarray], None] | None = None,
) -> FocalF12Increment:
    """Calculate only the explicit fixed-cusp F12 pair energy and its relaxed AO weights.

    Each fitting source must use the same joint AO order as one_electron_sources.
    The JK panel contains raw Coulomb operators. The F12 panel also contains
    f12, f12_squared, f12_over_coulomb and double_commutator, with their metrics
    and 3-center tensors. All weights are positive energy cotangents. The
    energy excludes HF, ordinary MP2 and CABS singles.
    Supplying occupied weights selects the corresponding linear combination of
    orbital increments and relaxes it on the same canonical reference.
    """
    from generativeqc_compiler.tensor import execute

    from ._f12_focal_frame import canonical_pair_frame
    from ._f12_frozen_core import resolve_frozen_occupied
    from ._f12_unrestricted import unrestricted_mp2_f12_ao_response
    from .elements import checked_integer

    cuda = request.backend == "cuda"
    if (
        request.role != "mp2_f12_correlation"
        or request.root_id not in ("rhf", "uhf")
        or request.backend not in ("cuda", "cpu_reference")
        or (cuda and run is None)
    ):
        raise ValueError(
            "selected F12 requires a canonical RHF or UHF pair request with its backend runner"
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

    if hasattr(executor, "batch"):
        from contextlib import contextmanager

        @contextmanager
        def checked_batch(g: Program):
            with executor.batch(g) as exec_one:
                def wrapped(inputs: Mapping[str, np.ndarray]) -> Execution:
                    out = exec_one(inputs)
                    if out.backend != (
                        "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
                    ):
                        raise RuntimeError(
                            "F12 electronic graph differs from its requested backend"
                        )
                    return out

                yield wrapped

        checked.batch = checked_batch  # type: ignore[attr-defined]

    space, ao, panel, proj = canonical_pair_frame(
        request,
        reference,
        one,
        jk,
        f12,
        expected_cabs_rank=expected_cabs_rank,
        relative_cutoff=relative_cutoff,
        library=library,
    )
    frozen = resolve_frozen_occupied(request, (space[0], space[1]))
    f12_weights_full = None
    if occupied_weights is not None:
        fa, fb = frozen
        oa, ob = space[0], space[1]
        if fa or fb:
            wa, wb = occupied_weights
            if len(wa) == oa - fa and len(wb) == ob - fb:
                f12_weights_full = (
                    np.concatenate((np.zeros(fa), wa)),
                    np.concatenate((np.zeros(fb), wb)),
                )
            else:
                f12_weights_full = occupied_weights
        else:
            f12_weights_full = occupied_weights
    terms, bars = unrestricted_mp2_f12_ao_response(
        space, ao, checked, frozen_occupied=frozen, jkfit=panel, component="f12",
        occupied_f12_weights=f12_weights_full, ao_bar_sink=ao_bar_sink,
    )
    weights = relaxed_f12_source_weights(
        request,
        reference,
        one,
        jk,
        panel,
        proj,
        bars,
        checked,
        budget=budget,
        relative_cutoff=relative_cutoff,
        library=library,
        canonical_library=canonical_library,
    )
    return FocalF12Increment(
        terms.f12_increment, terms.active_pairs, proj.rank, *weights
    )


def relaxed_f12_source_weights(
    request: FocalRequest,
    reference: FocalReference,
    one: FocalOneElectron,
    jk: FocalF12Sources,
    panel: dict[str, np.ndarray],
    proj: CABSProjector,
    bars: dict[str, np.ndarray],
    checked: Callable,
    *,
    budget: int,
    relative_cutoff: float,
    library: ctypes.CDLL | None,
    canonical_library: ctypes.CDLL | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Relax OBS and OptRI coefficient weights and retain the two raw integral fit identities."""
    from generativeqc_compiler.method.canonical_frame_response import (
        orbital_rotation_cotangent,
    )
    from generativeqc_compiler.mp2.f12_frame_response import frame_kernel_cotangent

    from ._canonical_frame import canonical_frame_seed
    from ._f12_focal_frame import METRIC_NAMES
    from ._f12_focal_singles_response import relaxed_weights
    from ._f12_spin_ao import _orbital_sources
    from ._f12_spin_fock import spin_fock, spin_fock_pullback

    cuda = request.backend == "cuda"
    spin_indices = reference._pair_spin_indices(request)
    occs = tuple(reference.occupied[i] for i in spin_indices)
    n = one.obs_ao_count
    atomic = len(one.overlap)
    complement = proj.frame
    spins = ("alpha", "beta")
    # Apply the same-space canonical multipliers to the physical OBS Fock,
    # then relax the remaining occupied/virtual torque through coupled HF.
    jkspace = (*occs, n, n, len(jk.whitener))
    fspace = (*occs, n, len(jk.whitener))
    if "__obs_coulomb__" in panel:
        obs_coulomb = panel["__obs_coulomb__"]
    elif "ao_coulomb" in panel:
        obs_coulomb = panel["ao_coulomb"][:, :n, :n]
    else:
        full_jk = panel["__ao_loader__"]("ao_coulomb")
        obs_coulomb = full_jk[:, :n, :n].copy()
        del full_jk
    obsao = {
        "metric/coulomb": panel["metric/coulomb"],
        "metric_whitener": panel["metric_whitener"],
        "ao_coulomb": obs_coulomb,
        "hcore_ao": reference.hcore,
        **{
            "coefficients_" + s: reference.coefficients[spin_indices[k]]
            for k, s in enumerate(spins)
        },
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
        eps = reference.orbital_energies[spin_indices[k]]
        if not np.allclose(focks["fock_" + spin], np.diag(eps), atol=2e-8, rtol=0):
            raise ValueError("F12 Fock mismatch with its native HF reference")
        rotation = checked(
            orbital_rotation_cotangent(n, n),
            {
                "coefficients": reference.coefficients[spin_indices[k]],
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
    cbar = np.zeros_like(reference.coefficients)
    for spin, index in zip(spins, spin_indices, strict=True):
        cbar[index] += (
            bars["coefficients_" + spin][:n, :n] + extra["coefficients_" + spin]
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
    return (
        {"overlap": relaxed["overlap"] + bars["overlap_ao"], "hcore": relaxed["hcore"]},
        {"metric": relaxed["metric"], "ao_coulomb": relaxed["three_center"]},
        {
            **{v: bars[k] for k, v in METRIC_NAMES.items()},
            **{k: v for k, v in bars.items() if k.startswith("ao_")},
        },
    )
