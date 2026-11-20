"""Construct a common canonical OBS/OptRI frame with separately attributed fitting sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._f12_focal_projector import focal_cabs_projector

if TYPE_CHECKING:
    import ctypes

    from ._f12_cabs import CABSProjector
    from ._f12_focal_increment import FocalF12Sources
    from ._f12_focal_one_electron import FocalOneElectron
    from ._f12_focal_reference import FocalReference
    from .experimental.f12_focal import FocalRequest

METRIC_NAMES = {
    "metric/coulomb": "metric",
    "metric/geminal": "metric_f12",
    "metric/squared_geminal": "metric_f12_squared",
    "metric/geminal_coulomb": "metric_f12_over_coulomb",
    "metric/double_commutator": "metric_double_commutator",
}


def canonical_pair_frame(
    request: FocalRequest,
    reference: FocalReference,
    one: FocalOneElectron,
    jk: FocalF12Sources,
    f12: FocalF12Sources,
    *,
    expected_cabs_rank: int | None,
    relative_cutoff: float,
    library: ctypes.CDLL | None,
) -> tuple[
    tuple[int, ...], dict[str, np.ndarray], dict[str, np.ndarray], CABSProjector
]:
    """Validate the two physical fitting metrics and build both extended spin frames."""
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
    spin_indices = reference._pair_spin_indices(request)
    occs = tuple(reference.occupied[k] for k in spin_indices)
    if (
        len(occs) != 2
        or any(not 0 <= o < n for o in occs)
        or not any(occs)
        or not np.allclose(one.overlap[:n, :n], reference.overlap, atol=2e-8, rtol=0)
        or not np.allclose(one.hcore[:n, :n], reference.hcore, atol=2e-8, rtol=0)
    ):
        raise ValueError("incompatible F12/HF orbital spaces")
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
        **{k: f12.integrals[v] for k, v in METRIC_NAMES.items()},
    }
    spins = ("alpha", "beta")
    for k, spin in enumerate(spins):
        frame = np.zeros((atomic, ext))
        frame[:n, :n] = reference.coefficients[spin_indices[k]]
        frame[:, n:] = complement
        ao["coefficients_" + spin] = frame
    panel = {
        "ao_coulomb": jk.integrals["ao_coulomb"],
        "metric/coulomb": jk.integrals["metric"],
        "metric_whitener": jk.whitener,
    }
    return space, ao, panel, proj
