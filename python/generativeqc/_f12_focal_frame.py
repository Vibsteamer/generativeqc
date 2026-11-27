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


def dense_pair_frame_key(
    request: FocalRequest,
    reference: FocalReference,
    one: FocalOneElectron,
    jk: FocalF12Sources,
    f12: FocalF12Sources,
    *,
    cabs_rank: int | None,
    relative_cutoff: float,
    library: ctypes.CDLL | None,
) -> tuple | None:
    """Identify the canonical frame and every dense operand retained by its projections."""
    from ._f12_spin_ao import _digest_array_full, _digest_dense_ao

    # Keep the rank contract identical to an uncached projector call.
    if isinstance(cabs_rank, (bool, np.bool_)) or not isinstance(cabs_rank, (int, np.integer)):
        return None
    if jk.ao_loader is not None or f12.ao_loader is not None:
        return None
    # The cached AO dictionaries retain these arrays by reference. Matching both
    # identity and content prevents reuse after mutation, including when the caller
    # replaces a mutated original with an equal copy of its initial contents.
    arrays = [
        ("overlap", one.overlap), ("hcore", one.hcore),
        ("jk_whitener", jk.whitener), ("f12_whitener", f12.whitener),
    ]
    for role, panel in (("jk", jk), ("f12", f12)):
        arrays.extend(((role, name), value) for name, value in sorted(panel.integrals.items()))
    digests = {}
    inventory = []
    for name, arr in arrays:
        if not isinstance(arr, np.ndarray) or arr.dtype != np.float64 or not arr.size:
            return None
        owner = id(arr)
        if owner not in digests:
            digest = _digest_dense_ao(arr)
            if digest is None:
                return None
            digests[owner] = digest
        inventory.append((name, owner, digests[owner]))

    return (
        request.atoms, request.charge, request.multiplicity, request.root_id,
        request.surface_id, request.basis.basis_id, request.role, request.backend,
        request.slater_normalization,
        tuple(
            (source.surface_id, source.basis_id, source.executed_backend)
            for source in (reference, one, jk, f12)
        ),
        reference.orbital_id, reference.jk_fit_id, reference.occupied,
        _digest_array_full(reference.coefficients), _digest_array_full(reference.overlap),
        _digest_array_full(reference.hcore), one.obs_ao_count,
        jk.fit_id, jk.beta, f12.fit_id, f12.beta, cabs_rank, relative_cutoff, library,
        tuple(inventory),
    )


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
        metric_fields = {
            ("metric" if name == "coulomb" else "metric_" + name): (q, q)
            for name in channels
        }
        if panel.ao_loader is not None:
            expected_fields = metric_fields
        else:
            expected_fields = {
                **metric_fields,
                **{("ao_" + name): (q, atomic, atomic) for name in channels},
            }
        if set(panel.integrals) != set(expected_fields) or any(
            a.shape != expected_fields[k] or a.dtype != np.float64 or not np.isfinite(a).all()
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
    ao: dict = {
        "overlap_ao": one.overlap,
        "hcore_ao": one.hcore,
        "metric_whitener": f12.whitener,
        **{k: v for k, v in f12.integrals.items() if k.startswith("ao_")},
        **{k: f12.integrals[v] for k, v in METRIC_NAMES.items()},
    }
    if f12.ao_loader is not None:
        ao["__ao_loader__"] = f12.ao_loader
    spins = ("alpha", "beta")
    for k, spin in enumerate(spins):
        frame = np.zeros((atomic, ext))
        frame[:n, :n] = reference.coefficients[spin_indices[k]]
        frame[:, n:] = complement
        ao["coefficients_" + spin] = frame
    panel: dict = {
        "metric/coulomb": jk.integrals["metric"],
        "metric_whitener": jk.whitener,
    }
    if jk.ao_loader is not None:
        panel["__ao_loader__"] = jk.ao_loader
        if jk.obs_coulomb is not None:
            panel["__obs_coulomb__"] = jk.obs_coulomb
    else:
        panel["ao_coulomb"] = jk.integrals["ao_coulomb"]
    return space, ao, panel, proj
