"""Relax a CABS correction through the canonical DF-HF occupied/virtual response."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable, Mapping

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
) -> dict[str, np.ndarray]:
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

    from ._f12_focal_projector import focal_cabs_projector
    from .f12_orbital_response import (
        _solve_orbital_response,
        unrestricted_nuclear_source_weights,
    )

    mb = checked_integer(
        budget_mib, "orbital response budget in MiB", low=1, high=(2**63 - 1) >> 20
    )
    occs = (
        reference.occupied if len(reference.occupied) == 2 else reference.occupied * 2
    )
    obs = one.obs_ao_count
    aux = len(whitener)
    space = (*occs, obs, aux)
    r = sum((obs - o) * o for o in occs)
    if request.backend == "cuda" and 8 * r * r > mb << 20:
        raise MemoryError("dense DF-HF orbital response exceeds the requested budget")
    projector = focal_cabs_projector(
        request,
        one,
        expected_rank=frame.shape[1],
        relative_cutoff=relative_cutoff,
        cotangent=weights["cabs_kernel"],
        library=library,
    )
    if not np.allclose(frame @ frame.T, projector.kernel, rtol=3e-7, atol=3e-7):
        raise ValueError(
            "CABS response must use the same fixed-rank projector as the energy"
        )
    ss = IndexSpace("jkfit", "auxiliary", aux)
    mo = IndexSpace("obs", "orbital", obs)
    a, b = Index("A", ss), Index("B", ss)
    i, j = Index("p", mo), Index("q", mo)
    g = input_tensor("raw", TensorSpec((b, i, j), role="input"))
    w = input_tensor("W", TensorSpec((a, b), role="input"))
    whiten = Program({"df": einsum("AB,Bpq->Apq", w, g)})
    central = {"whitener": whitener}
    factors = {}
    drivers = []
    overlap = projector.overlap_bar.copy()
    for k, (spin, o) in enumerate(zip(("alpha", "beta"), occs, strict=True)):
        index = k if len(reference.occupied) == 2 else 0
        coeff = reference.coefficients[index]
        eps = reference.orbital_energies[index]
        bar = weights["obs_coefficients"][index]
        if len(reference.occupied) == 1:
            bar = bar * 0.5
        df = run(
            whiten,
            {
                "W": whitener,
                "raw": np.ascontiguousarray(raw["raw_coulomb_" + spin][:, :obs, :obs]),
            },
        ).outputs["df"]
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
        inputs = {"coefficients": coeff, "frame_bar": bar}
        rotation = run(orbital_rotation_cotangent(obs, obs), inputs).outputs[
            "rotation_gradient"
        ]
        torque = rotation - rotation.T
        if any(
            b.size and np.max(np.abs(b)) > 1e-8
            for b in (torque[:o, :o], torque[o:, o:])
        ):
            raise ValueError(
                "CABS observable fails occupied or virtual orbital gauge invariance"
            )
        # Occupied C'=C_vir U + C_occ (-S'_oo/2). Virtual C' follows from
        # orthogonality, giving -S'_ov-U.T for its occupied component.
        mask = np.full((obs, obs), -0.5)
        mask[:o, o:] = -1.0
        mask[o:, :o] = 0.0
        overlap[:obs, :obs] += run(
            orbital_metric_cotangent(obs, obs), {**inputs, "metric_blocks": mask}
        ).outputs["overlap_bar"]
        if o:
            drivers.append(torque[o:, :o].ravel())
            factors.update(
                {
                    key + "_" + spin: val
                    for key, val in (
                        ("vo", df[:, o:, :o]),
                        ("oo", df[:, :o, :o]),
                        ("vv", df[:, o:, o:]),
                        ("gap", eps[o:, None] - eps[None, :o]),
                    )
                }
            )
    driver = np.concatenate(drivers)
    action = unrestricted_df_orbital_action(*space)

    def matvec(vector: np.ndarray) -> np.ndarray:
        inputs = dict(factors)
        offset = 0
        for spin, o in zip(("alpha", "beta"), occs, strict=True):
            if o:
                count = (obs - o) * o
                inputs["trial_" + spin] = np.ascontiguousarray(
                    vector[offset : offset + count].reshape(obs - o, o)
                )
                offset += count
        out = run(action, inputs).outputs
        return np.concatenate(
            [
                out[spin].ravel()
                for spin, o in zip(("alpha", "beta"), occs, strict=True)
                if o
            ]
        )

    if request.backend == "cuda":
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
    else:
        from inspect import signature

        from scipy.sparse.linalg import LinearOperator, minres

        z, status = minres(
            LinearOperator((r, r), matvec=matvec, dtype=np.float64),
            driver,
            **{"rtol" if "rtol" in signature(minres).parameters else "tol": 1e-11},
            maxiter=max(64, 4 * r),
        )
        if status:
            raise RuntimeError(
                f"DF-HF coupled orbital response did not converge ({status})"
            )
    if np.linalg.norm(matvec(z) - driver) > 1e-9 * max(1.0, np.linalg.norm(driver)):
        raise ValueError("DF-HF orbital response residual exceeds its tolerance")
    # With the direct occupied/virtual torque as driver the electronic observable
    # has zero explicit canonical Fock/energy seeds in the nuclear equation.
    bars = unrestricted_nuclear_source_weights(space, central, z, run)
    overlap[:obs, :obs] += bars["overlap_ao"]
    h = weights["hcore"].copy()
    h[:obs, :obs] += bars["hcore_ao"]
    three = weights["three_center"].copy()
    three[:, :obs, :obs] += bars["ao_coulomb"]
    return {
        "overlap": overlap,
        "hcore": h,
        "three_center": three,
        "metric": weights["metric"] + bars["metric/coulomb"],
    }
