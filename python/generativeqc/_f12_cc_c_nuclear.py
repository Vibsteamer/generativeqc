"""Stationary spin-resolved fixed-cusp C-source adjoint on a shared UHF OBS/CABS frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.cc.uccsd_f12_sources import unrestricted_f12_pair_sources
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_cross_frame_overlap
from generativeqc_compiler.tensor import Program, transpose_program

from generativeqc._f12_cc_fock_pullback import FockResponse, extended_fock_response
from generativeqc._f12_spin_ao import (
    _orbital_sources,
    project_ao_operator_pullback_tiled,
    project_ao_operator_tiled,
)
from generativeqc._f12_spin_pairs import occupied_pair_blocks

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def unrestricted_cc_c_ao_weights(
    space: tuple[int, int, int, int, int],
    ao: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    jkfit: Mapping[str, np.ndarray] | None = None,
    fock_response: FockResponse | None = None,
) -> dict[str, np.ndarray]:
    """Transpose ordered active C blocks through the geminal and extended Fock on both spins.

    C contains both virtual-complementary Fock couplings to the fitted geminal.
    All occupied orbitals contribute to the Fock. Frozen occupied indices only
    restrict which pairs receive stationary source weights.
    """
    oa, ob, obs, ext, aux = space
    occ, frozen = (
        dict(zip("ab", (oa, ob), strict=True)),
        dict(zip("ab", frozen_occupied, strict=True)),
    )
    if not (
        0 <= frozen["a"] <= oa < obs < ext
        and 0 <= frozen["b"] <= ob < obs
        and oa + ob
        and aux > 0
    ):
        raise ValueError(
            "Invalid unrestricted C occupied, virtual or complementary space"
        )
    for name in weights:
        spins = name.removeprefix("c_f12_")
        if (
            len(spins) != 4
            or any(s not in occ for s in spins)
            or sorted(spins[:2]) != sorted(spins[2:])
            or name != "c_f12_" + spins
        ):
            raise ValueError("Invalid C spin weights")
        l, r, p, q = spins
        shape = (occ[l] - frozen[l], occ[r] - frozen[r], obs - occ[p], obs - occ[q])
        if weights[name].shape != shape or weights[name].dtype != np.float64:
            raise ValueError("Invalid C source extent or precision")
    if not weights:
        raise ValueError("At least one stationary C source is required")
    wh = ao["metric_whitener"]
    if not np.allclose(
        wh @ ao["metric/coulomb"] @ wh.T, np.eye(aux), rtol=0, atol=2e-7
    ):
        raise ValueError("The C source requires its common full-rank Coulomb metric")
    spins = ("alpha", "beta")
    combined = len(ao["overlap_ao"])
    fc = {
        k: ao[k]
        for k in (
            "hcore_ao",
            "ao_coulomb",
            "metric/coulomb",
            "metric_whitener",
            "coefficients_alpha",
            "coefficients_beta",
        )
    }
    data, _ = _orbital_sources(space, fc, run, None, projected=None, role="fock")
    fsrc = {
        k.removeprefix("source/"): v for k, v in data.items() if k.startswith("source/")
    }
    fock, fock_pullback = fock_response or extended_fock_response(space, ao, jkfit, run)
    fseeds = {"fock_" + s: np.zeros((ext, ext)) for s in spins}
    fbar = {k: np.zeros_like(v) for k, v in fsrc.items()}
    cross = unrestricted_cross_frame_overlap(combined, ext)
    cross_inputs = {
        k: ao[k] for k in ("coefficients_alpha", "coefficients_beta", "overlap_ao")
    }
    overlap = run(cross, cross_inputs).outputs["cross_overlap"]
    obar = np.zeros_like(overlap)
    metric = ao["metric/geminal"]
    mbar = np.zeros_like(metric)
    gbar = np.zeros_like(ao["ao_f12"])
    frames = {}
    for spin in spins:
        n = occ[spin[0]]
        if n == frozen[spin[0]]:
            continue
        ray = project_ao_operator_tiled(
            combined,
            ext,
            n,
            aux,
            "raw_f12",
            "ao_f12",
            ao["coefficients_" + spin],
            ao["ao_f12"],
            run,
        )
        frames[spin] = (n, ao["coefficients_" + spin], ray, np.zeros_like(ray))
    for left, right, nl, nr, _, pairs in occupied_pair_blocks(
        space, frozen_occupied=frozen_occupied
    ):
        graph = unrestricted_f12_pair_sources(
            (nl, obs - nl, ext - obs),
            (nr, obs - nr, ext - obs),
            aux,
            channel=left + "_" + right,
        )
        rev = transpose_program(
            Program({"c_vv": graph.outputs["c_vv"]}),
            ("c_vv",),
            max_elements=max(1_000_000, ext * ext),
        ).program
        l, r = left[0], right[0]
        for i, j in pairs:
            ii, jj = i - frozen[l], j - frozen[r]
            seed = np.zeros((obs - nl, obs - nr))
            for p, q in ((l, r),) if l == r else ((l, r), (r, l)):
                for u, v, x, y, sgn in ((l, r, ii, jj, 1), (r, l, jj, ii, -1)):
                    k = "c_f12_" + u + v + p + q
                    if k in weights:
                        tile = weights[k][x, y]
                        seed += sgn * (tile if (p, q) == (l, r) else -tile.T)
            fields = {
                "bar_c_vv": seed,
                "fit_left": fsrc["fit_" + left][:, i],
                "fit_right": fsrc["fit_" + right][:, j],
                "raw_geminal_left": frames[left][2][:, i],
                "raw_geminal_right": frames[right][2][:, j],
                "metric_geminal": metric,
                "fock_left": fock["fock_" + left],
            }
            if left != right:
                fields.update(fock_right=fock["fock_" + right], cross_overlap=overlap)
            res = run(
                rev, {k: np.ascontiguousarray(x) for k, x in fields.items()}
            ).outputs
            if left != right:
                obar += res["bar_cross_overlap"]
                fseeds["fock_" + right] += res["bar_fock_right"]
            fseeds["fock_" + left] += res["bar_fock_left"]
            for spin, index, side in ((left, i, "left"), (right, j, "right")):
                frames[spin][3][:, index] += res["bar_raw_geminal_" + side]
                fbar["fit_" + spin][:, index] += res["bar_fit_" + side]
            mbar += res["bar_metric_geminal"]
    _, bars = _orbital_sources(
        space,
        fc,
        run,
        {"source/" + k: v for k, v in fbar.items()},
        projected=data,
        role="fock",
    )
    assert bars is not None
    fock_pullback(fseeds, bars)
    for spin, (n_occ, coeffs, _, seed) in frames.items():
        project_ao_operator_pullback_tiled(
            combined,
            ext,
            n_occ,
            aux,
            "raw_f12",
            "ao_f12",
            coeffs,
            ao["ao_f12"],
            seed,
            run,
            bar_coefficients=bars["coefficients_" + spin],
            bar_ao=gbar,
        )
    cv = transpose_program(cross, ("cross_overlap",)).program
    cg = run(cv, {**cross_inputs, "bar_cross_overlap": obar}).outputs
    for k in spins:
        bars["coefficients_" + k] += cg["bar_coefficients_" + k]
    bars.update(overlap_ao=cg["bar_overlap_ao"], ao_f12=gbar)
    bars["metric/geminal"] = mbar
    return bars
