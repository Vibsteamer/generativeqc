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
from generativeqc._f12_spin_pairs import batch_execute, occupied_pair_blocks

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _validate_c_weights(
    space: tuple[int, int, int, int, int],
    weights: Mapping[str, np.ndarray],
    frozen_occupied: tuple[int, int],
) -> None:
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


def _same_spin_c_weights(weights: Mapping[str, np.ndarray]) -> bool:
    ka, kb = "c_f12_aaaa", "c_f12_bbbb"
    if (ka in weights) != (kb in weights):
        return False
    return ka not in weights or np.array_equal(weights[ka], weights[kb])


def _accumulate_c_mo_pullback(
    space: tuple[int, int, int, int, int],
    data: Mapping[str, np.ndarray],
    fock: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
    run: Callable,
    dbar: dict[str, np.ndarray],
    fbar: dict[str, np.ndarray],
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    native_library=None,
    host_reference: bool = True,
    budget_bytes: int = 0,
) -> None:
    """Accumulate MO-space C_vv cotangents into dbar and fbar.

    A bound CUDA kernel requires the endpoint device budget in `budget_bytes`.
    """
    from ._f12_cc_fixed_sources_native import (
        REQUIRE_AOT_CCFIXED_ENV,
        require_aot_ccfixed,
    )
    from ._f12_cc_pair_pullback_native import (
        bind_pair_pullback,
        checked_device_budget,
        run_c_pullback_native,
    )

    _validate_c_weights(space, weights, frozen_occupied)
    oa, ob, obs, ext, aux = space
    frozen = dict(zip("ab", frozen_occupied, strict=True))
    native_fn = bind_pair_pullback(native_library, "c", host_reference)
    if native_fn is not None:
        budget_bytes = checked_device_budget(
            budget_bytes, host_reference=host_reference
        )
    spin_sym_aa_bb = (
        oa == ob
        and frozen["a"] == frozen["b"]
        and oa > frozen["a"]
        and _same_spin_c_weights(weights)
        and np.array_equal(fock["fock_alpha"], fock["fock_beta"])
        and np.array_equal(data["source/fit_alpha"], data["source/fit_beta"])
        and np.array_equal(data["raw/geminal/alpha"], data["raw/geminal/beta"])
    )

    for left, right, nl, nr, _, pairs in occupied_pair_blocks(
        space, frozen_occupied=frozen_occupied
    ):
        if spin_sym_aa_bb and left == "beta" and right == "beta":
            continue
        pair_list = list(pairs)
        if not pair_list:
            continue
        mirror_bb = spin_sym_aa_bb and left == "alpha" and right == "alpha"
        l, r = left[0], right[0]
        if native_fn is not None:
            bar_c_all = np.zeros((len(pair_list), obs - nl, obs - nr), dtype=np.float64)
            for idx, (i, j) in enumerate(pair_list):
                ii, jj = i - frozen[l], j - frozen[r]
                for p, q in ((l, r),) if l == r else ((l, r), (r, l)):
                    for u, v, x, y, sgn in ((l, r, ii, jj, 1), (r, l, jj, ii, -1)):
                        k = "c_f12_" + u + v + p + q
                        if k in weights:
                            tile = weights[k][x, y]
                            bar_c_all[idx] += sgn * (tile if (p, q) == (l, r) else -tile.T)
            res = run_c_pullback_native(
                native_fn,
                same_spin=(left == right),
                nl=nl,
                nr=nr,
                obs=obs,
                ext=ext,
                aux=aux,
                pairs=pair_list,
                fit_l=data["source/fit_" + left],
                fit_r=data["source/fit_" + right],
                raw_g_l=data[f"raw/geminal/{left}"],
                raw_g_r=data[f"raw/geminal/{right}"],
                metric_g=data["metric/geminal"],
                fock_l=fock["fock_" + left],
                fock_r=fock["fock_" + right],
                cross_overlap=None if left == right else data["cross_overlap"],
                bar_c_all=bar_c_all,
                budget_bytes=budget_bytes,
            )
            dbar["source/fit_" + left] += res["bar_fit_l"]
            dbar[f"raw/geminal/{left}"] += res["bar_raw_g_l"]
            dbar["metric/geminal"] += res["bar_metric_g"]
            fbar["fock_" + left] += res["bar_fock_l"]
            if left != right:
                dbar["source/fit_" + right] += res["bar_fit_r"]
                dbar[f"raw/geminal/{right}"] += res["bar_raw_g_r"]
                fbar["fock_" + right] += res["bar_fock_r"]
                dbar["cross_overlap"] += res["bar_cross_overlap"]
            elif mirror_bb:
                dbar["source/fit_beta"] += res["bar_fit_l"]
                dbar["raw/geminal/beta"] += res["bar_raw_g_l"]
                dbar["metric/geminal"] += res["bar_metric_g"]
                fbar["fock_beta"] += res["bar_fock_l"]
            continue
        if require_aot_ccfixed():
            raise RuntimeError(
                f"{REQUIRE_AOT_CCFIXED_ENV}=1 forbids TensorIR fallback in "
                "_accumulate_c_mo_pullback (native_library missing or symbol unavailable)"
            )
        pairs = pair_list
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
        shared = {
            "metric_geminal": np.ascontiguousarray(data["metric/geminal"]),
            "fock_left": np.ascontiguousarray(fock["fock_" + left]),
        }
        if left != right:
            shared["fock_right"] = np.ascontiguousarray(fock["fock_" + right])
            shared["cross_overlap"] = np.ascontiguousarray(data["cross_overlap"])

        l, r = left[0], right[0]
        with batch_execute(run, rev) as exec_rev:
            for i, j in pairs:
                ii, jj = i - frozen[l], j - frozen[r]
                seed = np.zeros((obs - nl, obs - nr))
                for p, q in ((l, r),) if l == r else ((l, r), (r, l)):
                    for u, v, x, y, sgn in ((l, r, ii, jj, 1), (r, l, jj, ii, -1)):
                        k = "c_f12_" + u + v + p + q
                        if k in weights:
                            tile = weights[k][x, y]
                            seed += sgn * (tile if (p, q) == (l, r) else -tile.T)
                fields = dict(shared)
                fields["bar_c_vv"] = np.ascontiguousarray(seed)
                fields["fit_left"] = np.ascontiguousarray(data["source/fit_" + left][:, i])
                fields["fit_right"] = np.ascontiguousarray(data["source/fit_" + right][:, j])
                fields["raw_geminal_left"] = np.ascontiguousarray(
                    data[f"raw/geminal/{left}"][:, i]
                )
                fields["raw_geminal_right"] = np.ascontiguousarray(
                    data[f"raw/geminal/{right}"][:, j]
                )
                res = exec_rev(fields).outputs
                if left != right:
                    dbar["cross_overlap"] += res["bar_cross_overlap"]
                    fbar["fock_" + right] += res["bar_fock_right"]
                fbar["fock_" + left] += res["bar_fock_left"]
                dbar[f"raw/geminal/{left}"][:, i] += res["bar_raw_geminal_left"]
                dbar[f"raw/geminal/{right}"][:, j] += res["bar_raw_geminal_right"]
                dbar["source/fit_" + left][:, i] += res["bar_fit_left"]
                dbar["source/fit_" + right][:, j] += res["bar_fit_right"]
                dbar["metric/geminal"] += res["bar_metric_geminal"]
                if mirror_bb:
                    fbar["fock_beta"] += res["bar_fock_left"]
                    dbar["raw/geminal/beta"][:, i] += res["bar_raw_geminal_left"]
                    dbar["raw/geminal/beta"][:, j] += res["bar_raw_geminal_right"]
                    dbar["source/fit_beta"][:, i] += res["bar_fit_left"]
                    dbar["source/fit_beta"][:, j] += res["bar_fit_right"]
                    dbar["metric/geminal"] += res["bar_metric_geminal"]


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
    _validate_c_weights(space, weights, frozen_occupied)
    oa, ob, obs, ext, aux = space
    occ, frozen = (
        dict(zip("ab", (oa, ob), strict=True)),
        dict(zip("ab", frozen_occupied, strict=True)),
    )
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
    data, _ = _orbital_sources(
        space, fc, run, None, projected=None, role="fock", occupied_coulomb=True
    )
    fock, fock_pullback = fock_response or extended_fock_response(space, ao, jkfit, run)
    fseeds = {"fock_" + s: np.zeros((ext, ext)) for s in spins}
    cross = unrestricted_cross_frame_overlap(combined, ext)
    cross_inputs = {
        k: ao[k] for k in ("coefficients_alpha", "coefficients_beta", "overlap_ao")
    }
    overlap = run(cross, cross_inputs).outputs["cross_overlap"]
    metric = ao["metric/geminal"]
    gbar = np.zeros_like(ao["ao_f12"])

    same_spin = oa == ob and np.array_equal(
        ao["coefficients_alpha"], ao["coefficients_beta"]
    )
    mo_data: dict[str, np.ndarray] = {
        "cross_overlap": overlap,
        "metric/geminal": metric,
    }
    dbar: dict[str, np.ndarray] = {
        "cross_overlap": np.zeros_like(overlap),
        "metric/geminal": np.zeros_like(metric),
    }
    frames = {}
    for spin in spins:
        n = occ[spin[0]]
        if n > 0:
            mo_data["source/fit_" + spin] = data["source/fit_" + spin]
            dbar["source/fit_" + spin] = np.zeros_like(data["source/fit_" + spin])
        if n == frozen[spin[0]]:
            continue
        if same_spin and spin == "beta" and "alpha" in frames:
            ray = frames["alpha"][2]
        else:
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
        mo_data[f"raw/geminal/{spin}"] = ray
        dbar[f"raw/geminal/{spin}"] = np.zeros_like(ray)
        frames[spin] = (n, ao["coefficients_" + spin], ray)

    _accumulate_c_mo_pullback(
        space,
        mo_data,
        fock,
        weights,
        run,
        dbar,
        fseeds,
        frozen_occupied=frozen_occupied,
    )

    fbar = {
        k: np.zeros_like(v)
        for k, v in data.items()
        if k.startswith("source/")
    }
    for spin in spins:
        sk = "source/fit_" + spin
        if sk in dbar:
            fbar[sk] = dbar[sk]
    _, bars = _orbital_sources(
        space,
        fc,
        run,
        fbar,
        projected=data,
        role="fock",
        occupied_coulomb=True,
    )
    assert bars is not None
    fock_pullback(fseeds, bars)
    for spin, (n_occ, coeffs, _) in frames.items():
        project_ao_operator_pullback_tiled(
            combined,
            ext,
            n_occ,
            aux,
            "raw_f12",
            "ao_f12",
            coeffs,
            ao["ao_f12"],
            dbar[f"raw/geminal/{spin}"],
            run,
            bar_coefficients=bars["coefficients_" + spin],
            bar_ao=gbar,
        )
    cv = transpose_program(cross, ("cross_overlap",)).program
    cg = run(cv, {**cross_inputs, "bar_cross_overlap": dbar["cross_overlap"]}).outputs
    for k in spins:
        bars["coefficients_" + k] += cg["bar_coefficients_" + k]
    bars.update(overlap_ao=cg["bar_overlap_ao"], ao_f12=gbar)
    bars["metric/geminal"] = dbar["metric/geminal"]
    return bars
