"""Nuclear and orbital pullback of the unrestricted fixed-geminal scalar V + B − X."""

from collections.abc import Callable, Mapping
from typing import NamedTuple

import numpy as np
from generativeqc_compiler.cc.uccsd_f12_quadratic import (
    unrestricted_fixed_geminal_quadratic_pair_sliced,
)
from generativeqc_compiler.cc.uccsd_f12_sources import unrestricted_f12_pair_geminals
from generativeqc_compiler.tensor import Program, transpose_program

from generativeqc._f12_cc_fock_pullback import FockResponse, extended_fock_response
from generativeqc._f12_spin_ao import _orbital_sources
from generativeqc._f12_spin_pairs import batch_execute, occupied_pair_blocks


class UnrestrictedCCScalarResponse(NamedTuple):
    linear_v: float
    b_energy: float
    x_fock_energy: float
    weights: dict[str, np.ndarray]


def _with_transpose(primal: Program) -> Program:
    reverse = transpose_program(primal, tuple(primal.outputs), max_elements=4_000_000).program
    return Program({**primal.outputs, **reverse.outputs})


def _is_spin_symmetric_scalar_data(
    space: tuple[int, int, int, int, int],
    data: Mapping[str, np.ndarray],
    fock: Mapping[str, np.ndarray],
    frozen_occupied: tuple[int, int],
) -> bool:
    oa, ob, _, _, _ = space
    if oa != ob or frozen_occupied[0] != frozen_occupied[1] or oa == 0:
        return False
    if not (
        np.array_equal(fock["fock_alpha"], fock["fock_beta"])
        and np.array_equal(fock["exchange_alpha"], fock["exchange_beta"])
        and np.array_equal(data["source/fit_alpha"], data["source/fit_beta"])
        and np.array_equal(
            data["source/raw_coulomb_alpha"][:, :oa],
            data["source/raw_coulomb_beta"][:, :ob],
        )
        and np.array_equal(data["cross_overlap"], data["cross_overlap"].T)
    ):
        return False
    return all(
        np.array_equal(data[f"raw/{op}/alpha"], data[f"raw/{op}/beta"])
        for op in ("geminal", "squared_geminal", "geminal_coulomb", "double_commutator")
    )


_QUAD_FIELDS = (
    ("fit", "source/fit_"),
    ("raw_geminal", "raw/geminal/"),
    ("raw_squared_geminal", "raw/squared_geminal/"),
    ("raw_double_commutator", "raw/double_commutator/"),
)
_X_FOCK_FIELDS = (
    ("fit", "source/fit_"),
    ("raw_geminal", "raw/geminal/"),
    ("raw_squared_geminal", "raw/squared_geminal/"),
)


def _accumulate_scalar_mo_pullback(
    space: tuple[int, int, int, int, int],
    data: Mapping[str, np.ndarray],
    fock: Mapping[str, np.ndarray],
    scalar_weight: float | np.ndarray,
    run: Callable,
    dbar: dict[str, np.ndarray],
    fbar: dict[str, np.ndarray],
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    native_library=None,
    host_reference: bool = True,
    budget_bytes: int = 0,
) -> tuple[float, float, float]:
    """Accumulate MO-space scalar V + B - X cotangents into dbar and fbar via GPU cuBLAS or TensorIR.

    A bound CUDA kernel requires the endpoint device budget in `budget_bytes`.
    """
    from ._f12_cc_fixed_sources_native import (
        REQUIRE_AOT_CCFIXED_ENV,
        require_aot_ccfixed,
    )
    from ._f12_cc_pair_pullback_native import (
        bind_pair_pullback,
        checked_device_budget,
        run_scalar_pullback_native,
    )

    oa, ob, obs, ext, aux = space
    weight = np.asarray(scalar_weight)
    if weight.shape or weight.dtype.kind not in "fiu" or not np.isfinite(weight):
        raise ValueError("A finite stationary scalar cotangent is required")
    occupied = {"alpha": oa, "beta": ob}
    frozen = {"alpha": frozen_occupied[0], "beta": frozen_occupied[1]}
    active = {s: slice(frozen[s], occupied[s]) for s in ("alpha", "beta")}
    spin_sym = _is_spin_symmetric_scalar_data(space, data, fock, frozen_occupied)
    native_fn = bind_pair_pullback(native_library, "scalar", host_reference)
    if native_fn is not None:
        budget_bytes = checked_device_budget(
            budget_bytes, host_reference=host_reference
        )

    # Precompute F_oo-mixed active slices once outside the pair loop (exact for arbitrary F_oo).
    fock_oo: dict[str, np.ndarray] = {}
    fock_mixed: dict[str, dict[str, np.ndarray]] = {}
    bar_fock_mixed: dict[str, dict[str, np.ndarray]] = {}
    for spin in ("alpha", "beta"):
        act = active[spin]
        if act.stop <= act.start:
            continue
        f_oo = np.ascontiguousarray(fock["fock_" + spin][act, act])
        fock_oo[spin] = f_oo
        if spin_sym and spin == "beta" and "alpha" in fock_mixed:
            fock_mixed["beta"] = fock_mixed["alpha"]
        else:
            fock_mixed[spin] = {
                name: np.einsum("ki,akp->aip", f_oo, data[prefix + spin][:, act, :])
                for name, prefix in _X_FOCK_FIELDS
            }
        bar_fock_mixed[spin] = {
            name: np.zeros_like(arr) for name, arr in fock_mixed[spin].items()
        }

    e_v = e_b = e_x = 0.0
    eye = np.eye(ext)
    seed = weight.astype(np.float64)
    for left, right, nl, nr, _, pairs in occupied_pair_blocks(
        space, frozen_occupied=frozen_occupied
    ):
        if spin_sym and left == "beta" and right == "beta":
            continue
        pair_list = list(pairs)
        if not pair_list:
            continue
        mirror_bb = spin_sym and left == "alpha" and right == "alpha"
        if native_fn is not None:
            lv, bv, xv, res = run_scalar_pullback_native(
                native_fn,
                same_spin=(left == right),
                nl=nl,
                nr=nr,
                fl=frozen[left],
                fr=frozen[right],
                obs=obs,
                ext=ext,
                aux=aux,
                scalar_weight=float(seed),
                pairs=pair_list,
                fit_l=data["source/fit_" + left],
                fit_r=data["source/fit_" + right],
                raw_coul_r=data["source/raw_coulomb_" + right],
                raw_g_l=data[f"raw/geminal/{left}"],
                raw_g_r=data[f"raw/geminal/{right}"],
                raw_gc_l=data[f"raw/geminal_coulomb/{left}"],
                raw_gc_r=data[f"raw/geminal_coulomb/{right}"],
                raw_sq_l=data[f"raw/squared_geminal/{left}"],
                raw_sq_r=data[f"raw/squared_geminal/{right}"],
                raw_dc_l=data[f"raw/double_commutator/{left}"],
                raw_dc_r=data[f"raw/double_commutator/{right}"],
                fmix_fit_l=fock_mixed[left]["fit"],
                fmix_fit_r=fock_mixed[right]["fit"],
                fmix_g_l=fock_mixed[left]["raw_geminal"],
                fmix_g_r=fock_mixed[right]["raw_geminal"],
                fmix_sq_l=fock_mixed[left]["raw_squared_geminal"],
                fmix_sq_r=fock_mixed[right]["raw_squared_geminal"],
                metric_g=data["metric/geminal"],
                metric_gc=data["metric/geminal_coulomb"],
                metric_sq=data["metric/squared_geminal"],
                metric_dc=data["metric/double_commutator"],
                fock_l=fock["fock_" + left],
                fock_r=fock["fock_" + right],
                exch_l=fock["exchange_" + left],
                exch_r=fock["exchange_" + right],
                cross_overlap=None if left == right else data["cross_overlap"],
                budget_bytes=budget_bytes,
            )
            mult = 2.0 if mirror_bb else 1.0
            e_v += mult * lv
            e_b += mult * bv
            e_x += mult * xv
            dbar["source/fit_" + left] += res["bar_fit_l"]
            dbar["source/raw_coulomb_" + right][:, :nr, :] += res["bar_raw_coul_r"]
            dbar[f"raw/geminal/{left}"] += res["bar_raw_g_l"]
            dbar[f"raw/geminal_coulomb/{left}"] += res["bar_raw_gc_l"]
            dbar[f"raw/squared_geminal/{left}"] += res["bar_raw_sq_l"]
            dbar[f"raw/double_commutator/{left}"] += res["bar_raw_dc_l"]
            bar_fock_mixed[left]["fit"] += res["bar_fmix_fit_l"]
            bar_fock_mixed[left]["raw_geminal"] += res["bar_fmix_g_l"]
            bar_fock_mixed[left]["raw_squared_geminal"] += res["bar_fmix_sq_l"]
            dbar["metric/geminal"] += mult * res["bar_metric_g"]
            dbar["metric/geminal_coulomb"] += mult * res["bar_metric_gc"]
            dbar["metric/squared_geminal"] += mult * res["bar_metric_sq"]
            dbar["metric/double_commutator"] += mult * res["bar_metric_dc"]
            fbar["fock_" + left] += res["bar_fock_l"]
            fbar["exchange_" + left] += res["bar_exch_l"]
            if left != right:
                dbar["source/fit_" + right] += res["bar_fit_r"]
                dbar[f"raw/geminal/{right}"] += res["bar_raw_g_r"]
                dbar[f"raw/geminal_coulomb/{right}"] += res["bar_raw_gc_r"]
                dbar[f"raw/squared_geminal/{right}"] += res["bar_raw_sq_r"]
                dbar[f"raw/double_commutator/{right}"] += res["bar_raw_dc_r"]
                bar_fock_mixed[right]["fit"] += res["bar_fmix_fit_r"]
                bar_fock_mixed[right]["raw_geminal"] += res["bar_fmix_g_r"]
                bar_fock_mixed[right]["raw_squared_geminal"] += res["bar_fmix_sq_r"]
                fbar["fock_" + right] += res["bar_fock_r"]
                fbar["exchange_" + right] += res["bar_exch_r"]
                dbar["cross_overlap"] += res["bar_cross_overlap"]
            elif mirror_bb:
                dbar["source/fit_beta"] += res["bar_fit_l"]
                dbar["source/raw_coulomb_beta"][:, :nr, :] += res["bar_raw_coul_r"]
                dbar["raw/geminal/beta"] += res["bar_raw_g_l"]
                dbar["raw/geminal_coulomb/beta"] += res["bar_raw_gc_l"]
                dbar["raw/squared_geminal/beta"] += res["bar_raw_sq_l"]
                dbar["raw/double_commutator/beta"] += res["bar_raw_dc_l"]
                bar_fock_mixed["beta"]["fit"] += res["bar_fmix_fit_l"]
                bar_fock_mixed["beta"]["raw_geminal"] += res["bar_fmix_g_l"]
                bar_fock_mixed["beta"]["raw_squared_geminal"] += res["bar_fmix_sq_l"]
                fbar["fock_beta"] += res["bar_fock_l"]
                fbar["exchange_beta"] += res["bar_exch_l"]
            continue
        if require_aot_ccfixed():
            raise RuntimeError(
                f"{REQUIRE_AOT_CCFIXED_ENV}=1 forbids TensorIR fallback in "
                "_accumulate_scalar_mo_pullback (native_library missing or symbol unavailable)"
            )
        pairs = pair_list
        ls, rs = (nl, obs - nl, ext - obs), (nr, obs - nr, ext - obs)
        channel = left + "_" + right
        vgraph = unrestricted_f12_pair_geminals(ls, rs, aux, channel=channel)
        linear = _with_transpose(
            Program({"linear_v_energy": vgraph.outputs["linear_v_energy"]})
        )
        quadratic = _with_transpose(
            unrestricted_fixed_geminal_quadratic_pair_sliced(
                ls, rs, aux, channel=channel
            )
        )
        shared_lin = {
            "metric_geminal": np.ascontiguousarray(data["metric/geminal"]),
            "metric_geminal_coulomb": np.ascontiguousarray(data["metric/geminal_coulomb"]),
            "bar_linear_v_energy": seed,
        }
        if left != right:
            shared_lin["cross_overlap"] = np.ascontiguousarray(data["cross_overlap"])

        shared_quad = {
            "fock_left": np.ascontiguousarray(fock["fock_" + left]),
            "exchange_left": np.ascontiguousarray(fock["exchange_" + left]),
            "metric_geminal": np.ascontiguousarray(data["metric/geminal"]),
            "metric_squared_geminal": np.ascontiguousarray(data["metric/squared_geminal"]),
            "metric_double_commutator": np.ascontiguousarray(data["metric/double_commutator"]),
            "bar_b_energy": seed,
            "bar_x_fock_energy": np.asarray(-seed),
        }
        if left != right:
            shared_quad["fock_right"] = np.ascontiguousarray(fock["fock_" + right])
            shared_quad["exchange_right"] = np.ascontiguousarray(fock["exchange_" + right])
            shared_quad["cross_overlap"] = np.ascontiguousarray(data["cross_overlap"])

        fl, fr = frozen[left], frozen[right]
        with batch_execute(run, linear) as exec_lin, batch_execute(run, quadratic) as exec_quad:
            for i, j in pairs:
                ii, jj = i - fl, j - fr
                values = dict(shared_lin)
                values["select_i"] = eye[i]
                values["select_j"] = eye[j]
                values["fit_left"] = np.ascontiguousarray(data["source/fit_" + left][:, i])
                values["fit_right"] = np.ascontiguousarray(data["source/fit_" + right][:, j])
                values["raw_coulomb_right"] = np.ascontiguousarray(
                    data["source/raw_coulomb_" + right][:, j]
                )
                for op in ("geminal", "geminal_coulomb"):
                    values[f"raw_{op}_left"] = np.ascontiguousarray(data[f"raw/{op}/{left}"][:, i])
                    values[f"raw_{op}_right"] = np.ascontiguousarray(data[f"raw/{op}/{right}"][:, j])
                vr = exec_lin(values).outputs
                lv = float(vr["linear_v_energy"])
                e_v += 2.0 * lv if mirror_bb else lv

                dbar["source/fit_" + left][:, i] += vr["bar_fit_left"]
                dbar["source/fit_" + right][:, j] += vr["bar_fit_right"]
                dbar["source/raw_coulomb_" + right][:, j] += vr["bar_raw_coulomb_right"]
                for op in ("geminal", "geminal_coulomb"):
                    dbar["metric/" + op] += vr["bar_metric_" + op]
                    dbar[f"raw/{op}/{left}"][:, i] += vr[f"bar_raw_{op}_left"]
                    dbar[f"raw/{op}/{right}"][:, j] += vr[f"bar_raw_{op}_right"]
                if left != right:
                    dbar["cross_overlap"] += vr["bar_cross_overlap"]
                if mirror_bb:
                    dbar["source/fit_beta"][:, i] += vr["bar_fit_left"]
                    dbar["source/fit_beta"][:, j] += vr["bar_fit_right"]
                    dbar["source/raw_coulomb_beta"][:, j] += vr["bar_raw_coulomb_right"]
                    for op in ("geminal", "geminal_coulomb"):
                        dbar["metric/" + op] += vr["bar_metric_" + op]
                        dbar[f"raw/{op}/beta"][:, i] += vr[f"bar_raw_{op}_left"]
                        dbar[f"raw/{op}/beta"][:, j] += vr[f"bar_raw_{op}_right"]

                qvals = dict(shared_quad)
                qvals["select_i_full"] = eye[i]
                qvals["select_j_full"] = eye[j]
                for name, prefix in _QUAD_FIELDS:
                    qvals[f"{name}_left"] = np.ascontiguousarray(data[prefix + left][:, i])
                    qvals[f"{name}_right"] = np.ascontiguousarray(data[prefix + right][:, j])
                for name, _ in _X_FOCK_FIELDS:
                    qvals[f"fock_{name}_left"] = np.ascontiguousarray(fock_mixed[left][name][:, ii])
                    qvals[f"fock_{name}_right"] = np.ascontiguousarray(fock_mixed[right][name][:, jj])

                qr = exec_quad(qvals).outputs
                bv = float(qr["b_energy"])
                xv = float(qr["x_fock_energy"])
                e_b += 2.0 * bv if mirror_bb else bv
                e_x += 2.0 * xv if mirror_bb else xv

                fbar["fock_" + left] += qr["bar_fock_left"]
                fbar["exchange_" + left] += qr["bar_exchange_left"]
                if left != right:
                    fbar["fock_" + right] += qr["bar_fock_right"]
                    fbar["exchange_" + right] += qr["bar_exchange_right"]
                    dbar["cross_overlap"] += qr["bar_cross_overlap"]
                for op in ("geminal", "squared_geminal", "double_commutator"):
                    dbar["metric/" + op] += qr["bar_metric_" + op]
                for name, prefix in _QUAD_FIELDS:
                    dbar[prefix + left][:, i] += qr[f"bar_{name}_left"]
                    dbar[prefix + right][:, j] += qr[f"bar_{name}_right"]
                for name, _ in _X_FOCK_FIELDS:
                    bar_fock_mixed[left][name][:, ii] += qr[f"bar_fock_{name}_left"]
                    bar_fock_mixed[right][name][:, jj] += qr[f"bar_fock_{name}_right"]

                if mirror_bb:
                    fbar["fock_beta"] += qr["bar_fock_left"]
                    fbar["exchange_beta"] += qr["bar_exchange_left"]
                    for op in ("geminal", "squared_geminal", "double_commutator"):
                        dbar["metric/" + op] += qr["bar_metric_" + op]
                    for name, prefix in _QUAD_FIELDS:
                        dbar[prefix + "beta"][:, i] += qr[f"bar_{name}_left"]
                        dbar[prefix + "beta"][:, j] += qr[f"bar_{name}_right"]
                    for name, _ in _X_FOCK_FIELDS:
                        bar_fock_mixed["beta"][name][:, ii] += qr[f"bar_fock_{name}_left"]
                        bar_fock_mixed["beta"][name][:, jj] += qr[f"bar_fock_{name}_right"]

    # Exact chain rule for X[a, i, p] = sum_k F_oo[k, i] * M[a, k, p].
    for spin, bar_map in bar_fock_mixed.items():
        act = active[spin]
        f_oo = fock_oo[spin]
        for name, prefix in _X_FOCK_FIELDS:
            bar_x = bar_map[name]
            m_act = data[prefix + spin][:, act, :]
            fbar["fock_" + spin][act, act] += np.einsum("akp,aip->ki", m_act, bar_x)
            dbar[prefix + spin][:, act, :] += np.einsum("ki,aip->akp", f_oo, bar_x)

    return e_v, e_b, e_x


def unrestricted_cc_scalar_ao_weights(
    space: tuple[int, int, int, int, int],
    ao: dict[str, np.ndarray],
    scalar_weight: float | np.ndarray,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    jkfit: Mapping[str, np.ndarray] | None = None,
    fock_response: FockResponse | None = None,
) -> UnrestrictedCCScalarResponse:
    """Return each unweighted energy and the weighted derivative of their signed sum."""
    _, _, _, _, aux = space
    weight = np.asarray(scalar_weight)
    if weight.shape or weight.dtype.kind not in "fiu" or not np.isfinite(weight):
        raise ValueError("A finite stationary scalar cotangent is required")
    if not np.allclose(
        ao["metric_whitener"] @ ao["metric/coulomb"] @ ao["metric_whitener"].T,
        np.eye(aux),
        atol=2e-7,
        rtol=0,
    ):
        raise ValueError("The full-rank whitener and Coulomb metric must match")
    data, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="pairs", occupied_coulomb=True
    )
    fock, fock_pullback = fock_response or extended_fock_response(space, ao, jkfit, run)
    dbar = {k: np.zeros_like(v) for k, v in data.items() if k != "space"}
    fbar = {k: np.zeros_like(v) for k, v in fock.items()}
    e_v, e_b, e_x = _accumulate_scalar_mo_pullback(
        space, data, fock, scalar_weight, run, dbar, fbar, frozen_occupied=frozen_occupied
    )
    _, physical = _orbital_sources(
        space, ao, run, dbar, projected=data, role="pairs", occupied_coulomb=True
    )
    assert physical is not None
    fock_pullback(fbar, physical)
    return UnrestrictedCCScalarResponse(e_v, e_b, e_x, physical)
