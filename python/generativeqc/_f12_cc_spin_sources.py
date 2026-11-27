"""Build external fixed-cusp CC sources from an unrestricted molecular DF frame."""

from collections.abc import Callable
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from typing import NamedTuple

import numpy as np
from generativeqc_compiler.cc.uccsd_f12_quadratic import (
    unrestricted_fixed_geminal_quadratic_pair,
    unrestricted_fixed_geminal_quadratic_pair_sliced,
)
from generativeqc_compiler.cc.uccsd_f12_sources import (
    batched_obs_w_projection,
    unrestricted_f12_pair_geminals,
    unrestricted_f12_pair_sources,
)
from generativeqc_compiler.tensor import Program, linearize

from ._f12_cc_fixed_sources_native import (
    AUTO_NATIVE_LIBRARY,
    REQUIRE_AOT_CCFIXED_ENV,
    bind_fixed_pair_sources,
    coulomb_fit_forward,
    require_aot_ccfixed,
    resolve_ccfixed_library,
    run_fixed_pair_sources_native,
)
from ._f12_spin_fock import _tile_groups
from ._f12_spin_pairs import batch_execute, occupied_pair_blocks

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


def _spin_symmetry_mismatch(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    frozen_occupied: tuple[int, int],
) -> str | None:
    oa, ob = space[0], space[1]
    if oa != ob or frozen_occupied[0] != frozen_occupied[1]:
        return f"occupations/frozen differ: ({oa},{ob}) / {frozen_occupied}"
    checks = [
        ("fock", fock["fock_alpha"], fock["fock_beta"]),
        ("exchange", fock["exchange_alpha"], fock["exchange_beta"]),
        ("fit", data["source/fit_alpha"], data["source/fit_beta"]),
        ("raw_coulomb", data["source/raw_coulomb_alpha"], data["source/raw_coulomb_beta"]),
    ]
    for op in ("geminal", "squared_geminal", "geminal_coulomb", "double_commutator"):
        checks.append((f"raw/{op}", data["raw/" + op + "/alpha"], data["raw/" + op + "/beta"]))
    for label, lhs, rhs in checks:
        if not np.allclose(lhs, rhs, atol=2e-8, rtol=1e-10):
            diff = float(np.max(np.abs(lhs - rhs)))
            return f"{label} alpha/beta max_abs_diff={diff:.3e}"
    cross = data.get("cross_overlap")
    if cross is not None and not np.allclose(cross, cross.T, atol=2e-7, rtol=1e-10):
        diff = float(np.max(np.abs(cross - cross.T)))
        return f"cross_overlap asymmetry={diff:.3e}"
    return None


def _is_spin_symmetric_f12_data(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    frozen_occupied: tuple[int, int],
) -> bool:
    return _spin_symmetry_mismatch(space, data, fock, frozen_occupied) is None


def spatial_rhf_cc_f12_fixed_sources(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    whitener: np.ndarray,
    run: Callable,
    *,
    frozen_occupied: int = 0,
    native_library: Path | None = AUTO_NATIVE_LIBRARY,  # type: ignore[assignment]
    host_reference: bool | None = None,
    max_workspace_bytes: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build compact spatial closed-shell (w_vo, w_vv, c_f12, fixed_geminal_energy).

    Exploits closed-shell spin symmetry: evaluates w_obs/c_vv only on alpha_beta
    (i <= j) and scalar V+B-X on alpha_beta (i <= j) and alpha_alpha (i < j,
    doubled for beta_beta). Executes natively via AOT cuBLAS/host-reference kernels
    when available, with bounded TensorIR fallback unless strict AOT is required.
    """
    oa, ob, obs, extended, auxiliary = space
    if oa != ob or not (0 <= frozen_occupied < oa < obs < extended):
        raise ValueError("Spatial RHF F12 sources require matched closed-shell dimensions")
    mismatch = _spin_symmetry_mismatch(space, data, fock, (frozen_occupied, frozen_occupied))
    if mismatch is not None:
        raise ValueError(f"Spatial RHF F12 sources require spin-symmetric RHF data ({mismatch})")
    o_act = oa - frozen_occupied
    n_vir = obs - oa
    occ_slice = slice(frozen_occupied, oa)
    vir_slice = slice(oa, obs)
    w_vo = np.zeros((o_act, o_act, n_vir, o_act), dtype=np.float64)
    w_vv = np.zeros((o_act, o_act, n_vir, n_vir), dtype=np.float64)
    c_f12 = np.zeros((o_act, o_act, n_vir, n_vir), dtype=np.float64)

    lib_path, use_host_ref = resolve_ccfixed_library(native_library, host_reference, run)
    raw_obs = np.ascontiguousarray(data["source/raw_coulomb_alpha"][:, :obs, :obs])
    fit_obs = coulomb_fit_forward(
        whitener,
        raw_obs,
        native_library=lib_path,
        host_reference=use_host_ref,
        run=run,
        max_workspace_bytes=max_workspace_bytes,
    )

    pair_fn = bind_fixed_pair_sources(lib_path, use_host_ref)
    if pair_fn is not None:
        linear_v = 0.0
        quadratic_energy = 0.0

        ab_pairs = [
            (i, j)
            for i in range(frozen_occupied, oa)
            for j in range(i, oa)
        ]
        ab_mult = np.array(
            [1.0 if i == j else 2.0 for i, j in ab_pairs], dtype=np.float64
        )
        w_obs_ab, c_vv_ab, e_v_ab, e_b_ab, e_x_ab = run_fixed_pair_sources_native(
            pair_fn,
            same_spin=False,
            compute_wc=True,
            compute_quad=True,
            nl=oa,
            nr=ob,
            fl=frozen_occupied,
            fr=frozen_occupied,
            obs=obs,
            ext=extended,
            aux=auxiliary,
            pairs=ab_pairs,
            pair_mult=ab_mult,
            fit_l=data["source/fit_alpha"],
            fit_r=data["source/fit_beta"],
            raw_coul_r=data["source/raw_coulomb_beta"],
            raw_g_l=data["raw/geminal/alpha"],
            raw_g_r=data["raw/geminal/beta"],
            raw_gc_l=data["raw/geminal_coulomb/alpha"],
            raw_gc_r=data["raw/geminal_coulomb/beta"],
            raw_sq_l=data["raw/squared_geminal/alpha"],
            raw_sq_r=data["raw/squared_geminal/beta"],
            raw_dc_l=data["raw/double_commutator/alpha"],
            raw_dc_r=data["raw/double_commutator/beta"],
            metric_g=data["metric/geminal"],
            metric_gc=data["metric/geminal_coulomb"],
            metric_sq=data["metric/squared_geminal"],
            metric_dc=data["metric/double_commutator"],
            fock_l=fock["fock_alpha"],
            fock_r=fock["fock_beta"],
            exch_l=fock["exchange_alpha"],
            exch_r=fock["exchange_beta"],
            cross_overlap=data["cross_overlap"],
            fit_obs_l=fit_obs,
            raw_obs_r=raw_obs,
            max_workspace_bytes=max_workspace_bytes,
        )
        assert w_obs_ab is not None and c_vv_ab is not None
        linear_v += e_v_ab
        quadratic_energy += e_b_ab - e_x_ab
        for idx, (i, j) in enumerate(ab_pairs):
            ii = i - frozen_occupied
            jj = j - frozen_occupied
            w = w_obs_ab[idx]
            c = c_vv_ab[idx]
            w_vo[ii, jj] = w[vir_slice, occ_slice]
            w_vv[ii, jj] = w[vir_slice, vir_slice]
            c_f12[ii, jj] = c
            if i < j:
                w_vo[jj, ii] = w[occ_slice, vir_slice].T
                w_vv[jj, ii] = w[vir_slice, vir_slice].T
                c_f12[jj, ii] = c.T

        if o_act >= 2:
            aa_pairs = [
                (i, j)
                for i in range(frozen_occupied, oa)
                for j in range(i + 1, oa)
            ]
            aa_mult = np.full(len(aa_pairs), 2.0, dtype=np.float64)
            _, _, e_v_aa, e_b_aa, e_x_aa = run_fixed_pair_sources_native(
                pair_fn,
                same_spin=True,
                compute_wc=False,
                compute_quad=True,
                nl=oa,
                nr=oa,
                fl=frozen_occupied,
                fr=frozen_occupied,
                obs=obs,
                ext=extended,
                aux=auxiliary,
                pairs=aa_pairs,
                pair_mult=aa_mult,
                fit_l=data["source/fit_alpha"],
                fit_r=None,
                raw_coul_r=data["source/raw_coulomb_alpha"],
                raw_g_l=data["raw/geminal/alpha"],
                raw_g_r=None,
                raw_gc_l=data["raw/geminal_coulomb/alpha"],
                raw_gc_r=None,
                raw_sq_l=data["raw/squared_geminal/alpha"],
                raw_sq_r=None,
                raw_dc_l=data["raw/double_commutator/alpha"],
                raw_dc_r=None,
                metric_g=data["metric/geminal"],
                metric_gc=data["metric/geminal_coulomb"],
                metric_sq=data["metric/squared_geminal"],
                metric_dc=data["metric/double_commutator"],
                fock_l=fock["fock_alpha"],
                fock_r=None,
                exch_l=fock["exchange_alpha"],
                exch_r=None,
                cross_overlap=None,
                fit_obs_l=None,
                raw_obs_r=None,
                max_workspace_bytes=max_workspace_bytes,
            )
            linear_v += e_v_aa
            quadratic_energy += e_b_aa - e_x_aa

        total_energy = float(linear_v + quadratic_energy)
        if not np.isfinite(total_energy):
            raise ValueError("Nonfinite spatial RHF fixed-geminal energy")
        return w_vo, w_vv, c_f12, total_energy

    if require_aot_ccfixed():
        raise RuntimeError(
            f"{REQUIRE_AOT_CCFIXED_ENV}=1 forbids TensorIR fallback in "
            "spatial_rhf_cc_f12_fixed_sources (native_library missing or "
            "gqc_f12_cc_fixed_pair_sources symbol unavailable)"
        )

    fit_obs_c = np.ascontiguousarray(fit_obs)
    fock_c = np.ascontiguousarray(fock["fock_alpha"])
    cross_c = np.ascontiguousarray(data["cross_overlap"])
    metric_gem_c = np.ascontiguousarray(data["metric/geminal"])
    metric_gc_c = np.ascontiguousarray(data["metric/geminal_coulomb"])
    selectors = np.eye(extended)
    dims = (oa, n_vir, extended - obs)
    occ_map = {"a": occ_slice, "b": occ_slice}

    linear_v = 0.0
    quadratic_energy = 0.0

    # 1. Opposite-spin (alpha_beta) i <= j: provides w_vo, w_vv, c_f12 and ab V+B-X.
    ab_lin_graph = unrestricted_f12_pair_sources(dims, dims, auxiliary, channel="alpha_beta")
    ab_quad_graph = unrestricted_fixed_geminal_quadratic_pair(
        dims, dims, auxiliary, channel="alpha_beta",
        frozen_occupied=(frozen_occupied, frozen_occupied),
    )
    ab_quad_shared = _quadratic_pair_fields(data, fock, "alpha", "beta", occ_map)
    ab_lin_shared = {
        "fit_obs_left": fit_obs_c,
        "raw_coulomb_obs_right": raw_obs,
        "fock_left": fock_c,
        "fock_right": fock_c,
        "cross_overlap": cross_c,
        "metric_geminal": metric_gem_c,
        "metric_geminal_coulomb": metric_gc_c,
    }
    with (
        batch_execute(run, ab_lin_graph) as exec_lin,
        batch_execute(run, ab_quad_graph) as exec_quad,
    ):
        for i in range(frozen_occupied, oa):
            ii = i - frozen_occupied
            fit_i = np.ascontiguousarray(data["source/fit_alpha"][:, i])
            gem_i = np.ascontiguousarray(data["raw/geminal/alpha"][:, i])
            gc_i = np.ascontiguousarray(data["raw/geminal_coulomb/alpha"][:, i])
            sel_i = np.ascontiguousarray(selectors[i])
            sel_i_act = np.ascontiguousarray(selectors[i, occ_slice])
            for j in range(i, oa):
                jj = j - frozen_occupied
                mult = 1.0 if i == j else 2.0
                lin_in = {
                    **ab_lin_shared,
                    "fit_left": fit_i,
                    "fit_right": np.ascontiguousarray(data["source/fit_beta"][:, j]),
                    "raw_coulomb_right": np.ascontiguousarray(
                        data["source/raw_coulomb_beta"][:, j]
                    ),
                    "raw_geminal_left": gem_i,
                    "raw_geminal_right": np.ascontiguousarray(data["raw/geminal/beta"][:, j]),
                    "raw_geminal_coulomb_left": gc_i,
                    "raw_geminal_coulomb_right": np.ascontiguousarray(
                        data["raw/geminal_coulomb/beta"][:, j]
                    ),
                    "select_i": sel_i,
                    "select_j": np.ascontiguousarray(selectors[j]),
                }
                res = exec_lin(lin_in).outputs
                linear_v += mult * float(res["linear_v_energy"])
                qres = exec_quad(
                    {
                        **ab_quad_shared,
                        "select_i": sel_i_act,
                        "select_j": np.ascontiguousarray(selectors[j, occ_slice]),
                    }
                ).outputs
                quadratic_energy += mult * (
                    float(qres["b_energy"]) - float(qres["x_fock_energy"])
                )
                w = res["w_obs"]
                c = res["c_vv"]
                w_vo[ii, jj] = w[vir_slice, occ_slice]
                w_vv[ii, jj] = w[vir_slice, vir_slice]
                c_f12[ii, jj] = c
                if i < j:
                    w_vo[jj, ii] = w[occ_slice, vir_slice].T
                    w_vv[jj, ii] = w[vir_slice, vir_slice].T
                    c_f12[jj, ii] = c.T

    # 2. Same-spin (alpha_alpha) i < j: scalar V+B-X only (doubled for beta_beta).
    if o_act >= 2:
        aa_full_graph = unrestricted_f12_pair_sources(
            dims, dims, auxiliary, channel="alpha_alpha"
        )
        aa_v_graph = Program({"linear_v_energy": aa_full_graph.outputs["linear_v_energy"]})
        aa_quad_graph = unrestricted_fixed_geminal_quadratic_pair(
            dims, dims, auxiliary, channel="alpha_alpha",
            frozen_occupied=(frozen_occupied, frozen_occupied),
        )
        aa_quad_shared = _quadratic_pair_fields(data, fock, "alpha", "alpha", occ_map)
        aa_lin_shared = {
            "metric_geminal": metric_gem_c,
            "metric_geminal_coulomb": metric_gc_c,
        }
        with (
            batch_execute(run, aa_v_graph) as exec_aa_v,
            batch_execute(run, aa_quad_graph) as exec_aa_q,
        ):
            for i in range(frozen_occupied, oa):
                fit_i = np.ascontiguousarray(data["source/fit_alpha"][:, i])
                gem_i = np.ascontiguousarray(data["raw/geminal/alpha"][:, i])
                gc_i = np.ascontiguousarray(data["raw/geminal_coulomb/alpha"][:, i])
                sel_i = np.ascontiguousarray(selectors[i])
                sel_i_act = np.ascontiguousarray(selectors[i, occ_slice])
                for j in range(i + 1, oa):
                    res = exec_aa_v(
                        {
                            **aa_lin_shared,
                            "fit_left": fit_i,
                            "fit_right": np.ascontiguousarray(data["source/fit_alpha"][:, j]),
                            "raw_coulomb_right": np.ascontiguousarray(
                                data["source/raw_coulomb_alpha"][:, j]
                            ),
                            "raw_geminal_left": gem_i,
                            "raw_geminal_right": np.ascontiguousarray(
                                data["raw/geminal/alpha"][:, j]
                            ),
                            "raw_geminal_coulomb_left": gc_i,
                            "raw_geminal_coulomb_right": np.ascontiguousarray(
                                data["raw/geminal_coulomb/alpha"][:, j]
                            ),
                            "select_i": sel_i,
                            "select_j": np.ascontiguousarray(selectors[j]),
                        }
                    ).outputs
                    linear_v += 2.0 * float(res["linear_v_energy"])
                    qres = exec_aa_q(
                        {
                            **aa_quad_shared,
                            "select_i": sel_i_act,
                            "select_j": np.ascontiguousarray(selectors[j, occ_slice]),
                        }
                    ).outputs
                    quadratic_energy += 2.0 * (
                        float(qres["b_energy"]) - float(qres["x_fock_energy"])
                    )

    total_energy = float(linear_v + quadratic_energy)
    if not np.isfinite(total_energy):
        raise ValueError("Nonfinite spatial RHF fixed-geminal energy")
    return w_vo, w_vv, c_f12, total_energy


def unrestricted_cc_f12_residual_sources(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    whitener: np.ndarray,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    native_library: Path | None = AUTO_NATIVE_LIBRARY,  # type: ignore[assignment]
    host_reference: bool | None = None,
    max_workspace_bytes: int = 0,
) -> tuple[dict[str, np.ndarray], float]:
    """Generate all W/C input blocks and the linear occupied V energy."""
    return _fixed_source_terms(
        space,
        data,
        fock,
        whitener,
        run,
        frozen_occupied,
        quadratic=False,
        native_library=native_library,
        host_reference=host_reference,
        max_workspace_bytes=max_workspace_bytes,
    )


def unrestricted_cc_f12_fixed_source_feeds(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    whitener: np.ndarray,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    native_library: Path | None = AUTO_NATIVE_LIBRARY,  # type: ignore[assignment]
    host_reference: bool | None = None,
    max_workspace_bytes: int = 0,
) -> dict[str, np.ndarray]:
    """Build W/C and the scalar V+B-X feeds from a single molecular DF frame."""
    feeds, energy = _fixed_source_terms(
        space,
        data,
        fock,
        whitener,
        run,
        frozen_occupied,
        quadratic=True,
        native_library=native_library,
        host_reference=host_reference,
        max_workspace_bytes=max_workspace_bytes,
    )
    if not np.isfinite(energy):
        raise ValueError("Nonfinite fixed-geminal energy")
    feeds["fixed_geminal_energy"] = np.asarray(energy, dtype=np.float64)
    return feeds


class _SourceDirection(NamedTuple):
    data: dict[str, np.ndarray]
    fock: dict[str, np.ndarray]
    coulomb_metric: np.ndarray


def unrestricted_cc_f12_fixed_source_tangent(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    whitener: np.ndarray,
    run: Callable,
    *,
    data_tangent: dict[str, np.ndarray],
    fock_tangent: dict[str, np.ndarray],
    coulomb_metric_tangent: np.ndarray,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> dict[str, np.ndarray]:
    """Return dW_vo, dW_vv, dC and d(V+B-X) in the CC equation feed layout."""
    direction = _SourceDirection(data_tangent, fock_tangent, coulomb_metric_tangent)
    feeds, scalar = _fixed_source_terms(
        space,
        data,
        fock,
        whitener,
        run,
        frozen_occupied,
        quadratic=True,
        direction=direction,
    )
    if not np.isfinite(scalar):
        raise ValueError("Nonfinite fixed-geminal energy tangent")
    feeds["fixed_geminal_energy"] = np.asarray(scalar, dtype=np.float64)
    return feeds


def _pair_kernel(graph: Program, tangent: bool) -> Program:
    if not tangent:
        return graph
    derivative = linearize(graph, None).program
    return Program({name: derivative.outputs["d_" + name] for name in graph.outputs})


def _linear_pair_fields(
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    fit_obs: dict[str, np.ndarray],
    raw_obs: dict[str, np.ndarray],
    sl: str,
    sr: str,
    i: int,
    j: int,
) -> dict[str, np.ndarray]:
    fields = {
        "fit_obs_left": fit_obs[sl],
        "raw_coulomb_obs_right": raw_obs[sr],
        "fock_left": fock["fock_" + sl],
        "fit_left": data["source/fit_" + sl][:, i],
        "fit_right": data["source/fit_" + sr][:, j],
        "raw_coulomb_right": data["source/raw_coulomb_" + sr][:, j],
    }
    if sl != sr:
        fields["cross_overlap"] = data["cross_overlap"]
        fields["fock_right"] = fock["fock_" + sr]
    for op in ("geminal", "geminal_coulomb"):
        fields["metric_" + op] = data["metric/" + op]
        fields["raw_" + op + "_left"] = data[f"raw/{op}/{sl}"][:, i]
        fields["raw_" + op + "_right"] = data[f"raw/{op}/{sr}"][:, j]
    return fields


def _quadratic_pair_fields(
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    left: str,
    right: str,
    occupied: dict[str, slice],
) -> dict[str, np.ndarray]:
    operators = ("geminal", "squared_geminal", "double_commutator")
    fields = {f"metric_{op}": data[f"metric/{op}"] for op in operators}
    if left != right:
        fields["cross_overlap"] = data["cross_overlap"]
    legs = ((left, "left"),) if left == right else ((left, "left"), (right, "right"))
    for spin, side in legs:
        active = occupied[spin[0]]
        fields[f"fock_{side}"] = fock[f"fock_{spin}"]
        fields[f"exchange_{side}"] = fock[f"exchange_{spin}"]
        fields[f"fit_{side}"] = data[f"source/fit_{spin}"][:, active]
        for op in operators:
            fields[f"raw_{op}_{side}"] = data[f"raw/{op}/{spin}"][:, active]
    return {name: np.ascontiguousarray(value) for name, value in fields.items()}


def _w_batch_dims(n_pairs: int, aux: int, obs: int) -> tuple[int, int]:
    target_bytes = 192 << 20
    pair_batch = min(n_pairs, 32)
    slice_bytes = max(1, pair_batch * obs * obs * 8)
    aux_tile = max(1, min(aux, target_bytes // slice_bytes))
    if aux_tile == 1 and pair_batch > 1:
        pair_batch = max(1, min(n_pairs, target_bytes // max(1, obs * obs * 8)))
    return pair_batch, aux_tile


def _fixed_source_terms(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    whitener: np.ndarray,
    run: Callable,
    frozen_occupied: tuple[int, int],
    *,
    quadratic: bool,
    direction: _SourceDirection | None = None,
    native_library: Path | None = AUTO_NATIVE_LIBRARY,  # type: ignore[assignment]
    host_reference: bool | None = None,
    max_workspace_bytes: int = 0,
) -> tuple[dict[str, np.ndarray], float]:
    oa, ob, obs, extended, auxiliary = space
    pairs = list(occupied_pair_blocks(space, frozen_occupied=frozen_occupied))
    occupied = dict(zip(("a", "b"), (oa, ob), strict=True))
    frozen = dict(zip(("a", "b"), frozen_occupied, strict=True))
    active = {spin: no - frozen[spin] for spin, no in occupied.items()}
    virtual = {spin: obs - no for spin, no in occupied.items()}
    occ_slice = {spin: slice(frozen[spin], no) for spin, no in occupied.items()}
    vir_slice = {spin: slice(no, obs) for spin, no in occupied.items()}

    sources = {}
    for i, j, p, q in product("ab", repeat=4):
        if sorted((i, j)) != sorted((p, q)):
            continue
        spins = i + j + p + q
        shape = (active[i], active[j], virtual[p])
        sources["w_vo_" + spins] = np.zeros((*shape, active[q]))
        sources["w_vv_" + spins] = np.zeros((*shape, virtual[q]))
        sources["c_f12_" + spins] = np.zeros((*shape, virtual[q]))

    if not pairs:
        return sources, 0.0

    spin_sym = (
        oa == ob
        and frozen["a"] == frozen["b"]
        and oa > 0
        and np.array_equal(fock["fock_alpha"], fock["fock_beta"])
        and np.array_equal(fock["exchange_alpha"], fock["exchange_beta"])
        and np.array_equal(data["source/fit_alpha"], data["source/fit_beta"])
        and np.array_equal(
            data["source/raw_coulomb_alpha"][:, :obs, :obs],
            data["source/raw_coulomb_beta"][:, :obs, :obs],
        )
        and np.array_equal(data["cross_overlap"], data["cross_overlap"].T)
        and all(
            np.array_equal(data[f"raw/{op}/alpha"], data[f"raw/{op}/beta"])
            for op in ("geminal", "squared_geminal", "geminal_coulomb", "double_commutator")
        )
    )

    lib_path, use_host_ref = resolve_ccfixed_library(native_library, host_reference, run)
    raw_obs = {
        spin: np.ascontiguousarray(data["source/raw_coulomb_" + spin][:, :obs, :obs])
        for spin in {tag for block in pairs for tag in (block.left, block.right)}
    }
    fit_spins = (
        {
            block.left
            for block in pairs
            if not (spin_sym and block.left == "beta" and block.right == "beta")
        }
        if direction is None
        else {block.left for block in pairs}
    )
    fit_obs = {
        spin: coulomb_fit_forward(
            whitener,
            raw_obs[spin],
            native_library=lib_path,
            host_reference=use_host_ref,
            run=run,
            max_workspace_bytes=max_workspace_bytes,
        )
        for spin in fit_spins
    }

    linear_v = 0.0
    quadratic_energy = 0.0

    if direction is None:
        pair_fn = bind_fixed_pair_sources(lib_path, use_host_ref)
        if pair_fn is not None:
            for sl, sr, nl, nr, _, pair_indices in pairs:
                if spin_sym and sl == "beta" and sr == "beta":
                    continue
                i_spin, j_spin = sl[0], sr[0]
                mirror_bb = spin_sym and sl == "alpha" and sr == "alpha"
                all_pairs = list(pair_indices)
                if not all_pairs:
                    continue
                if spin_sym and sl == "alpha" and sr == "beta":
                    eval_pairs = [(i, j) for i, j in all_pairs if i <= j]
                else:
                    eval_pairs = all_pairs
                same_spin = sl == sr
                fl, fr = frozen[i_spin], frozen[j_spin]
                pair_mult = np.array(
                    [
                        2.0 if (mirror_bb or (spin_sym and not same_spin and i < j)) else 1.0
                        for i, j in eval_pairs
                    ],
                    dtype=np.float64,
                )
                w_eval, c_eval, e_v, e_b, e_x = run_fixed_pair_sources_native(
                    pair_fn,
                    same_spin=same_spin,
                    compute_wc=True,
                    compute_quad=quadratic,
                    nl=nl,
                    nr=nr,
                    fl=fl,
                    fr=fr,
                    obs=obs,
                    ext=extended,
                    aux=auxiliary,
                    pairs=eval_pairs,
                    pair_mult=pair_mult,
                    fit_l=data["source/fit_" + sl],
                    fit_r=None if same_spin else data["source/fit_" + sr],
                    raw_coul_r=data["source/raw_coulomb_" + sr],
                    raw_g_l=data["raw/geminal/" + sl],
                    raw_g_r=None if same_spin else data["raw/geminal/" + sr],
                    raw_gc_l=data["raw/geminal_coulomb/" + sl],
                    raw_gc_r=None if same_spin else data["raw/geminal_coulomb/" + sr],
                    raw_sq_l=data["raw/squared_geminal/" + sl] if quadratic else None,
                    raw_sq_r=(
                        None
                        if (same_spin or not quadratic)
                        else data["raw/squared_geminal/" + sr]
                    ),
                    raw_dc_l=data["raw/double_commutator/" + sl] if quadratic else None,
                    raw_dc_r=(
                        None
                        if (same_spin or not quadratic)
                        else data["raw/double_commutator/" + sr]
                    ),
                    metric_g=data["metric/geminal"],
                    metric_gc=data["metric/geminal_coulomb"],
                    metric_sq=data["metric/squared_geminal"] if quadratic else None,
                    metric_dc=data["metric/double_commutator"] if quadratic else None,
                    fock_l=fock["fock_" + sl],
                    fock_r=None if same_spin else fock["fock_" + sr],
                    exch_l=fock["exchange_" + sl] if quadratic else None,
                    exch_r=None if (same_spin or not quadratic) else fock["exchange_" + sr],
                    cross_overlap=None if same_spin else data["cross_overlap"],
                    fit_obs_l=fit_obs[sl],
                    raw_obs_r=raw_obs[sr],
                    max_workspace_bytes=max_workspace_bytes,
                )
                assert w_eval is not None and c_eval is not None
                linear_v += e_v
                if quadratic:
                    quadratic_energy += e_b - e_x

                eval_map = {pair: idx for idx, pair in enumerate(eval_pairs)}
                for i, j in all_pairs:
                    if (i, j) in eval_map:
                        idx = eval_map[(i, j)]
                        w, c = w_eval[idx], c_eval[idx]
                    else:
                        idx = eval_map[(j, i)]
                        w, c = w_eval[idx].T, c_eval[idx].T
                    target_orders = [(i_spin, j_spin, w, c)]
                    if not same_spin:
                        target_orders.append((j_spin, i_spin, -w.T, -c.T))
                    for p, q, w_tile, c_tile in target_orders:
                        for name, tile in (
                            ("w_vo_", w_tile[vir_slice[p], occ_slice[q]]),
                            ("w_vv_", w_tile[vir_slice[p], vir_slice[q]]),
                            ("c_f12_", c_tile),
                        ):
                            ii, jj = i - fl, j - fr
                            sources[name + i_spin + j_spin + p + q][ii, jj] = tile
                            sources[name + j_spin + i_spin + p + q][jj, ii] = -tile
                            if mirror_bb:
                                sources[name + "bbbb"][ii, jj] = tile
                                sources[name + "bbbb"][jj, ii] = -tile
            return sources, linear_v + quadratic_energy if quadratic else linear_v

        if require_aot_ccfixed():
            raise RuntimeError(
                f"{REQUIRE_AOT_CCFIXED_ENV}=1 forbids TensorIR fallback in "
                "_fixed_source_terms (native_library missing or "
                "gqc_f12_cc_fixed_pair_sources symbol unavailable)"
            )

        selectors = np.eye(extended)
        # Fast primal path: auxiliary-tiled batched OBS W projection + 2D-sliced quadratic B-X.
        fock_mixed: dict[str, dict[str, np.ndarray]] = {}
        if quadratic:
            for spin in ("alpha", "beta"):
                act = occ_slice[spin[0]]
                if act.stop <= act.start:
                    continue
                if spin_sym and spin == "beta" and "alpha" in fock_mixed:
                    fock_mixed["beta"] = fock_mixed["alpha"]
                else:
                    f_oo = np.ascontiguousarray(fock["fock_" + spin][act, act])
                    fock_mixed[spin] = {
                        name: np.einsum("ki,akp->aip", f_oo, data[prefix + spin][:, act, :])
                        for name, prefix in _X_FOCK_FIELDS
                    }

        for sl, sr, nl, nr, _, pair_indices in pairs:
            if spin_sym and sl == "beta" and sr == "beta":
                continue
            i_spin, j_spin = sl[0], sr[0]
            mirror_bb = spin_sym and sl == "alpha" and sr == "alpha"
            all_pairs = list(pair_indices)
            if not all_pairs:
                continue
            if spin_sym and sl == "alpha" and sr == "beta":
                eval_pairs = [(i, j) for i, j in all_pairs if i <= j]
            else:
                eval_pairs = all_pairs
            n_eval = len(eval_pairs)
            left = (nl, obs - nl, extended - obs)
            right = (nr, obs - nr, extended - obs)
            channel = sl + "_" + sr

            gem_graph = unrestricted_f12_pair_geminals(
                left, right, auxiliary, channel=channel
            )
            shared_lin = {
                "fock_left": np.ascontiguousarray(fock["fock_" + sl]),
                "metric_geminal": np.ascontiguousarray(data["metric/geminal"]),
                "metric_geminal_coulomb": np.ascontiguousarray(
                    data["metric/geminal_coulomb"]
                ),
            }
            if sl != sr:
                shared_lin["cross_overlap"] = np.ascontiguousarray(data["cross_overlap"])
                shared_lin["fock_right"] = np.ascontiguousarray(fock["fock_" + sr])

            if quadratic:
                quad_graph = unrestricted_fixed_geminal_quadratic_pair_sliced(
                    left, right, auxiliary, channel=channel
                )
                shared_quad = {
                    "fock_left": shared_lin["fock_left"],
                    "exchange_left": np.ascontiguousarray(fock["exchange_" + sl]),
                    "metric_geminal": shared_lin["metric_geminal"],
                    "metric_squared_geminal": np.ascontiguousarray(
                        data["metric/squared_geminal"]
                    ),
                    "metric_double_commutator": np.ascontiguousarray(
                        data["metric/double_commutator"]
                    ),
                }
                if sl != sr:
                    shared_quad["fock_right"] = shared_lin["fock_right"]
                    shared_quad["exchange_right"] = np.ascontiguousarray(
                        fock["exchange_" + sr]
                    )
                    shared_quad["cross_overlap"] = shared_lin["cross_overlap"]
            else:
                quad_graph = None

            f_obs_eval = np.empty((n_eval, obs, obs), dtype=np.float64)
            w_eval = np.empty((n_eval, obs, obs), dtype=np.float64)
            c_eval: list[np.ndarray] = []
            fl, fr = frozen[i_spin], frozen[j_spin]
            quad_cm = (
                batch_execute(run, quad_graph)
                if quad_graph is not None
                else nullcontext(None)
            )
            with batch_execute(run, gem_graph) as exec_lin, quad_cm as exec_quad:
                for idx, (i, j) in enumerate(eval_pairs):
                    rows = dict(shared_lin)
                    rows["fit_left"] = np.ascontiguousarray(data["source/fit_" + sl][:, i])
                    rows["fit_right"] = np.ascontiguousarray(data["source/fit_" + sr][:, j])
                    rows["raw_coulomb_right"] = np.ascontiguousarray(
                        data["source/raw_coulomb_" + sr][:, j]
                    )
                    for op in ("geminal", "geminal_coulomb"):
                        rows[f"raw_{op}_left"] = np.ascontiguousarray(
                            data[f"raw/{op}/{sl}"][:, i]
                        )
                        rows[f"raw_{op}_right"] = np.ascontiguousarray(
                            data[f"raw/{op}/{sr}"][:, j]
                        )
                    rows["select_i"] = selectors[i]
                    rows["select_j"] = selectors[j]
                    result = exec_lin(rows).outputs
                    lv = float(result["linear_v_energy"])
                    mult = 2.0 if (mirror_bb or (spin_sym and sl != sr and i < j)) else 1.0
                    linear_v += mult * lv
                    f_obs_eval[idx] = result["geminal_obs"]
                    w_eval[idx] = result["geminal_coulomb_obs"]
                    c_eval.append(result["c_vv"])
                    if quadratic:
                        assert exec_quad is not None
                        qvals = dict(shared_quad)
                        qvals["select_i_full"] = selectors[i]
                        qvals["select_j_full"] = selectors[j]
                        for name, prefix in _QUAD_FIELDS:
                            qvals[f"{name}_left"] = np.ascontiguousarray(
                                data[prefix + sl][:, i]
                            )
                            qvals[f"{name}_right"] = np.ascontiguousarray(
                                data[prefix + sr][:, j]
                            )
                        ii, jj = i - fl, j - fr
                        for name, _ in _X_FOCK_FIELDS:
                            qvals[f"fock_{name}_left"] = np.ascontiguousarray(
                                fock_mixed[sl][name][:, ii]
                            )
                            qvals[f"fock_{name}_right"] = np.ascontiguousarray(
                                fock_mixed[sr][name][:, jj]
                            )
                        values = exec_quad(qvals).outputs
                        qv = float(values["b_energy"]) - float(values["x_fock_energy"])
                        quadratic_energy += mult * qv

            # Auxiliary-tiled batched OBS Coulomb projection: w_eval -= sum_a D_a f_obs R_a^T
            pair_batch, aux_tile = _w_batch_dims(n_eval, auxiliary, obs)
            for k_len, k_slices in _tile_groups(n_eval, pair_batch):
                for p_len, p_slices in _tile_groups(auxiliary, aux_tile):
                    proj_prog = batched_obs_w_projection(obs, p_len, k_len)
                    with batch_execute(run, proj_prog) as exec_proj:
                        for ks in k_slices:
                            f_b = np.ascontiguousarray(f_obs_eval[ks])
                            for ps in p_slices:
                                w_eval[ks] -= exec_proj(
                                    {
                                        "geminal_obs_batch": f_b,
                                        "fit_obs_left_tile": np.ascontiguousarray(
                                            fit_obs[sl][ps]
                                        ),
                                        "raw_coulomb_obs_right_tile": np.ascontiguousarray(
                                            raw_obs[sr][ps]
                                        ),
                                    }
                                ).outputs["w_proj_batch"]
            del f_obs_eval

            eval_map = {pair: idx for idx, pair in enumerate(eval_pairs)}
            for i, j in all_pairs:
                if (i, j) in eval_map:
                    idx = eval_map[(i, j)]
                    w, c = w_eval[idx], c_eval[idx]
                else:
                    idx = eval_map[(j, i)]
                    w, c = w_eval[idx].T, c_eval[idx].T
                target_orders = [(i_spin, j_spin, w, c)]
                if sl != sr:
                    target_orders.append((j_spin, i_spin, -w.T, -c.T))
                for p, q, w_tile, c_tile in target_orders:
                    for name, tile in (
                        ("w_vo_", w_tile[vir_slice[p], occ_slice[q]]),
                        ("w_vv_", w_tile[vir_slice[p], vir_slice[q]]),
                        ("c_f12_", c_tile),
                    ):
                        ii, jj = i - fl, j - fr
                        sources[name + i_spin + j_spin + p + q][ii, jj] = tile
                        sources[name + j_spin + i_spin + p + q][jj, ii] = -tile
                        if mirror_bb:
                            sources[name + "bbbb"][ii, jj] = tile
                            sources[name + "bbbb"][jj, ii] = -tile
        return sources, linear_v + quadratic_energy if quadratic else linear_v

    # Tangent path (used by directional-derivative diagnostic tests)
    selectors = np.eye(extended)
    d_raw_obs = {
        spin: np.ascontiguousarray(
            direction.data["source/raw_coulomb_" + spin][:, :obs, :obs]
        )
        for spin in raw_obs
    }
    d_fit_obs = {
        spin: coulomb_fit_forward(
            whitener,
            d_raw_obs[spin]
            - (direction.coulomb_metric @ fit.reshape(auxiliary, -1)).reshape(
                auxiliary, obs, obs
            ),
            native_library=lib_path,
            host_reference=use_host_ref,
            run=run,
            max_workspace_bytes=max_workspace_bytes,
        )
        for spin, fit in fit_obs.items()
    }
    for sl, sr, nl, nr, _, pair_indices in pairs:
        i_spin, j_spin = sl[0], sr[0]
        left = (nl, obs - nl, extended - obs)
        right = (nr, obs - nr, extended - obs)
        channel = sl + "_" + sr
        graph = _pair_kernel(
            unrestricted_f12_pair_sources(left, right, auxiliary, channel=channel), True
        )
        if quadratic:
            quadratic_graph = _pair_kernel(
                unrestricted_fixed_geminal_quadratic_pair(
                    left,
                    right,
                    auxiliary,
                    channel=channel,
                    frozen_occupied=(frozen[i_spin], frozen[j_spin]),
                ),
                True,
            )
            quadratic_fields = _quadratic_pair_fields(data, fock, sl, sr, occ_slice)
            tangent_fields = _quadratic_pair_fields(
                direction.data, direction.fock, sl, sr, occ_slice
            )
            quadratic_fields.update({"d_" + k: v for k, v in tangent_fields.items()})
        quad_cm = (
            batch_execute(run, quadratic_graph) if quadratic else nullcontext(None)
        )
        with batch_execute(run, graph) as exec_lin, quad_cm as exec_quad:
            for i, j in pair_indices:
                rows = _linear_pair_fields(data, fock, fit_obs, raw_obs, sl, sr, i, j)
                tangent_rows = _linear_pair_fields(
                    direction.data, direction.fock, d_fit_obs, d_raw_obs, sl, sr, i, j
                )
                rows.update({"d_" + k: v for k, v in tangent_rows.items()})
                rows.update(select_i=selectors[i], select_j=selectors[j])
                result = exec_lin(
                    {k: np.ascontiguousarray(x) for k, x in rows.items()}
                ).outputs
                linear_v += float(result["linear_v_energy"])
                if quadratic:
                    assert exec_quad is not None
                    values = exec_quad(
                        {
                            **quadratic_fields,
                            "select_i": selectors[i, occ_slice[i_spin]],
                            "select_j": selectors[j, occ_slice[j_spin]],
                        }
                    ).outputs
                    quadratic_energy += float(values["b_energy"]) - float(
                        values["x_fock_energy"]
                    )
                w, c = result["w_obs"], result["c_vv"]
                target_orders = [(i_spin, j_spin, w, c)]
                if sl != sr:
                    target_orders.append((j_spin, i_spin, -w.T, -c.T))
                for p, q, w_tile, c_tile in target_orders:
                    for name, tile in (
                        ("w_vo_", w_tile[vir_slice[p], occ_slice[q]]),
                        ("w_vv_", w_tile[vir_slice[p], vir_slice[q]]),
                        ("c_f12_", c_tile),
                    ):
                        ii, jj = i - frozen[i_spin], j - frozen[j_spin]
                        sources[name + i_spin + j_spin + p + q][ii, jj] = tile
                        sources[name + j_spin + i_spin + p + q][jj, ii] = -tile
    return sources, linear_v + quadratic_energy if quadratic else linear_v
