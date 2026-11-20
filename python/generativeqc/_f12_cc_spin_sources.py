"""Build external fixed-cusp CC sources from an unrestricted molecular DF frame."""

from collections.abc import Callable
from itertools import product

import numpy as np
from generativeqc_compiler.cc.uccsd_f12_quadratic import (
    unrestricted_fixed_geminal_quadratic_pair,
)
from generativeqc_compiler.cc.uccsd_f12_sources import unrestricted_f12_pair_sources
from generativeqc_compiler.mp2.f12_density_fitting import coulomb_fit_response
from generativeqc_compiler.tensor import Program

from ._f12_spin_pairs import occupied_pair_blocks


def unrestricted_cc_f12_residual_sources(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    whitener: np.ndarray,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> tuple[dict[str, np.ndarray], float]:
    """Generate all W/C input blocks and the linear occupied V energy.

    The spin-frame data use the same fields as ``unrestricted_orbital_sources``.
    ``whitener.T @ whitener`` is the inverse common Coulomb metric. The
    caller sets the backend through ``run``. Frozen orbitals are excluded
    from the stored active CC sources, but are retained in the projector
    and extended Fock. Each occupied pair is processed once. The returned
    scalar is solely the linear V contribution and still needs the common
    B and occupied-Fock/X terms before solving the full CC equations.
    """
    return _fixed_source_terms(
        space, data, fock, whitener, run, frozen_occupied, quadratic=False
    )


def unrestricted_cc_f12_fixed_source_feeds(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    whitener: np.ndarray,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> dict[str, np.ndarray]:
    """Build W/C and the scalar V+B-X feeds from a single molecular DF frame.

    The extended Fock and positive exchange follow the spin Fock source
    convention. The frame supplies all raw and fitted operators from the same
    reference and auxiliary space. Frozen occupied orbitals are excluded from
    the CC amplitudes and the occupied Fock contraction of X.
    """
    feeds, energy = _fixed_source_terms(
        space, data, fock, whitener, run, frozen_occupied, quadratic=True
    )
    if not np.isfinite(energy):
        raise ValueError("Nonfinite fixed-geminal energy")
    feeds["fixed_geminal_energy"] = np.asarray(energy, dtype=np.float64)
    return feeds


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


def _fixed_source_terms(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    whitener: np.ndarray,
    run: Callable,
    frozen_occupied: tuple[int, int],
    *,
    quadratic: bool,
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

    fit_program = coulomb_fit_response(auxiliary, obs * obs)
    fit_graph = Program({"fit": fit_program.outputs["fit"]})
    raw_obs = {
        spin: np.ascontiguousarray(data["source/raw_coulomb_" + spin][:, :obs, :obs])
        for spin in {tag for block in pairs for tag in (block.left, block.right)}
    }
    fit_obs = {
        spin: run(fit_graph, {
            "metric_whitener": whitener,
            "three_center": raw_obs[spin].reshape(auxiliary, -1),
        }).outputs["fit"].reshape(auxiliary, obs, obs)
        for spin in {block.left for block in pairs}
    }

    selectors = np.eye(extended)
    linear_v = 0.
    quadratic_energy = 0.
    for sl, sr, nl, nr, _, pair_indices in pairs:
        i_spin, j_spin = sl[0], sr[0]
        left = (nl, obs - nl, extended - obs)
        right = (nr, obs - nr, extended - obs)
        channel = sl + "_" + sr
        graph = unrestricted_f12_pair_sources(
            left, right, auxiliary, channel=channel
        )
        if quadratic:
            quadratic_graph = unrestricted_fixed_geminal_quadratic_pair(
                left, right, auxiliary, channel=channel,
                frozen_occupied=(frozen[i_spin], frozen[j_spin]),
            )
            quadratic_fields = _quadratic_pair_fields(data, fock, sl, sr, occ_slice)
        shared = {
            "fit_obs_left": fit_obs[sl],
            "raw_coulomb_obs_right": raw_obs[sr],
            "fock_left": fock["fock_" + sl],
            **{f"metric_{operator}": data[f"metric/{operator}"]
               for operator in ("geminal", "geminal_coulomb")},
        }
        if sl != sr:
            shared["cross_overlap"] = data["cross_overlap"]
            shared["fock_right"] = fock["fock_" + sr]
        for i, j in pair_indices:
            rows = {**shared,
                    "select_i": selectors[i], "select_j": selectors[j],
                    "fit_left": data["source/fit_" + sl][:, i],
                    "fit_right": data["source/fit_" + sr][:, j],
                    "raw_coulomb_right": data["source/raw_coulomb_" + sr][:, j]}
            for operator in ("geminal", "geminal_coulomb"):
                rows[f"raw_{operator}_left"] = data[f"raw/{operator}/{sl}"][:, i]
                rows[f"raw_{operator}_right"] = data[f"raw/{operator}/{sr}"][:, j]
            result = run(graph, {k: np.ascontiguousarray(x) for k, x in rows.items()}).outputs
            linear_v += float(result["linear_v_energy"])
            if quadratic:
                values = run(quadratic_graph, {
                    **quadratic_fields,
                    "select_i": selectors[i, occ_slice[i_spin]],
                    "select_j": selectors[j, occ_slice[j_spin]],
                }).outputs
                quadratic_energy += float(values["b_energy"]) - float(values["x_fock_energy"])
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
