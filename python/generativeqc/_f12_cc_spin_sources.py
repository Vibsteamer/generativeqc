"""Build external fixed-cusp CC sources from an unrestricted molecular DF frame."""

from collections.abc import Callable
from itertools import product

import numpy as np
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
    for sl, sr, nl, nr, _, pair_indices in pairs:
        i_spin, j_spin = sl[0], sr[0]
        graph = unrestricted_f12_pair_sources(
            (nl, obs - nl, extended - obs),
            (nr, obs - nr, extended - obs),
            auxiliary,
            channel=sl + "_" + sr,
        )
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
    return sources, linear_v
