"""All-active small-molecule CC-F12 correlation from a native common DF and orbital frame."""

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from generativeqc_compiler.cc.f12_residual import df_standard_v
from generativeqc_compiler.mp2.f12_density_fitting import coulomb_fit_response
from generativeqc_compiler.mp2.f12_occupied_pair_overlap import all_occupied_vx
from generativeqc_compiler.tensor import Program
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks.f12_h2_cc_molecular_correlation import _small_df_coulomb
from benchmarks.f12_molecular_pairs import factor_inputs, pair_plan
from benchmarks.f12_restricted_multipair_cc import solve_all_pair_cc
from benchmarks.f12_restricted_multipair_lambda import (
    prepare_lagrangian_gradient,
    relaxed_source_weights,
    solve_lambda,
)


@lru_cache(maxsize=10)
def _bc_graph(o: int, n: int, r: int, a: int, diagonal: bool) -> Program:
    graph = pair_plan(o, n - o, r - n, diagonal=diagonal, auxiliaries=a)
    keys = (
        ("b_direct", "c_direct") if diagonal else ("b_direct", "b_exchange", "c_direct")
    )
    return Program(
        {key: graph.outputs[key] for key in keys}, provenance=graph.provenance
    )


def _live_inputs(graph: Program, source: dict) -> dict:
    return {
        node.attrs["name"]: source[node.attrs["name"]]
        for node in graph.live_nodes
        if node.op == "input"
    }


def _add_pair_weights(target: dict, weights: dict, m: int, n: int) -> None:
    for name, weight in weights.items():
        key = name.removeprefix("bar_")
        if key.endswith(("_i", "_j")):
            target[key[:-2]][:, m if key.endswith("_i") else n] += weight
        else:
            target[key] += weight


def molecular_cc_correlation(
    mol: dict, fock: dict, whitener: np.ndarray, run: Callable, *, pullback: bool
) -> tuple:
    """Return all relaxed electronic weights to the same HF/CABS molecular response chain.

    The fixed cusp geminal energy samples only B_mn,mn and B_mn,nm. Other B
    entries do not enter this approximation. All X and V occupied rows are
    generated so that the occupied Fock response is retained.
    """
    o, n, r = map(int, mol["space"])
    a = len(whitener)
    v = n - o
    if not 1 <= o <= 8 or not 1 <= v or n > 48:
        raise ValueError(
            "bounded all-pair molecular diagnostic requires 1–8 occupied and at most 48 orbitals"
        )
    raw = np.ascontiguousarray(mol["raw_coulomb_all"][:, :n, :n])
    fitg = coulomb_fit_response(a, n * n)
    fit_inputs = {
        "metric_whitener": whitener,
        "three_center": raw.reshape(a, -1),
        "fit_bar": np.zeros((a, n * n)),
    }
    fit = run(fitg, fit_inputs).outputs["fit"].reshape(raw.shape)
    gg = _small_df_coulomb(n, a)
    gfeed = {"fit": fit, "raw": raw}
    eri = run(gg, gfeed).outputs["eri"]
    table = all_occupied_vx(o, v, r - n, a)
    vf = _live_inputs(table, mol)
    rows = run(table, vf).outputs
    cc = {
        "f": fock["fock"][:n, :n],
        "eri": eri,
        "v": rows["v"],
        "x": rows["x"],
        "b": np.zeros((o, o, o, o)),
        "c": np.empty((o, o, v, v)),
        "external": np.empty((o, o, n, n)),
    }
    allowed_b = np.zeros_like(cc["b"], dtype=bool)
    external_graphs = {m: df_standard_v(n, a, diagonal=m) for m in (False, True)}
    records = []
    eye = np.eye(n)
    for left, right in zip(*np.triu_indices(o), strict=True):
        m, k = int(left), int(right)
        diagonal = m == k
        ext = external_graphs[diagonal]
        extfeed = {"fit_all": fit, "raw_coulomb_all": raw, "select_m": eye[m]}
        if not diagonal:
            extfeed["select_n"] = eye[k]
        for field in ("f12", "f12_over_coulomb"):
            src = mol["raw_" + field]
            extfeed["raw_" + field + "_m"] = np.ascontiguousarray(src[:, m, :n])
            if not diagonal:
                extfeed["raw_" + field + "_n"] = np.ascontiguousarray(src[:, k, :n])
            extfeed["metric_" + field] = mol["metric_" + field]
        z = run(ext, extfeed).outputs["v"]
        cc["external"][m, k] = z
        bc = _bc_graph(o, n, r, a, diagonal)
        bcfeed = _live_inputs(bc, factor_inputs(mol, m, k, fock))
        terms = run(bc, bcfeed).outputs
        cc["b"][m, k, m, k] = terms["b_direct"]
        cc["c"][m, k] = terms["c_direct"]
        allowed_b[m, k, m, k] = True
        if not diagonal:
            cc["external"][k, m] = z.T
            cc["c"][k, m] = terms["c_direct"].T
            cc["b"][k, m, k, m] = terms["b_direct"]
            cc["b"][k, m, m, k] = cc["b"][m, k, k, m] = terms["b_exchange"]
            allowed_b[k, m, k, m] = allowed_b[k, m, m, k] = allowed_b[m, k, k, m] = True
        if pullback:
            records.append((m, k, ext, extfeed, bc, bcfeed))
    solved = solve_all_pair_cc(cc, run)
    energies = solved["component_energies_Eh"]
    details = {
        "ordinary_cc_correlation_Eh": energies[0],
        "common_geminal_Eh": energies[1],
        "extra_f12b_geminal_Eh": energies[2],
        "cc_iterations": len(solved["residual_history"]),
        "cc_residual": solved["residual_history"][-1],
    }
    if not pullback:
        return float(sum(energies)), {}, details
    t1, t2 = solved["t1"], solved["t2"]
    gradient = prepare_lagrangian_gradient(cc, t1, t2, run)
    dual = solve_lambda(gradient, cc["f"], o)
    details.update(
        lambda_iterations=len(dual["residual_history"]),
        lambda_residual=dual["residual_history"][-1],
    )
    relaxed = relaxed_source_weights(cc, t1, t2, dual["lambda1"], dual["lambda2"], run)
    if np.any(relaxed["b"][~allowed_b] != 0):
        raise ValueError(
            "fixed-geminal equations requested an uncomputed interpair B element"
        )
    keys = (
        "fit",
        "raw_coulomb",
        "raw_f12",
        "raw_f12_squared",
        "raw_f12_over_coulomb",
        "raw_double_commutator",
        "metric_f12",
        "metric_f12_squared",
        "metric_f12_over_coulomb",
        "metric_double_commutator",
        "raw_coulomb_all",
    )
    ans = {key: np.zeros_like(mol[key]) for key in keys}
    ans["fock"] = np.zeros_like(fock["fock"])
    ans["exchange"] = np.zeros_like(fock["exchange"])
    ans["fock"][:n, :n] = relaxed["f"]
    gr = run(
        transpose_program(gg, ("eri",)).program, {**gfeed, "bar_eri": relaxed["eri"]}
    ).outputs
    fitbar = gr["bar_fit"].copy()
    rawbar = gr["bar_raw"].copy()
    vr = run(
        transpose_program(table, ("v", "x")).program,
        {**vf, "bar_v": relaxed["v"], "bar_x": relaxed["x"]},
    ).outputs
    for key, x in vr.items():
        ans[key.removeprefix("bar_")] += x
    for m, k, ext, extfeed, bc, bcfeed in records:
        zbar = relaxed["external"][m, k].copy()
        cbar = relaxed["c"][m, k].copy()
        bbar = relaxed["b"][m, k, m, k]
        seeds = {}
        if m != k:
            zbar += relaxed["external"][k, m].T
            cbar += relaxed["c"][k, m].T
            bbar += relaxed["b"][k, m, k, m]
            seeds["bar_b_exchange"] = np.asarray(
                relaxed["b"][k, m, m, k] + relaxed["b"][m, k, k, m]
            )
        seeds.update(bar_b_direct=np.asarray(bbar), bar_c_direct=cbar)
        bpull = run(
            transpose_program(bc, tuple(bc.outputs)).program, {**bcfeed, **seeds}
        ).outputs
        _add_pair_weights(ans, bpull, m, k)
        xpull = run(
            transpose_program(ext, ("v",)).program, {**extfeed, "bar_v": zbar}
        ).outputs
        fitbar += xpull["bar_fit_all"]
        rawbar += xpull["bar_raw_coulomb_all"]
        for field in ("f12", "f12_over_coulomb"):
            ans["metric_" + field] += xpull["bar_metric_" + field]
            ans["raw_" + field][:, m, :n] += xpull["bar_raw_" + field + "_m"]
            if m != k:
                ans["raw_" + field][:, k, :n] += xpull["bar_raw_" + field + "_n"]
    fitpull = run(fitg, {**fit_inputs, "fit_bar": fitbar.reshape(a, -1)}).outputs
    ans["raw_coulomb_all"][:, :n, :n] += rawbar + fitpull["three_center_bar"].reshape(
        raw.shape
    )
    ans["metric"] = fitpull["metric_bar"]
    return float(sum(energies)), ans, details
