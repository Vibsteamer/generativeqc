"""Two-electron closed-shell CC-F12 correlation and its relaxed molecular-source weights."""

from collections.abc import Callable

import numpy as np
from generativeqc_compiler.cc.f12_residual import df_standard_v
from generativeqc_compiler.mp2.f12_density_fitting import coulomb_fit_response
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks.f12_molecular_pairs import factor_inputs, pair_plan
from benchmarks.f12_restricted_h2_solver import solve_fixed_sources


def _small_df_coulomb(orbitals: int, auxiliary: int) -> Program:
    # The bounded two-electron diagnostic permits dense ordinary ERIs for the
    # generated RCCSD residual. The normal multi-pair driver uses its own DF path.
    orb = IndexSpace("obs", "orbital", orbitals)
    aux = IndexSpace("auxiliary", "auxiliary", auxiliary)
    p = Index("p", aux)
    i, j, k, l = (Index(a, orb) for a in "ijkl")
    fit = input_tensor("fit", TensorSpec((p, i, j), role="input", differentiable=True))
    raw = input_tensor("raw", TensorSpec((p, k, l), role="input", differentiable=True))
    return Program({"eri": einsum("pij,pkl->ijkl", fit, raw)})


def h2_cc_correlation(
    mol: dict, fock: dict, whitener: np.ndarray, run: Callable, *, pullback: bool
) -> tuple:
    occ, obs, ri = map(int, mol["space"])
    if occ != 1 or not 2 <= obs <= 24:
        raise ValueError(
            "one occupied pair and at most 24 ordinary spatial orbitals required"
        )
    naux = len(whitener)
    raw = np.ascontiguousarray(mol["raw_coulomb_all"][:, :obs, :obs])
    fg = coulomb_fit_response(naux, obs * obs)
    fin = {
        "metric_whitener": whitener,
        "three_center": raw.reshape(naux, -1),
        "fit_bar": np.zeros((naux, obs * obs)),
    }
    fit = run(fg, fin).outputs["fit"].reshape(raw.shape)
    gg = _small_df_coulomb(obs, naux)
    gfeed = {"fit": fit, "raw": raw}
    eri = run(gg, gfeed).outputs["eri"]
    vg = df_standard_v(obs, naux, diagonal=True)
    vfeed = {"fit_all": fit, "raw_coulomb_all": raw, "select_m": np.eye(obs)[0]}
    ops = ("f12", "f12_over_coulomb")
    for op in ops:
        vfeed["raw_" + op + "_m"] = np.ascontiguousarray(mol["raw_" + op][:, 0, :obs])
        vfeed["metric_" + op] = mol["metric_" + op]
    external = run(vg, vfeed).outputs["v"]
    inherited = pair_plan(occ, obs - occ, ri - obs, diagonal=True, auxiliaries=naux)
    # B, X, C and the occupied V have the full strong-orthogonality projector;
    # the standard approximate V with virtual indices is independently built above.
    selected = {"b_direct": "b", "x_direct": "x", "v_direct": "v", "c_direct": "c"}
    pg = Program(
        {k: inherited.outputs[k] for k in selected}, provenance=inherited.provenance
    )
    pair_raw = factor_inputs(mol, 0, 0, fock)
    pf = {
        q.attrs["name"]: pair_raw[q.attrs["name"]]
        for q in pg.live_nodes
        if q.op == "input"
    }
    projected = run(pg, pf).outputs
    cc = {
        "eri": eri,
        "f": np.ascontiguousarray(fock["fock"][:obs, :obs]),
        "vext": external,
    }
    cc.update(
        {
            k: np.asarray(projected[name]).reshape((1, 1))
            if k != "c"
            else projected[name]
            for name, k in selected.items()
        }
    )
    solved = solve_fixed_sources(
        cc, run, source_pullback=pullback, solve_adjoint=pullback
    )
    comps = solved["ordered_component_energies_Eh"]
    info = {
        "ordinary_cc_correlation_Eh": comps[0],
        "common_geminal_Eh": comps[1],
        "extra_f12b_geminal_Eh": comps[2],
        "cc_iterations": solved["iterations"],
        "cc_residual": solved["residual_history"][-1],
        "lambda_iterations": len(solved["lambda_residual_history"]),
        "lambda_residual": solved["lambda_residual_history"][-1] if pullback else None,
    }
    if not pullback:
        return sum(comps), {}, info
    relaxed = solved["source_cotangents"]
    seeds = {
        "bar_" + name: relaxed[key].reshape(projected[name].shape)
        for name, key in selected.items()
    }
    pb = run(transpose_program(pg, tuple(selected)).program, {**pf, **seeds}).outputs
    vb = run(
        transpose_program(vg, ("v",)).program, {**vfeed, "bar_v": relaxed["vext"]}
    ).outputs
    gb = run(
        transpose_program(gg, ("eri",)).program, {**gfeed, "bar_eri": relaxed["eri"]}
    ).outputs
    fb = run(
        fg, {**fin, "fit_bar": (vb["bar_fit_all"] + gb["bar_fit"]).reshape(naux, -1)}
    ).outputs
    result = {
        "metric": fb["metric_bar"].copy(),
        "raw_coulomb_all": np.zeros_like(mol["raw_coulomb_all"]),
        "fock": np.zeros_like(fock["fock"]),
        "exchange": np.zeros_like(fock["exchange"]),
    }
    for name in (
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
    ):
        result[name] = np.zeros_like(mol[name])
    result["raw_coulomb_all"][:, :obs, :obs] = (
        gb["bar_raw"]
        + vb["bar_raw_coulomb_all"]
        + fb["three_center_bar"].reshape(raw.shape)
    )
    result["fock"][:obs, :obs] = relaxed["f"]
    for op in ops:
        result["raw_" + op][:, 0, :obs] += vb["bar_raw_" + op + "_m"]
        result["metric_" + op] += vb["bar_metric_" + op]
    for key, value in pb.items():
        name = key.removeprefix("bar_")
        if name.endswith("_i"):
            result[name[:-2]][:, 0] += value
        else:
            result[name] += value
    return sum(comps), result, info
