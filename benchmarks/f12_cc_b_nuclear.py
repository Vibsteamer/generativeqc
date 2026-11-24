"""Nuclear density weights of every B entry consumed by the fixed-cusp restricted CC equations."""

from collections.abc import Callable

import numpy as np
from scipy.linalg import solve_triangular

from benchmarks.f12_ao_response import OPERATORS, evaluate
from benchmarks.f12_molecular_pairs import factor_inputs, pair_plan
from generativeqc_compiler.tensor import Program
from generativeqc_compiler.tensor.ad_program import transpose_program


def fixed_cusp_b_weights(
    data: dict, weight: np.ndarray, run: Callable
) -> tuple[dict, dict]:
    """Return the sampled direct/exchange B entries and their stationary AO transpose.

    The first output maps (m,n,k,l) to values. Reject weights outside this
    fixed-cusp sampled domain. The full-rank Coulomb factor is rebuilt from
    the raw metric so physical metric derivatives follow the same solve.
    """
    o, obs, ri = map(int, data["space"])
    if weight.shape != (o,) * 4:
        raise ValueError("An occupied four-index B cotangent is required")
    pairs = tuple(zip(*np.triu_indices(o), strict=True))
    allowed = np.zeros(weight.shape, dtype=bool)
    for i, j in pairs:
        allowed[i, j, i, j] = allowed[j, i, j, i] = allowed[i, j, j, i] = allowed[
            j, i, i, j
        ] = True
    if np.any(weight[~allowed]):
        raise ValueError(
            "An inactive interpair B entry is weighted outside the fixed-cusp domain"
        )
    values = {}

    def selected(mol, fock, whitener, run, *, pullback):
        a = len(whitener)
        ans = {
            k: np.zeros_like(mol[k])
            for k in (
                "fit",
                "raw_coulomb",
                "raw_coulomb_all",
                *("raw_" + op for op in OPERATORS),
                *("metric_" + op for op in OPERATORS),
            )
        }
        ans["metric"] = np.zeros_like(whitener)
        ans.update({k: np.zeros_like(x) for k, x in fock.items()})
        graphs = {}
        reverses = {}
        for diagonal in (True, False):
            graph = pair_plan(o, obs - o, ri - obs, diagonal=diagonal, auxiliaries=a)
            cols = ("b_direct",) if diagonal else ("b_direct", "b_exchange")
            graphs[diagonal] = Program(
                {k: graph.outputs[k] for k in cols}, provenance=graph.provenance
            )
            reverses[diagonal] = transpose_program(graphs[diagonal], cols).program
        for i, j in pairs:
            diag = i == j
            g = graphs[diag]
            source = factor_inputs(mol, i, j, fock)
            feed = {
                n.attrs["name"]: source[n.attrs["name"]]
                for n in g.live_nodes
                if n.op == "input"
            }
            cols = run(g, feed).outputs
            direct = float(cols["b_direct"])
            values[i, j, i, j] = values[j, i, j, i] = direct
            seed = {"bar_b_direct": np.asarray(weight[i, j, i, j])}
            if not diag:
                values[i, j, j, i] = values[j, i, i, j] = float(cols["b_exchange"])
                seed["bar_b_direct"] = np.asarray(
                    weight[i, j, i, j] + weight[j, i, j, i]
                )
                seed["bar_b_exchange"] = np.asarray(
                    weight[i, j, j, i] + weight[j, i, i, j]
                )
            response = run(reverses[diag], {**feed, **seed}).outputs
            for key, entry in response.items():
                name = key.removeprefix("bar_")
                if name.endswith(("_i", "_j")):
                    ans[name[:-2]][:, i if name.endswith("_i") else j] += entry
                else:
                    ans[name] += entry
        scalar = sum(weight[idx] * value for idx, value in values.items())
        return scalar, ans, {"fixed_cusp_b_functional": float(scalar)}

    j = data["metric"]
    whitener = solve_triangular(np.linalg.cholesky(j), np.eye(len(j)), lower=True)
    _, bars, _ = evaluate(
        {**data, "metric_whitener": whitener},
        run,
        pullback=True,
        molecular_correlation=selected,
    )
    if np.any(bars["ao_f12_over_coulomb"]) or np.any(bars["metric_f12_over_coulomb"]):
        raise ValueError("Unexpected external radial operator in B")
    ops = ("coulomb", "f12", "f12_squared", "double_commutator")
    return values, {
        "ao": {k: bars["ao_" + k] for k in ops},
        "metric": {k: bars["metric" if k == "coulomb" else "metric_" + k] for k in ops},
        "hcore": bars["hcore_ao"],
        "coefficients": bars["coefficients"],
    }
