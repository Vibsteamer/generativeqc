"""Directional nuclear force of all DF-UHF MP2-F12 spin pairs at a real geometry."""

import argparse
import ctypes
import json
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_singles
from generativeqc._f12_spin_pairs import occupied_pair_blocks, pair_operands
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_extended_df_fock
from generativeqc_compiler.tensor import Program, linearize

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner


def evaluate(
    directory: Path, run: Callable, singles: Callable, checkpoint: Callable
) -> list[dict]:
    with np.load(directory / "nh2_parent_D_f12.npz", allow_pickle=False) as infile:
        central = dict(infile)
    with np.load(directory / "source_tangents.npz", allow_pickle=False) as infile:
        tangent = dict(infile)
    reference = json.loads(
        (directory / "NH2_parent_D_independent_nuclear_fd.json").read_text()
    )
    oa, ob, obs, n, aux = map(int, central["space"])
    spins = ("alpha", "beta")
    occ = dict(zip(spins, (oa, ob), strict=True))
    dirs = [central, tangent]
    src = [
        {k.removeprefix("source/"): a for k, a in t.items() if k.startswith("source/")}
        for t in dirs
    ]
    fg = unrestricted_extended_df_fock(oa, ob, n, aux)
    F = run(fg, src[0]).outputs

    def physical_jvp(
        graph: Program, live: Iterable[str], primal: dict, motion: dict
    ) -> dict:
        feeds = {**primal, **{"d_" + k: motion[k] for k in live}}
        need = {node.attrs["name"] for node in graph.live_nodes if node.op == "input"}
        return run(graph, {k: np.ascontiguousarray(feeds[k]) for k in need}).outputs

    fg_j = linearize(fg, tuple(src[1])).program
    fdir = physical_jvp(fg_j, src[1], *src)
    dF = {k.removeprefix("d_"): a for k, a in fdir.items()}
    dsingles = []
    for sp in spins:
        _, fbar = singles(F["fock_" + sp], occ[sp], spin_weight=1, pullback=True)
        dsingles.append(float(np.vdot(fbar, dF["fock_" + sp])))
    gradients = {
        "ordinary_UMP2_Eh": 0.0,
        "F12_increment_Eh": 0.0,
        "CABS_singles_Eh": sum(dsingles),
        "UHF_Eh": float(tangent["reference_gradient"]),
    }
    logs = []
    for sl, sr, nl, nr, g, pairs in occupied_pair_blocks((oa, ob, obs, n, aux)):
        live = tuple(
            node.attrs["name"]
            for node in g.live_nodes
            if node.op == "input"
            and node.spec.differentiable
            and not node.attrs["name"].startswith("select_")
        )
        j = linearize(g, live, outputs=("ordinary_pair_energy", "pair_energy")).program
        for i, k in pairs:
            feeds = [
                pair_operands(t, s, f, sl, sr, nl, nr, n, i, k)
                for t, s, f in zip(dirs, src, [F, dF], strict=True)
            ]
            diff = physical_jvp(j, live, *feeds)
            dg, df = (
                float(diff["d_ordinary_pair_energy"]),
                float(diff["d_pair_energy"]),
            )
            if not np.all(np.isfinite([dg, df])):
                raise ValueError(f"Non-finite {sl}/{sr} pair derivative {i},{k}")
            gradients["ordinary_UMP2_Eh"] += dg
            gradients["F12_increment_Eh"] += df
            logs.append(
                {
                    "case": "NH2_D_f12",
                    "spin": sl + "_" + sr,
                    "ij": [i, k],
                    "ordinary_gradient": dg,
                    "F12_gradient": df,
                }
            )
            checkpoint(logs)
    gradients["total_Eh"] = sum(gradients.values())
    errors = {}
    for h, entry in reference["steps"].items():
        plus, minus = entry["plus_minus"]
        for key, value in gradients.items():
            fd = (np.sum(plus[key]) - np.sum(minus[key])) / (2 * float(h))
            error = abs(value - fd)
            errors[f"{key}_{h}"] = float(error)
            if not np.isfinite(error) or error > 3e-7:
                raise ValueError(
                    f"{key} analytical {value} vs two-geometry derivative {fd} at {h}"
                )
    logs.append(
        {
            "case": "NH2_D_f12",
            "pairs": len(logs),
            "atom": reference["atom"],
            "axis": reference["axis"],
            "force_Eh_per_bohr": -gradients["total_Eh"],
            "component_gradients": gradients,
            "two_step_errors_Eh_per_bohr": errors,
        }
    )
    checkpoint(logs)
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--component-library", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_molecular_graph_budget(parser)
    a = parser.parse_args()
    run, metrics = cuda_runner(a.arch, a.cache, max_bytes=a.budget_mib << 20)

    def save(rows: list[dict]) -> None:
        t = a.output.with_suffix(".json.tmp")
        t.write_text(
            json.dumps({"rows": rows, "programs": metrics}, indent=2, allow_nan=False)
            + "\n"
        )
        t.replace(a.output)

    evaluate(
        a.fixtures,
        run,
        partial(cabs_singles, library=ctypes.CDLL(a.component_library)),
        save,
    )


if __name__ == "__main__":
    main()
