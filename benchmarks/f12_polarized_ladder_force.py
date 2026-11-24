"""True radial UHF-F12 force components of both triplet H2 geometries on a cardinal interval."""

import argparse
import ctypes
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_singles
from generativeqc_compiler.mp2.f12_unrestricted import projected_same_spin_f12_pair
from generativeqc_compiler.tensor import linearize

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner


def evaluate(
    directory: Path, run: Callable, singles: Callable, save: Callable
) -> list[dict]:
    rows = []
    high = json.loads((directory / "high_force.json").read_text())
    if not high["passed"]:
        raise ValueError(
            "Conventional high-correlation force has not passed both displacement controls"
        )
    sources = json.loads((directory / "force_points.json").read_text())
    for point in sources:
        ident = point["id"]
        radius = point["R"]
        orbit = point["ordinal"]
        with np.load(directory / (ident + ".npz"), allow_pickle=False) as data:
            o, v, c, a = map(int, data["space"])
            base = {
                k.removeprefix("base/"): data[k] for k in data if k.startswith("base/")
            }
            dots = {
                k.removeprefix("dot/"): data[k] for k in data if k.startswith("dot/")
            }
            ghf = float(data["reference_gradient"])
        pair = projected_same_spin_f12_pair(o, v, c, auxiliaries=a)
        g = linearize(
            pair, tuple(dots), outputs=("ordinary_pair_energy", "pair_energy")
        ).program
        inputs = {**base, **{"d_" + k: x for k, x in dots.items()}}
        need = {t.attrs["name"] for t in g.live_nodes if t.op == "input"}
        dy = run(g, {k: np.ascontiguousarray(inputs[k]) for k in need}).outputs
        _, bar = singles(base["fock"], o, spin_weight=1, pullback=True)
        gs = float(np.vdot(bar, dots["fock"]))
        cmp = {
            "UHF": ghf,
            "UMP2": float(dy["d_ordinary_pair_energy"]),
            "F12": float(dy["d_pair_energy"]),
            "CABS": gs,
        }
        cmp["total"] = sum(cmp.values())
        errors = {
            h: {k: abs(x - cmp[k]) for k, x in targets.items()}
            for h, targets in point["steps_Eh_per_bohr"].items()
        }
        err = max(v for elem in errors.values() for v in elem.values())
        if len(errors) != 2 or not np.isfinite(err) or err > 3e-7:
            raise ValueError(
                f"Reoptimized molecular two-step derivative differs at {ident}: {errors}"
            )
        ghigh = next(
            x["post_gradient_Eh_per_bohr"]
            for x in high["radii"]
            if abs(x["R"] - radius) < 1e-13
        )
        rows.append(
            {
                "case": "polarized_H2",
                "orbital_cardinal": orbit,
                "radius_bohr": radius,
                "gradient_components_Eh_per_bohr": cmp,
                "native_electronic_correlation_gradient": cmp["F12"] + cmp["UMP2"],
                "force_on_upper_H_Eh_per_bohr": -cmp["total"],
                "external_aug_T_high_gradient": ghigh,
                "focal_force_on_upper_H_Eh_per_bohr": -(cmp["total"] + ghigh),
                "two_step_errors": errors,
            }
        )
        save(rows)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--component-library", required=True)
    add_molecular_graph_budget(p)
    a = p.parse_args()
    run, metrics = cuda_runner(a.arch, a.cache, max_bytes=a.budget_mib << 20)

    def save(rows: list[dict]) -> None:
        tmp = a.output.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"programs": metrics, "rows": rows}, indent=2, allow_nan=False)
            + "\n"
        )
        tmp.replace(a.output)

    evaluate(
        a.fixtures,
        run,
        partial(cabs_singles, library=ctypes.CDLL(a.component_library)),
        save,
    )


if __name__ == "__main__":
    main()
