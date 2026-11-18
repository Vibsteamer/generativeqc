"""Trace open-shell NH2 MP2-F12 from common AO inputs to an analytical nuclear force."""

import argparse
import ctypes
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_singles
from generativeqc._f12_spin_ao import unrestricted_orbital_sources
from generativeqc._f12_spin_pairs import molecular_pair_energies
from generativeqc._f12_unrestricted import unrestricted_mp2_f12_source_pullback
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_extended_df_fock
from generativeqc_compiler.mp2.f12_unrestricted_force import (
    unrestricted_f12_nuclear_source_action,
)

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner
from benchmarks.f12_molecular_metric import metric_whitener


def evaluate(
    directory: Path, run: Callable, singles: Callable, whiten: Callable
) -> list[dict]:
    ref = json.loads((directory / "reference.json").read_bytes())
    with np.load(directory / "central_ao_sources.npz", allow_pickle=False) as f:
        ao = {k: f[k] for k in f if k != "metric_whitener"}
    w, info = whiten(ao["metric/coulomb"])
    if info["rank"] != len(w):
        raise ValueError("The full-rank shared Coulomb metric branch is required")
    ao["metric_whitener"] = w
    space = tuple(ref["space"])
    data, _ = unrestricted_orbital_sources(space, ao, run)
    spins = (("alpha", space[0]), ("beta", space[1]))
    args = {
        k.removeprefix("source/"): v for k, v in data.items() if k.startswith("source/")
    }
    fg = unrestricted_extended_df_fock(*space[:2], *space[3:])
    fock = run(fg, args).outputs
    sc = sum(
        singles(fock["fock_" + sp], occ, spin_weight=1, pullback=False)[0]
        for sp, occ in spins
    )
    pair = list(molecular_pair_energies(space, data, args, fock, run))
    mp2 = sum(p.mp2 for p in pair)
    f12 = sum(p.f12 for p in pair)
    total = ref["UHF_Eh"] + sc + mp2 + f12
    eerr = max(
        abs(total - ref["total_Eh"]),
        abs(mp2 + ref["UHF_Eh"] - ref["Psi4_UMP2_total_Eh"]),
        abs(sc - sum(ref["CABS_singles_Eh"])),
        abs(f12 - ref["F12_increment_Eh"]),
    )
    if len(pair) != 36 or not np.isfinite(eerr) or eerr > 2e-8:
        raise ValueError(f"Molecular energy mismatch: {len(pair)} pairs, error {eerr}")
    weights = unrestricted_mp2_f12_source_pullback(space, data, run, singles=singles)
    _, bars = unrestricted_orbital_sources(space, ao, run, orbital_weights=weights)
    keys = sorted(bars)
    flat = np.concatenate([bars[k].ravel() for k in keys])
    with np.load(directory / "physical_ao_tangent.npz", allow_pickle=False) as physical:
        if set(physical) != set(keys):
            raise ValueError(
                "Incomplete physical AO/metric/alpha-beta tangent source set"
            )
        tangent = np.concatenate([physical[k].ravel() for k in keys])
    result = run(
        unrestricted_f12_nuclear_source_action(len(flat)),
        {
            "weights": flat,
            "tangents": tangent,
            "reference_gradient": np.array(ref["UHF_component_gradient"]),
        },
    )
    force = float(result.outputs["force"])
    errs = {k: abs(force + v) for k, v in ref["H1_z_finite_gradients"].items()}
    direct_error = abs(force - ref["H1_z_direct_force"])
    if not np.isfinite(force) or max(direct_error, *errs.values()) > 3e-7:
        raise ValueError(
            f"H1-z force mismatch with complete molecular displacements {errs}, {direct_error}"
        )
    return [
        {
            "case": "NH2_D_f12",
            "active_pairs": len(pair),
            "UMP2_Eh": mp2,
            "F12_Eh": f12,
            "CABS_singles_Eh": sc,
            "total_Eh": total,
            "reference_Eh": ref["UHF_Eh"],
            "molecular_energy_error": eerr,
            "physical_ao_terms": keys,
            "explicit_raw_plus_orbital_analytic_force_Eh_per_bohr": force,
            "full_energy_finite_difference_errors": errs,
            "direct_derivative_error": direct_error,
            "metric": info,
        }
    ]


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
    rows = evaluate(
        a.fixtures,
        run,
        partial(cabs_singles, library=ctypes.CDLL(a.component_library)),
        partial(metric_whitener, library=a.component_library),
    )
    a.output.write_text(
        json.dumps({"rows": rows, "programs": metrics}, indent=2, allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
