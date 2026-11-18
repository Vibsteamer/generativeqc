"""Evaluate the b triplet H2 UMP2-F12 and CABS corrections at both physical radii."""

import argparse
import ctypes
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_singles
from generativeqc_compiler.mp2.f12_unrestricted import projected_same_spin_f12_pair

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner


def evaluate(directory: Path, run: Callable, singles: Callable) -> list[dict]:
    output = []
    basis = None
    for radius in (28, 30):
        ref = json.loads((directory / f"R{radius}.json").read_text())
        with np.load(directory / f"R{radius}.npz", allow_pickle=False) as frozen:
            occ, virt, cabs, fit = map(int, frozen["space"])
            operands = {k: frozen[k] for k in frozen if k != "space"}
        if occ != 2 or ref["multiplicity"] != 3 or abs(ref["R"] - radius / 10) > 1e-14:
            raise ValueError("The two fixed all-alpha geometries are required")
        current = (
            ref["OBS"],
            ref["CABS"],
            ref["fit_HF_and_MP2_and_F12"],
            ref["beta_bohr-1"],
            ref["cabs_overlap_cutoff_relative"],
        )
        if basis is not None and current != basis:
            raise ValueError("Different protocols at the two distances")
        if (
            not np.allclose(
                ref["occupied_inversion_parity"], [1.0, -1.0], atol=3e-8, rtol=0
            )
            or ref["S_non_sigma_occupied_amplitudes_l2"] > 1e-7
        ):
            raise ValueError("Inconsistent spin triplet reference symmetry")
        basis = current
        graph = projected_same_spin_f12_pair(occ, virt, cabs, auxiliaries=fit)
        pair = run(
            graph, {k: np.ascontiguousarray(v) for k, v in operands.items()}
        ).outputs
        sing = float(singles(operands["fock"], occ, spin_weight=1, pullback=False)[0])
        ump, ef = float(pair["ordinary_pair_energy"]), float(pair["pair_energy"])
        energy = ref["UHF_Eh"] + ump + ef + sing
        err = max(
            abs(energy - ref["total_Eh"]),
            abs(ump + ref["UHF_Eh"] - ref["PySCF_UMP2_total_Eh"]),
            abs(ef - ref["F12_Eh"]),
            abs(sing - ref["CABS_Eh"]),
        )
        if not np.isfinite(err) or err > 2e-8:
            raise ValueError(
                f"Polarized {current[0]} R={radius / 10} energy mismatch: {err}"
            )
        output.append(
            {
                "case": "polarized_H2",
                "basis": current[0],
                "radius_bohr": radius / 10,
                "total_Eh": energy,
                "F12_Eh": ef,
                "UMP2_Eh": ump,
                "CABS_Eh": sing,
                "UHF_Eh": ref["UHF_Eh"],
                "max_component_error_Eh": err,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component-library", required=True)
    add_molecular_graph_budget(parser)
    args = parser.parse_args()
    run, metrics = cuda_runner(args.arch, args.cache, max_bytes=args.budget_mib << 20)
    rows = evaluate(
        args.fixtures,
        run,
        partial(cabs_singles, library=ctypes.CDLL(args.component_library)),
    )
    args.output.write_text(
        json.dumps({"rows": rows, "programs": metrics}, indent=2, allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
