"""Measure the native CABS singles component on frozen Psi4 Fock matrices."""

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_singles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-library", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.component_library)) if args.component_library else None
    path = (
        Path(__file__).resolve().parents[1]
        / "tests/reference_data/f12/psi4_cabs_singles.npz"
    )
    rows = []
    with np.load(path, allow_pickle=False) as data:
        for name in ("he", "h2", "h2_df", "water_df"):
            fock = data[name + "_fock"]
            nocc = int(data[name + "_occupied"])
            for response in (False, True):
                samples = []
                for _ in range(4):
                    start = time.perf_counter()
                    value, _ = cabs_singles(
                        fock, nocc, spin_weight=2, pullback=response, library=lib
                    )
                    samples.append(time.perf_counter() - start)
                reference = float(data[name + "_energy"])
                rows.append(
                    {
                        "case": name,
                        "occupied": nocc,
                        "dimension": len(fock),
                        "pullback": response,
                        "energy_Eh": value,
                        "psi4_Eh": reference,
                        "error_Eh": value - reference,
                        "wall_seconds": samples,
                    }
                )
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
