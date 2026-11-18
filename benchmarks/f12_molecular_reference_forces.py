"""Qualify the real F12 and singles source on one live CUDA DF-RHF reference."""

import argparse
import ctypes
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._canonical_frame import canonical_frame_seed
from generativeqc._f12_cabs import cabs_projector, cabs_singles
from generativeqc._f12_molecular_response import molecular_reference_response

from benchmarks._f12_cuda import cuda_runner
from benchmarks.df_reference_response import DFReferenceSession
from benchmarks.f12_molecular_frame import chain
from benchmarks.f12_molecular_one_electron import cases, native_contraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "source",
        "frame",
        "one-electron-reference",
        "scf-inputs",
        "oracle",
        "component-library",
        "canonical-library",
        "session-library",
        "cache",
        "output",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--arch", required=True)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.component_library.resolve()))
    canonical = partial(
        canonical_frame_seed, library=ctypes.CDLL(str(args.canonical_library.resolve()))
    )
    run, programs = cuda_runner(args.arch, args.cache)
    reverse = partial(
        chain,
        run=run,
        projector=partial(cabs_projector, library=lib),
        singles=partial(cabs_singles, library=lib, spin_weight=2),
    )
    scf = json.loads((args.scf_inputs / "reference.json").read_text())
    layouts = {case["name"]: case for case in scf["rows"]}
    gates = scf["gates"]
    rows = []
    with (
        np.load(args.source, allow_pickle=False) as source,
        np.load(args.frame, allow_pickle=False) as frames,
        np.load(args.one_electron_reference, allow_pickle=False) as refs,
    ):
        for name, data, case, ref in cases(source, frames, refs):
            start = time.monotonic()
            target = json.loads((args.oracle / (name + "_orbital.json")).read_text())
            source_file = args.scf_inputs / layouts[name]["central_input"]
            with DFReferenceSession(args.session_library, source_file) as session:
                observed = molecular_reference_response(
                    session,
                    data,
                    ref["native_to_source"],
                    reverse,
                    run,
                    canonical=canonical,
                )
            explicit = native_contraction(
                case,
                observed.explicit_weights["overlap"],
                observed.explicit_weights["hcore_ao"],
                ref["native_to_source"],
            )
            forces = observed.reference_forces - explicit.gradient
            oracle = np.asarray(target["HF_and_orbital_forces"])
            one = np.asarray(target["one_electron_gradient"])
            errors = {
                "HF_energy_Eh": abs(observed.reference_energy - target["E_HF"]),
                "increment_energy_Eh": abs(
                    observed.increment_energy - target["E_increment"]
                ),
                "same_reference_force_Eh_per_bohr": float(
                    np.max(abs(observed.reference_forces - oracle))
                ),
                "explicit_one_electron_Eh_per_bohr": float(
                    np.max(abs(explicit.gradient - one))
                ),
                "partial_force_Eh_per_bohr": float(
                    np.max(abs(forces - (oracle - one)))
                ),
            }
            remaining = sorted(set(observed.explicit_weights) - {"overlap", "hcore_ao"})
            operators = (
                "coulomb",
                "f12",
                "f12_squared",
                "f12_over_coulomb",
                "double_commutator",
            )
            expected = (
                {"ao_" + op for op in operators}
                | {"metric_" + op for op in operators[1:]}
                | {"metric"}
            )
            passed = (
                set(remaining) == expected
                and errors["HF_energy_Eh"] <= gates["energy_Eh"]
                and errors["increment_energy_Eh"] <= 5e-11
                and errors["same_reference_force_Eh_per_bohr"]
                <= gates["force_Eh_per_bohr"]
                and errors["explicit_one_electron_Eh_per_bohr"] <= 3e-8
                and errors["partial_force_Eh_per_bohr"] <= gates["force_Eh_per_bohr"]
                and observed.diagnostics["orbital_residual"]
                <= gates["response_residual"]
                and observed.diagnostics["canonical_bytes"] > 0
                and explicit.resources["device_bytes"] > 0
            )
            rows.append(
                {
                    "case": name,
                    "passed": passed,
                    "partial_force_Eh_per_bohr": forces.tolist(),
                    "errors": errors,
                    "outstanding_nuclear_weights": remaining,
                    "response": observed.diagnostics,
                    "one_electron_resources": explicit.resources,
                    "wall_including_compilation_s": time.monotonic() - start,
                }
            )
    passed = (
        bool(programs)
        and all(row["passed"] for row in rows)
        and {r["case"] for r in rows} == {"h2", "water"}
    )
    args.output.write_text(
        json.dumps(
            {
                "scope": "HF, response of the F12/singles orbital source and explicit one-electron contraction",
                "passed": passed,
                "rows": rows,
                "programs": programs,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    if not passed:
        raise SystemExit("physical reference response check failed")


if __name__ == "__main__":
    main()
