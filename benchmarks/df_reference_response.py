"""Native DF-RHF/Z/Pulay composition, with separately prepared CPU oracles.

The controlled scalar drives both Fock and occupied-frame cotangents. It tests
the molecular response connection needed by F12; it is not an F12 energy.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import time
from pathlib import Path

import numpy as np
from typing_extensions import Self

DP = ct.POINTER(ct.c_double)


def pointer(array: np.ndarray) -> DP:
    return array.ctypes.data_as(DP)


class DFReferenceSession:
    """Own a forward DF plan and its exact native reference until close()."""

    def __init__(self, library: Path, source: Path) -> None:
        self.library = ct.CDLL(str(library.resolve()))
        self.handle = ct.c_void_p()
        functions = {
            "create": [
                ct.c_char_p,
                ct.c_int,
                ct.POINTER(ct.c_void_p),
                ct.POINTER(ct.c_size_t),
            ],
            "snapshot": [
                ct.c_void_p,
                DP,
                ct.c_size_t,
                DP,
                ct.c_size_t,
                DP,
                ct.c_size_t,
            ],
            "response": [
                ct.c_void_p,
                DP,
                DP,
                ct.c_size_t,
                ct.c_size_t,
                ct.c_size_t,
                DP,
                ct.c_size_t,
                DP,
                ct.c_size_t,
            ],
        }
        for name, args in functions.items():
            function = getattr(self.library, "gqc_df_session_" + name)
            function.argtypes = [*args, ct.c_void_p, ct.c_size_t]
            function.restype = ct.c_int
        self.library.gqc_df_session_destroy.argtypes = [ct.c_void_p]
        self.library.gqc_df_session_destroy.restype = None
        dimensions = (ct.c_size_t * 3)()
        self._call("create", str(source).encode(), 0, ct.byref(self.handle), dimensions)
        self.n, self.occupied, self.atoms = dimensions

    def _call(self, name: str, *args: object) -> None:
        error = ct.create_string_buffer(4096)
        code = getattr(self.library, "gqc_df_session_" + name)(*args, error, len(error))
        if code:
            raise RuntimeError(error.value.decode())

    def close(self) -> None:
        if self.handle:
            self.library.gqc_df_session_destroy(self.handle)
            self.handle = ct.c_void_p()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def snapshot(self) -> dict:
        matrices = np.empty((5, self.n, self.n))
        energies = np.empty(self.n)
        info = np.empty(6)
        self._call(
            "snapshot",
            self.handle,
            pointer(matrices),
            matrices.size,
            pointer(energies),
            energies.size,
            pointer(info),
            info.size,
        )
        return dict(
            zip(
                ["coefficients", "overlap", "fock", "hcore", "density"],
                matrices,
                strict=True,
            ),
            energies=energies,
            energy=info[0],
            scf_seconds=info[1],
            iterations=int(info[2]),
            commutator=info[3],
            density_drift=info[4],
            eigen_residual=info[5],
        )

    def response(
        self,
        bar_fock: np.ndarray,
        bar_frame: np.ndarray,
        *,
        maximum_bytes: int = 43 << 30,
    ) -> tuple[np.ndarray, dict]:
        f, c = [
            np.ascontiguousarray(a, dtype=np.float64) for a in (bar_fock, bar_frame)
        ]
        if f.shape != (self.n, self.n) or c.shape != f.shape:
            raise ValueError(
                "Response seeds must be square matrices in the native AO/MO frame"
            )
        forces = np.full((self.atoms, 3), np.nan)
        info = np.full(6, np.nan)
        self._call(
            "response",
            self.handle,
            pointer(f),
            pointer(c),
            f.size,
            maximum_bytes,
            (5 * self.n**2 + self.n) * 8,  # Detached forward snapshot remains live.
            pointer(forces),
            forces.size,
            pointer(info),
            info.size,
        )
        return forces, dict(
            zip(
                [
                    "seconds",
                    "orbital_residual",
                    "stationarity",
                    "reference_residual",
                    "capacity_bytes",
                    "orbital_actions",
                ],
                map(float, info),
                strict=True,
            )
        )


def controlled_source(
    coefficients: np.ndarray, energies: np.ndarray, occupied: int
) -> tuple:
    """Gauge-invariant occupied projector and occupied/virtual Fock traces.

    The fixed AO matrix is part of the validation scalar, not a new Hamiltonian.
    Same-space rotations and orbital phases cannot change this objective.
    """
    n = len(energies)
    raw = np.random.default_rng(1818).normal(scale=1e-3, size=(n, n))
    weight = (raw + raw.T) / 2
    trace = np.r_[np.full(occupied, 1e-3), np.full(n - occupied, -2e-4)]
    c = coefficients[:, :occupied]
    value = 2 * np.vdot(c, weight @ c) + np.vdot(trace, energies)
    bar_c = np.zeros_like(coefficients)
    bar_c[:, :occupied] = 4 * weight @ c
    return float(value), np.diag(trace), bar_c


def persist(path: Path, values: dict) -> None:
    """Retain actual native values even when a later scientific gate fails."""
    path.write_text(
        json.dumps(values, indent=2, allow_nan=False, default=lambda a: a.tolist())
        + "\n"
    )


def check(
    library: Path, directory: Path, case: dict, gates: dict, artifacts: Path
) -> dict:
    started = time.monotonic()
    with DFReferenceSession(library, directory / case["central_input"]) as native:
        state = native.snapshot()
        persist(artifacts / f"df-session-{case['name']}-state.json", state)
        independent = case["reference"]
        errors = {
            "energy": abs(state["energy"] - independent["energy"]),
            "orbital_energies": float(
                np.max(abs(state["energies"] - independent["energies"]))
            ),
        }
        for key in ("overlap", "hcore", "fock", "density"):
            errors[key] = float(np.max(abs(state[key] - independent[key])))
        value, f, c = controlled_source(
            state["coefficients"], state["energies"], native.occupied
        )
        total = state["energy"] + value
        errors["controlled_energy"] = abs(total - independent["controlled_energy"])
        forces, response = native.response(f, c)
        hf_forces, hf_response = native.response(np.zeros_like(f), np.zeros_like(c))
        persist(
            artifacts / f"df-session-{case['name']}-response.json",
            {
                "forces": forces,
                "hf_forces": hf_forces,
                "response": response,
                "hf_response": hf_response,
                "controlled_energy": total,
                "bar_fock": f,
                "bar_frame": c,
            },
        )
        errors["independent_hf_force"] = float(
            np.max(abs(hf_forces - independent["forces"]))
        )
        if not all(np.isfinite(v) for v in errors.values()):
            raise ValueError(f"{case['name']}: nonfinite reference error: {errors}")
        if max(errors[k] for k in ("energy", "controlled_energy")) > gates["energy_Eh"]:
            raise ValueError(
                f"{case['name']}: native/independent energy mismatch: {errors}"
            )
        if (
            max(
                errors[k]
                for k in ("orbital_energies", "overlap", "hcore", "fock", "density")
            )
            > gates["state_absolute"]
        ):
            raise ValueError(f"{case['name']}: mismatched DF reference: {errors}")
        if errors["independent_hf_force"] > gates["force_Eh_per_bohr"]:
            raise ValueError(f"{case['name']}: HF force mismatch: {errors}")
        for result in (response, hf_response):
            if (
                max(result[k] for k in ("orbital_residual", "stationarity"))
                > gates["response_residual"]
            ):
                raise ValueError(
                    f"{case['name']}: unresolved native orbital response: {result}"
                )
        # A rejected request must leave the same reference usable; no hidden
        # direct-integral or CPU retry is allowed after capacity/seed rejection.
        refusals = []
        for label, source_f, budget in [
            ("budget", f, 1),
            ("nonfinite", np.full_like(f, np.nan), 43 << 30),
        ]:
            untouched_force = np.full_like(forces, 12345.0)
            untouched_diagnostics = np.full(6, 54321.0)
            try:
                native._call(
                    "response",
                    native.handle,
                    pointer(source_f),
                    pointer(c),
                    f.size,
                    budget,
                    0,
                    pointer(untouched_force),
                    untouched_force.size,
                    pointer(untouched_diagnostics),
                    untouched_diagnostics.size,
                )
            except RuntimeError as exc:
                refusals.append({"case": label, "message": str(exc)})
            else:
                raise ValueError(f"Unbounded or nonfinite request accepted: {label}")
            np.testing.assert_array_equal(untouched_force, 12345.0)
            np.testing.assert_array_equal(untouched_diagnostics, 54321.0)
        np.testing.assert_array_equal(
            native.snapshot()["coefficients"], state["coefficients"]
        )
    native.close()  # Idempotent Python owner close; native null handle is safe.
    try:
        native.snapshot()
    except RuntimeError:
        pass
    else:
        raise ValueError("Closed reference session remained usable")

    differences, scf_seconds = [], state["scf_seconds"]
    for direction in case["directions"]:
        vector = np.asarray(direction["vector"])
        slope = -float(np.vdot(forces, vector))
        hf_slope = -float(np.vdot(hf_forces, vector))
        for pair in direction["steps"]:
            energies, hf_energies = [], []
            for sign in ("plus", "minus"):
                with DFReferenceSession(
                    library, directory / pair[sign + "_input"]
                ) as displaced:
                    moved = displaced.snapshot()
                    correction = controlled_source(
                        moved["coefficients"], moved["energies"], displaced.occupied
                    )[0]
                    hf_energies.append(float(moved["energy"]))
                    energies.append(float(moved["energy"] + correction))
                    scf_seconds += moved["scf_seconds"]
            step = pair["step_bohr"]
            error = {
                "native_fd": abs((energies[0] - energies[1]) / (2 * step) - slope),
                "native_hf_fd": abs(
                    (hf_energies[0] - hf_energies[1]) / (2 * step) - hf_slope
                ),
                "independent_fd": abs(pair["controlled_slope"] - slope),
                "independent_hf_fd": abs(pair["hf_slope"] - hf_slope),
            }
            persist(
                artifacts
                / f"df-session-{case['name']}-{direction['name']}-{step}.json",
                {
                    "native_energies": energies,
                    "native_hf_energies": hf_energies,
                    "analytic_slope": slope,
                    "analytic_hf_slope": hf_slope,
                    "errors": error,
                },
            )
            if (
                not all(np.isfinite(v) for v in error.values())
                or max(error.values()) > gates["force_Eh_per_bohr"]
            ):
                raise ValueError(
                    f"{case['name']}: DF response finite difference failed: {error}"
                )
            differences.append(
                {
                    "direction": direction["name"],
                    "step_bohr": step,
                    "native_energies": energies,
                    "native_hf_energies": hf_energies,
                    "analytic_slope": slope,
                    "analytic_hf_slope": hf_slope,
                    "errors": error,
                }
            )
    translation = float(np.max(abs(forces.sum(axis=0))))
    if translation > gates["translation_Eh_per_bohr"] or not np.isfinite(forces).all():
        raise ValueError(f"{case['name']}: nonfinite or non-translational force")
    return {
        "case": case["name"],
        "passed": True,
        "basis": case["basis"],
        "auxiliary": case["auxiliary"],
        "nbf": native.n,
        "occupied": native.occupied,
        "reference_model": "same-owner-ri-jk",
        "scope": "Native DF-HF plus controlled frame/Fock source response; not a full F12 energy or force",
        "reference_errors": errors,
        "reference_energy_Eh": float(state["energy"]),
        "controlled_energy_Eh": float(total),
        "forces_Eh_per_bohr": forces.tolist(),
        "hf_forces_Eh_per_bohr": hf_forces.tolist(),
        "translation": translation,
        "response": response,
        "hf_response": hf_response,
        "refusals": refusals,
        "finite_differences": differences,
        "central_endpoint_seconds": state["scf_seconds"] + response["seconds"],
        "all_scf_seconds": scf_seconds,
        "qualification_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--session-library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = json.loads((args.input_dir / "reference.json").read_text())
    report = {"passed": False, "rows": []}
    try:
        for row in fixture["rows"]:
            report["rows"].append(
                check(
                    args.session_library,
                    args.input_dir,
                    row,
                    fixture["gates"],
                    args.output.parent,
                )
            )
        report["passed"] = True
    finally:
        persist(args.output, report)


if __name__ == "__main__":
    main()
