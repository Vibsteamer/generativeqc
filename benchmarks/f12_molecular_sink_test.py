"""Compare native molecular integral adjoints against full independent Libint2 gradients."""

import argparse
import ctypes as ct
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

_OPCODES = {
    "coulomb": 0,
    "f12": 0,
    "f12_over_coulomb": 1,
    "f12_squared": 2,
    "double_commutator": 3,
}
_VALUE_TOLERANCE = 1.0e-8
_GRADIENT_TOLERANCE = 3.0e-8
_TRANSLATION_TOLERANCE = 3.0e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind(library: Path, kernel: str, host_reference: bool) -> Callable:
    integers = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    doubles = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    suffix = "_host_reference" if host_reference else ""
    function = getattr(
        ct.CDLL(str(library.resolve())),
        f"gqc_f12_weighted_shells_{kernel}{suffix}",
    )
    function.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        integers,
        integers,
        doubles,
        doubles,
        doubles,
        integers,
        integers,
        doubles,
        doubles,
        ct.c_uint64,
        ct.POINTER(ct.c_uint64),
    ]
    function.restype = ct.c_int
    return function


def channel_name(group: str, operator: str) -> str:
    if operator not in _OPCODES or group not in ("metric", "ao"):
        raise ValueError(f"unknown integral channel {group}/{operator}")
    return (
        "metric"
        if group == "metric" and operator == "coulomb"
        else group + "_" + operator
    )


def contract_geometry(
    fixture: Mapping[str, np.ndarray],
    manifest: dict,
    functions: Mapping[str, Callable],
    group: str,
    operator: str,
    budget: int,
    host_reference: bool,
) -> tuple[str, dict]:
    """Execute one physical AO channel from shells, centers and Cartesian adjoints."""
    label = channel_name(group, operator)
    t0 = time.monotonic()
    blocks = fixture[group + "_blocks"]
    offsets = fixture[group + "_offsets"]
    weights = fixture["weights_" + label]
    angular = fixture["angular"]
    atom = fixture["atom"]
    exponents = fixture["exponents"]
    centers = fixture["centers"]
    natoms = len(manifest["elements"])
    nblocks, ncenter = blocks.shape
    if (
        offsets.shape != (nblocks + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(weights)
    ):
        raise ValueError(f"incomplete Cartesian shell offsets for {label}")
    if ncenter != (2 if group == "metric" else 3):
        raise ValueError(f"invalid number of shell centers for {label}")
    geometry = np.asarray(manifest["geometry_bohr"])
    if (
        geometry.shape != (natoms, 3)
        or not np.isfinite(geometry).all()
        or atom.shape != (len(angular),)
        or np.any(atom < 0)
        or np.any(atom >= natoms)
        or not np.array_equal(geometry[atom], centers)
    ):
        raise ValueError("inconsistent shell and atom geometry")
    output = np.full((nblocks, 10), np.nan)
    size = ct.c_uint64()
    kernel = "coulomb" if operator == "coulomb" else "geminal"
    before_kernel = time.monotonic()
    status = functions[kernel](
        ncenter,
        _OPCODES[operator],
        0.0 if operator == "coulomb" else manifest["options"]["f12_beta"],
        len(angular),
        len(exponents),
        nblocks,
        len(weights),
        angular,
        fixture["prim_offsets"],
        exponents,
        fixture["radial"],
        centers,
        blocks,
        offsets,
        weights,
        output,
        budget,
        ct.byref(size),
    )
    kernel_seconds = time.monotonic() - before_kernel
    if status:
        raise RuntimeError(f"{kernel} {label} returned status {status}")
    if not np.isfinite(output).all():
        raise ValueError(f"non-finite or missing shell output for {label}")
    # Metric stores centers A and C. Three-center stores A, B, C.
    if ncenter == 2 and np.any(output[:, 4:7] != 0):
        raise ValueError("two-center derivative has nonzero absent B center")
    xyz_offsets = (1, 7) if ncenter == 2 else (1, 4, 7)
    gradient = np.zeros((natoms, 3))
    for j, start in enumerate(xyz_offsets):
        np.add.at(gradient, atom[blocks[:, j]], output[:, start : start + 3])
    value = float(output[:, 0].sum())
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError(f"non-finite molecular value or gradient for {label}")
    translation = float(np.max(np.abs(gradient.sum(axis=0))))
    return label, {
        "contraction_checks_passed": bool(
            translation <= _TRANSLATION_TOLERANCE
            and size.value <= budget
            and (host_reference or size.value > 0)
        ),
        "weighted_value_Eh": value,
        "gradient_Eh_per_bohr": gradient.tolist(),
        "max_translation_residual_Eh_per_bohr": translation,
        "shell_blocks": nblocks,
        "cartesian_weights": len(weights),
        "seconds": time.monotonic() - t0,
        "kernel_and_transfer_seconds": kernel_seconds,
        "device_bytes": size.value,
    }


def compare_reference(label: str, row: dict, reference: dict) -> dict:
    """Validate one computed result against independent signed value and gradient."""
    if (
        row["shell_blocks"] != reference["shell_blocks"]
        or row["cartesian_weights"] != reference["cartesian_weights"]
    ):
        raise ValueError(f"incomplete shell blocks for {label}")
    energy = float(reference["weighted_value_Eh"])
    target = np.asarray(reference["gradient_Eh_per_bohr"], dtype=np.float64)
    gradient = np.asarray(row["gradient_Eh_per_bohr"])
    if (
        not np.isfinite(energy)
        or target.shape != gradient.shape
        or not np.isfinite(target).all()
    ):
        raise ValueError(f"invalid independent reference for {label}")
    de = abs(row["weighted_value_Eh"] - energy)
    dg = float(np.max(np.abs(gradient - target)))
    return {
        **row,
        "reference_status": "compared",
        "weighted_value_error_Eh": de,
        "max_gradient_error_Eh_per_bohr": dg,
        "passed": row["contraction_checks_passed"]
        and de <= _VALUE_TOLERANCE
        and dg <= _GRADIENT_TOLERANCE,
    }


def contract(
    fixture: Mapping[str, np.ndarray],
    manifest: dict,
    functions: Mapping[str, Callable],
    group: str,
    operator: str,
    budget: int,
    host_reference: bool,
) -> tuple[str, dict]:
    """Original fixed-fixture gate requires a reference for every channel."""
    label, row = contract_geometry(
        fixture, manifest, functions, group, operator, budget, host_reference
    )
    return label, compare_reference(label, row, manifest["channels"][label])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "geminal-library", "coulomb-library", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--host-reference", action="store_true", help="Offline host ABI proof"
    )
    parser.add_argument("--budget-mib", type=int, default=512)
    args = parser.parse_args()
    if args.budget_mib <= 0:
        parser.error("budget must be positive")
    libraries = {"geminal": args.geminal_library, "coulomb": args.coulomb_library}
    functions = {k: bind(p, k, args.host_reference) for k, p in libraries.items()}
    t0 = time.monotonic()
    with np.load(args.fixture, allow_pickle=False) as fixture:
        manifest = json.loads(str(fixture["manifest"]))
        report = {
            "case": manifest["case"],
            "component": "weighted_molecular_nuclear_sink",
            "scope": "ten fixed-adjoint 2c/3c molecular operator contractions",
            "backend": "host_reference_fp64" if args.host_reference else "cuda_fp64",
            "source": manifest["source"],
            "fixture_sha256": sha256(args.fixture),
            "library_sha256": {k: sha256(path) for k, path in libraries.items()},
            "reference_inputs": manifest["inputs"],
            "tolerances": {
                "weighted_value_Eh": _VALUE_TOLERANCE,
                "max_gradient_Eh_per_bohr": _GRADIENT_TOLERANCE,
                "max_translation_Eh_per_bohr": _TRANSLATION_TOLERANCE,
            },
            "elements": manifest["elements"],
            "geometry_bohr": manifest["geometry_bohr"],
            "options": manifest["options"],
            "rows": {},
        }
        for group in ("metric", "ao"):
            for op in _OPCODES:
                label, row = contract(
                    fixture,
                    manifest,
                    functions,
                    group,
                    op,
                    args.budget_mib << 20,
                    args.host_reference,
                )
                report["rows"][label] = row
    rows = report["rows"]
    if len(rows) != 10 or set(rows) != set(manifest["channels"]):
        raise ValueError("incomplete molecular integral channels")
    report["passed"] = all(row["passed"] for row in rows.values())
    report["seconds"] = time.monotonic() - t0
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise SystemExit("molecular sink exceeded its numerical or resource gate")


if __name__ == "__main__":
    main()
