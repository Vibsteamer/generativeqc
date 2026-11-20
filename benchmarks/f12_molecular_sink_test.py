"""Compare native molecular integral adjoints against full independent Libint2 gradients."""

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from generativeqc._f12_focal_integral_kernel import (
    _GRADIENT_TOLERANCE,
    _OPCODES,
    _TRANSLATION_TOLERANCE,
    _VALUE_TOLERANCE,
    bind,
    channel_name,
    compare_reference,
    contract_geometry,
)

__all__ = [
    "bind",
    "channel_name",
    "compare_reference",
    "contract",
    "contract_geometry",
    "sha256",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
