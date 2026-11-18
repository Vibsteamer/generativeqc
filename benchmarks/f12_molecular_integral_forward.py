"""Build molecular two-/three-center spherical AO tensors from native shell integrals."""

import argparse
import ctypes as ct
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from benchmarks.f12_molecular_sink_live import ShellProjection
from benchmarks.f12_molecular_sink_test import _OPCODES, sha256


def bind_forward(library: Path, kind: str, host_reference: bool) -> Callable:
    ints = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    doubles = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    name = f"gqc_f12_cartesian_shells_{kind}" + (
        "_host_reference" if host_reference else ""
    )
    func = getattr(ct.CDLL(str(library.resolve())), name)
    func.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ints,
        ints,
        doubles,
        doubles,
        doubles,
        ints,
        ints,
        doubles,
        ct.c_uint64,
        ct.POINTER(ct.c_uint64),
    ]
    func.restype = ct.c_int
    return func


def forward_integrals(
    layout: Mapping[str, np.ndarray],
    library: Path,
    *,
    host_reference: bool = False,
    budget_mib: int = 512,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Return ten native integral tensors and their execution records, keyed by operator."""
    if not 0 < budget_mib <= (2**64 - 1) >> 20:
        raise ValueError("invalid forward device-memory budget")
    budget = budget_mib << 20
    common = {
        k: layout[k]
        for k in (
            "angular",
            "prim_offsets",
            "exponents",
            "radial",
            "centers",
            "metric_blocks",
            "ao_blocks",
            "ao_offsets",
            "metric_offsets",
            "manifest",
            "spherical_rows",
            "ao_sizes",
            "orbital_shell_count",
            "atom",
        )
    }
    for angular in np.unique(common["angular"]):
        key = f"transform_{angular}"
        common[key] = layout[key]
    manifest = json.loads(str(common["manifest"]))
    beta = manifest["options"]["f12_beta"]
    geometry = np.asarray(manifest["geometry_bohr"])
    if not np.array_equal(geometry[common["atom"]], common["centers"]):
        raise ValueError("molecular and shell coordinates disagree")
    funcs = {
        k: bind_forward(library, k, host_reference) for k in ("geminal", "coulomb")
    }
    tensors, resources = {}, {}
    for group in ("metric", "ao"):
        projection = ShellProjection(common, group)
        offsets, blocks = projection.offsets, projection.blocks
        nblocks, ncenters = blocks.shape
        ncart = int(offsets[-1])
        for op, opcode in _OPCODES.items():
            label = group if op == "coulomb" and group == "metric" else group + "_" + op
            start = time.monotonic()
            raw = np.full(ncart, np.nan)
            workspace = ct.c_uint64()
            result = funcs["coulomb" if op == "coulomb" else "geminal"](
                ncenters,
                opcode,
                beta,
                len(common["angular"]),
                len(common["exponents"]),
                nblocks,
                ncart,
                common["angular"],
                common["prim_offsets"],
                common["exponents"],
                common["radial"],
                common["centers"],
                blocks,
                offsets,
                raw,
                budget,
                ct.byref(workspace),
            )
            kernel_seconds = time.monotonic() - start
            if result:
                raise RuntimeError(f"{label} native forward returned status {result}")
            if not np.isfinite(raw).all() or not (0 < workspace.value <= budget):
                raise ValueError(
                    f"{label} incomplete native outputs or invalid workspace"
                )
            tensors[label] = projection.spherical_values(raw)
            resources[label] = {
                "backend": "host_reference_fp64" if host_reference else "cuda_fp64",
                "shell_blocks": nblocks,
                "cartesian_integrals": ncart,
                "spherical_elements": tensors[label].size,
                "kernel_and_transfer_seconds": kernel_seconds,
                "host_spherical_assembly_seconds": time.monotonic()
                - start
                - kernel_seconds,
                "workspace_bytes": workspace.value,
                "device_bytes": 0 if host_reference else workspace.value,
            }
    return tensors, resources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("layout", "source", "library", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--host-reference", action="store_true")
    args = parser.parse_args()
    with np.load(args.layout, allow_pickle=False) as layout:
        metadata = json.loads(str(layout["manifest"]))
        sources, resources = forward_integrals(
            layout, args.library, host_reference=args.host_reference
        )
    expected_sha = metadata["inputs"]["source_fixture_sha256"]
    if sha256(args.source) != expected_sha:
        raise ValueError("independent source file does not match fixture provenance")
    case = metadata["case"]
    rows = {}
    with np.load(args.source, allow_pickle=False) as ref:
        for label, values in sources.items():
            expected = ref[case + "/" + label]
            if expected.shape != values.shape or not np.isfinite(values).all():
                raise ValueError(f"{label} incomplete spherical tensor")
            diff = values - expected
            err = float(np.max(np.abs(diff)))
            rows[label] = {
                **resources[label],
                "shape": values.shape,
                "max_abs_error": err,
                "rms_error": float(np.sqrt(np.mean(diff**2))),
                "max_abs_reference": float(np.max(np.abs(expected))),
                "passed": err <= 1.0e-9,
            }
    passed = len(rows) == 10 and all(r["passed"] for r in rows.values())
    report = {
        "case": case,
        "scope": "full spherical two- and three-center molecular forward tensors",
        "source_sha256": expected_sha,
        "layout_sha256": sha256(args.layout),
        "library_sha256": sha256(args.library),
        "tolerance": 1e-9,
        "passed": passed,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not passed:
        raise SystemExit("molecular forward failed full tensor oracle comparison")


if __name__ == "__main__":
    main()
