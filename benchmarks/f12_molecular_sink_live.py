"""Contract live molecular F12 adjoints in the orbital and fitting AO bases."""

import argparse
import json
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from benchmarks.f12_molecular_sink_test import (
    _OPCODES,
    _TRANSLATION_TOLERANCE,
    bind,
    channel_name,
    compare_reference,
    contract_geometry,
)

if TYPE_CHECKING:
    from generativeqc.experimental.f12_focal import FocalRequest


class ShellProjection:
    """Real orbital-product symmetry and real-spherical/Cartesian adjoint map."""

    def __init__(self, layout: Mapping[str, np.ndarray], group: str) -> None:
        if group not in ("metric", "ao"):
            raise ValueError(f"unknown integral group {group}")
        self.blocks = layout[group + "_blocks"]
        self.offsets = layout[group + "_offsets"]
        self.spherical_rows = layout["spherical_rows"]
        angular = layout["angular"]
        norb, naux = map(int, layout["ao_sizes"])
        orbital_shells = int(layout["orbital_shell_count"])
        ncenters = 2 if group == "metric" else 3
        self.shape = (naux, naux) if ncenters == 2 else (naux, norb, norb)
        if (
            not 0 < orbital_shells < len(angular)
            or self.spherical_rows.shape[0] != len(angular)
            or self.blocks.ndim != 2
            or self.blocks.shape[1] != ncenters
            or (self.blocks < 0).any()
            or (self.blocks >= len(angular)).any()
        ):
            raise ValueError("invalid shell layout")
        for start, end, size in (
            (0, orbital_shells, norb),
            (orbital_shells, len(angular), naux),
        ):
            represented = np.concatenate(
                [
                    self.spherical_rows[i, : 2 * angular[i] + 1]
                    for i in range(start, end)
                ]
            )
            if not np.array_equal(np.sort(represented), np.arange(size)):
                raise ValueError(
                    "spherical row map must be bijective within each basis"
                )
        # Prove coverage even for a production geometry without a fixed reference.
        # Triangular keys identify either orientation of the same symmetric shell pair.
        naux_shells = len(angular) - orbital_shells
        pair_basis = naux_shells if ncenters == 2 else orbital_shells
        triangular = pair_basis * (pair_basis + 1) // 2
        expected = triangular if ncenters == 2 else naux_shells * triangular
        if len(self.blocks) != expected:
            raise ValueError("incomplete shell enumeration")
        pairs = self.blocks[:, :2].astype(np.int64)
        if ncenters == 2:
            pairs -= orbital_shells
        if np.any(pairs < 0) or np.any(pairs >= pair_basis):
            raise ValueError("shell pair is outside its basis")
        upper = pairs.max(axis=1)
        keys = upper * (upper + 1) // 2 + pairs.min(axis=1)
        if ncenters == 3:
            auxiliary = self.blocks[:, 2].astype(np.int64) - orbital_shells
            if np.any(auxiliary < 0):
                raise ValueError(
                    "three-center shell block is missing its auxiliary shell"
                )
            keys += triangular * auxiliary
        if len(np.unique(keys)) != expected:
            raise ValueError("duplicate shell blocks")
        self.transforms = {l: layout[f"transform_{l}"] for l in np.unique(angular)}
        angulars = angular[self.blocks]
        cart_sizes = np.prod((angulars + 1) * (angulars + 2) // 2, axis=1)
        if self.offsets[0] != 0 or not np.array_equal(
            np.diff(self.offsets), cart_sizes
        ):
            raise ValueError("inconsistent Cartesian shell offsets")
        kinds, inverse = np.unique(angulars, axis=0, return_inverse=True)
        self.groups = [
            (np.flatnonzero(inverse == j), tuple(int(l) for l in ls))
            for j, ls in enumerate(kinds)
        ]

    def chunks(self) -> Iterator[tuple]:
        for group, angular in self.groups:
            transforms = [self.transforms[l] for l in angular]
            for first in range(0, len(group), 128):
                indices = group[first : first + 128]
                blocks = self.blocks[indices]
                rows = [
                    self.spherical_rows[blocks[:, j], : 2 * l + 1]
                    for j, l in enumerate(angular)
                ]
                positions = self.offsets[indices, None] + np.arange(
                    self.offsets[indices[0] + 1] - self.offsets[indices[0]]
                )
                yield rows, transforms, positions, blocks[:, 0] != blocks[:, 1]

    def pack(self, weights: np.ndarray) -> np.ndarray:
        """Convert supplied spherical cotangents into one Cartesian weight per integral."""
        if (
            weights.dtype != np.float64
            or weights.shape != self.shape
            or not np.isfinite(weights).all()
        ):
            raise ValueError(f"expected finite FP64 adjoints with shape {self.shape}")
        packed = np.empty(self.offsets[-1], dtype=np.float64)
        for rows, transforms, positions, different in self.chunks():
            a, b = rows[:2]
            ta, tb = transforms[:2]
            if len(rows) == 2:
                values = weights[a[:, :, None], b[:, None, :]].copy()
                values[different] += weights[
                    b[different, None, :], a[different, :, None]
                ]
                raw = np.einsum("iab,aA,bB->iAB", values, ta, tb, optimize=True)
            else:
                p = rows[2]
                values = weights[
                    p[:, :, None, None], a[:, None, :, None], b[:, None, None, :]
                ].copy()
                values[different] += weights[
                    p[different, :, None, None],
                    b[different, None, None, :],
                    a[different, None, :, None],
                ]
                raw = np.einsum(
                    "iPab,aA,bB,PQ->iABQ", values, ta, tb, transforms[2], optimize=True
                )
            packed[positions] = raw.reshape(positions.shape)
        return packed

    def spherical_values(self, cartesian: np.ndarray) -> np.ndarray:
        """Assemble native forward integral values using the transpose of weight packing."""
        if cartesian.shape != (self.offsets[-1],) or cartesian.dtype != np.float64:
            raise ValueError("expected one FP64 value for each Cartesian integral")
        values = np.empty(self.shape)
        for rows, transforms, positions, different in self.chunks():
            a, b = rows[:2]
            ta, tb = transforms[:2]
            raw = cartesian[positions].reshape(
                len(a), *(t.shape[1] for t in transforms)
            )
            if len(rows) == 2:
                block = np.einsum("iAB,aA,bB->iab", raw, ta, tb, optimize=True)
                values[a[:, :, None], b[:, None, :]] = block
                values[b[different, None, :], a[different, :, None]] = block[different]
            else:
                p = rows[2]
                block = np.einsum(
                    "iABQ,aA,bB,PQ->iPab", raw, ta, tb, transforms[2], optimize=True
                )
                values[
                    p[:, :, None, None], a[:, None, :, None], b[:, None, None, :]
                ] = block
                values[
                    p[different, :, None, None],
                    b[different, None, None, :],
                    a[different, None, :, None],
                ] = block[different]
        return values


def contract_live_weights(
    layout: "Mapping[str, np.ndarray] | FocalRequest",
    explicit_weights: Mapping[str, np.ndarray],
    library: Path,
    *,
    host_reference: bool = False,
    budget_mib: int = 512,
    coulomb_library: Path | None = None,
    singles_only: bool = False,
) -> dict:
    """Run each channel on the supplied AO adjoints and sum its nuclear gradient.

    Geometry-only and reference-bearing layouts use the same native contraction.
    Independent errors are available when the manifest contains all ten central
    channel references. CPU shape transforms and per-atom shell reductions
    surround the compiled integral kernels.
    """
    if not 0 < budget_mib <= (2**64 - 1) >> 20:
        raise ValueError("invalid molecular sink budget")
    start = time.monotonic()
    if not isinstance(layout, Mapping):
        from benchmarks.f12_focal_shell_layout import focal_shell_layout

        singles_only = singles_only or layout.role == "hf_cabs"
        layout = focal_shell_layout(
            layout, library, host_reference=host_reference, budget_mib=budget_mib
        )
    metadata = json.loads(str(layout["manifest"]))
    channels = {
        channel_name(group, op) for group in ("metric", "ao") for op in _OPCODES
    }
    references = metadata.get("channels")
    if references is not None and (
        not isinstance(references, dict) or set(references) != channels
    ):
        raise ValueError("partial or unexpected central channel reference set")
    labels = metadata.get("integral_channels")
    if labels is not None and (len(labels) != len(channels) or set(labels) != channels):
        raise ValueError("partial or unexpected declared integral channels")
    operators = ("coulomb",) if singles_only else tuple(_OPCODES)
    needed = {channel_name(group, op) for group in ("metric", "ao") for op in operators}
    if set(explicit_weights) - {"overlap", "hcore_ao"} != needed:
        raise ValueError(
            "the explicit adjoint set must contain all required integral channels"
        )
    functions = {
        "coulomb": bind(coulomb_library or library, "coulomb", host_reference),
    }
    if not singles_only:
        functions["geminal"] = bind(library, "geminal", host_reference)
    else:
        # Stored channel fixtures describe the combined pair + singles functional.
        references = None
    topology = {k: layout[k] for k in layout if not k.startswith("weights_")}
    rows = {}
    total = np.zeros((len(metadata["elements"]), 3))
    for group in ("metric", "ao"):
        projection = ShellProjection(topology, group)
        for operator in operators:
            label = channel_name(group, operator)
            pack_start = time.monotonic()
            weights = projection.pack(explicit_weights[label])
            pack_seconds = time.monotonic() - pack_start
            _, row = contract_geometry(
                {**topology, "weights_" + label: weights},
                metadata,
                functions,
                group,
                operator,
                budget_mib << 20,
                host_reference,
            )
            if references is not None:
                row = compare_reference(label, row, references[label])
            else:
                row = {**row, "reference_status": "unavailable", "passed": None}
            total += row["gradient_Eh_per_bohr"]
            rows[label] = {**row, "host_packing_seconds": pack_seconds}
    if not np.isfinite(total).all():
        raise ValueError("non-finite total explicit integral gradient")
    residual = float(abs(total.sum(axis=0)).max())
    return {
        "case": metadata["case"],
        "backend": "host_reference_fp64" if host_reference else "cuda_fp64",
        "rows": rows,
        "gradient_Eh_per_bohr": total.tolist(),
        "max_device_bytes": max(row["device_bytes"] for row in rows.values()),
        "seconds": time.monotonic() - start,
        "max_translation_residual_Eh_per_bohr": residual,
        "passes_contraction_checks": bool(
            all(row["contraction_checks_passed"] for row in rows.values())
            and residual <= _TRANSLATION_TOLERANCE
        ),
        "reference_channel_count": len(references) if references is not None else 0,
        "passes_central_channel_references": all(row["passed"] for row in rows.values())
        if references is not None
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("layout", "weights", "library", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--coulomb-library", type=Path)
    parser.add_argument("--host-reference", action="store_true")
    args = parser.parse_args()
    with (
        np.load(args.layout, allow_pickle=False) as topology,
        np.load(args.weights, allow_pickle=False) as weights,
    ):
        report = contract_live_weights(
            topology,
            weights,
            args.library,
            host_reference=args.host_reference,
            coulomb_library=args.coulomb_library,
        )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passes_contraction_checks"]:
        raise SystemExit(
            "molecular channel contraction exceeded its numerical or resource gate"
        )
    if report["passes_central_channel_references"] is False:
        raise SystemExit(
            "central molecular weights disagree with independent reference"
        )


if __name__ == "__main__":
    main()
