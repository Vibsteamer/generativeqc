"""Contract live molecular F12 adjoints in the orbital and fitting AO bases."""

import json
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._f12_focal_integral_kernel import (
    _OPCODES,
    _TRANSLATION_TOLERANCE,
    bind,
    channel_name,
    compare_reference,
    contract_geometry,
)

if TYPE_CHECKING:
    from .experimental.f12_focal import FocalRequest


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

    _MAX_CARTESIAN_CHUNK = 128_000_000

    def block_chunks(self, max_cart: int | None = None) -> list[tuple[int, int]]:
        """Partition blocks into contiguous ranges each <= max_cart Cartesian integrals."""
        limit = self._MAX_CARTESIAN_CHUNK if max_cart is None else int(max_cart)
        limit = min(limit, (2**31) - 1)
        nblocks = len(self.blocks)
        total = int(self.offsets[-1])
        if total <= limit:
            return [(0, nblocks)]
        ranges: list[tuple[int, int]] = []
        b0 = 0
        while b0 < nblocks:
            base = int(self.offsets[b0])
            target = base + limit
            b1 = int(np.searchsorted(self.offsets, target, side="right")) - 1
            if b1 <= b0:
                if int(self.offsets[b0 + 1] - base) >= 2**31:
                    raise MemoryError("single shell block exceeds int32 Cartesian offset limit")
                b1 = b0 + 1
            ranges.append((b0, b1))
            b0 = b1
        return ranges

    def chunks_for_slice(self, b0: int, b1: int) -> Iterator[tuple]:
        base = int(self.offsets[b0])
        for group, angular in self.groups:
            lo = int(np.searchsorted(group, b0, side="left"))
            hi = int(np.searchsorted(group, b1, side="left"))
            if lo >= hi:
                continue
            sub = group[lo:hi]
            transforms = [self.transforms[l] for l in angular]
            stride = int(self.offsets[sub[0] + 1] - self.offsets[sub[0]])
            arange = np.arange(stride, dtype=np.int64)
            for first in range(0, len(sub), 128):
                indices = sub[first : first + 128]
                blocks = self.blocks[indices]
                rows = [
                    self.spherical_rows[blocks[:, j], : 2 * l + 1]
                    for j, l in enumerate(angular)
                ]
                positions = (self.offsets[indices].astype(np.int64) - base)[:, None] + arange
                yield rows, transforms, positions, blocks[:, 0] != blocks[:, 1]

    def chunks(self) -> Iterator[tuple]:
        yield from self.chunks_for_slice(0, len(self.blocks))

    def pack_slice(self, weights: np.ndarray, b0: int, b1: int) -> np.ndarray:
        """Convert supplied spherical cotangents for blocks [b0, b1) into Cartesian weights."""
        if (
            weights.dtype != np.float64
            or weights.shape != self.shape
            or not np.isfinite(weights).all()
        ):
            raise ValueError(f"expected finite FP64 adjoints with shape {self.shape}")
        ncart = int(self.offsets[b1] - self.offsets[b0])
        packed = np.empty(ncart, dtype=np.float64)
        for rows, transforms, positions, different in self.chunks_for_slice(b0, b1):
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

    def pack(self, weights: np.ndarray) -> np.ndarray:
        """Convert supplied spherical cotangents into one Cartesian weight per integral."""
        return self.pack_slice(weights, 0, len(self.blocks))

    def spherical_values_into(
        self, cartesian: np.ndarray, values: np.ndarray, b0: int, b1: int
    ) -> None:
        """Assemble forward integral values for blocks [b0, b1) into preallocated values."""
        ncart = int(self.offsets[b1] - self.offsets[b0])
        if cartesian.shape != (ncart,) or cartesian.dtype != np.float64:
            raise ValueError("expected one FP64 value for each Cartesian integral")
        for rows, transforms, positions, different in self.chunks_for_slice(b0, b1):
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

    def spherical_values(self, cartesian: np.ndarray) -> np.ndarray:
        """Assemble native forward integral values using the transpose of weight packing."""
        values = np.empty(self.shape)
        self.spherical_values_into(cartesian, values, 0, len(self.blocks))
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
        from ._f12_focal_shells import focal_shell_layout

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
        chunks = projection.block_chunks()
        for operator in operators:
            label = channel_name(group, operator)
            pack_seconds = 0.0
            if len(chunks) == 1:
                pack_start = time.monotonic()
                weights = projection.pack(explicit_weights[label])
                pack_seconds = time.monotonic() - pack_start
                _, row = contract_geometry(
                    {
                        **topology,
                        group + "_offsets": np.ascontiguousarray(projection.offsets, dtype=np.int32),
                        "weights_" + label: weights,
                    },
                    metadata,
                    functions,
                    group,
                    operator,
                    budget_mib << 20,
                    host_reference,
                )
            else:
                grad_sum = np.zeros((len(metadata["elements"]), 3), dtype=np.float64)
                val_sum = 0.0
                max_dev = 0
                kernel_s = 0.0
                total_s = 0.0
                checks_ok = True
                for b0, b1 in chunks:
                    pack_start = time.monotonic()
                    packed = projection.pack_slice(explicit_weights[label], b0, b1)
                    pack_seconds += time.monotonic() - pack_start
                    c_blocks = np.ascontiguousarray(projection.blocks[b0:b1])
                    c_offsets = np.ascontiguousarray(
                        projection.offsets[b0 : b1 + 1] - projection.offsets[b0],
                        dtype=np.int32,
                    )
                    _, crow = contract_geometry(
                        {
                            **topology,
                            group + "_blocks": c_blocks,
                            group + "_offsets": c_offsets,
                            "weights_" + label: packed,
                        },
                        metadata,
                        functions,
                        group,
                        operator,
                        budget_mib << 20,
                        host_reference,
                    )
                    del packed
                    checks_ok = checks_ok and bool(crow["contraction_checks_passed"])
                    val_sum += float(crow["weighted_value_Eh"])
                    grad_sum += np.asarray(crow["gradient_Eh_per_bohr"], dtype=np.float64)
                    max_dev = max(max_dev, int(crow["device_bytes"]))
                    kernel_s += float(crow["kernel_and_transfer_seconds"])
                    total_s += float(crow["seconds"])
                translation = float(np.max(np.abs(grad_sum.sum(axis=0))))
                row = {
                    "contraction_checks_passed": bool(
                        checks_ok and translation <= _TRANSLATION_TOLERANCE
                    ),
                    "weighted_value_Eh": val_sum,
                    "gradient_Eh_per_bohr": grad_sum.tolist(),
                    "max_translation_residual_Eh_per_bohr": translation,
                    "shell_blocks": len(projection.blocks),
                    "cartesian_weights": int(projection.offsets[-1]),
                    "seconds": total_s,
                    "kernel_and_transfer_seconds": kernel_s,
                    "device_bytes": max_dev,
                }
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


def contract_single_channel(
    layout: Mapping[str, np.ndarray],
    group: str,
    operator: str,
    weight: np.ndarray,
    library: Path,
    *,
    host_reference: bool = False,
    budget_mib: int = 512,
    coulomb_library: Path | None = None,
) -> np.ndarray:
    """Contract ONE single (group, operator) spherical adjoint directly into its [natoms, 3] nuclear gradient."""
    if not 0 < budget_mib <= (2**64 - 1) >> 20:
        raise ValueError("invalid molecular sink budget")
    if group not in ("metric", "ao") or operator not in _OPCODES:
        raise ValueError(f"invalid channel ({group!r}, {operator!r})")
    metadata = json.loads(str(layout["manifest"]))
    kind = "coulomb" if operator == "coulomb" else "geminal"
    functions = {
        kind: bind(
            (coulomb_library or library) if kind == "coulomb" else library,
            kind,
            host_reference,
        )
    }
    topology = {k: layout[k] for k in layout if not k.startswith("weights_")}
    projection = ShellProjection(topology, group)
    label = channel_name(group, operator)
    grad = np.zeros((len(metadata["elements"]), 3), dtype=np.float64)
    for b0, b1 in projection.block_chunks():
        packed = projection.pack_slice(weight, b0, b1)
        c_blocks = np.ascontiguousarray(projection.blocks[b0:b1])
        c_offsets = np.ascontiguousarray(
            projection.offsets[b0 : b1 + 1] - projection.offsets[b0], dtype=np.int32
        )
        _, row = contract_geometry(
            {
                **topology,
                group + "_blocks": c_blocks,
                group + "_offsets": c_offsets,
                "weights_" + label: packed,
            },
            metadata,
            functions,
            group,
            operator,
            budget_mib << 20,
            host_reference,
        )
        del packed
        if not row["contraction_checks_passed"]:
            raise ArithmeticError(
                f"The molecular shell nuclear derivative failed contraction checks for {label}"
            )
        grad += np.asarray(row["gradient_Eh_per_bohr"], dtype=np.float64)
    if not np.isfinite(grad).all():
        raise ValueError(f"non-finite explicit integral gradient for {label}")
    if float(np.max(np.abs(grad.sum(axis=0)))) > _TRANSLATION_TOLERANCE:
        raise ArithmeticError(
            f"The molecular shell nuclear derivative failed translation checks for {label}"
        )
    return grad
