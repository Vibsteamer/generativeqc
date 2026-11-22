"""Build molecular two-/three-center spherical AO tensors from native shell integrals."""

import ctypes as ct
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._f12_focal_integral_kernel import _OPCODES
from ._f12_focal_integral_sink import (
    ShellProjection,
    contract_live_weights,
    contract_single_channel,
)
from ._f12_focal_metric import metric_whitener
from ._f12_focal_shells import focal_shell_layout

__all__ = [
    "contract_live_weights",
    "contract_single_channel",
    "focal_shell_layout",
    "forward_integrals",
    "forward_metrics_only",
    "forward_single_ao_operator",
    "metric_whitener",
]

if TYPE_CHECKING:
    from .experimental.f12_focal import FocalRequest


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
    layout: "Mapping[str, np.ndarray] | FocalRequest",
    library: Path,
    *,
    host_reference: bool = False,
    coulomb_only: bool = False,
    budget_mib: int = 512,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Evaluate supplied shell tables or build them from a focal geometry and basis.

    Focal sources are ordered OBS first, then OptRI. Each subspace uses the
    public basis order. The two- and three-center fitting indices share one
    independently numbered fitting basis.
    """
    if not 0 < budget_mib <= (2**64 - 1) >> 20:
        raise ValueError("invalid forward device-memory budget")
    budget = budget_mib << 20
    if not isinstance(layout, Mapping):
        coulomb_only = coulomb_only or layout.role == "hf_cabs"
        layout = focal_shell_layout(
            layout, library, host_reference=host_reference, budget_mib=budget_mib
        )
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
    beta = 0.0 if coulomb_only else manifest["options"]["f12_beta"]
    geometry = np.asarray(manifest["geometry_bohr"])
    if not np.array_equal(geometry[common["atom"]], common["centers"]):
        raise ValueError("molecular and shell coordinates disagree")
    kinds = ("coulomb",) if coulomb_only else ("coulomb", "geminal")
    funcs = {k: bind_forward(library, k, host_reference) for k in kinds}
    ops = {"coulomb": _OPCODES["coulomb"]} if coulomb_only else _OPCODES
    tensors, resources = {}, {}
    for group in ("metric", "ao"):
        projection = ShellProjection(common, group)
        offsets, blocks = projection.offsets, projection.blocks
        nblocks, ncenters = blocks.shape
        ncart_total = int(offsets[-1])
        chunks = projection.block_chunks()
        for op, opcode in ops.items():
            label = group if op == "coulomb" and group == "metric" else group + "_" + op
            start = time.monotonic()
            values = np.empty(projection.shape, dtype=np.float64)
            kernel_seconds = 0.0
            max_workspace = 0
            for b0, b1 in chunks:
                c_blocks = np.ascontiguousarray(blocks[b0:b1])
                c_offsets = np.ascontiguousarray(offsets[b0 : b1 + 1] - offsets[b0], dtype=np.int32)
                c_ncart = int(c_offsets[-1])
                raw = np.full(c_ncart, np.nan)
                workspace = ct.c_uint64()
                t_k0 = time.monotonic()
                result = funcs["coulomb" if op == "coulomb" else "geminal"](
                    ncenters,
                    opcode,
                    beta,
                    len(common["angular"]),
                    len(common["exponents"]),
                    len(c_blocks),
                    c_ncart,
                    common["angular"],
                    common["prim_offsets"],
                    common["exponents"],
                    common["radial"],
                    common["centers"],
                    c_blocks,
                    c_offsets,
                    raw,
                    budget,
                    ct.byref(workspace),
                )
                kernel_seconds += time.monotonic() - t_k0
                if result:
                    raise RuntimeError(f"{label} native forward returned status {result}")
                if not np.isfinite(raw).all() or not (0 < workspace.value <= budget):
                    raise ValueError(
                        f"{label} incomplete native outputs or invalid workspace"
                    )
                max_workspace = max(max_workspace, int(workspace.value))
                projection.spherical_values_into(raw, values, b0, b1)
                del raw
            tensors[label] = values
            resources[label] = {
                "backend": "host_reference_fp64" if host_reference else "cuda_fp64",
                "shell_blocks": nblocks,
                "cartesian_integrals": ncart_total,
                "spherical_elements": tensors[label].size,
                "kernel_and_transfer_seconds": kernel_seconds,
                "host_spherical_assembly_seconds": time.monotonic()
                - start
                - kernel_seconds,
                "workspace_bytes": max_workspace,
                "device_bytes": 0 if host_reference else max_workspace,
            }
    return tensors, resources


def _prepare_forward_common(layout: Mapping[str, np.ndarray], coulomb_only: bool):
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
    beta = 0.0 if coulomb_only else manifest["options"]["f12_beta"]
    geometry = np.asarray(manifest["geometry_bohr"])
    if not np.array_equal(geometry[common["atom"]], common["centers"]):
        raise ValueError("molecular and shell coordinates disagree")
    return common, beta


def forward_metrics_only(
    layout: Mapping[str, np.ndarray],
    library: Path,
    *,
    host_reference: bool = False,
    coulomb_only: bool = False,
    budget_mib: int = 512,
) -> dict[str, np.ndarray]:
    """Build only the tiny [Q, Q] metric matrices without allocating any [Q, ext, ext] AO panel."""
    if not 0 < budget_mib <= (2**64 - 1) >> 20:
        raise ValueError("invalid forward device-memory budget")
    budget = budget_mib << 20
    common, beta = _prepare_forward_common(layout, coulomb_only)
    kinds = ("coulomb",) if coulomb_only else ("coulomb", "geminal")
    funcs = {k: bind_forward(library, k, host_reference) for k in kinds}
    ops = {"coulomb": _OPCODES["coulomb"]} if coulomb_only else _OPCODES
    projection = ShellProjection(common, "metric")
    offsets, blocks = projection.offsets, projection.blocks
    nblocks, ncenters = blocks.shape
    ncart = int(offsets[-1])
    tensors = {}
    for op, opcode in ops.items():
        label = "metric" if op == "coulomb" else "metric_" + op
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
        if result:
            raise RuntimeError(f"{label} native forward returned status {result}")
        if not np.isfinite(raw).all() or not (0 < workspace.value <= budget):
            raise ValueError(f"{label} incomplete native outputs or invalid workspace")
        tensors[label] = projection.spherical_values(raw)
        del raw
    return tensors


def forward_single_ao_operator(
    layout: Mapping[str, np.ndarray],
    library: Path,
    op: str,
    *,
    host_reference: bool = False,
    coulomb_only: bool = False,
    budget_mib: int = 512,
) -> np.ndarray:
    """Build exactly ONE [Q, ext, ext] spherical AO operator tensor and immediately release its Cartesian scratch."""
    if not 0 < budget_mib <= (2**64 - 1) >> 20:
        raise ValueError("invalid forward device-memory budget")
    if op not in _OPCODES:
        raise ValueError(f"unknown F12 operator {op!r}")
    budget = budget_mib << 20
    common, beta = _prepare_forward_common(layout, coulomb_only or op == "coulomb")
    func = bind_forward(library, "coulomb" if op == "coulomb" else "geminal", host_reference)
    projection = ShellProjection(common, "ao")
    offsets, blocks = projection.offsets, projection.blocks
    _, ncenters = blocks.shape
    label = "ao_" + op
    values = np.empty(projection.shape, dtype=np.float64)
    for b0, b1 in projection.block_chunks():
        c_blocks = np.ascontiguousarray(blocks[b0:b1])
        c_offsets = np.ascontiguousarray(offsets[b0 : b1 + 1] - offsets[b0], dtype=np.int32)
        ncart = int(c_offsets[-1])
        raw = np.full(ncart, np.nan)
        workspace = ct.c_uint64()
        result = func(
            ncenters,
            _OPCODES[op],
            beta,
            len(common["angular"]),
            len(common["exponents"]),
            len(c_blocks),
            ncart,
            common["angular"],
            common["prim_offsets"],
            common["exponents"],
            common["radial"],
            common["centers"],
            c_blocks,
            c_offsets,
            raw,
            budget,
            ct.byref(workspace),
        )
        if result:
            raise RuntimeError(f"{label} native forward returned status {result}")
        if not np.isfinite(raw).all() or not (0 < workspace.value <= budget):
            raise ValueError(f"{label} incomplete native outputs or invalid workspace")
        projection.spherical_values_into(raw, values, b0, b1)
        del raw
    return values
