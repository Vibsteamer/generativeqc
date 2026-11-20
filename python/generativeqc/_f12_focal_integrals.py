"""Build molecular two-/three-center spherical AO tensors from native shell integrals."""

import ctypes as ct
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._f12_focal_integral_kernel import _OPCODES
from ._f12_focal_integral_sink import ShellProjection, contract_live_weights
from ._f12_focal_metric import metric_whitener
from ._f12_focal_shells import focal_shell_layout

__all__ = [
    "contract_live_weights",
    "focal_shell_layout",
    "forward_integrals",
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
        ncart = int(offsets[-1])
        for op, opcode in ops.items():
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
