"""Build molecular two-/three-center spherical AO tensors from native shell integrals."""

import ctypes as ct
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._f12_focal_integral_kernel import (
    _OPCODES,
    DEFAULT_SLATER_FIT_NORMALIZATION,
    LEGACY_SLATER_NORM_CODE,
    bind_spherical_forward,
    slater_normalization_code,
    spherical_layout_descriptor,
)
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


def bind_forward(
    library: Path, kind: str, host_reference: bool, *, extended: bool = False
) -> Callable:
    ints = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    doubles = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    ext = "_ex" if extended else ""
    name = f"gqc_f12_cartesian_shells_{kind}{ext}" + (
        "_host_reference" if host_reference else ""
    )
    func = getattr(ct.CDLL(str(library.resolve())), name)
    args = [ct.c_int, ct.c_int, ct.c_double]
    if extended:
        args.append(ct.c_int)
    args.extend(
        [
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
    )
    func.argtypes = args
    func.restype = ct.c_int
    return func


def _bind_fallback_forward(
    library: Path, kind: str, host_reference: bool, norm_code: int
) -> tuple[Callable, bool]:
    if kind != "coulomb":
        try:
            return bind_forward(library, kind, host_reference, extended=True), True
        except AttributeError as exc:
            if norm_code != LEGACY_SLATER_NORM_CODE:
                raise RuntimeError(
                    "Configured SlaterFitNormalization requires updated integral "
                    "sink library exporting *_ex or *_spherical_* symbols"
                ) from exc
    return bind_forward(library, kind, host_reference, extended=False), False


def _call_spherical_forward(
    common: Mapping[str, np.ndarray],
    desc: dict,
    func: Callable,
    group: str,
    opcode: int,
    beta: float,
    norm_code: int,
    budget: int,
    label: str,
) -> tuple[np.ndarray, int, float]:
    ncenters = 2 if group == "metric" else 3
    shape = (
        (desc["naux_sph"], desc["naux_sph"])
        if ncenters == 2
        else (desc["naux_sph"], desc["norb_sph"], desc["norb_sph"])
    )
    values = np.empty(shape, dtype=np.float64)
    workspace = ct.c_uint64()
    t0 = time.monotonic()
    rc = func(
        ncenters,
        opcode,
        beta,
        norm_code,
        desc["nshell"],
        desc["orbital_shell_count"],
        len(common["exponents"]),
        desc["norb_sph"],
        desc["naux_sph"],
        np.ascontiguousarray(common["angular"], dtype=np.int32),
        np.ascontiguousarray(common["prim_offsets"], dtype=np.int32),
        np.ascontiguousarray(common["exponents"], dtype=np.float64),
        np.ascontiguousarray(common["radial"], dtype=np.float64),
        np.ascontiguousarray(common["centers"], dtype=np.float64),
        desc["sph_offsets"],
        desc["transform_table"],
        values,
        budget,
        ct.byref(workspace),
    )
    elapsed = time.monotonic() - t0
    if rc:
        raise RuntimeError(f"{label} native spherical forward returned status {rc}")
    if not (0 < workspace.value <= budget):
        raise ValueError(f"{label} invalid native spherical workspace")
    return values, int(workspace.value), elapsed


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
    beta = 0.0 if coulomb_only else float(manifest["options"]["f12_beta"])
    geometry = np.asarray(manifest["geometry_bohr"])
    if not np.array_equal(geometry[common["atom"]], common["centers"]):
        raise ValueError("molecular and shell coordinates disagree")
    return common, beta


def forward_integrals(
    layout: "Mapping[str, np.ndarray] | FocalRequest",
    library: Path,
    *,
    host_reference: bool = False,
    coulomb_only: bool = False,
    budget_mib: int = 512,
    slater_normalization: str = DEFAULT_SLATER_FIT_NORMALIZATION,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Evaluate supplied shell tables or build them from a focal geometry and basis.

    Focal sources are ordered OBS first, then OptRI. Each subspace uses the
    public basis order. The two- and three-center fitting indices share one
    independently numbered fitting basis.
    """
    if not 0 < budget_mib <= (2**64 - 1) >> 20:
        raise ValueError("invalid forward device-memory budget")
    norm_code = slater_normalization_code(slater_normalization)
    budget = budget_mib << 20
    if not isinstance(layout, Mapping):
        coulomb_only = coulomb_only or layout.role == "hf_cabs"
        layout = focal_shell_layout(
            layout, library, host_reference=host_reference, budget_mib=budget_mib
        )
    common, beta = _prepare_forward_common(layout, coulomb_only)
    kinds = ("coulomb",) if coulomb_only else ("coulomb", "geminal")
    sph_funcs = {k: bind_spherical_forward(library, k, host_reference) for k in kinds}
    fallback_funcs: dict[str, tuple[Callable, bool]] = {}
    ops = {"coulomb": _OPCODES["coulomb"]} if coulomb_only else _OPCODES
    tensors, resources = {}, {}
    for group in ("metric", "ao"):
        # Keep tiny [Q, Q] metric on exact ShellProjection to preserve bit-identical
        # M^(-1/2) whitener conditioning; dispatch heavy [Q, ext, ext] AO directly on GPU.
        desc = None if group == "metric" else spherical_layout_descriptor(common, group)
        projection = None
        chunks = None
        nblocks = len(common[group + "_blocks"])
        ncart_total = int(common[group + "_offsets"][-1])
        for op, opcode in ops.items():
            label = group if op == "coulomb" and group == "metric" else group + "_" + op
            kind = "coulomb" if op == "coulomb" else "geminal"
            start = time.monotonic()
            sph_fn = sph_funcs.get(kind)
            if desc is not None and sph_fn is not None:
                values, max_workspace, kernel_seconds = _call_spherical_forward(
                    common,
                    desc,
                    sph_fn,
                    group,
                    opcode,
                    beta,
                    norm_code,
                    budget,
                    label,
                )
            else:
                if projection is None:
                    projection = ShellProjection(common, group)
                    chunks = projection.block_chunks()
                if kind not in fallback_funcs:
                    fallback_funcs[kind] = _bind_fallback_forward(
                        library, kind, host_reference, norm_code
                    )
                fn, is_ext = fallback_funcs[kind]
                offsets, blocks = projection.offsets, projection.blocks
                _, ncenters = blocks.shape
                values = np.empty(projection.shape, dtype=np.float64)
                kernel_seconds = 0.0
                max_workspace = 0
                for b0, b1 in chunks:
                    c_blocks = np.ascontiguousarray(blocks[b0:b1])
                    c_offsets = np.ascontiguousarray(
                        offsets[b0 : b1 + 1] - offsets[b0], dtype=np.int32
                    )
                    c_ncart = int(c_offsets[-1])
                    raw = np.full(c_ncart, np.nan)
                    workspace = ct.c_uint64()
                    t_k0 = time.monotonic()
                    if is_ext:
                        result = fn(
                            ncenters,
                            opcode,
                            beta,
                            norm_code,
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
                    else:
                        result = fn(
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
                "host_spherical_assembly_seconds": max(
                    0.0, time.monotonic() - start - kernel_seconds
                ),
                "workspace_bytes": max_workspace,
                "device_bytes": 0 if host_reference else max_workspace,
            }
    return tensors, resources


def forward_metrics_only(
    layout: Mapping[str, np.ndarray],
    library: Path,
    *,
    host_reference: bool = False,
    coulomb_only: bool = False,
    budget_mib: int = 512,
    slater_normalization: str = DEFAULT_SLATER_FIT_NORMALIZATION,
) -> dict[str, np.ndarray]:
    """Build only the tiny [Q, Q] metric matrices without allocating any [Q, ext, ext] AO panel."""
    if not 0 < budget_mib <= (2**64 - 1) >> 20:
        raise ValueError("invalid forward device-memory budget")
    norm_code = slater_normalization_code(slater_normalization)
    budget = budget_mib << 20
    common, beta = _prepare_forward_common(layout, coulomb_only)
    kinds = ("coulomb",) if coulomb_only else ("coulomb", "geminal")
    sph_funcs = {k: bind_spherical_forward(library, k, host_reference) for k in kinds}
    fallback_funcs: dict[str, tuple[Callable, bool]] = {}
    ops = {"coulomb": _OPCODES["coulomb"]} if coulomb_only else _OPCODES
    # Keep tiny [Q, Q] metric on exact ShellProjection to preserve bit-identical M^(-1/2).
    desc = None
    projection = None
    tensors = {}
    for op, opcode in ops.items():
        label = "metric" if op == "coulomb" else "metric_" + op
        kind = "coulomb" if op == "coulomb" else "geminal"
        sph_fn = sph_funcs.get(kind)
        if desc is not None and sph_fn is not None:
            values, _, _ = _call_spherical_forward(
                common, desc, sph_fn, "metric", opcode, beta, norm_code, budget, label
            )
            tensors[label] = values
        else:
            if projection is None:
                projection = ShellProjection(common, "metric")
            if kind not in fallback_funcs:
                fallback_funcs[kind] = _bind_fallback_forward(
                    library, kind, host_reference, norm_code
                )
            fn, is_ext = fallback_funcs[kind]
            offsets, blocks = projection.offsets, projection.blocks
            nblocks, ncenters = blocks.shape
            ncart = int(offsets[-1])
            raw = np.full(ncart, np.nan)
            workspace = ct.c_uint64()
            if is_ext:
                result = fn(
                    ncenters,
                    opcode,
                    beta,
                    norm_code,
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
            else:
                result = fn(
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
    slater_normalization: str = DEFAULT_SLATER_FIT_NORMALIZATION,
) -> np.ndarray:
    """Build exactly ONE [Q, ext, ext] spherical AO operator tensor and immediately release scratch."""
    if not 0 < budget_mib <= (2**64 - 1) >> 20:
        raise ValueError("invalid forward device-memory budget")
    if op not in _OPCODES:
        raise ValueError(f"unknown F12 operator {op!r}")
    norm_code = slater_normalization_code(slater_normalization)
    budget = budget_mib << 20
    common, beta = _prepare_forward_common(layout, coulomb_only or op == "coulomb")
    kind = "coulomb" if op == "coulomb" else "geminal"
    label = "ao_" + op
    sph_fn = bind_spherical_forward(library, kind, host_reference)
    desc = spherical_layout_descriptor(common, "ao") if sph_fn is not None else None
    if sph_fn is not None and desc is not None:
        values, _, _ = _call_spherical_forward(
            common, desc, sph_fn, "ao", _OPCODES[op], beta, norm_code, budget, label
        )
        return values

    fn, is_ext = _bind_fallback_forward(library, kind, host_reference, norm_code)
    projection = ShellProjection(common, "ao")
    offsets, blocks = projection.offsets, projection.blocks
    _, ncenters = blocks.shape
    values = np.empty(projection.shape, dtype=np.float64)
    for b0, b1 in projection.block_chunks():
        c_blocks = np.ascontiguousarray(blocks[b0:b1])
        c_offsets = np.ascontiguousarray(
            offsets[b0 : b1 + 1] - offsets[b0], dtype=np.int32
        )
        ncart = int(c_offsets[-1])
        raw = np.full(ncart, np.nan)
        workspace = ct.c_uint64()
        if is_ext:
            result = fn(
                ncenters,
                _OPCODES[op],
                beta,
                norm_code,
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
        else:
            result = fn(
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
