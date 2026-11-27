"""Unrestricted 3C(FIX) orbital sources and their reverse map to a common AO basis."""

from collections import OrderedDict
from collections.abc import Callable
import ctypes as ct
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import threading
from typing import Literal
import weakref

import numpy as np
from generativeqc_compiler.mp2.f12_density_fitting import (
    coulomb_fit_response,
    molecular_orbital_sources,
)
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_cross_frame_overlap
from generativeqc_compiler.tensor import Program, transpose_program

from ._f12_ao_mo_native import (
    AoMoNativeOps,
    detect_host_reference,
    resolve_ao_mo_ops,
    run_ao_mo_bilin_fwd,
    run_ao_mo_bilin_rev,
    run_ao_mo_project_fwd,
    run_ao_mo_project_rev,
    run_coulomb_fit_fwd,
    run_coulomb_fit_rev,
)
from ._f12_spin_fock import _tile_groups, default_auxiliary_batch_size
from ._f12_spin_pairs import batch_execute

_OPERATORS = {
    "hcore": "hcore",
    "coulomb": "raw_coulomb",
    "geminal": "raw_f12",
    "squared_geminal": "raw_f12_squared",
    "geminal_coulomb": "raw_f12_over_coulomb",
    "double_commutator": "raw_double_commutator",
}

_THREE_CENTER_OPS = (
    "coulomb",
    "geminal",
    "squared_geminal",
    "geminal_coulomb",
    "double_commutator",
)

_PROJECTED_CACHE_MAX_ENTRIES = 2
_PROJECTED_CACHE_MAX_BYTES = 2 << 30
_DENSE_FULL_HASH_MAX_BYTES = 32 << 20


def had_reuse_disabled() -> bool:
    """Return True when ha-D / projected MO source reuse is explicitly disabled."""
    return os.environ.get("GENERATIVEQC_F12_DISABLE_HAD_REUSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _freeze_readonly(arr: np.ndarray) -> np.ndarray:
    if arr.flags.writeable:
        if arr.base is None:
            arr.setflags(write=False)
        else:
            arr = np.array(arr, copy=True)
            arr.setflags(write=False)
    return arr


def _readonly_view(arr: np.ndarray) -> np.ndarray:
    view = arr.view()
    view.setflags(write=False)
    return view


def _is_truly_readonly(arr: np.ndarray) -> bool:
    """Return True only when arr and its entire NumPy base chain are read-only."""
    cur: object = arr
    while isinstance(cur, np.ndarray):
        if cur.flags.writeable:
            return False
        cur = cur.base
    return cur is None


def _digest_array_full(arr: np.ndarray) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.dtype.str.encode("ascii"))
    if arr.flags.c_contiguous:
        h.update(memoryview(arr).cast("B"))
    elif arr.ndim >= 1:
        for idx in range(arr.shape[0]):
            sl = np.ascontiguousarray(arr[idx])
            h.update(memoryview(sl).cast("B"))
    else:
        c = np.ascontiguousarray(arr)
        h.update(memoryview(c).cast("B"))
    return h.digest()


def _digest_dense_ao(arr: np.ndarray) -> tuple | None:
    if arr.nbytes <= _DENSE_FULL_HASH_MAX_BYTES:
        return ("full", _digest_array_full(arr))
    # Never cache across calls for writable large (>32 MiB) AO panels using partial
    # sampling: unsampled slices could be mutated silently. Only truly read-only
    # large arrays may hit the cross-call cache, verified by a full content digest.
    if not _is_truly_readonly(arr):
        return None
    return ("full_ro", _digest_array_full(arr))


def _entry_nbytes(
    arrays: dict[str, np.ndarray],
    overlap_anchor: np.ndarray,
    metric_anchor: np.ndarray,
) -> int:
    seen: set[int] = set()
    total = 0
    for arr in (*arrays.values(), overlap_anchor, metric_anchor):
        base = arr.base if isinstance(arr.base, np.ndarray) else arr
        ident = id(base)
        if ident not in seen:
            seen.add(ident)
            total += int(base.nbytes)
    return total


@dataclass
class _ProjectedPairCacheEntry:
    key: tuple
    space: tuple[int, int, int, int, int]
    same_spin: bool
    # Retaining the small 2D overlap/metric arrays while the entry stays in the
    # bounded cache prevents Python/allocator address reuse from ever aliasing id().
    overlap_anchor: np.ndarray
    metric_anchor: np.ndarray
    panel_ref: weakref.ReferenceType | None
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    coulomb_rows: dict[str, int] = field(default_factory=dict)
    nbytes: int = 0


_PROJECTED_CACHE_LOCK = threading.RLock()
_PROJECTED_CACHE: OrderedDict[tuple, _ProjectedPairCacheEntry] = OrderedDict()
_PROJECTED_CACHE_STATS: dict[str, int] = {
    "hits": 0,
    "partial_hits": 0,
    "misses": 0,
    "op_hits": 0,
    "op_misses": 0,
    "evictions": 0,
    "projection_calls": 0,
    "coulomb_fit_calls": 0,
}


def clear_projected_source_cache(*, reset_stats: bool = False) -> None:
    """Clear the bounded projected MO pair-source cache."""
    with _PROJECTED_CACHE_LOCK:
        _PROJECTED_CACHE.clear()
        if reset_stats:
            for key in _PROJECTED_CACHE_STATS:
                _PROJECTED_CACHE_STATS[key] = 0


def projected_source_cache_info() -> dict[str, int]:
    """Return a snapshot of projected MO source cache occupancy and counters."""
    with _PROJECTED_CACHE_LOCK:
        return {
            **_PROJECTED_CACHE_STATS,
            "size": len(_PROJECTED_CACHE),
            "max_entries": _PROJECTED_CACHE_MAX_ENTRIES,
            "total_bytes": sum(entry.nbytes for entry in _PROJECTED_CACHE.values()),
            "max_bytes": _PROJECTED_CACHE_MAX_BYTES,
        }


def _evict_projected_cache_to_budget() -> None:
    with _PROJECTED_CACHE_LOCK:
        total_bytes = sum(entry.nbytes for entry in _PROJECTED_CACHE.values())
        while _PROJECTED_CACHE and (
            len(_PROJECTED_CACHE) > _PROJECTED_CACHE_MAX_ENTRIES
            or total_bytes > _PROJECTED_CACHE_MAX_BYTES
        ):
            _, evicted = _PROJECTED_CACHE.popitem(last=False)
            total_bytes -= evicted.nbytes
            _PROJECTED_CACHE_STATS["evictions"] += 1


def _build_projected_cache_key(
    space: tuple[int, int, int, int, int],
    ao: dict[str, np.ndarray],
    run: Callable,
    ops: AoMoNativeOps | None,
    same_spin: bool,
    host_reference: bool | None,
) -> tuple[tuple | None, object | None]:
    oa, ob, obs, ext, aux = space
    del oa, ob, obs
    nc = len(ao["coefficients_alpha"])
    if (
        "overlap_ao" not in ao
        or "metric/coulomb" not in ao
        or "metric_whitener" not in ao
    ):
        return None, None
    overlap = ao["overlap_ao"]
    metric_coul = ao["metric/coulomb"]
    whitener = ao["metric_whitener"]
    for op in _THREE_CENTER_OPS:
        if ("metric/" + op) not in ao:
            return None, None

    ao_loader = ao.get("__ao_loader__")
    panel_owner: object | None = None
    panel_stream_tag: tuple | None = None
    op_tags: list[tuple] = []
    for op in _THREE_CENTER_OPS:
        ao_name = "ao_" + _OPERATORS[op].removeprefix("raw_")
        if ao_name in ao:
            digest = _digest_dense_ao(ao[ao_name])
            if digest is None:
                return None, None
            op_tags.append((op, "arr", digest))
        elif ao_loader is not None:
            if panel_stream_tag is None:
                # Only this evaluator derives its output entirely from the shell
                # tables and options. Arbitrary callbacks can change their panels
                # while exposing the same owner, manifest and fitting metadata.
                from ._f12_focal_molecular import _Panel

                panel_stream_tag = _Panel.stream_key(ao_loader)
                if panel_stream_tag is None:
                    return None, None
                panel_owner = ao_loader.__self__
            op_tags.append((op, "stream", panel_stream_tag))
        else:
            return None, None

    batch_width = default_auxiliary_batch_size(aux, max(nc, ext))
    ops_tag = (
        ("aot", bool(ops.host_reference), ops.library_name)
        if ops is not None
        else ("tensorir", bool(detect_host_reference(run, ao, host_reference)))
    )
    c_alpha_digest = _digest_array_full(ao["coefficients_alpha"])
    c_beta_digest = (
        c_alpha_digest if same_spin else _digest_array_full(ao["coefficients_beta"])
    )
    overlap_digest = _digest_array_full(overlap)
    hcore_digest = _digest_array_full(ao["hcore_ao"]) if "hcore_ao" in ao else None
    whitener_digest = _digest_array_full(whitener)
    metric_digests = tuple(
        _digest_array_full(ao["metric/" + op]) for op in _THREE_CENTER_OPS
    )
    key = (
        space,
        nc,
        batch_width,
        ops_tag,
        id(overlap),
        id(metric_coul),
        c_alpha_digest,
        c_beta_digest,
        overlap_digest,
        hcore_digest,
        whitener_digest,
        metric_digests,
        tuple(op_tags),
    )
    return key, panel_owner


def _projected_cache_entry_valid(
    entry: _ProjectedPairCacheEntry,
    ao: dict[str, np.ndarray],
    panel_owner: object | None,
) -> bool:
    if ao["overlap_ao"] is not entry.overlap_anchor:
        return False
    if ao["metric/coulomb"] is not entry.metric_anchor:
        return False
    if panel_owner is not None:
        if entry.panel_ref is None or entry.panel_ref() is not panel_owner:
            return False
    return True


def _solve_coulomb_fit_fwd(
    ops: AoMoNativeOps | None,
    run: Callable,
    whitener: np.ndarray,
    raw_occ: np.ndarray,
    aux: int,
    occ: int,
    ext: int,
) -> np.ndarray:
    with _PROJECTED_CACHE_LOCK:
        _PROJECTED_CACHE_STATS["coulomb_fit_calls"] += 1
    raw = np.ascontiguousarray(raw_occ).reshape(aux, occ * ext)
    if ops is not None:
        return run_coulomb_fit_fwd(ops, whitener, raw).reshape(aux, occ, ext)
    solve = coulomb_fit_response(aux, occ * ext)
    solved = run(
        Program({"fit": solve.outputs["fit"]}),
        {"metric_whitener": whitener, "three_center": raw},
    ).outputs
    return solved["fit"].reshape(aux, occ, ext)


def project_ao_operator_tiled(
    nc: int,
    ext: int,
    occ: int,
    aux: int,
    field: str,
    ao_name: str,
    coefficients: np.ndarray,
    ao_mat: np.ndarray,
    run: Callable,
    *,
    batch_size: int | None = None,
    shell_library: Path | str | ct.CDLL | None = None,
    host_reference: bool | None = None,
    _ops: AoMoNativeOps | None = None,
) -> np.ndarray:
    """Transform one 3-center AO operator [aux, nc, nc] into the MO frame in bounded auxiliary tiles."""
    left_dim = ext if field == "raw_coulomb" else occ
    ops = (
        _ops
        if _ops is not None
        else resolve_ao_mo_ops(
            run,
            None,
            shell_library=shell_library,
            host_reference=host_reference,
        )
    )
    if ops is not None:
        return run_ao_mo_project_fwd(
            ops,
            nc,
            ext,
            left_dim,
            aux,
            coefficients,
            ao_mat,
            q_chunk_hint=0 if batch_size is None else batch_size,
        )
    width = (
        default_auxiliary_batch_size(aux, max(nc, ext))
        if batch_size is None
        else batch_size
    )
    coeffs = np.ascontiguousarray(coefficients)
    if width >= aux:
        frame = molecular_orbital_sources(nc, ext, occ, aux)
        graph = Program({field: frame.outputs[field]})
        return run(graph, {"coefficients": coeffs, ao_name: np.ascontiguousarray(ao_mat)}).outputs[field]
    out = np.empty((aux, left_dim, ext), dtype=np.float64)
    for tile_len, slices in _tile_groups(aux, width):
        frame = molecular_orbital_sources(nc, ext, occ, tile_len)
        graph = Program({field: frame.outputs[field]})
        with batch_execute(run, graph) as exec_one:
            for rows in slices:
                out[rows] = exec_one(
                    {"coefficients": coeffs, ao_name: np.ascontiguousarray(ao_mat[rows])}
                ).outputs[field]
    return out


def project_ao_operator_pullback_tiled(
    nc: int,
    ext: int,
    occ: int,
    aux: int,
    field: str,
    ao_name: str,
    coefficients: np.ndarray,
    ao_mat: np.ndarray | None,
    seed: np.ndarray,
    run: Callable,
    *,
    bar_coefficients: np.ndarray | None = None,
    bar_ao: np.ndarray | None = None,
    batch_size: int | None = None,
    seed_scale: float = 1.0,
    three_center_bar: np.ndarray | None = None,
    inputs_mode: Literal["both", "coefficients", "ao"] = "both",
    shell_library: Path | str | ct.CDLL | None = None,
    host_reference: bool | None = None,
    _ops: AoMoNativeOps | None = None,
) -> np.ndarray | None:
    """Pull MO operator cotangents [aux, *, ext] back to AO [aux, nc, nc] and/or coefficients in bounded tiles."""
    left_dim = ext if field == "raw_coulomb" else occ
    ops = (
        _ops
        if _ops is not None
        else resolve_ao_mo_ops(
            run,
            None,
            shell_library=shell_library,
            host_reference=host_reference,
        )
    )
    if ops is not None:
        return run_ao_mo_project_rev(
            ops,
            nc,
            ext,
            left_dim,
            aux,
            coefficients,
            ao_mat,
            seed,
            bar_coefficients=bar_coefficients,
            bar_ao=bar_ao,
            seed_scale=seed_scale,
            three_center_bar=three_center_bar,
            inputs_mode=inputs_mode,
            q_chunk_hint=0 if batch_size is None else batch_size,
        )
    width = (
        default_auxiliary_batch_size(aux, max(nc, ext))
        if batch_size is None
        else batch_size
    )
    coeffs = np.ascontiguousarray(coefficients)
    if inputs_mode != "coefficients" and bar_ao is None:
        if ao_mat is None:
            bar_ao = np.zeros((aux, nc, nc), dtype=np.float64)
        else:
            bar_ao = np.zeros_like(ao_mat)
    req_inputs = (
        ("coefficients",)
        if inputs_mode == "coefficients"
        else ((ao_name,) if inputs_mode == "ao" else None)
    )
    for tile_len, slices in _tile_groups(aux, width):
        frame = molecular_orbital_sources(nc, ext, occ, tile_len)
        graph = Program({field: frame.outputs[field]})
        rev = transpose_program(graph, (field,), inputs=req_inputs).program
        with batch_execute(run, rev) as exec_one:
            pad_coul = field == "raw_coulomb" and seed.ndim == 3 and seed.shape[1] < ext
            tc_rows = three_center_bar.shape[1] if three_center_bar is not None else 0
            for rows in slices:
                if pad_coul:
                    seed_tile = np.zeros((tile_len, ext, ext), dtype=np.float64)
                    seed_tile[:, : seed.shape[1], :] = seed[rows]
                    if three_center_bar is not None:
                        seed_tile[:, :tc_rows] += three_center_bar[rows]
                    if seed_scale != 1.0:
                        seed_tile *= seed_scale
                elif three_center_bar is not None:
                    seed_tile = np.array(seed[rows], dtype=np.float64, copy=True)
                    seed_tile[:, :tc_rows] += three_center_bar[rows]
                    if seed_scale != 1.0:
                        seed_tile *= seed_scale
                elif seed_scale != 1.0:
                    seed_tile = np.ascontiguousarray(seed[rows] * seed_scale)
                else:
                    seed_tile = np.ascontiguousarray(seed[rows])
                feeds = {"coefficients": coeffs, "bar_" + field: seed_tile}
                if inputs_mode != "ao":
                    assert ao_mat is not None
                    feeds[ao_name] = np.ascontiguousarray(ao_mat[rows])
                weights = exec_one(feeds).outputs
                if bar_coefficients is not None and "bar_coefficients" in weights:
                    bar_coefficients += weights["bar_coefficients"]
                if bar_ao is not None and ("bar_" + ao_name) in weights:
                    bar_ao[rows] += weights["bar_" + ao_name]
    return bar_ao


def unrestricted_orbital_sources(
    space: tuple[int, int, int, int, int],
    ao: dict[str, np.ndarray],
    run: Callable,
    *,
    orbital_weights: dict[str, np.ndarray] | None = None,
    shell_library: Path | str | ct.CDLL | None = None,
    host_reference: bool | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Project shared AO hcore, three-center operators, and metrics to both spin frames.

    The second return value transfers an optional electronic source adjoint to
    the independent spin coefficients and the common AO Hamiltonian, Coulomb
    and geminal metrics, overlap, and all integral factors. A single full-rank
    whitener must satisfy W.T @ W = inverse(Coulomb metric). Its physical
    response is taken through the defining metric equation. Only one operator
    tile per spin is transformed at a time; this keeps workspace independent
    of the number of operators.
    """
    return _orbital_sources(
        space,
        ao,
        run,
        orbital_weights,
        projected=None,
        shell_library=shell_library,
        host_reference=host_reference,
    )


def _orbital_sources(
    space: tuple[int, int, int, int, int],
    ao: dict[str, np.ndarray],
    run: Callable,
    orbital_weights: dict[str, np.ndarray] | None,
    *,
    projected: dict[str, np.ndarray] | None,
    role: Literal["shared", "pairs", "fock", "ordinary_jk", "ordinary_fock"] = "shared",
    ao_bar_sink: Callable[[str, np.ndarray], None] | None = None,
    occupied_coulomb: bool = False,
    obs_coulomb: bool = False,
    skip_coulomb_fit_solve: bool = False,
    shell_library: Path | str | ct.CDLL | None = None,
    host_reference: bool | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    # A combined energy/response evaluation owns the forward buffers through
    # the reverse pass. Standalone evaluations build their own projections.
    oa, ob, obs, ext, aux = space
    valid_frame = obs == ext if role in ("ordinary_jk", "ordinary_fock") else obs < ext
    if not (0 <= oa < obs and 0 <= ob < obs and valid_frame and oa + ob > 0):
        raise ValueError("Invalid UHF occupied, OBS, or extended dimensions")
    if occupied_coulomb and obs_coulomb:
        raise ValueError("occupied_coulomb and obs_coulomb are mutually exclusive")
    nc = len(ao["coefficients_alpha"])
    if nc <= 0 or ext > nc or aux <= 0:
        raise ValueError("Invalid combined AO, extended orbital, or auxiliary dimensions")
    occupied = {"alpha": oa, "beta": ob}
    same_spin = oa == ob and np.array_equal(
        ao["coefficients_alpha"], ao["coefficients_beta"]
    )
    ao_loader = ao.get("__ao_loader__")
    ops = resolve_ao_mo_ops(
        run,
        ao,
        shell_library=shell_library,
        host_reference=host_reference,
    )
    cache_eligible = (
        not had_reuse_disabled()
        and projected is None
        and orbital_weights is None
        and role in ("shared", "pairs")
    )
    cache_key: tuple | None = None
    cache_entry: _ProjectedPairCacheEntry | None = None
    cached_arrays: dict[str, np.ndarray] | None = None
    cached_coulomb_rows: dict[str, int] | None = None
    cache_panel_owner: object | None = None
    if cache_eligible:
        cache_key, cache_panel_owner = _build_projected_cache_key(
            space, ao, run, ops, same_spin, host_reference
        )
        if cache_key is not None:
            with _PROJECTED_CACHE_LOCK:
                cand = _PROJECTED_CACHE.get(cache_key)
                if cand is not None and _projected_cache_entry_valid(
                    cand, ao, cache_panel_owner
                ):
                    cache_entry = cand
                    _PROJECTED_CACHE.move_to_end(cache_key)
                    cached_arrays = dict(cand.arrays)
                    cached_coulomb_rows = dict(cand.coulomb_rows)
    new_arrays: dict[str, np.ndarray] = {}
    new_coulomb_rows: dict[str, int] = {}
    computed_any = False

    needs_cross = role in ("shared", "pairs")
    cross = None
    csrc = None
    if needs_cross:
        csrc = {
            k: ao[k] for k in ("overlap_ao", "coefficients_alpha", "coefficients_beta")
        }
        if ops is None:
            cross = unrestricted_cross_frame_overlap(nc, ext)
    if projected is None:
        data = {"space": np.array(space)}
        if needs_cross:
            if cached_arrays is not None and "cross_overlap" in cached_arrays:
                data["cross_overlap"] = _readonly_view(cached_arrays["cross_overlap"])
                with _PROJECTED_CACHE_LOCK:
                    _PROJECTED_CACHE_STATS["op_hits"] += 1
            else:
                if ops is not None:
                    cross_val = run_ao_mo_bilin_fwd(
                        ops,
                        ao["overlap_ao"],
                        ao["coefficients_alpha"],
                        ao["coefficients_beta"],
                    )
                else:
                    assert cross is not None and csrc is not None
                    cross_val = run(cross, csrc).outputs["cross_overlap"]
                if cache_eligible and cache_key is not None:
                    cross_val = _freeze_readonly(cross_val)
                    new_arrays["cross_overlap"] = cross_val
                    data["cross_overlap"] = _readonly_view(cross_val)
                    computed_any = True
                    with _PROJECTED_CACHE_LOCK:
                        _PROJECTED_CACHE_STATS["op_misses"] += 1
                else:
                    data["cross_overlap"] = cross_val
    else:
        data = projected
    bars = None
    if orbital_weights is not None:
        skip_keys = {
            "metric_whitener",
            "__ao_loader__",
            "__obs_coulomb__",
            "__shell_library__",
        }
        bars = {
            k: np.zeros_like(v)
            for k, v in ao.items()
            if k not in skip_keys and not (ao_bar_sink is not None and k.startswith("ao_"))
        }
        if needs_cross:
            if ops is not None:
                run_ao_mo_bilin_rev(
                    ops,
                    ao["overlap_ao"],
                    ao["coefficients_alpha"],
                    ao["coefficients_beta"],
                    orbital_weights["cross_overlap"],
                    bar_a=bars["overlap_ao"],
                    bar_c_left=bars["coefficients_alpha"],
                    bar_c_right=bars["coefficients_beta"],
                )
            else:
                assert cross is not None and csrc is not None
                cross_rev = transpose_program(cross, ("cross_overlap",)).program
                result = run(
                    cross_rev,
                    {**csrc, "bar_cross_overlap": orbital_weights["cross_overlap"]},
                ).outputs
                for name, value in result.items():
                    bars[name.removeprefix("bar_")] += value
    operators = tuple(_OPERATORS)
    if role == "ordinary_jk":
        operators = ("coulomb",)
    elif role in ("fock", "ordinary_fock"):
        operators = ("hcore", "coulomb")
    elif role == "pairs":
        operators = tuple(op for op in operators if op != "hcore")
    for op in operators:
        if op not in ("hcore", "coulomb"):
            name = "metric/" + op
            data[name] = ao[name]
            if bars is not None:
                bars[name] += orbital_weights[name]
    hcore_frames = (
        None
        if ops is not None
        else {
            spin: molecular_orbital_sources(nc, ext, occ, 1)
            for spin, occ in occupied.items()
        }
    )

    def _spin_has_field(spin_name: str, occ_count: int, op_name: str, field_name: str) -> bool:
        if ops is not None:
            return op_name in ("hcore", "coulomb") or occ_count > 0
        assert hcore_frames is not None
        return field_name in hcore_frames[spin_name].outputs

    # Iterate operators outermost so streaming mode holds at most ONE [Q, ext, ext] AO panel
    # (and at most ONE [Q, ext, ext] cotangent) in host RAM across both spin channels.
    for op in operators:
        field = _OPERATORS[op]
        ao_name = "hcore_ao" if op == "hcore" else "ao_" + field.removeprefix("raw_")
        if not any(_spin_has_field(spin, occ, op, field) for spin, occ in occupied.items()):
            continue

        if op == "hcore":
            if (
                projected is None
                and cached_arrays is not None
                and all(
                    ("source/hcore_" + s) in cached_arrays
                    for s, o in occupied.items()
                    if _spin_has_field(s, o, op, field)
                )
            ):
                for spin, occ in occupied.items():
                    if not _spin_has_field(spin, occ, op, field):
                        continue
                    key = "source/hcore_" + spin
                    data[key] = _readonly_view(cached_arrays[key])
                with _PROJECTED_CACHE_LOCK:
                    _PROJECTED_CACHE_STATS["op_hits"] += 1
                continue
            ao_mat = ao[ao_name]
            if bars is not None and ao_name not in bars:
                bars[ao_name] = np.zeros_like(ao_mat)
            for spin, occ in occupied.items():
                if not _spin_has_field(spin, occ, op, field):
                    continue
                co = "coefficients_" + spin
                key = "source/hcore_" + spin
                if ops is not None:
                    if projected is None:
                        if same_spin and spin == "beta":
                            h_val = (
                                _readonly_view(new_arrays["source/hcore_alpha"])
                                if cache_eligible and cache_key is not None
                                else data["source/hcore_alpha"].copy()
                            )
                        else:
                            h_val = run_ao_mo_bilin_fwd(ops, ao_mat, ao[co], ao[co])
                        if cache_eligible and cache_key is not None:
                            h_val = _freeze_readonly(h_val)
                            new_arrays[key] = h_val
                            data[key] = _readonly_view(h_val)
                        else:
                            data[key] = h_val
                    if bars is not None:
                        run_ao_mo_bilin_rev(
                            ops,
                            ao_mat,
                            ao[co],
                            ao[co],
                            orbital_weights[key],
                            bar_a=bars[ao_name],
                            bar_c_left=bars[co],
                            bar_c_right=bars[co],
                        )
                else:
                    assert hcore_frames is not None
                    graph = Program({field: hcore_frames[spin].outputs[field]})
                    item = {"coefficients": ao[co], ao_name: ao_mat}
                    if projected is None:
                        h_val = run(graph, item).outputs[field]
                        if cache_eligible and cache_key is not None:
                            h_val = _freeze_readonly(h_val)
                            new_arrays[key] = h_val
                            data[key] = _readonly_view(h_val)
                        else:
                            data[key] = h_val
                    if bars is not None:
                        rev = transpose_program(graph, (field,)).program
                        weights = run(rev, {**item, "bar_" + field: orbital_weights[key]}).outputs
                        bars[co] += weights["bar_coefficients"]
                        bars[ao_name] += weights["bar_" + ao_name]
                        del weights
            if projected is None and cache_eligible and cache_key is not None:
                computed_any = True
                with _PROJECTED_CACHE_LOCK:
                    _PROJECTED_CACHE_STATS["op_misses"] += 1
            continue

        use_occ_coul = occupied_coulomb and op == "coulomb"
        use_obs_coul = obs_coulomb and op == "coulomb"
        graph_field = "raw_f12" if (use_occ_coul or use_obs_coul) else field
        graph_ao_name = "ao_f12" if (use_occ_coul or use_obs_coul) else ao_name

        def _op_key(spin_name: str) -> str:
            return (
                "source/" + field + "_" + spin_name
                if op == "coulomb"
                else "raw/" + op + "/" + spin_name
            )

        req_coul_rows = (
            {
                spin: (occ if use_occ_coul else (obs if use_obs_coul else ext))
                for spin, occ in occupied.items()
            }
            if op == "coulomb"
            else {}
        )

        # Serve pure-forward pair/shared operators from the bounded cache BEFORE
        # calling ao_loader(ao_name) so streaming panels avoid redundant shell eval.
        if projected is None and cached_arrays is not None and cached_coulomb_rows is not None:
            if op == "coulomb":
                coul_ready = all(
                    _op_key(s) in cached_arrays
                    and cached_coulomb_rows.get(s, -1) >= req_coul_rows[s]
                    for s, o in occupied.items()
                    if _spin_has_field(s, o, op, field)
                )
                if coul_ready:
                    for spin, occ in occupied.items():
                        if not _spin_has_field(spin, occ, op, field):
                            continue
                        key = _op_key(spin)
                        if same_spin and spin == "beta":
                            data[key] = data[_op_key("alpha")]
                            if occ and not skip_coulomb_fit_solve:
                                data["source/fit_beta"] = data["source/fit_alpha"]
                                if "source/fit_alpha" in new_arrays:
                                    new_arrays["source/fit_beta"] = new_arrays["source/fit_alpha"]
                            continue
                        cached_coul = cached_arrays[key]
                        req_rows = req_coul_rows[spin]
                        sliced = (
                            cached_coul
                            if cached_coul.shape[1] == req_rows
                            else cached_coul[:, :req_rows, :]
                        )
                        data[key] = _readonly_view(sliced)
                        if occ and not skip_coulomb_fit_solve:
                            fit_key = "source/fit_" + spin
                            if fit_key in cached_arrays:
                                data[fit_key] = _readonly_view(cached_arrays[fit_key])
                            else:
                                fit_val = _freeze_readonly(
                                    _solve_coulomb_fit_fwd(
                                        ops,
                                        run,
                                        ao["metric_whitener"],
                                        data[key][:, :occ],
                                        aux,
                                        occ,
                                        ext,
                                    )
                                )
                                new_arrays[fit_key] = fit_val
                                data[fit_key] = _readonly_view(fit_val)
                                computed_any = True
                    with _PROJECTED_CACHE_LOCK:
                        _PROJECTED_CACHE_STATS["op_hits"] += 1
                    continue
            else:
                op_ready = all(
                    _op_key(s) in cached_arrays
                    for s, o in occupied.items()
                    if _spin_has_field(s, o, op, field)
                )
                if op_ready:
                    for spin, occ in occupied.items():
                        if not _spin_has_field(spin, occ, op, field):
                            continue
                        key = _op_key(spin)
                        if same_spin and spin == "beta":
                            data[key] = data[_op_key("alpha")]
                        else:
                            data[key] = _readonly_view(cached_arrays[key])
                    with _PROJECTED_CACHE_LOCK:
                        _PROJECTED_CACHE_STATS["op_hits"] += 1
                    continue

        stream_this = ao_loader is not None and ao_name not in ao
        ao_mat = ao_loader(ao_name) if stream_this else ao[ao_name]
        sink_this = bars is not None and ao_bar_sink is not None
        bar_ao_mat = None
        target_bar_ao = None

        # Forward pass (and metric solve for Coulomb)
        if projected is None:
            for spin, occ in occupied.items():
                if not _spin_has_field(spin, occ, op, field):
                    continue
                key = _op_key(spin)
                if same_spin and spin == "beta":
                    data[key] = data[_op_key("alpha")]
                    if op == "coulomb" and occ and not skip_coulomb_fit_solve:
                        data["source/fit_beta"] = data["source/fit_alpha"]
                    if cache_eligible and cache_key is not None:
                        new_arrays[key] = new_arrays[_op_key("alpha")]
                        if op == "coulomb":
                            new_coulomb_rows["beta"] = new_coulomb_rows["alpha"]
                            if (
                                occ
                                and not skip_coulomb_fit_solve
                                and "source/fit_alpha" in new_arrays
                            ):
                                new_arrays["source/fit_beta"] = new_arrays["source/fit_alpha"]
                    continue
                proj_occ = obs if use_obs_coul else occ
                if use_occ_coul and occ == 0:
                    proj_val = np.empty((aux, 0, ext), dtype=np.float64)
                else:
                    with _PROJECTED_CACHE_LOCK:
                        _PROJECTED_CACHE_STATS["projection_calls"] += 1
                    proj_val = project_ao_operator_tiled(
                        nc,
                        ext,
                        proj_occ,
                        aux,
                        graph_field,
                        graph_ao_name,
                        ao["coefficients_" + spin],
                        ao_mat,
                        run,
                        _ops=ops,
                    )
                if cache_eligible and cache_key is not None:
                    proj_val = _freeze_readonly(proj_val)
                    new_arrays[key] = proj_val
                    data[key] = _readonly_view(proj_val)
                    if op == "coulomb":
                        new_coulomb_rows[spin] = req_coul_rows[spin]
                else:
                    data[key] = proj_val
                if op == "coulomb" and occ and not skip_coulomb_fit_solve:
                    fit_key = "source/fit_" + spin
                    if cached_arrays is not None and fit_key in cached_arrays:
                        data[fit_key] = _readonly_view(cached_arrays[fit_key])
                    else:
                        fit_val = _solve_coulomb_fit_fwd(
                            ops,
                            run,
                            ao["metric_whitener"],
                            data[key][:, :occ],
                            aux,
                            occ,
                            ext,
                        )
                        if cache_eligible and cache_key is not None:
                            fit_val = _freeze_readonly(fit_val)
                            new_arrays[fit_key] = fit_val
                            data[fit_key] = _readonly_view(fit_val)
                        else:
                            data[fit_key] = fit_val
            if cache_eligible and cache_key is not None:
                computed_any = True
                with _PROJECTED_CACHE_LOCK:
                    _PROJECTED_CACHE_STATS["op_misses"] += 1

        # Reverse pass (two-phase for large full-ext Coulomb so ao_mat and bar_ao never coexist)
        if bars is not None:
            assert orbital_weights is not None
            can_share_rev = (
                same_spin
                and all(_spin_has_field(s, occupied[s], op, field) for s in ("alpha", "beta"))
                and np.array_equal(orbital_weights[_op_key("alpha")], orbital_weights[_op_key("beta")])
                and (
                    op != "coulomb"
                    or oa == 0
                    or skip_coulomb_fit_solve
                    or np.array_equal(orbital_weights["source/fit_alpha"], orbital_weights["source/fit_beta"])
                )
            )
            # Compute Coulomb metric response terms [aux, occ, ext] without copying full [aux, ext, ext] seeds
            tc_bars: dict[str, np.ndarray | None] = {}
            rev_spins = (("alpha", oa),) if can_share_rev else tuple(occupied.items())
            for spin, occ in rev_spins:
                if not _spin_has_field(spin, occ, op, field):
                    continue
                key = _op_key(spin)
                if op == "coulomb" and occ and not skip_coulomb_fit_solve:
                    raw = np.ascontiguousarray(data[key][:, :occ]).reshape(aux, occ * ext)
                    fit_key = "source/fit_" + spin
                    if ops is not None:
                        tc_bars[spin] = run_coulomb_fit_rev(
                            ops,
                            ao["metric_whitener"],
                            raw,
                            np.ascontiguousarray(orbital_weights[fit_key]).reshape(aux, occ * ext),
                            bars["metric/coulomb"],
                            metric_scale=2.0 if can_share_rev else 1.0,
                        ).reshape(aux, occ, ext)
                    else:
                        solve = coulomb_fit_response(aux, occ * ext)
                        fit_arg = {
                            "metric_whitener": ao["metric_whitener"],
                            "three_center": raw,
                            "fit_bar": orbital_weights[fit_key].reshape(aux, -1),
                        }
                        if projected is not None:
                            solve = Program(
                                {k: solve.outputs[k] for k in ("three_center_bar", "metric_bar")}
                            )
                        solved = run(solve, fit_arg).outputs
                        tc_bars[spin] = solved["three_center_bar"].reshape(aux, occ, ext)
                        bars["metric/coulomb"] += (2.0 if can_share_rev else 1.0) * solved["metric_bar"]
                else:
                    tc_bars[spin] = None

            split_ao = stream_this
            if split_ao:
                # Phase 1: compute bar_coefficients using ao_mat + seed, then free ao_mat before allocating bar_ao
                for spin, occ in rev_spins:
                    if not _spin_has_field(spin, occ, op, field) or (use_occ_coul and occ == 0):
                        continue
                    key = _op_key(spin)
                    proj_occ = obs if use_obs_coul else occ
                    bar_c_target = np.zeros_like(bars["coefficients_alpha"]) if can_share_rev else bars["coefficients_" + spin]
                    project_ao_operator_pullback_tiled(
                        nc,
                        ext,
                        proj_occ,
                        aux,
                        graph_field,
                        graph_ao_name,
                        ao["coefficients_" + spin],
                        ao_mat,
                        orbital_weights[key],
                        run,
                        bar_coefficients=bar_c_target,
                        bar_ao=None,
                        seed_scale=2.0 if can_share_rev else 1.0,
                        three_center_bar=tc_bars[spin],
                        inputs_mode="coefficients",
                        _ops=ops,
                    )
                    if can_share_rev:
                        bars["coefficients_alpha"] += 0.5 * bar_c_target
                        bars["coefficients_beta"] += 0.5 * bar_c_target
                del ao_mat
                ao_mat = None
                if sink_this:
                    bar_ao_mat = np.zeros((aux, nc, nc), dtype=np.float64)
                    target_bar_ao = bar_ao_mat
                else:
                    if ao_name not in bars:
                        bars[ao_name] = np.zeros((aux, nc, nc), dtype=np.float64)
                    target_bar_ao = bars[ao_name]
                # Phase 2: compute bar_ao using ONLY seed (ao_mat is already freed!)
                for spin, occ in rev_spins:
                    if not _spin_has_field(spin, occ, op, field) or (use_occ_coul and occ == 0):
                        continue
                    key = _op_key(spin)
                    proj_occ = obs if use_obs_coul else occ
                    project_ao_operator_pullback_tiled(
                        nc,
                        ext,
                        proj_occ,
                        aux,
                        graph_field,
                        graph_ao_name,
                        ao["coefficients_" + spin],
                        None,
                        orbital_weights[key],
                        run,
                        bar_coefficients=None,
                        bar_ao=target_bar_ao,
                        seed_scale=2.0 if can_share_rev else 1.0,
                        three_center_bar=tc_bars[spin],
                        inputs_mode="ao",
                        _ops=ops,
                    )
            else:
                if sink_this:
                    bar_ao_mat = np.zeros((aux, nc, nc), dtype=np.float64)
                    target_bar_ao = bar_ao_mat
                else:
                    if ao_name not in bars:
                        bars[ao_name] = np.zeros((aux, nc, nc), dtype=np.float64)
                    target_bar_ao = bars[ao_name]
                for spin, occ in rev_spins:
                    if not _spin_has_field(spin, occ, op, field) or (use_occ_coul and occ == 0):
                        continue
                    key = _op_key(spin)
                    proj_occ = obs if use_obs_coul else occ
                    bar_c_target = np.zeros_like(bars["coefficients_alpha"]) if can_share_rev else bars["coefficients_" + spin]
                    project_ao_operator_pullback_tiled(
                        nc,
                        ext,
                        proj_occ,
                        aux,
                        graph_field,
                        graph_ao_name,
                        ao["coefficients_" + spin],
                        ao_mat,
                        orbital_weights[key],
                        run,
                        bar_coefficients=bar_c_target,
                        bar_ao=target_bar_ao,
                        seed_scale=2.0 if can_share_rev else 1.0,
                        three_center_bar=tc_bars[spin],
                        _ops=ops,
                    )
                    if can_share_rev:
                        bars["coefficients_alpha"] += 0.5 * bar_c_target
                        bars["coefficients_beta"] += 0.5 * bar_c_target
        # Drop target_bar_ao alias BEFORE invoking ao_bar_sink and clear bar_ao_mat
        # immediately afterward so the previous [Q, ext, ext] panel is freed before
        # the next operator allocates its AO buffer.
        target_bar_ao = None
        if ao_mat is not None:
            del ao_mat
            ao_mat = None
        if sink_this:
            assert bar_ao_mat is not None
            ao_bar_sink(ao_name, bar_ao_mat)
            del bar_ao_mat
            bar_ao_mat = None
    if cache_eligible and cache_key is not None:
        with _PROJECTED_CACHE_LOCK:
            existing = _PROJECTED_CACHE.get(cache_key)
            if existing is not None and not _projected_cache_entry_valid(
                existing, ao, cache_panel_owner
            ):
                existing = None
            if cache_entry is not None and existing is not None:
                if not computed_any:
                    _PROJECTED_CACHE_STATS["hits"] += 1
                else:
                    _PROJECTED_CACHE_STATS["partial_hits"] += 1
                    existing.arrays.update(new_arrays)
                    existing.coulomb_rows.update(new_coulomb_rows)
                    existing.nbytes = _entry_nbytes(
                        existing.arrays,
                        existing.overlap_anchor,
                        existing.metric_anchor,
                    )
                    if existing.nbytes > _PROJECTED_CACHE_MAX_BYTES:
                        _PROJECTED_CACHE.pop(cache_key, None)
                        _PROJECTED_CACHE_STATS["evictions"] += 1
                    else:
                        _PROJECTED_CACHE.move_to_end(cache_key)
                        _evict_projected_cache_to_budget()
            elif cache_entry is not None and not computed_any:
                _PROJECTED_CACHE_STATS["hits"] += 1
            else:
                _PROJECTED_CACHE_STATS["misses"] += 1
                merged_arrays = (
                    {**cached_arrays, **new_arrays}
                    if cached_arrays is not None
                    else new_arrays
                )
                merged_rows = (
                    {**cached_coulomb_rows, **new_coulomb_rows}
                    if cached_coulomb_rows is not None
                    else new_coulomb_rows
                )
                nbytes = _entry_nbytes(
                    merged_arrays, ao["overlap_ao"], ao["metric/coulomb"]
                )
                if nbytes <= _PROJECTED_CACHE_MAX_BYTES:
                    panel_ref = (
                        weakref.ref(cache_panel_owner)
                        if cache_panel_owner is not None
                        else None
                    )
                    _PROJECTED_CACHE[cache_key] = _ProjectedPairCacheEntry(
                        key=cache_key,
                        space=space,
                        same_spin=same_spin,
                        overlap_anchor=ao["overlap_ao"],
                        metric_anchor=ao["metric/coulomb"],
                        panel_ref=panel_ref,
                        arrays=merged_arrays,
                        coulomb_rows=merged_rows,
                        nbytes=nbytes,
                    )
                    _PROJECTED_CACHE.move_to_end(cache_key)
                    _evict_projected_cache_to_budget()
    return data, bars
