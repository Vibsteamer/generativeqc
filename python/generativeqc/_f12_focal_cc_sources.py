"""Spin-resolved fixed-cusp CC equation inputs at one molecular geometry."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
import hashlib
import struct
import threading
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from generativeqc_compiler.common.arrays import immutable

from ._f12_ao_mo_native import resolve_ao_mo_ops
from ._f12_cc_fixed_sources_native import AUTO_NATIVE_LIBRARY
from ._f12_cc_fock_pullback import extended_fock_response
from ._f12_cc_spin_sources import unrestricted_cc_f12_fixed_source_feeds
from ._f12_focal_frame import canonical_pair_frame
from ._f12_spin_ao import (
    _freeze_readonly,
    _orbital_sources,
    _readonly_view,
    clear_projected_source_cache,
    had_reuse_disabled,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_molecular import MolecularFocalProvider
    from ._f12_focal_reference import FocalReference
    from ._uccsd_rhf_virtual_pairs import RHFDFLadderOptions
    from ._uccsd_uhf_virtual_ladder import UHFDFLadderOptions
    from .experimental.f12_focal import FocalRequest

_HAD_CC_CACHE_MAX_ENTRIES = 2
_HAD_CC_CACHE_MAX_BYTES = 1 << 30


@dataclass
class _HadCCSourceCacheEntry:
    key: tuple
    cabs_rank: int
    f12_fit_id: str
    frozen_occupied: tuple[int, int]
    fock: dict[str, np.ndarray]
    feeds: dict[str, np.ndarray]
    nbytes: int


_HAD_CC_CACHE_LOCK = threading.RLock()
_HAD_CC_CACHE: OrderedDict[tuple, _HadCCSourceCacheEntry] = OrderedDict()
_HAD_CC_CACHE_STATS: dict[str, int] = {
    "hits": 0,
    "misses": 0,
    "evictions": 0,
    "fock_reuses": 0,
    "fixed_feed_reuses": 0,
    "extended_fock_calls": 0,
    "fixed_feed_calls": 0,
}


def clear_had_cc_source_cache(*, reset_stats: bool = False) -> None:
    """Clear the bounded ha-D CC fixed-source / extended-Fock cache."""
    with _HAD_CC_CACHE_LOCK:
        _HAD_CC_CACHE.clear()
        if reset_stats:
            for key in _HAD_CC_CACHE_STATS:
                _HAD_CC_CACHE_STATS[key] = 0


def clear_had_source_caches(*, reset_stats: bool = False) -> None:
    """Clear both the projected MO cache and the ha-D CC fixed-source cache."""
    clear_projected_source_cache(reset_stats=reset_stats)
    clear_had_cc_source_cache(reset_stats=reset_stats)


def had_cc_source_cache_info() -> dict[str, int]:
    """Return a snapshot of ha-D CC fixed-source cache occupancy and counters."""
    with _HAD_CC_CACHE_LOCK:
        return {
            **_HAD_CC_CACHE_STATS,
            "size": len(_HAD_CC_CACHE),
            "max_entries": _HAD_CC_CACHE_MAX_ENTRIES,
            "total_bytes": sum(entry.nbytes for entry in _HAD_CC_CACHE.values()),
            "max_bytes": _HAD_CC_CACHE_MAX_BYTES,
        }


def _freeze_f64_array(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim > 0:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    elif arr.base is not None or arr is v:
        arr = arr.copy()
    return _freeze_readonly(arr)


def _reference_digest(ref: FocalReference) -> bytes:
    h = hashlib.sha256()
    coeffs = np.ascontiguousarray(ref.coefficients, dtype=np.float64)
    eigs = np.ascontiguousarray(ref.orbital_energies, dtype=np.float64)
    h.update(str(coeffs.shape).encode("ascii"))
    h.update(memoryview(coeffs).cast("B"))
    h.update(str(eigs.shape).encode("ascii"))
    h.update(memoryview(eigs).cast("B"))
    h.update(struct.pack("<d", float(ref.energy)))
    return h.digest()


def _mapping_nbytes(*mappings: Mapping[str, np.ndarray]) -> int:
    seen: set[int] = set()
    total = 0
    for mapping in mappings:
        for arr in mapping.values():
            base = arr.base if isinstance(arr.base, np.ndarray) else arr
            ident = id(base)
            if ident not in seen:
                seen.add(ident)
                total += int(base.nbytes)
    return total


def _evict_had_cc_cache_to_budget() -> None:
    with _HAD_CC_CACHE_LOCK:
        total_bytes = sum(entry.nbytes for entry in _HAD_CC_CACHE.values())
        while _HAD_CC_CACHE and (
            len(_HAD_CC_CACHE) > _HAD_CC_CACHE_MAX_ENTRIES
            or total_bytes > _HAD_CC_CACHE_MAX_BYTES
        ):
            _, evicted = _HAD_CC_CACHE.popitem(last=False)
            total_bytes -= evicted.nbytes
            _HAD_CC_CACHE_STATS["evictions"] += 1


def _had_cc_source_cache_key(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    ref: FocalReference,
    cabs_rank: int,
    frozen: tuple[int, int],
    space: tuple[int, int, int, int, int],
    ao: Mapping[str, np.ndarray],
    checked: object,
    cuda: bool,
) -> tuple:
    from ._f12_frozen_core import FROZEN_CORE_DEFINITION_VERSION

    _, _, obs, ext, aux = space
    basis = request.basis
    assert basis.f12_fit is not None and basis.cabs is not None
    geom_key = tuple(
        (a.atomic_number, tuple((v if v else 0.0).hex() for v in a.position))
        for a in request.atoms
    )
    ao_mo_ops = resolve_ao_mo_ops(
        checked,  # type: ignore[arg-type]
        ao,
        shell_library=provider.shell_library,
        host_reference=(not cuda),
    )
    ops_tag = (
        ("aot", bool(ao_mo_ops.host_reference), ao_mo_ops.library_name)
        if ao_mo_ops is not None
        else ("tensorir", not cuda)
    )
    return (
        geom_key,
        request.surface_id,
        basis.basis_id,
        request.root_id,
        request.correlation_space_id,
        FROZEN_CORE_DEFINITION_VERSION,
        ref.frozen_frame.occupied_order if ref.frozen_frame is not None else None,
        request.charge,
        request.multiplicity,
        request.backend,
        basis.orbital.identity,
        basis.jk_fit.identity,
        basis.f12_fit.identity,
        basis.cabs.identity,
        basis.beta,
        int(cabs_rank),
        float(provider.relative_cutoff),
        provider.slater_normalization,
        tuple(ref.occupied),
        frozen,
        int(obs),
        int(ext),
        int(aux),
        _reference_digest(ref),
        ops_tag,
    )


@dataclass(frozen=True)
class FocalUCCFixedSources:
    """W_vo, W_vv and C in the returned reference's canonical OBS spin frame.

    feeds contains the six spin orderings per tensor and fixed_geminal_energy
    (V+B-X in Eh). The reference holds the same OBS coefficient matrices, metric,
    occupancy, orbital energies, surface, basis and backend used by the tensors.
    """

    reference: FocalReference
    cabs_rank: int
    f12_fit_id: str
    feeds: dict[str, np.ndarray]
    frozen_occupied: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class FocalUCCEquationSources:
    """Ordinary and fixed-geminal CC inputs for the same canonical DF-HF determinant."""

    fixed: FocalUCCFixedSources
    mp2_fit_id: str
    ordinary: dict[str, np.ndarray]
    rhf_df_ladder: RHFDFLadderOptions | None = None
    virtual_factors: dict[str, np.ndarray] | None = None
    uhf_df_ladder: UHFDFLadderOptions | None = None


@overload
def molecular_ucc_sources(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    *,
    include_ordinary: Literal[False],
    reference_forces: bool = False,
    reference: FocalReference | None = None,
    expected_cabs_rank: int | None = None,
) -> FocalUCCFixedSources: ...


@overload
def molecular_ucc_sources(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    *,
    include_ordinary: Literal[True],
    reference_forces: bool = False,
    reference: FocalReference | None = None,
    expected_cabs_rank: int | None = None,
) -> FocalUCCEquationSources: ...


def molecular_ucc_sources(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    *,
    include_ordinary: bool,
    reference_forces: bool = False,
    reference: FocalReference | None = None,
    expected_cabs_rank: int | None = None,
) -> FocalUCCFixedSources | FocalUCCEquationSources:
    """Calculate fixed-geminal UCC equation inputs on an all-active singlet RHF or UHF root."""
    from generativeqc_compiler.tensor import execute

    from ._f12_focal_one_electron import one_electron_sources
    from ._f12_focal_reference import df_reference, partition_focal_reference
    from ._f12_frozen_core import resolve_frozen_occupied

    if (
        request.role != "mp2_f12_correlation"
        or request.root_id not in ("rhf", "uhf")
        or (request.root_id == "rhf" and request.multiplicity != 1)
    ):
        raise ValueError(
            "fixed-cusp UCC sources require a UHF or singlet RHF molecular pair request"
        )
    if (
        request.basis.f12_fit is None
        or request.basis.beta is None
        or request.basis.cabs is None
    ):
        raise ValueError(
            "fixed-cusp UCC sources require F12Fit, OptRI and a Gaussian exponent"
        )
    if request.backend not in ("cuda", "cpu_reference"):
        raise ValueError("fixed-cusp UCC sources require an explicit device backend")
    if include_ordinary and request.basis.mp2_fit is None:
        raise ValueError(
            "ordinary UCC Hamiltonian inputs require an explicit correlation fitting basis"
        )
    cuda = request.backend == "cuda"
    if cuda and (
        provider.run is None
        or provider.metric_library is None
        or provider.response_library is None
    ):
        raise ValueError(
            "CUDA fixed-cusp UCC sources require the device graph, metric and projector providers"
        )
    executor = provider.run or partial(execute, max_bytes=provider.budget_mib << 20)

    def checked(graph: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        out = executor(graph, inputs)
        if out.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError("UCC fixed-source graph executed on a different backend")
        return out

    if hasattr(executor, "batch"):
        from contextlib import contextmanager

        @contextmanager
        def checked_batch(graph: Program):
            with executor.batch(graph) as exec_one:
                def wrapped(inputs: Mapping[str, np.ndarray]) -> Execution:
                    out = exec_one(inputs)
                    if out.backend != (
                        "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
                    ):
                        raise RuntimeError(
                            "UCC fixed-source graph executed on a different backend"
                        )
                    return out

                yield wrapped

        checked.batch = checked_batch  # type: ignore[attr-defined]

    ref = reference
    if ref is None:
        ref = df_reference(
            request,
            forces=reference_forces,
            budget_mib=provider.budget_mib,
            **provider._seed_options(request),
        )
    elif (
        ref.surface_id != request.surface_id
        or ref.basis_id != request.basis.basis_id
        or ref.executed_backend != request.backend
        or ref.orbital_id != request.basis.orbital.identity
        or ref.jk_fit_id != request.basis.jk_fit.identity
        or (reference_forces and ref.forces is None)
    ):
        raise ValueError("CC equation sources require the matching canonical reference")
    ref = partition_focal_reference(request, ref, provider.budget_mib)
    one, (jk, mp2_panel, corr) = provider._cached_pair_sources(
        request, derivatives=False, budget=provider.budget_mib
    )
    space, ao, panel, proj = canonical_pair_frame(
        request,
        ref,
        one,
        jk.joint(request, explicit=False),
        corr.joint(request, explicit=True),
        expected_cabs_rank=expected_cabs_rank,
        relative_cutoff=provider.relative_cutoff,
        library=provider.response_library,
    )
    oa, ob, obs, ext, _ = space
    spin_indices = ref._pair_spin_indices(request)
    frozen = resolve_frozen_occupied(request, (oa, ob))
    reuse_enabled = not had_reuse_disabled()
    cc_key: tuple | None = None
    cached_entry: _HadCCSourceCacheEntry | None = None
    if reuse_enabled:
        cc_key = _had_cc_source_cache_key(
            provider,
            request,
            ref,
            proj.rank,
            frozen,
            space,
            ao,
            checked,
            cuda,
        )
        with _HAD_CC_CACHE_LOCK:
            cached_entry = _HAD_CC_CACHE.get(cc_key)
            if cached_entry is not None:
                _HAD_CC_CACHE.move_to_end(cc_key)
                _HAD_CC_CACHE_STATS["hits"] += 1
                _HAD_CC_CACHE_STATS["fock_reuses"] += 1
                _HAD_CC_CACHE_STATS["fixed_feed_reuses"] += 1

    data: dict[str, np.ndarray] | None = None
    if cached_entry is not None:
        fock = {k: _readonly_view(v) for k, v in cached_entry.fock.items()}
        feeds = {k: _readonly_view(v) for k, v in cached_entry.feeds.items()}
        fixed = FocalUCCFixedSources(
            ref,
            cached_entry.cabs_rank,
            cached_entry.f12_fit_id,
            feeds,
            cached_entry.frozen_occupied,
        )
        if not include_ordinary:
            return fixed
    else:
        if reuse_enabled:
            with _HAD_CC_CACHE_LOCK:
                _HAD_CC_CACHE_STATS["misses"] += 1
        data, _ = _orbital_sources(
            space,
            ao,
            checked,
            None,
            projected=None,
            role="pairs",
            obs_coulomb=True,
            shell_library=provider.shell_library,
            host_reference=(not cuda),
        )
        with _HAD_CC_CACHE_LOCK:
            _HAD_CC_CACHE_STATS["extended_fock_calls"] += 1
        fock = extended_fock_response(
            space,
            ao,
            panel,
            checked,
            cuda=cuda,
            library=provider.response_library,
            native_library=provider.shell_library,
            host_reference=(not cuda),
        )[0]
        for spin, k in zip(("alpha", "beta"), spin_indices, strict=True):
            if not np.allclose(
                fock["fock_" + spin][:obs, :obs],
                np.diag(ref.orbital_energies[k]),
                atol=2e-8,
                rtol=0,
            ):
                raise ValueError(
                    "UCC extended Fock disagrees with the canonical DF reference"
                )
        with _HAD_CC_CACHE_LOCK:
            _HAD_CC_CACHE_STATS["fixed_feed_calls"] += 1
        feeds = unrestricted_cc_f12_fixed_source_feeds(
            space,
            data,
            fock,
            corr.whitener,
            checked,
            frozen_occupied=frozen,
            native_library=(
                provider.shell_library
                if provider.shell_library is not None
                else AUTO_NATIVE_LIBRARY
            ),
            host_reference=(not cuda),
        )
        if reuse_enabled and cc_key is not None:
            frozen_fock = {k: _freeze_f64_array(v) for k, v in fock.items()}
            frozen_feeds = {k: _freeze_f64_array(v) for k, v in feeds.items()}
            nbytes = _mapping_nbytes(frozen_fock, frozen_feeds)
            if nbytes <= _HAD_CC_CACHE_MAX_BYTES:
                with _HAD_CC_CACHE_LOCK:
                    _HAD_CC_CACHE[cc_key] = _HadCCSourceCacheEntry(
                        key=cc_key,
                        cabs_rank=proj.rank,
                        f12_fit_id=corr.fit_id,
                        frozen_occupied=frozen,
                        fock=frozen_fock,
                        feeds=frozen_feeds,
                        nbytes=nbytes,
                    )
                    _HAD_CC_CACHE.move_to_end(cc_key)
                    _evict_had_cc_cache_to_budget()
            fock = {k: _readonly_view(v) for k, v in frozen_fock.items()}
            feeds = {k: _readonly_view(v) for k, v in frozen_feeds.items()}
        fixed = FocalUCCFixedSources(ref, proj.rank, corr.fit_id, feeds, frozen)
        if not include_ordinary:
            return fixed

    from ._uccsd_df_hamiltonian import unrestricted_cc_df_hamiltonian

    assert request.basis.mp2_fit is not None
    fit_id = request.basis.mp2_fit.identity
    if fit_id == corr.fit_id:
        if data is None:
            data, _ = _orbital_sources(
                space,
                ao,
                checked,
                None,
                projected=None,
                role="pairs",
                obs_coulomb=True,
                shell_library=provider.shell_library,
                host_reference=(not cuda),
            )
        raw, whitener = data, corr.whitener
    else:
        # A separate correlation fit needs only the ordinary OBS×OBS Coulomb panel.
        mp = mp2_panel if mp2_panel.fit_id == fit_id else provider._panel(request, "mp2_fit")
        whitener = mp.whitener
        obs_ao = {
            "ao_coulomb": np.ascontiguousarray(mp.obs_coulomb(obs)),
            "metric_whitener": whitener,
            **{
                "coefficients_" + spin: ref.coefficients[k]
                for spin, k in zip(("alpha", "beta"), spin_indices, strict=True)
            },
        }
        raw, _ = _orbital_sources(
            (oa, ob, obs, obs, len(whitener)),
            obs_ao,
            checked,
            None,
            projected=None,
            role="ordinary_jk",
            shell_library=provider.shell_library,
            host_reference=(not cuda),
        )
    ladder = provider.rhf_df_ladder if request.root_id == "rhf" else None
    uhf_ladder = provider.uhf_df_ladder if request.root_id == "uhf" else None
    factors: dict[str, np.ndarray] | None = (
        {} if ladder is not None or uhf_ladder is not None else None
    )
    ordinary = unrestricted_cc_df_hamiltonian(
        (oa, ob, obs, ext, len(whitener)),
        raw,
        fock,
        whitener,
        checked,
        frozen_occupied=frozen,
        max_output_bytes=(provider.budget_mib << 20)
        - sum(v.nbytes for v in feeds.values()),
        retain_rhf_virtual_factors=factors if ladder is not None else None,
        retain_uhf_virtual_factors=factors if uhf_ladder is not None else None,
    )
    if factors is not None:
        # The solver borrows these permanently read-only buffers instead of
        # retaining a second copy of all ordinary and fixed MO sources.
        for mapping in (ordinary, feeds):
            for name, value in mapping.items():
                mapping[name] = immutable(value)
    return FocalUCCEquationSources(fixed, fit_id, ordinary, ladder, factors, uhf_ladder)
