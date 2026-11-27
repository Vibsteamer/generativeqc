"""Forward-only HF+CABS and MP2-F12/3C(FIX) on one molecular determinant."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import fsum, isfinite
from typing import TYPE_CHECKING

from ._f12_focal_tplus_sources import (
    OccupiedMP2F12,
    SpinOccupied,
    add_distinct_pair,
    add_occupied_pair_matrix,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    import numpy as np
    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_molecular import MolecularFocalProvider
    from ._f12_focal_reference import FocalReference
    from ._f12_focal_ump2 import FocalCoulombSource
    from ._f12_spin_pairs import PairEnergy
    from .experimental.f12_focal import FocalRequest


@dataclass(frozen=True)
class FocalLowOrderEnergy:
    """Single-cardinal energies in Eh, with optional CABS singles and aa, ab, bb MP2 channels."""

    reference: FocalReference
    cabs_rank: int
    cabs_spin_energies: tuple[float, float] | None
    ordinary_mp2_spin_energies: tuple[float, float, float]
    f12_increment: float
    occupied_mp2_f12: OccupiedMP2F12 | None = None

    @property
    def hf_cabs_energy(self) -> float:
        if self.cabs_spin_energies is None:
            raise ValueError("HF+CABS was not requested on this basis")
        return self.reference.energy + fsum(self.cabs_spin_energies)

    @property
    def mp2_f12_correlation_energy(self) -> float:
        return fsum((*self.ordinary_mp2_spin_energies, self.f12_increment))

    @property
    def energy(self) -> float:
        return self.hf_cabs_energy + self.mp2_f12_correlation_energy


@dataclass(frozen=True)
class FocalLowOrderForces:
    """Component energies and analytic forces in Eh/bohr for the requested terms on one root."""

    energies: FocalLowOrderEnergy
    hf_cabs_forces: np.ndarray | None
    mp2_f12_correlation_forces: np.ndarray

    @property
    def forces(self) -> np.ndarray:
        if self.hf_cabs_forces is None:
            raise ValueError("HF+CABS was not requested on this basis")
        return self.hf_cabs_forces + self.mp2_f12_correlation_forces


def _ordinary_pair_energies_blas(
    b_left: np.ndarray,
    b_right: np.ndarray,
    eo_left: np.ndarray,
    eo_right: np.ndarray,
    ev_left: np.ndarray,
    ev_right: np.ndarray,
    *,
    same: bool,
) -> np.ndarray:
    """Compute ordered DF-MP2 pair_energies[i, j] in exact FP64 BLAS with zero TensorIR."""
    import numpy as np

    q_dim, ni, vl = b_left.shape
    _, nj, vr = b_right.shape
    pair_e = np.zeros((ni, nj), dtype=np.float64)
    if not ni or not nj or not vl or not vr or (same and ni < 2):
        return pair_e
    ev_sum = ev_left[:, None] + ev_right[None, :]
    if same:
        for i in range(ni - 1):
            rem = ni - i - 1
            br_rem = b_right[:, i + 1 :, :].reshape(q_dim, rem * vr)
            g_block = (b_left[:, i, :].T @ br_rem).reshape(vl, rem, vr)
            for offset in range(rem):
                j = i + 1 + offset
                g = g_block[:, offset, :]
                diff = g - g.T
                denom = (eo_left[i] + eo_right[j]) - ev_sum
                val = 0.25 * float(np.sum((diff * diff) / denom))
                pair_e[i, j] = val
                pair_e[j, i] = val
    else:
        br_flat = b_right.reshape(q_dim, nj * vr)
        for i in range(ni):
            g_block = (b_left[:, i, :].T @ br_flat).reshape(vl, nj, vr)
            for j in range(nj):
                g = g_block[:, j, :]
                denom = (eo_left[i] + eo_right[j]) - ev_sum
                pair_e[i, j] = float(np.sum((g * g) / denom))
    return pair_e


def _ordinary_energies(
    reference: FocalReference,
    spin_indices: tuple[int, int],
    source: FocalCoulombSource,
    run: Callable,
    occupied_mp2: SpinOccupied | None = None,
    frozen_occupied: tuple[int, int] = (0, 0),
    *,
    shell_library: object = None,
    response_library: object = None,
    budget_mib: int,
    cuda: bool | None = None,
) -> tuple[float, float, float]:
    import numpy as np

    from ._f12_cphf_relax_native import resolve_cphf_relax, run_whiten_3c_native
    from ._f12_cv_dfmp2_native import bind_delta_cv_native
    from ._f12_mp2_pair_native import (
        aot_mp2f12_disabled,
        bind_ordinary_df_mp2,
        run_ordinary_df_mp2_native,
    )
    from ._f12_ram_hygiene import disclose_host_assisted_stage

    n, q = len(reference.hcore), len(source.metric)
    spins = ("alpha", "beta")
    occs = tuple(reference.occupied[k] for k in spin_indices)
    active_occs = tuple(o - f for o, f in zip(occs, frozen_occupied, strict=True))
    is_cuda = (reference.executed_backend == "cuda") if cuda is None else bool(cuda)
    same_spin = (
        occs[0] == occs[1]
        and frozen_occupied[0] == frozen_occupied[1]
        and (
            spin_indices[0] == spin_indices[1]
            or (
                np.array_equal(
                    reference.coefficients[spin_indices[0]],
                    reference.coefficients[spin_indices[1]],
                )
                and np.array_equal(
                    reference.orbital_energies[spin_indices[0]],
                    reference.orbital_energies[spin_indices[1]],
                )
            )
        )
    )

    if occupied_mp2 is None and not aot_mp2f12_disabled():
        native_ord = bind_ordinary_df_mp2(
            host_reference=not is_cuda,
            shell_library=shell_library,  # type: ignore[arg-type]
        )
        if native_ord is not None:
            return run_ordinary_df_mp2_native(
                native_ord,
                same_spin=same_spin,
                compute_response=False,
                oa=occs[0],
                fa=frozen_occupied[0],
                ob=occs[1],
                fb=frozen_occupied[1],
                budget_bytes=budget_mib << 20,
                coeff_a=reference.coefficients[spin_indices[0]],
                coeff_b=reference.coefficients[spin_indices[1]],
                eps_a=reference.orbital_energies[spin_indices[0]],
                eps_b=reference.orbital_energies[spin_indices[1]],
                ao_coul=source.three_center,
                whitener=source.whitener,
            )[0]

    use_aot_factors = not aot_mp2f12_disabled()
    aot_cv = (
        bind_delta_cv_native(  # type: ignore[arg-type]
            shell_library, host_reference=not is_cuda, budget_bytes=budget_mib << 20
        )
        if use_aot_factors and shell_library is not None
        else None
    )
    whiten_fn = (
        resolve_cphf_relax(
            shell_library,  # type: ignore[arg-type]
            "whiten_3c",
            fallback_library=response_library,  # type: ignore[arg-type]
            host_reference=not is_cuda,
        )
        if use_aot_factors and aot_cv is None
        else None
    )

    controls: dict[str, dict[str, np.ndarray]] = {}
    used_aot_b = False
    for idx_s, (name, k, occ, f) in enumerate(
        zip(spins, spin_indices, occs, frozen_occupied, strict=True)
    ):
        if not (occ - f):
            continue
        eps = reference.orbital_energies[k]
        if idx_s == 1 and same_spin and "alpha" in controls:
            controls["beta"] = {
                "B_beta": controls["alpha"]["B_alpha"],
                "eo_beta": eps[f:occ],
                "ev_beta": eps[occ:],
            }
            continue
        c_mat = reference.coefficients[k]
        co = np.ascontiguousarray(c_mat[:, f:occ], dtype=np.float64)
        cv = np.ascontiguousarray(c_mat[:, occ:], dtype=np.float64)
        if aot_cv is not None:
            b = aot_cv.run_mp2_factors_fwd(
                source.three_center, source.whitener, co, cv
            )
            used_aot_b = True
        elif whiten_fn is not None:
            raw_ia = np.einsum(
                "Auv,ui,va->Aia",
                source.three_center,
                co,
                cv,
                optimize=True,
            )
            b = run_whiten_3c_native(
                whiten_fn,
                source.whitener,
                raw_ia,
                budget_bytes=min(budget_mib << 20, 512 << 20),
            )
            used_aot_b = True
        else:
            from generativeqc_compiler.mp2.ordinary_df_factors import ordinary_df_factors
            from generativeqc_compiler.tensor import Program

            factor = ordinary_df_factors(n, occ, q, frozen=f)
            values = {
                "coefficients": c_mat,
                "ao_coulomb": source.three_center,
                "whitener": source.whitener,
            }
            b = run(Program({"B": factor.outputs["B"]}), values).outputs["B"]
        controls[name] = {
            "B_" + name: b,
            "eo_" + name: eps[f:occ],
            "ev_" + name: eps[occ:],
        }

    if occupied_mp2 is not None and (use_aot_factors and (used_aot_b or not is_cuda)):
        if is_cuda:
            disclose_host_assisted_stage(
                "ordinary_mp2_occupied_partition",
                requested_backend="cuda",
                detail="host FP64 BLAS pair-energy contraction for split-fit occupied MP2 partition",
            )
        energies: list[float] = []
        aa_matrix: np.ndarray | None = None
        for i, j in ((0, 0), (0, 1), (1, 1)):
            ni, nj = active_occs[i], active_occs[j]
            if not ni or not nj or (i == j and ni < 2):
                energies.append(0.0)
                continue
            if i == 1 and j == 1 and same_spin and aa_matrix is not None:
                value = aa_matrix
            else:
                left, right = spins[i], spins[j]
                value = _ordinary_pair_energies_blas(
                    controls[left]["B_" + left],
                    controls[right]["B_" + right],
                    controls[left]["eo_" + left],
                    controls[right]["eo_" + right],
                    controls[left]["ev_" + left],
                    controls[right]["ev_" + right],
                    same=(i == j),
                )
                if i == 0 and j == 0:
                    aa_matrix = value
            add_occupied_pair_matrix(occupied_mp2[i], occupied_mp2[j], value)
            energies.append(float(value.sum()))
        return tuple(energies)

    from generativeqc_compiler.mp2.unrestricted_df_response import df_ump2_pair_response
    from generativeqc_compiler.tensor import Program

    energies = []
    for i, j in ((0, 0), (0, 1), (1, 1)):
        ni, nj = active_occs[i], active_occs[j]
        oi, oj = occs[i], occs[j]
        if not ni or not nj or (i == j and ni < 2):
            energies.append(0.0)
            continue
        left, right = spins[i], spins[j]
        pair = df_ump2_pair_response(
            left + "_" + right, q, ni, n - oi, nj, n - oj, 1,
            occupied_partition=occupied_mp2 is not None,
        )
        output = "energy" if occupied_mp2 is None else "pair_energies"
        value = run(Program({output: pair.outputs[output]}), {**controls[left], **controls[right]}).outputs[output]
        if occupied_mp2 is not None:
            add_occupied_pair_matrix(occupied_mp2[i], occupied_mp2[j], value)
            value = value.sum()
        energies.append(float(value))
    return tuple(energies)


def _check_request(provider: MolecularFocalProvider, request: FocalRequest) -> int:
    from .elements import checked_integer

    basis = request.basis
    if (
        request.role != "mp2_f12_correlation"
        or request.root_id not in ("rhf", "uhf")
        or (request.root_id == "rhf" and request.multiplicity != 1)
        or any(
            x is None for x in (basis.cabs, basis.mp2_fit, basis.f12_fit, basis.beta)
        )
    ):
        raise ValueError(
            "low-order energies require an RHF/UHF MP2-F12 request with its three fitting roles and OptRI"
        )
    if request.slater_normalization != provider.slater_normalization:
        raise ValueError("MP2-F12 request and provider Slater normalization must match")
    cuda = request.backend == "cuda"
    if request.backend not in ("cuda", "cpu_reference") or (
        cuda
        and (
            provider.run is None
            or provider.metric_library is None
            or provider.response_library is None
        )
    ):
        raise ValueError(
            "low-order energies require their physical backend, CUDA metric library and graph runner"
        )
    return checked_integer(
        provider.budget_mib,
        "molecular focal budget in MiB",
        low=1,
        high=(2**63 - 1) >> 20,
    )


def _fock_uses_canonical_spectrum(
    fock: Mapping[str, np.ndarray],
    reference: FocalReference,
    spin_indices: tuple[int, int],
    obs: int,
) -> bool:
    import numpy as np

    return all(
        np.array_equal(fock["fock_" + spin].diagonal()[:obs], reference.orbital_energies[k])
        for spin, k in zip(("alpha", "beta"), spin_indices, strict=True)
    )


def _canonical_mp2_with_physical_f12_pairs(
    physical_pairs: Iterator[PairEnergy],
    pair_energies: Callable[..., Iterator[PairEnergy]],
    fock: Mapping[str, np.ndarray],
    reference: FocalReference,
    spin_indices: tuple[int, int],
    obs: int,
    budget_mib: int,
) -> Iterator[PairEnergy]:
    """Take ordinary occupied-pair denominators from the canonical reference."""
    import numpy as np

    if sum(fock["fock_" + spin].nbytes for spin in ("alpha", "beta")) > budget_mib << 20:
        raise MemoryError("Canonical Fock copies for occupied-pair energies exceed the numeric budget")
    canonical_fock = dict(fock)
    for spin, k in zip(("alpha", "beta"), spin_indices, strict=True):
        key = "fock_" + spin
        matrix = fock[key].copy()
        np.fill_diagonal(matrix[:obs, :obs], reference.orbital_energies[k])
        canonical_fock[key] = matrix
    canonical_pairs = pair_energies(fock=canonical_fock, include_ordinary=True)
    for geminal, ordinary in zip(physical_pairs, canonical_pairs, strict=True):
        if geminal.spins != ordinary.spins or geminal.occupied_indices != ordinary.occupied_indices:
            raise ValueError("The physical and canonical pair kernels returned different occupied pairs")
        yield geminal._replace(mp2=ordinary.mp2)


def molecular_low_order_energy(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    expected_cabs_rank: int | None,
    *,
    include_hf_cabs: bool = True,
    include_occupied_partitions: bool = False,
    reference: FocalReference | None = None,
) -> FocalLowOrderEnergy:
    import numpy as np
    from generativeqc_compiler.tensor import execute

    from ._f12_cabs_fock import physical_fock
    from ._f12_focal_frame import canonical_pair_frame, dense_pair_frame_key
    from ._f12_focal_molecular import df_reference, one_electron_sources
    from ._f12_focal_reference import partition_focal_reference
    from ._f12_focal_singles import _singles_from_fock
    from ._f12_mp2_pair_native import aot_mp2f12_disabled
    from ._f12_spin_ao import _orbital_sources
    from ._f12_spin_fock import spin_fock
    from ._f12_spin_pairs import molecular_pair_energies

    budget = _check_request(provider, request)
    cuda = request.backend == "cuda"
    runner = provider.run or partial(execute, max_bytes=budget << 20)

    def run(graph: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        result = runner(graph, inputs)
        if result.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError("low-order energy graph executed on another backend")
        return result

    if hasattr(runner, "batch"):
        from contextlib import contextmanager

        @contextmanager
        def run_batch(graph: Program):
            with runner.batch(graph) as exec_one:
                def wrapped(inputs: Mapping[str, np.ndarray]) -> Execution:
                    result = exec_one(inputs)
                    if result.backend != (
                        "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
                    ):
                        raise RuntimeError("low-order energy graph executed on another backend")
                    return result

                yield wrapped

        run.batch = run_batch  # type: ignore[attr-defined]

    if reference is not None:
        if (
            reference.surface_id != request.surface_id
            or reference.basis_id != request.basis.basis_id
            or reference.executed_backend != request.backend
            or reference.orbital_id != request.basis.orbital.identity
            or reference.jk_fit_id != request.basis.jk_fit.identity
        ):
            raise ValueError(
                "Supplied reference disagrees with the low-order energy request"
            )
        ref = reference
    else:
        ref = df_reference(
            request, forces=False, budget_mib=budget, **provider._seed_options(request)
        )
    ref = partition_focal_reference(request, ref, budget)
    one, (jk, mp2, f12) = provider._cached_pair_sources(
        request, derivatives=False, budget=budget
    )
    jk_source = jk.joint(request, explicit=False)
    f12_source = f12.joint(request, explicit=True)
    space, ao, jkfit, projector = canonical_pair_frame(
        request,
        ref,
        one,
        jk_source,
        f12_source,
        expected_cabs_rank=expected_cabs_rank,
        relative_cutoff=provider.relative_cutoff,
        library=provider.response_library,
    )
    oa, ob, obs, ext, _ = space
    distinct_jk = jk is not f12
    def sources(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            k.removeprefix("source/"): v
            for k, v in raw.items()
            if k.startswith("source/")
        }

    indices = ref._pair_spin_indices(request)
    singles = None
    cache_key = dense_pair_frame_key(
        request, ref, one, jk_source, f12_source, cabs_rank=projector.rank,
        relative_cutoff=provider.relative_cutoff, library=provider.response_library,
    )
    jk_projected: dict[str, np.ndarray] | None = None
    if distinct_jk:
        # Compute and discard JKFit Fock + CABS singles before allocating the F12 pair tensors
        # so multi-GiB JK and F12 panels never coexist in host RAM when streaming.
        shared = ("overlap_ao", "hcore_ao", "coefficients_alpha", "coefficients_beta")
        fock_ao = {**{k: ao[k] for k in shared}, **jkfit}
        fock_data, _ = _orbital_sources(
            (oa, ob, obs, ext, len(jk.whitener)),
            fock_ao,
            run,
            None,
            projected=None,
            role="fock",
        )
        fock = spin_fock((oa, ob, ext, len(jk.whitener)), sources(fock_data), run)
        if cache_key is not None:
            jk_projected = fock_data
        else:
            del fock_data
        if include_hf_cabs:
            frames = (ao["coefficients_alpha"], ao["coefficients_beta"])
            jk_ao_coul = (
                jkfit["__ao_loader__"]("ao_coulomb")
                if "__ao_loader__" in jkfit
                else jkfit["ao_coulomb"]
            )
            singles_focks, _ = physical_fock(
                one.hcore,
                jk_ao_coul,
                jk.whitener,
                frames,
                (oa, ob),
                run,
                cuda,
                provider.response_library,
            )
            del jk_ao_coul
            singles = tuple(
                _singles_from_fock(
                    singles_fock,
                    ref.orbital_energies[k],
                    ref.occupied[k],
                    cuda,
                    False,
                    provider.response_library,
                )[1]
                for singles_fock, k in zip(singles_focks, indices, strict=True)
            )
        data, _ = _orbital_sources(
            space, ao, run, None, projected=None, role="pairs", occupied_coulomb=True
        )
    else:
        data, _ = _orbital_sources(space, ao, run, None, projected=None, role="shared")
        if cache_key is not None:
            jk_projected = data
        fock = spin_fock((oa, ob, ext, len(jk.whitener)), sources(data), run)
        if include_hf_cabs:
            frames = (ao["coefficients_alpha"], ao["coefficients_beta"])
            jk_ao_coul = (
                jkfit["__ao_loader__"]("ao_coulomb")
                if "__ao_loader__" in jkfit
                else jkfit["ao_coulomb"]
            )
            singles_focks, _ = physical_fock(
                one.hcore,
                jk_ao_coul,
                jk.whitener,
                frames,
                (oa, ob),
                run,
                cuda,
                provider.response_library,
            )
            del jk_ao_coul
            singles = tuple(
                _singles_from_fock(
                    singles_fock,
                    ref.orbital_energies[k],
                    ref.occupied[k],
                    cuda,
                    False,
                    provider.response_library,
                )[1]
                for singles_fock, k in zip(singles_focks, indices, strict=True)
            )
    if cache_key is not None:
        pair_data = data
        if not distinct_jk:
            # The response treats JK and F12 as separate roles even when the
            # energy shared their AO projection. Only JK needs all Coulomb rows.
            pair_data = {k: v for k, v in data.items() if not k.startswith("source/hcore_")}
            for spin, occ in (("alpha", oa), ("beta", ob)):
                key = "source/raw_coulomb_" + spin
                pair_data[key] = data[key][:, :occ]
        provider._pair_source_cache["projected_frame"] = (
            cache_key,
            (space, ao, jkfit, projector, (pair_data, jk_projected)),
        )
    else:
        provider._pair_source_cache.pop("projected_frame", None)
    from ._f12_frozen_core import active_occupied_counts, resolve_frozen_occupied

    frozen = resolve_frozen_occupied(request, (oa, ob))
    act_a, act_b = active_occupied_counts((oa, ob), frozen)
    frozen_by_spin = {"alpha": frozen[0], "beta": frozen[1]}
    channels = {("alpha", "alpha"): [], ("alpha", "beta"): [], ("beta", "beta"): []}
    increments = []
    occupied = (
        OccupiedMP2F12((np.zeros(act_a), np.zeros(act_b)), (np.zeros(act_a), np.zeros(act_b)))
        if include_occupied_partitions
        else None
    )
    shared_pairs = mp2 is f12 and _fock_uses_canonical_spectrum(fock, ref, indices, obs)
    pair_energies = partial(
        molecular_pair_energies, space, data, sources(data), run=run,
        frozen_occupied=frozen, shell_library=provider.shell_library,
        budget_mib=budget, backend=request.backend,
    )
    pairs = pair_energies(fock=fock, include_ordinary=shared_pairs)
    if (
        cuda and occupied is not None and mp2 is f12 and not shared_pairs
        and provider.shell_library is not None and not aot_mp2f12_disabled()
    ):
        # The AOT pair ABI releases its device workspace before yielding scalar
        # pairs. The TensorIR oracle instead computes canonical MP2 after the
        # physical pass has closed its batch.
        pairs = _canonical_mp2_with_physical_f12_pairs(
            pairs, pair_energies, fock, ref, indices, obs, budget,
        )
        shared_pairs = True
    for item in pairs:
        if item.mp2 is not None:
            channels[item.spins].append(item.mp2)
        increments.append(item.f12)
        if occupied is not None:
            shifted = (
                item.occupied_indices[0] - frozen_by_spin[item.spins[0]],
                item.occupied_indices[1] - frozen_by_spin[item.spins[1]],
            )
            add_distinct_pair(occupied.f12_increment, item.spins, shifted, item.f12)
            if shared_pairs and item.mp2 is not None:
                add_distinct_pair(occupied.mp2, item.spins, shifted, item.mp2)
    ordinary = (
        tuple(fsum(values) for values in channels.values())
        if shared_pairs
        else _ordinary_energies(
            ref,
            indices,
            mp2.obs(request, obs),
            run,
            None if occupied is None else occupied.mp2,
            frozen_occupied=frozen,
            shell_library=provider.shell_library,
            response_library=provider.response_library,
            budget_mib=budget,
            cuda=cuda,
        )
    )
    result = FocalLowOrderEnergy(
        ref, projector.rank, singles, ordinary, fsum(increments), occupied
    )
    if not isfinite(result.mp2_f12_correlation_energy) or (
        include_hf_cabs and not isfinite(result.energy)
    ):
        raise ArithmeticError("nonfinite molecular low-order energy")
    return result
