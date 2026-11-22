"""Forward-only all-active HF+CABS and MP2-F12/3C(FIX) on one molecular determinant."""

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
    from collections.abc import Callable, Mapping

    import numpy as np
    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_molecular import MolecularFocalProvider
    from ._f12_focal_reference import FocalReference
    from ._f12_focal_ump2 import FocalCoulombSource
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


def _ordinary_energies(
    reference: FocalReference,
    spin_indices: tuple[int, int],
    source: FocalCoulombSource,
    run: Callable,
    occupied_mp2: SpinOccupied | None = None,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> tuple[float, float, float]:
    from generativeqc_compiler.mp2.ordinary_df_factors import ordinary_df_factors
    from generativeqc_compiler.mp2.unrestricted_df_response import df_ump2_pair_response
    from generativeqc_compiler.tensor import Program

    n, q = len(reference.hcore), len(source.metric)
    controls = {}
    spins = ("alpha", "beta")
    occs = tuple(reference.occupied[k] for k in spin_indices)
    active_occs = tuple(o - f for o, f in zip(occs, frozen_occupied, strict=True))
    for name, k, occ, f in zip(spins, spin_indices, occs, frozen_occupied, strict=True):
        if occ - f:
            factor = ordinary_df_factors(n, occ, q, frozen=f)
            values = {
                "coefficients": reference.coefficients[k],
                "ao_coulomb": source.three_center,
                "whitener": source.whitener,
            }
            b = run(Program({"B": factor.outputs["B"]}), values).outputs["B"]
            eps = reference.orbital_energies[k]
            controls[name] = {
                "B_" + name: b,
                "eo_" + name: eps[f:occ],
                "ev_" + name: eps[occ:],
            }
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


def molecular_low_order_energy(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    expected_cabs_rank: int | None,
    *,
    include_hf_cabs: bool = True,
    include_occupied_partitions: bool = False,
) -> FocalLowOrderEnergy:
    import numpy as np
    from generativeqc_compiler.tensor import execute

    from ._f12_cabs_fock import physical_fock
    from ._f12_focal_frame import canonical_pair_frame
    from ._f12_focal_molecular import df_reference, one_electron_sources
    from ._f12_focal_singles import _singles_from_fock
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

    ref = df_reference(
        request, forces=False, budget_mib=budget, **provider._seed_options(request)
    )
    one, (jk, mp2, f12) = provider._cached_pair_sources(
        request, derivatives=False, budget=budget
    )
    space, ao, jkfit, projector = canonical_pair_frame(
        request,
        ref,
        one,
        jk.joint(request, explicit=False),
        f12.joint(request, explicit=True),
        expected_cabs_rank=expected_cabs_rank,
        relative_cutoff=provider.relative_cutoff,
        library=provider.response_library,
    )
    oa, ob, obs, ext, _ = space
    distinct_jk = jk is not f12
    data, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="pairs" if distinct_jk else "shared"
    )
    fock_data = data
    if distinct_jk:
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

    def sources(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            k.removeprefix("source/"): v
            for k, v in raw.items()
            if k.startswith("source/")
        }

    fock = spin_fock((oa, ob, ext, len(jk.whitener)), sources(fock_data), run)
    indices = ref._pair_spin_indices(request)
    singles = None
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
    # Use the identical ordinary canonical source in both the occupied
    # partition and its relaxed reverse, even when the fitted panels coincide.
    shared_pairs = mp2 is f12 and occupied is None
    for item in molecular_pair_energies(
        space, data, sources(data), fock, run, frozen_occupied=frozen, include_ordinary=shared_pairs
    ):
        if item.mp2 is not None:
            channels[item.spins].append(item.mp2)
        increments.append(item.f12)
        if occupied is not None:
            shifted = (
                item.occupied_indices[0] - frozen_by_spin[item.spins[0]],
                item.occupied_indices[1] - frozen_by_spin[item.spins[1]],
            )
            add_distinct_pair(occupied.f12_increment, item.spins, shifted, item.f12)
    ordinary = (
        tuple(fsum(values) for values in channels.values())
        if shared_pairs
        else _ordinary_energies(
            ref, indices, mp2.obs(request, obs), run,
            None if occupied is None else occupied.mp2,
            frozen_occupied=frozen,
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
