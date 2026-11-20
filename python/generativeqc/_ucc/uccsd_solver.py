"""Shared unrestricted packed CC amplitudes and orbital-energy iteration."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from heapq import nlargest, nsmallest
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.resources import checked_bytes

from .solver import _DIIS, SolverOptions

if TYPE_CHECKING:
    from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
    from typing_extensions import Self


class _SpinOrbitalEnergies(Protocol):
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray

    def nocc(self, spin: str) -> int: ...


def _representatives(
    layout: UCCSDLayout, values: Mapping[str, np.ndarray], *, residuals: bool = False
) -> np.ndarray:
    """Read raw excitation representatives without projecting numerical residuals.

    Residual roundoff need not satisfy bit-exact full-array antisymmetry. Reading
    the physical representatives is distinct from accepting an invalid supplied
    amplitude: initial amplitudes always pass layout.pack's exact admission.
    """
    pieces = []
    for block in layout.blocks:
        if block.packed_layout.storage_elements == 0:
            pieces.append(np.empty(0))
            continue
        name = block.name.replace("t", "r", 1) if residuals else block.name
        full = values[name]
        if block.same_spin:
            occ_first, occ_second = np.triu_indices(block.spec.shape[0], 1)
            vir_first, vir_second = np.triu_indices(block.spec.shape[2], 1)
            full = full[
                occ_first[:, None],
                occ_second[:, None],
                vir_first[None, :],
                vir_second[None, :],
            ]
        pieces.append(full.ravel())
    return np.concatenate(pieces)


def _denominator_minimum(
    snapshot: _SpinOrbitalEnergies, layout: UCCSDLayout, options: SolverOptions
) -> float:
    """Check actual excitation extrema and proposal shifts before allocation."""
    minima = []
    for block in layout.blocks:
        if block.packed_layout.storage_elements == 0:
            continue
        occupied, virtual = [], []
        for axis in block.spec.indices:
            energies = getattr(snapshot, f"orbital_energies_{axis.space.spin}")
            count = snapshot.nocc(axis.space.spin)
            (occupied if axis.space.kind == "occupied" else virtual).append(
                energies[:count] if axis.space.kind == "occupied" else energies[count:]
            )
        try:
            with np.errstate(over="raise", invalid="raise"):
                if block.same_spin:
                    largest = np.sum(nlargest(2, occupied[0])) - np.sum(
                        nsmallest(2, virtual[0])
                    )
                    smallest = np.sum(nsmallest(2, occupied[0])) - np.sum(
                        nlargest(2, virtual[0])
                    )
                else:
                    largest = sum(np.max(energies) for energies in occupied) - sum(
                        np.min(energies) for energies in virtual
                    )
                    smallest = sum(np.min(energies) for energies in occupied) - sum(
                        np.max(energies) for energies in virtual
                    )
                shifted = smallest - len(occupied) * options.level_shift
        except FloatingPointError as error:
            raise ValueError("nonfinite UCCSD denominator extrema") from error
        if not all(np.isfinite(value) for value in (largest, smallest, shifted)):
            raise ValueError("nonfinite UCCSD denominator extrema")
        if largest >= 0 or -largest <= options.denominator_threshold:
            raise ValueError(
                f"near-zero or non-Aufbau UCCSD denominator in {block.name}"
            )
        minima.append(float(-largest))
    return min(minima) if minima else 0.0


def _denominators(
    snapshot: _SpinOrbitalEnergies, layout: UCCSDLayout, shift: float
) -> np.ndarray:
    pieces = []
    for block in layout.blocks:
        if block.packed_layout.storage_elements == 0:
            pieces.append(np.empty(0))
            continue
        if block.same_spin:
            spin = block.spec.indices[0].space.spin
            energies = getattr(snapshot, f"orbital_energies_{spin}")
            occupied = snapshot.nocc(spin)
            occ_first, occ_second = np.triu_indices(occupied, 1)
            vir_first, vir_second = np.triu_indices(len(energies) - occupied, 1)
            pairs_occ = energies[occ_first] + energies[occ_second]
            pairs_vir = energies[occupied + vir_first] + energies[occupied + vir_second]
            values = pairs_occ[:, None] - pairs_vir[None, :] - 2 * shift
        else:
            values = np.zeros(block.spec.shape)
            for position, axis in enumerate(block.spec.indices):
                energies = getattr(snapshot, f"orbital_energies_{axis.space.spin}")
                occupied = snapshot.nocc(axis.space.spin)
                chosen = (
                    energies[:occupied]
                    if axis.space.kind == "occupied"
                    else energies[occupied:]
                )
                shape = [1] * len(block.spec.shape)
                shape[position] = len(chosen)
                values += (1 if axis.space.kind == "occupied" else -1) * chosen.reshape(
                    shape
                )
            values -= (1 if block.name.startswith("t1") else 2) * shift
        pieces.append(values.ravel())
    return immutable(np.concatenate(pieces))


def _iteration_bytes(
    layout: UCCSDLayout, options: SolverOptions, feed_bytes: int
) -> int:
    full_bytes = sum(block.full_layout.storage_bytes(8) for block in layout.blocks)
    packed_bytes = layout.packed_size * 8
    return checked_bytes(
        feed_bytes
        + 4 * full_bytes
        + (12 + 4 * options.diis_size) * packed_bytes
        + 32 * (options.diis_size + 1) ** 2,
        "UCCSD iteration storage",
    )


@dataclass(frozen=True)
class UCCSDIteration:
    """One evaluated physical state; changes are never inferred from proposals."""

    iteration: int
    correlation_energy: float
    energy_change: float | None
    residual_max: float
    block_residual_max: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class UCCSDResult:
    """Owned last finite excitation state with explicit convergence provenance."""

    status: str
    reason: str
    correlation_energy: float | None
    total_energy: float | None
    amplitudes: Mapping[str, np.ndarray]
    packed_amplitudes: np.ndarray
    history: tuple[UCCSDIteration, ...]
    provenance: Mapping[str, Any]

    @property
    def converged(self) -> bool:
        return self.status == "converged"


class _PackedUCCSD:
    """Shared packed amplitude iteration for admitted spin-orbital equations."""

    def _execution_configuration(self) -> tuple:
        return (
            self.executor.backend,
            self.executor.max_bytes,
            getattr(self.executor, "max_work", None),
            getattr(self.executor, "max_nodes", None),
            getattr(self.executor, "device", None),
            getattr(self.executor.compiler, "target", None),
        )

    def evaluate(self, packed: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate physical residuals on the selected backend with owned feeds."""
        if self._closed:
            raise RuntimeError("prepared UCCSD is closed")
        if self._execution_configuration() != self._executor_configuration:
            raise RuntimeError(
                "UCCSD executor configuration changed; prepare a fresh owner"
            )
        amplitudes = self.layout.unpack(packed)
        outputs = self.executor.execute(self.program, {**self.feeds, **amplitudes})
        expected = {
            "correlation_energy": (),
            **{
                block.name.replace("t", "r", 1): block.spec.shape
                for block in self.layout.blocks
            },
        }
        if outputs.keys() != expected.keys():
            raise ValueError("UCCSD executor returned incomplete physical outputs")
        validated = {}
        for name, shape in expected.items():
            value = np.asarray(outputs[name])
            if value.dtype != np.float64 or value.shape != shape:
                raise ValueError("UCCSD executor violated the FP64 shape contract")
            if not np.isfinite(value).all():
                raise FloatingPointError(
                    "UCCSD executor returned nonfinite physical outputs"
                )
            validated[name] = value
        return validated

    def solve(
        self, initial: Mapping[str, np.ndarray] | np.ndarray | None = None
    ) -> UCCSDResult:
        """Solve with packed excitation DIIS and fresh physical convergence gates."""
        if self._closed:
            raise RuntimeError("prepared UCCSD is closed")
        if initial is None:
            current = np.zeros(self.layout.packed_size)
        elif isinstance(initial, Mapping):
            current = self.layout.pack(initial)
        else:
            current = self.layout.pack(self.layout.unpack(initial))
        last_finite = current.copy()
        diis = _DIIS(self.options.diis_size)
        history = []
        energy = previous_energy = last_finite_energy = None
        status, reason = "iteration_limit", "maximum UCCSD iterations reached"
        evaluations = 0
        started = time.perf_counter()
        try:
            for iteration in range(self.options.max_iterations + 1):
                outputs = self.evaluate(current)
                evaluations += 1
                current_energy = float(outputs["correlation_energy"])
                maxima = tuple(
                    (
                        block.name.replace("t", "r", 1),
                        float(
                            np.max(
                                np.abs(outputs[block.name.replace("t", "r", 1)]),
                                initial=0,
                            )
                        ),
                    )
                    for block in self.layout.blocks
                )
                residual = max(value for _, value in maxima)
                delta = (
                    None
                    if previous_energy is None
                    else abs(current_energy - previous_energy)
                )
                history.append(
                    UCCSDIteration(iteration, current_energy, delta, residual, maxima)
                )
                last_finite, energy = current.copy(), current_energy
                last_finite_energy = current_energy
                if residual <= self.options.residual_tolerance and (
                    not self.layout.packed_size
                    or (delta is not None and delta <= self.options.energy_tolerance)
                ):
                    recheck = self.evaluate(current)
                    evaluations += 1
                    last_finite_energy = float(recheck["correlation_energy"])
                    fresh_residual = max(
                        float(np.max(np.abs(value), initial=0))
                        for name, value in recheck.items()
                        if name != "correlation_energy"
                    )
                    if (
                        fresh_residual <= self.options.residual_tolerance
                        and abs(float(recheck["correlation_energy"]) - energy)
                        <= self.options.energy_tolerance
                    ):
                        status, reason = (
                            "converged",
                            "energy change and all five freshly evaluated physical residuals passed",
                        )
                        break
                if iteration == self.options.max_iterations:
                    break
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    trial = (
                        current
                        + (1 - self.options.damping)
                        * _representatives(self.layout, outputs, residuals=True)
                        / self.denominators
                    )
                trial_outputs = self.evaluate(trial)
                evaluations += 1
                last_finite, last_finite_energy = (
                    trial.copy(),
                    float(trial_outputs["correlation_energy"]),
                )
                error = _representatives(self.layout, trial_outputs, residuals=True)
                current = diis.update(trial, error)
                previous_energy = energy
        except (FloatingPointError, OverflowError) as error:
            status, reason = "nonfinite", str(error)
        except ValueError as error:
            if not any(word in str(error).lower() for word in ("finite", "overflow")):
                raise
            status, reason = "nonfinite", str(error)
        packed = immutable(last_finite)
        amplitudes = MappingProxyType(
            {
                name: immutable(value)
                for name, value in self.layout.unpack(packed).items()
            }
        )
        provenance = MappingProxyType(
            {
                "reference_id": self.reference.identity,
                "hamiltonian_id": self.reference.hamiltonian_id,
                "integral_source_id": self.integral_source_id,
                "integral_hash": self.integral_hash,
                "equation_hash": self.program.logical_hash,
                "tensor_backend": self.executor.backend,
                "integral_backend": self.integral_backend,
                "source_backend": self.source_backend,
                "fock_backend": "host-fp64",
                "iteration_backend": "host-packed-excitation-diis",
                "provider_statistics": self.provider_statistics,
                "minimum_absolute_denominator": self.minimum_absolute_denominator,
                "solver_numeric_bound_bytes": self.logical_required_bytes,
                "provider_peak_bytes": self.provider_peak_bytes,
                "integral_request_count": self.integral_request_count,
                "physical_evaluations": evaluations,
                "diis_restarts": diis.restarts,
                "options": MappingProxyType(asdict(self.options)),
                "preparation_seconds": self.preparation_seconds,
                "solve_seconds": time.perf_counter() - started,
                "complete_endpoint_seconds": self.preparation_seconds
                + time.perf_counter()
                - started,
            }
        )
        return UCCSDResult(
            status,
            reason,
            last_finite_energy,
            None
            if last_finite_energy is None
            else self.reference.reference_energy + last_finite_energy,
            amplitudes,
            packed,
            tuple(history),
            provenance,
        )

    def close(self) -> None:
        """Release owned feeds; never close borrowed providers or executors."""
        self.feeds = MappingProxyType({})
        self.denominators = np.empty(0)
        self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("prepared UCCSD is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
