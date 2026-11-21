"""Private fixed-source UCCSD-F12b solver for matched canonical spin-MO feeds."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.cc.uccsd_equations import (
    build_f12_df_remainder_program,
    build_f12_program,
)
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDLayout
from generativeqc_compiler.cc.uccsd_response import (
    _response_programs,
    build_f12_response_programs,
    build_f12_rhf_repeated_lambda_programs,
    fixed_source_lagrangian_direction,
    molecular_lagrangian_direction,
)
from generativeqc_compiler.cc.uccsd_rhf_virtual_pairs import (
    build_uccsd_rhf_virtual_pair_programs,
)
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.resources import MAX_BYTES, checked_bytes
from generativeqc_compiler.mp2.f12_density_fitting import fitted_integral_block
from generativeqc_compiler.tensor.cpu import emit_cpu
from generativeqc_compiler.tensor.cuda_plan import plan_cuda

from generativeqc._uccsd_rhf_virtual_pairs import (
    RHFDFLadderOptions,
    prepare_rhf_f12_lambda_action,
    prepare_rhf_virtual_pair_operators,
    rhf_virtual_pair_storage,
)
from generativeqc.response_solver import DiagonalPreconditioner, solve

from .implicit import ImplicitSolveError, ResponseGMRES
from .native_tensor_cpu import NativeCCTensorExecutor
from .native_tensor_cuda import CudaCCTensorExecutor
from .solver import SolverOptions
from .uccsd_lambda import (
    UCCSDLambdaOptions,
    UCCSDLambdaResult,
    _Counter,
    _PackedUCCSDLambda,
    _Resource,
)
from .uccsd_solver import (
    UCCSDIteration,
    UCCSDResult,
    _denominator_minimum,
    _denominators,
    _iteration_bytes,
    _PackedUCCSD,
)

if TYPE_CHECKING:
    from typing import Any

    from generativeqc_compiler.cc.uccsd_response import UCCSDResponsePrograms
    from generativeqc_compiler.tensor import Program

    from .uccsd_lambda import UCCSDParameterCotangent


def _digest(values: np.ndarray) -> str:
    return sha256(values.astype("<f8", copy=False).tobytes()).hexdigest()


def _source_key(name: str) -> bool:
    if name == "fixed_geminal_energy":
        return True
    stem, _, spins = name.rpartition("_")
    return stem in {"w_vo", "w_vv", "c_f12"} and spins in {
        "aaaa",
        "abab",
        "abba",
        "baab",
        "baba",
        "bbbb",
    }


def _has_immutable_buffer(value: np.ndarray) -> bool:
    if value.flags.writeable or not value.flags.c_contiguous:
        return False
    backing = value.base
    while isinstance(backing, np.ndarray):
        backing = backing.base
    return isinstance(backing, bytes)


def _checked_array(
    name: str, value: np.ndarray, shape: tuple[int, ...], *, share: bool = False
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.float64
        or value.shape != shape
    ):
        raise ValueError(f"{name} must be a native FP64 ndarray of shape {shape}")
    if share and _has_immutable_buffer(value):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        return value
    return immutable(value, shape=shape)


def _action_resource(
    executor: NativeCCTensorExecutor | CudaCCTensorExecutor, program: Program
) -> _Resource:
    cuda = isinstance(executor, CudaCCTensorExecutor)
    _, resources = emit_cpu(
        program,
        max_bytes=MAX_BYTES if cuda else executor.max_bytes,
        max_work=MAX_BYTES if cuda else executor.max_work,
        max_nodes=getattr(executor, "max_nodes", 16384),
    )
    native = (
        plan_cuda(
            program, executor.compiler.target, max_bytes=executor.max_bytes
        ).peak_bytes
        if cuda
        else resources["required_bytes"]
    )
    return _Resource(native, resources["scalar_work"])


def _run(
    executor: NativeCCTensorExecutor | CudaCCTensorExecutor,
    program: Program,
    feeds: Mapping[str, np.ndarray],
) -> SimpleNamespace:
    return SimpleNamespace(outputs=executor.execute(program, feeds))


@dataclass(frozen=True)
class _FixedSourceReference:
    layout: UCCSDLayout
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    reference_energy: float
    identity: str
    hamiltonian_id: str

    def nocc(self, spin: str) -> int:
        return {"alpha": self.layout.oa, "beta": self.layout.ob}[spin]

    @property
    def numeric_bytes(self) -> int:
        return self.orbital_energies_alpha.nbytes + self.orbital_energies_beta.nbytes


class _PreparedFixedSourceUCCSD(_PackedUCCSD):
    """Solve the fixed-source equations with the selected native tensor executor.

    The caller qualifies and supplies the ordinary and F12 feeds in the same
    canonical spin frame. This owner binds detached inputs and proposal
    denominators to the equation identity and uses shared packed DIIS. It does
    not construct a molecular reference, fit, or orbital response.
    """

    def __init__(
        self,
        layout: UCCSDLayout,
        ordinary_feeds: Mapping[str, np.ndarray],
        fixed_source_feeds: Mapping[str, np.ndarray],
        *,
        orbital_energies_alpha: np.ndarray,
        orbital_energies_beta: np.ndarray,
        reference_id: str,
        hamiltonian_id: str,
        reference_energy: float,
        options: SolverOptions | None = None,
        executor: NativeCCTensorExecutor | CudaCCTensorExecutor | None = None,
        rhf_df_ladder: RHFDFLadderOptions | None = None,
        virtual_factors: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        started = time.perf_counter()
        if not isinstance(layout, UCCSDLayout):
            raise TypeError("fixed-source UCCSD-F12b requires UCCSDLayout")
        if not all(
            isinstance(key, str) and key for key in (reference_id, hamiltonian_id)
        ):
            raise ValueError(
                "fixed-source UCCSD-F12b requires reference and Hamiltonian identities"
            )
        if not math.isfinite(reference_energy):
            raise ValueError(
                "fixed-source UCCSD-F12b requires a finite reference energy"
            )
        options = SolverOptions() if options is None else options
        if not isinstance(options, SolverOptions):
            raise TypeError("fixed-source UCCSD-F12b requires SolverOptions")
        if executor is None:
            executor = NativeCCTensorExecutor(max_bytes=options.max_bytes)
        if not isinstance(executor, (NativeCCTensorExecutor, CudaCCTensorExecutor)):
            raise TypeError(
                "fixed-source UCCSD-F12b requires a native CPU or CUDA tensor executor"
            )
        self.layout, self.options, self.executor = layout, options, executor
        self.rhf_df_ladder = rhf_df_ladder
        self.virtual_factors = None
        self.spatial_operators = MappingProxyType({})
        self.bare_programs = None
        self.bounded_primal_work = 0
        if rhf_df_ladder is None:
            if virtual_factors is not None:
                raise ValueError(
                    "Unexpected virtual factors for dense fixed-source UCCSD"
                )
            self.program = build_f12_program(layout)
        else:
            if (
                not isinstance(rhf_df_ladder, RHFDFLadderOptions)
                or virtual_factors is None
            ):
                raise ValueError(
                    "Bounded fixed-source UCCSD requires RHF factors and limits"
                )
            self.program = build_f12_df_remainder_program(layout)
        inputs = {
            node.attrs["name"]: node.spec.shape
            for node in self.program.live_nodes
            if node.op == "input" and node.attrs["name"] not in BLOCK_NAMES
        }
        source_keys = {name for name in inputs if _source_key(name)}
        ordinary_keys = inputs.keys() - source_keys
        if ordinary_feeds.keys() != ordinary_keys:
            raise ValueError("fixed-source UCCSD-F12b ordinary feed keys differ")
        if not source_keys <= fixed_source_feeds.keys() or any(
            not isinstance(name, str) or not _source_key(name)
            for name in fixed_source_feeds
        ):
            raise ValueError("fixed-source UCCSD-F12b fixed feed keys differ")
        feed_bytes = sum(8 * math.prod(shape) for shape in inputs.values())
        reference_bytes = 8 * (layout.oa + layout.va + layout.ob + layout.vb)
        bounded_resident = 0
        pair_preparation_bytes = 0
        if rhf_df_ladder is not None:
            assert virtual_factors is not None
            if set(virtual_factors) != {"fit_left", "raw_right"}:
                raise ValueError(
                    "Bounded CC requires exactly one ordinary RHF fitting factor pair"
                )
            fit, raw = (virtual_factors[name] for name in ("fit_left", "raw_right"))
            if (
                fit.ndim != 3
                or fit.shape[1:] != (layout.va, layout.va)
                or raw.shape != fit.shape
            ):
                raise ValueError(
                    "Bounded CC fitting factors do not match the RHF virtual space"
                )
            if any(value.flags.writeable for value in (fit, raw)):
                raise ValueError("Bounded CC requires immutable shared MP2Fit factors")
            if not np.array_equal(orbital_energies_alpha, orbital_energies_beta):
                raise ValueError(
                    "Bounded CC requires the same RHF orbital energies for both spins"
                )
            self.bare_programs = build_uccsd_rhf_virtual_pair_programs(layout)
            auxiliary = fit.shape[0]
            preparation, operator_bytes = rhf_virtual_pair_storage(
                layout.va, auxiliary, rhf_df_ladder.row_tile
            )
            n_pairs = layout.va * (layout.va + 1) // 2
            width = min(n_pairs, rhf_df_ladder.row_tile)
            n_full, final = divmod(n_pairs, width)
            fit_graph = _action_resource(
                executor, fitted_integral_block(auxiliary, width, n_pairs, coulomb=True)
            )
            fit_work, fit_peak = n_full * fit_graph.scalar_work, fit_graph.native_bytes
            if final:
                tail = _action_resource(
                    executor,
                    fitted_integral_block(auxiliary, final, n_pairs, coulomb=True),
                )
                fit_work += tail.scalar_work
                fit_peak = max(fit_peak, tail.native_bytes)
            remainder, bare = (
                _action_resource(executor, graph)
                for graph in (self.program, self.bare_programs.primal)
            )
            self.primal_action_work = remainder.scalar_work + bare.scalar_work
            self.primal_preparation_work = fit_work
            self.bounded_primal_work = checked_bytes(
                fit_work + (3 * options.max_iterations + 2) * self.primal_action_work,
                "bounded RHF amplitude work",
            )
            if self.bounded_primal_work > rhf_df_ladder.max_primal_work:
                raise MemoryError("Bounded RHF amplitude semantic-work budget exceeded")
            # The molecular caller retains its inputs. Count any copies
            # required when they lack permanently read-only buffers.
            duplicates = sum(
                value.nbytes
                for value in (
                    *ordinary_feeds.values(),
                    *fixed_source_feeds.values(),
                    fit,
                    raw,
                )
                if not _has_immutable_buffer(value)
            )
            shared = (
                duplicates
                + fit.nbytes
                + raw.nbytes
                + sum(
                    value.nbytes
                    for name, value in fixed_source_feeds.items()
                    if name not in source_keys
                )
            )
            bounded_resident = shared + operator_bytes
            pair_preparation_bytes = feed_bytes + shared + preparation + fit_peak
            self.bounded_extra_bytes = bounded_resident
        self.host_iteration_bytes = _iteration_bytes(
            layout, options, feed_bytes + reference_bytes + bounded_resident
        )
        if isinstance(executor, CudaCCTensorExecutor):
            self.execution_peak_bytes = plan_cuda(
                self.program, executor.compiler.target, max_bytes=executor.max_bytes
            ).peak_bytes
        else:
            _, resources = emit_cpu(
                self.program,
                max_bytes=executor.max_bytes,
                max_work=executor.max_work,
                max_nodes=executor.max_nodes,
            )
            self.execution_peak_bytes = resources["required_bytes"]
        if self.bare_programs is not None:
            full = sum(block.full_layout.storage_bytes(8) for block in layout.blocks)
            self.execution_peak_bytes = max(
                self.execution_peak_bytes, bare.native_bytes + full
            )
        preparation_bytes = (
            feed_bytes
            + reference_bytes
            + 8 * layout.packed_size
            + 8 * max(map(math.prod, inputs.values()))
        )
        self.logical_required_bytes = checked_bytes(
            max(
                preparation_bytes,
                self.host_iteration_bytes + self.execution_peak_bytes,
                pair_preparation_bytes,
            ),
            "fixed-source UCCSD-F12b simultaneous numeric storage",
        )
        if self.logical_required_bytes > options.max_bytes:
            raise MemoryError("fixed-source UCCSD-F12b solver numeric budget exceeded")
        if rhf_df_ladder is not None:
            fit = _checked_array("fit_left", fit, fit.shape, share=True)
            raw = _checked_array("raw_right", raw, raw.shape, share=True)
            self.virtual_factors = MappingProxyType({"fit_left": fit, "raw_right": raw})
            self.spatial_operators = MappingProxyType(
                prepare_rhf_virtual_pair_operators(
                    fit,
                    raw,
                    lambda graph, feeds: _run(executor, graph, feeds),
                    row_tile=rhf_df_ladder.row_tile,
                    max_bytes=preparation,
                )
            )
        alpha = _checked_array(
            "orbital_energies_alpha",
            orbital_energies_alpha,
            (layout.oa + layout.va,),
        )
        beta = _checked_array(
            "orbital_energies_beta",
            orbital_energies_beta,
            (layout.ob + layout.vb,),
        )
        reference_hash = canonical_hash(
            {
                "id": reference_id,
                "alpha": _digest(alpha),
                "beta": _digest(beta),
                "energy": float(reference_energy),
            }
        )
        self.reference = _FixedSourceReference(
            layout, alpha, beta, float(reference_energy), reference_hash, hamiltonian_id
        )
        self.minimum_absolute_denominator = _denominator_minimum(
            self.reference, layout, options
        )
        self.denominators = _denominators(self.reference, layout, options.level_shift)
        feeds = {
            name: _checked_array(
                name,
                (fixed_source_feeds if name in source_keys else ordinary_feeds)[name],
                shape,
                share=rhf_df_ladder is not None,
            )
            for name, shape in inputs.items()
        }
        self.feeds = MappingProxyType(feeds)
        self.integral_hash = canonical_hash(
            {
                name: _digest(value)
                for name, value in {
                    **feeds,
                    **({} if self.virtual_factors is None else self.virtual_factors),
                }.items()
            }
        )
        self.integral_source_id = reference_id
        self.integral_backend = (
            "supplied-canonical-spin-MO-feeds"
            if self.bare_programs is None
            else "bounded-rhf-spatial-virtual-pairs"
        )
        self.source_backend = "supplied-fixed-geminal-sources"
        self.integral_request_count = self.provider_peak_bytes = 0
        self.provider_statistics = MappingProxyType({})
        self._executor_configuration = self._execution_configuration()
        self._closed = False
        self.preparation_seconds = time.perf_counter() - started

    @property
    def equation_hash(self) -> str:
        if self.bare_programs is None:
            return super().equation_hash
        return canonical_hash(
            {
                "remainder": self.program.logical_hash,
                "bare": self.bare_programs.primal.logical_hash,
            }
        )

    def _evaluate_physical(self, amplitudes: Mapping[str, np.ndarray]) -> dict:
        remainder = super()._evaluate_physical(amplitudes)
        if self.bare_programs is not None:
            bare = self.executor.execute(
                self.bare_programs.primal, {**self.spatial_operators, **amplitudes}
            )
            if bare.keys() != self.bare_programs.primal.outputs.keys():
                raise ValueError("Bounded RHF ladder returned incomplete doubles")
            for name, value in bare.items():
                remainder[name] = remainder[name] + value
        return remainder

    def _work_provenance(self, evaluations: int) -> dict[str, int]:
        if self.bare_programs is None:
            return {}
        return {
            "planned_semantic_scalar_work": self.bounded_primal_work,
            "executed_semantic_scalar_work": (
                self.primal_preparation_work + evaluations * self.primal_action_work
            ),
        }

    def close(self) -> None:
        self.spatial_operators = MappingProxyType({})
        self.virtual_factors = None
        super().close()

    def release_repeated_operators(self) -> None:
        """Release the prepared spatial matrices after amplitude and dual qualification."""
        self.spatial_operators = MappingProxyType({})

    def _replay_fixed_source_amplitudes(
        self, packed_amplitudes: np.ndarray, correlation_energy: float
    ) -> UCCSDResult:
        """Bind the exact supplied amplitudes after native energy/residual replay."""
        if isinstance(correlation_energy, bool) or not math.isfinite(
            correlation_energy
        ):
            raise ValueError("archived UCCSD correlation energy must be finite")
        packed = _checked_array(
            "packed UCCSD amplitudes", packed_amplitudes, (self.layout.packed_size,)
        )
        outputs = self.evaluate(packed)
        energy = float(outputs["correlation_energy"])
        drift = abs(energy - correlation_energy)
        maxima = tuple(
            (name, float(np.max(np.abs(residual), initial=0)))
            for name, residual in outputs.items()
            if name != "correlation_energy"
        )
        residual = max(value for _, value in maxima)
        if (
            drift > self.options.energy_tolerance
            or residual > self.options.residual_tolerance
        ):
            raise ImplicitSolveError(
                "archived UCCSD amplitudes failed fresh physical replay"
            )
        return UCCSDResult(
            "converged",
            "archived energy and all freshly evaluated physical residuals passed",
            energy,
            self.reference.reference_energy + energy,
            MappingProxyType(
                {
                    name: immutable(value)
                    for name, value in self.layout.unpack(packed).items()
                }
            ),
            packed,
            (UCCSDIteration(0, energy, float(drift), residual, maxima),),
            MappingProxyType(
                {
                    "reference_id": self.reference.identity,
                    "hamiltonian_id": self.reference.hamiltonian_id,
                    "integral_hash": self.integral_hash,
                    "equation_hash": self.equation_hash,
                    "tensor_backend": self.executor.backend,
                    "iteration_backend": "archived-state-physical-replay",
                    "physical_evaluations": 1,
                    **self._work_provenance(1),
                    "solver_numeric_bound_bytes": self.logical_required_bytes,
                }
            ),
        )


def _gap_workspace_bytes(dimension: int) -> int:
    # Reserve the frozen gaps, the diagonal/inverse snapshots, and two
    # temporary vectors for construction and application.
    return checked_bytes(5 * dimension * 8, "packed orbital-gap workspace")


@dataclass(frozen=True, kw_only=True)
class _OrbitalGapGMRES(ResponseGMRES):
    """Right-precondition packed Lambda with the unshifted Fock gaps."""

    gaps: np.ndarray = field(repr=False, compare=False)
    _gap_hash: str = field(init=False, repr=False)
    backend: str = field(default="host-orbital-gap-response-gmres", init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.gaps) is not np.ndarray
            or self.gaps.dtype != np.float64
            or self.gaps.shape != (self.dimension,)
            or not np.isfinite(self.gaps).all()
            or np.any(self.gaps <= 0)
        ):
            raise ValueError(
                "packed orbital gaps must be a finite positive FP64 vector"
            )
        gaps = DiagonalPreconditioner(self.gaps).diagonal
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "_gap_hash", _digest(gaps))

    @property
    def workspace_bytes(self) -> int:
        return checked_bytes(
            super().workspace_bytes + _gap_workspace_bytes(self.dimension),
            "packed orbital-gap GMRES workspace",
        )

    @property
    def identity(self) -> str:
        return canonical_hash(
            {"gmres": super().identity, "unshifted_packed_gap_sha256": self._gap_hash}
        )

    def solve(self, operator: Any, rhs: Any) -> Any:
        if operator.dimension != self.dimension:
            raise ValueError("response callback dimension mismatch")
        gap_bytes = _gap_workspace_bytes(self.dimension)
        return solve(
            operator,
            rhs,
            options=replace(
                self.options,
                max_workspace_bytes=self.options.max_workspace_bytes - gap_bytes,
            ),
            preconditioner=DiagonalPreconditioner(self.gaps),
        )


class _PreparedFixedSourceUCCSDLambda(_PackedUCCSDLambda):
    """Qualify the native fixed-source Lagrangian using packed GMRES actions."""

    _parent_type = _PreparedFixedSourceUCCSD

    def _build_response_programs(
        self, layout: UCCSDLayout, *, max_elements: int
    ) -> UCCSDResponsePrograms:
        if self.prepared.bare_programs is None:
            return build_f12_response_programs(layout, max_elements=max_elements)
        self._repeated = build_f12_rhf_repeated_lambda_programs(
            layout, max_retained_bytes=self.options.max_bytes, max_elements=max_elements
        )
        return _response_programs(self.prepared.program, max_elements)

    def _extra_response_bytes(self) -> int:
        if self._repeated is None:
            return 0
        full = sum(block.full_layout.storage_bytes(8) for block in self.layout.blocks)
        return (
            self.prepared.bounded_extra_bytes
            + self._repeated.remainder.retained_bytes
            + 2 * full
        )

    def _response_setup_work(self) -> int:
        return 0 if self._repeated is None else self._repeated_preparation.scalar_work

    def _resource(self, program: Any) -> _Resource:
        if self._repeated is None:
            return super()._resource(program)
        identity = program.logical_hash
        if identity == self.prepared.program.logical_hash:
            return _Resource(
                self.prepared.execution_peak_bytes, self.prepared.primal_action_work
            )
        if identity not in (
            self.programs.residual_vjp.derivative_hash,
            self.programs.stationarity_vjp.derivative_hash,
        ):
            return super()._resource(program)
        if self._repeated_preparation is None:
            self._repeated_preparation = _action_resource(
                self.prepared.executor, self._repeated.remainder.prepare
            )
            replay = _action_resource(
                self.prepared.executor, self._repeated.remainder.replay
            )
            bare = _action_resource(self.prepared.executor, self._repeated.bare)
            self._repeated_resource = _Resource(
                max(
                    replay.native_bytes,
                    bare.native_bytes,
                    self._repeated_preparation.native_bytes,
                ),
                replay.scalar_work + bare.scalar_work,
            )
        if identity == self.programs.residual_vjp.derivative_hash:
            return self._repeated_resource
        energy = self._resources[self.programs.energy_vjp.derivative_hash]
        return _Resource(
            max(energy.native_bytes, self._repeated_resource.native_bytes),
            energy.scalar_work + self._repeated_resource.scalar_work,
        )

    def _execute(
        self, program: Any, extra: Mapping[str, np.ndarray], counter: _Counter
    ) -> dict:
        identity = program.logical_hash
        if self._repeated is None or identity not in (
            self.programs.residual_vjp.derivative_hash,
            self.programs.stationarity_vjp.derivative_hash,
        ):
            return super()._execute(program, extra, counter)
        if self._repeated_action is None:
            counter.scalar_work += self._response_setup_work()
            if counter.scalar_work > self.options.max_total_work:
                raise MemoryError(
                    "RHF Lambda preparation exceeds the semantic-work budget"
                )
            counter.native_calls += 1
            self._repeated_action = prepare_rhf_f12_lambda_action(
                self.layout,
                self.packed_amplitudes,
                self.prepared.feeds,
                self.prepared.spatial_operators,
                lambda graph, feeds: _run(self.prepared.executor, graph, feeds),
                max_retained_bytes=self.options.max_bytes,
                programs=self._repeated,
            )
        self._charge(program, counter)
        counter.native_calls += 1
        bars = self._repeated_action.bars(extra)
        if identity == self.programs.stationarity_vjp.derivative_hash:
            energy = self.prepared.executor.execute(
                self.programs.energy_vjp.program, {**self.feeds, **extra}
            )
            counter.native_calls += 1
            for name, value in energy.items():
                bars[name] += value
        if any(not np.isfinite(value).all() for value in bars.values()):
            raise FloatingPointError(
                "RHF Lambda action returned nonfinite physical bars"
            )
        self._check_open()
        return bars

    def _qualify_parent(self, counter: _Counter) -> tuple[float, float]:
        result = super()._qualify_parent(counter)
        if self._repeated is not None:
            counter.native_calls += 1
        return result

    def close(self) -> None:
        self._repeated_action = None
        super().close()

    def release_repeated_operators(self) -> None:
        """Finish the repeat-action epoch before grouped source differentiation."""
        self._check_open()
        self._repeated_action = None
        self.prepared.release_repeated_operators()

    def __init__(
        self,
        prepared: _PreparedFixedSourceUCCSD,
        state: UCCSDResult,
        *,
        options: UCCSDLambdaOptions | None = None,
        solver: Any = None,
        orbital_gap_preconditioner: bool = False,
    ) -> None:
        self._repeated = self._repeated_preparation = self._repeated_action = None
        if type(orbital_gap_preconditioner) is not bool:
            raise TypeError("orbital_gap_preconditioner must be a bool")
        if orbital_gap_preconditioner:
            if solver is not None:
                raise ValueError(
                    "orbital-gap preconditioning requires the built-in GMRES"
                )
            if not isinstance(prepared, _PreparedFixedSourceUCCSD) or prepared._closed:
                raise TypeError(
                    "UCCSD Lambda requires a live _PreparedFixedSourceUCCSD"
                )
            options = UCCSDLambdaOptions() if options is None else options
            if not isinstance(options, UCCSDLambdaOptions):
                raise TypeError("UCCSD Lambda requires UCCSDLambdaOptions")
            dimension = prepared.layout.packed_size
            if dimension:
                workspace_bytes = checked_bytes(
                    ResponseGMRES(dimension, options.gmres).workspace_bytes
                    + _gap_workspace_bytes(dimension),
                    "packed orbital-gap GMRES workspace",
                )
                if workspace_bytes > min(
                    options.max_bytes, options.gmres.max_workspace_bytes
                ):
                    raise MemoryError("packed orbital-gap workspace budget exceeded")
                gaps = -_denominators(prepared.reference, prepared.layout, 0.0)
                solver = _OrbitalGapGMRES(
                    dimension=dimension, options=options.gmres, gaps=gaps
                )
        super().__init__(prepared, state, options=options, solver=solver)

    def fixed_source_directional_derivative(
        self, result: UCCSDLambdaResult, direction: Mapping[str, np.ndarray]
    ) -> float:
        """Requalify the center and contract a complete W/C/V+B-X direction.

        For nuclear input tangents the scalar is the F12-source contribution
        to the energy gradient. The ordinary Hamiltonian and reference terms
        have their own derivatives.
        """
        return self._correlation_direction(result, direction, ordinary=None)

    def molecular_directional_derivative(
        self,
        result: UCCSDLambdaResult,
        *,
        ordinary_direction: Mapping[str, np.ndarray],
        fixed_source_direction: Mapping[str, np.ndarray],
        reference_derivative: float,
    ) -> float:
        """Derivative of the reference plus stationary CCSD-F12b correlation.

        Both mappings must describe the same coordinate of the same moving
        spin-MO frame. The ordinary mapping carries dF and antisymmetrized dERI.
        Supply the full reference derivative, including nuclear repulsion.
        A separately defined CABS singles correction contributes independently.
        """
        if not isinstance(ordinary_direction, Mapping):
            raise TypeError("Expected a complete ordinary Hamiltonian direction")
        if isinstance(reference_derivative, bool) or not math.isfinite(
            reference_derivative
        ):
            raise ValueError("Expected a finite UHF reference derivative")
        derivative = self._correlation_direction(
            result, fixed_source_direction, ordinary=ordinary_direction
        )
        return math.fsum((reference_derivative, derivative))

    def _correlation_direction(
        self,
        result: UCCSDLambdaResult,
        fixed: Mapping[str, np.ndarray],
        *,
        ordinary: Mapping[str, np.ndarray] | None,
    ) -> float:
        self._check_open()
        if self._repeated is not None:
            raise ValueError(
                "Bounded RHF directions require the stationary source cotangent route"
            )
        expected = {k: v for k, v in self.prepared.feeds.items() if _source_key(k)}
        if not expected.keys() <= fixed.keys() or any(
            not isinstance(k, str) or not _source_key(k) for k in fixed
        ):
            raise ValueError("Incomplete or unknown fixed-source direction")
        direction = dict(fixed)
        if ordinary is None:
            program = fixed_source_lagrangian_direction(self.layout)
        else:
            all_feeds = self.prepared.feeds
            if ordinary.keys() != all_feeds.keys() - expected.keys():
                raise ValueError("Incomplete or unknown ordinary Hamiltonian direction")
            direction.update(ordinary)
            expected = all_feeds
            program = molecular_lagrangian_direction(self.layout)
        resource = self._resource(program)
        direction_bytes = sum(v.nbytes for v in expected.values())
        peak = checked_bytes(
            self.base_bytes + 2 * direction_bytes + resource.native_bytes,
            "CC Lagrangian direction numeric storage",
        )
        work = self.qualification_scalar_work + resource.scalar_work
        if peak > self.options.max_bytes or work > self.options.max_total_work:
            raise MemoryError(
                "CC Lagrangian direction exceeds numeric/semantic-work budget"
            )
        self._resources[program.logical_hash] = resource
        packed, _ = self.qualify(result)
        fields = {
            "d_" + k: _checked_array(k, direction[k], v.shape)
            for k, v in expected.items()
        }
        output = self._execute(program, {**fields, **self._seeds(packed)}, _Counter())
        (derivative,) = output.values()
        return float(derivative)

    def _replay_multipliers(self, multipliers: np.ndarray) -> UCCSDLambdaResult:
        """Bind archived multipliers only after this owner's physical checks."""
        packed = _checked_array(
            "packed Lambda multipliers", multipliers, (self.layout.packed_size,)
        )
        counter = _Counter()
        primal, drift = self._qualify_parent(counter)
        norm, maximum = self._stationarity(packed, counter)
        return UCCSDLambdaResult(
            packed,
            {
                name.replace("t", "l", 1): value
                for name, value in self.layout.unpack(packed).items()
            },
            self.state_identity,
            self._lambda_identity(packed),
            primal,
            drift,
            norm,
            norm,
            maximum,
            0,
            0,
            {
                "tensor_backend": self.prepared.executor.backend,
                "numeric_bound_bytes": self.peak_bytes,
                "executed_semantic_scalar_work": counter.scalar_work,
                "native_calls": counter.native_calls,
                "scope": "archived-state-physical-replay",
            },
        )

    def _replay_fixed_source_parameter_cotangent(
        self, packed_multipliers: np.ndarray, parameter: str
    ) -> UCCSDParameterCotangent:
        """Check a supplied packed dual against this native center and take its VJP."""
        self._check_open()
        started = time.perf_counter()
        packed = _checked_array(
            "packed Lambda multipliers", packed_multipliers, (self.layout.packed_size,)
        )
        return self._parameter_cotangent_from_packed(
            packed,
            parameter,
            lambda_identity=self._lambda_identity(packed),
            started=started,
        )
