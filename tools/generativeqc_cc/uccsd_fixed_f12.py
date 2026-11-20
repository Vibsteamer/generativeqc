"""Private fixed-source UCCSD-F12b solver for matched canonical spin-MO feeds."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from generativeqc.response_solver import DiagonalPreconditioner, solve
from generativeqc_compiler.cc.uccsd_equations import build_f12_program
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDLayout
from generativeqc_compiler.cc.uccsd_response import build_f12_response_programs
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.resources import checked_bytes
from generativeqc_compiler.tensor.cpu import emit_cpu
from generativeqc_compiler.tensor.cuda_plan import plan_cuda

from tools.generativeqc_response.implicit import ResponseGMRES

from .native_tensor_cpu import NativeCCTensorExecutor
from .native_tensor_cuda import CudaCCTensorExecutor
from .solver import SolverOptions
from .uccsd_lambda import UCCSDLambdaOptions, _PackedUCCSDLambda
from .uccsd_solver import (
    _denominator_minimum,
    _denominators,
    _iteration_bytes,
    _PackedUCCSD,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from .uccsd_solver import UCCSDResult


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


def _checked_array(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != shape:
        raise ValueError(f"{name} must be a native FP64 ndarray of shape {shape}")
    return immutable(value, shape=shape)


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
            raise ValueError("fixed-source UCCSD-F12b requires a finite reference energy")
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
        self.program = build_f12_program(layout)
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
        self.host_iteration_bytes = _iteration_bytes(
            layout, options, feed_bytes + reference_bytes
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
        preparation_bytes = (
            feed_bytes
            + reference_bytes
            + 8 * layout.packed_size
            + 8 * max(map(math.prod, inputs.values()))
        )
        self.logical_required_bytes = checked_bytes(
            max(preparation_bytes, self.host_iteration_bytes + self.execution_peak_bytes),
            "fixed-source UCCSD-F12b simultaneous numeric storage",
        )
        if self.logical_required_bytes > options.max_bytes:
            raise MemoryError("fixed-source UCCSD-F12b solver numeric budget exceeded")
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
            )
            for name, shape in inputs.items()
        }
        self.feeds = MappingProxyType(feeds)
        self.integral_hash = canonical_hash(
            {name: _digest(value) for name, value in feeds.items()}
        )
        self.integral_source_id = reference_id
        self.integral_backend = "supplied-canonical-spin-MO-feeds"
        self.source_backend = "supplied-fixed-geminal-sources"
        self.integral_request_count = self.provider_peak_bytes = 0
        self.provider_statistics = MappingProxyType({})
        self._executor_configuration = self._execution_configuration()
        self._closed = False
        self.preparation_seconds = time.perf_counter() - started


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
            raise ValueError("packed orbital gaps must be a finite positive FP64 vector")
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
    _build_response_programs = staticmethod(build_f12_response_programs)

    def __init__(
        self,
        prepared: _PreparedFixedSourceUCCSD,
        state: UCCSDResult,
        *,
        options: UCCSDLambdaOptions | None = None,
        solver: Any = None,
        orbital_gap_preconditioner: bool = False,
    ) -> None:
        if type(orbital_gap_preconditioner) is not bool:
            raise TypeError("orbital_gap_preconditioner must be a bool")
        if orbital_gap_preconditioner:
            if solver is not None:
                raise ValueError("orbital-gap preconditioning requires the built-in GMRES")
            if not isinstance(prepared, _PreparedFixedSourceUCCSD) or prepared._closed:
                raise TypeError("UCCSD Lambda requires a live _PreparedFixedSourceUCCSD")
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
