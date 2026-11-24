"""Execute generated implicit VJPs through the existing #179 solver callback.

This is an explicit host-orchestrated development adapter, not a public force
endpoint or a native/device-resident Krylov loop. Tensor execution is injected
and identified; the default is the NumPy reference interpreter. CUDA executors
must be explicitly prepared by their resource owner. No backend fallback occurs.
"""

from __future__ import annotations

import hashlib
import math
import threading
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from generativeqc._ucc.implicit import (
    CheckedAdjointResult,
    ImplicitSolveError,
    ResponseGMRES,
    TransposeSolver,
    _array,
    _checked_bytes,
    _immutable,
    checked_transpose_solve,
    transpose_solver_contract,
)
from generativeqc.response_problem import ResponseCompatibilityError, ResponseProblem
from generativeqc.response_solver import _vector_norm
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.method import ImplicitVJPPlan
from generativeqc_compiler.method.implicit import PREFIX
from generativeqc_compiler.tensor import execute


@dataclass(frozen=True)
class ReferenceTensorExecutor:
    """Identified CPU reference execution of the very same generated programs."""

    plan: ImplicitVJPPlan
    backend: str = field(default="numpy-cpu-interpreter", init=False)
    workspace_bytes: int = field(default=0, init=False)

    @property
    def plan_identity(self) -> typing.Any:
        return self.plan.identity

    @property
    def identity(self) -> typing.Any:
        return canonical_hash({"plan": self.plan_identity, "backend": self.backend})

    def execute(self, stage: typing.Any, feeds: typing.Any) -> typing.Any:
        return execute(
            self.plan.programs[stage],
            feeds,
            max_bytes=self.plan.reference_workspace_bytes,
        )


@dataclass(frozen=True, init=False, eq=False, repr=False)
class ResponseTransposeBinding:
    """Bind a generated #465 plan to one live #179 response operator.

    The generated residual remains the mathematical source of parameter VJPs.
    Only the transpose Jacobian action may be delegated to a qualified native
    response operator, and the operator/reference/layout/resource identities
    become part of the execution contract.
    """

    plan_identity: str
    problem_identity: str
    reference_identity: str
    operator_identity: str
    backend_identity: str
    backend: str
    dimension: int
    host_workspace_bytes: int
    device_workspace_bytes: int
    identity: str

    def __init__(self, plan: typing.Any, operator: typing.Any) -> None:
        if not isinstance(plan, ImplicitVJPPlan):
            raise TypeError("response binding requires an ImplicitVJPPlan")
        problem = getattr(operator, "problem", None)
        if not isinstance(problem, ResponseProblem):
            raise TypeError(
                "response binding requires an operator with ResponseProblem"
            )
        if problem.method != "rhf":
            raise ResponseCompatibilityError(
                "implicit response binding currently supports RHF only"
            )
        spec = plan.spec
        if getattr(operator, "dimension", None) != spec.dimension:
            raise ResponseCompatibilityError("implicit/response dimension mismatch")
        if (
            problem.operator_identity != spec.operator_identity
            or getattr(operator, "identity", None) != spec.operator_identity
        ):
            raise ResponseCompatibilityError(
                "implicit/response operator identity mismatch"
            )
        if problem.layout.identity != spec.state_layout:
            raise ResponseCompatibilityError("implicit/response state layout mismatch")
        if problem.layout.identity != spec.residual_layout:
            raise ResponseCompatibilityError(
                "implicit/response residual layout mismatch"
            )
        if problem.gauge != spec.gauge:
            raise ResponseCompatibilityError("implicit/response gauge mismatch")
        backend_object = getattr(operator, "backend", None)
        backend_identity = getattr(backend_object, "identity", None)
        if not isinstance(backend_identity, str) or not backend_identity.strip():
            raise ValueError("response operator backend must have a stable identity")
        host = _checked_bytes(
            getattr(operator, "host_workspace_bytes", None),
            "response operator host workspace",
        )
        device = _checked_bytes(
            getattr(operator, "device_workspace_bytes", None),
            "response operator device workspace",
        )
        resource_identity = getattr(operator, "resource_identity", None)
        if not isinstance(resource_identity, str) or not resource_identity.strip():
            raise ValueError("response operator must publish a resource identity")
        backend = f"{operator.__class__.__name__}/{backend_object.__class__.__name__}"
        identity = canonical_hash(
            {
                "plan": plan.identity,
                "problem": problem.identity,
                "reference": problem.reference_identity,
                "operator": spec.operator_identity,
                "backend": backend_identity,
                "resources": resource_identity,
                "host_workspace_bytes": host,
                "device_workspace_bytes": device,
                "contract": "response-transpose-binding-v1",
            }
        )
        object.__setattr__(self, "plan_identity", plan.identity)
        object.__setattr__(self, "problem_identity", problem.identity)
        object.__setattr__(self, "reference_identity", problem.reference_identity)
        object.__setattr__(self, "operator_identity", spec.operator_identity)
        object.__setattr__(self, "backend_identity", backend_identity)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "dimension", spec.dimension)
        object.__setattr__(self, "host_workspace_bytes", host)
        object.__setattr__(self, "device_workspace_bytes", device)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "_problem", problem)
        object.__setattr__(self, "_operator", operator)
        object.__setattr__(self, "_resource_identity", resource_identity)

    def assert_current(self, reference_identity: typing.Any) -> None:
        if reference_identity != self.reference_identity:
            raise ResponseCompatibilityError(
                "response operator belongs to a different reference"
            )
        self._problem.assert_compatible(self._operator.problem)
        validate = getattr(self._operator.backend, "validate_reference", None)
        if validate is not None:
            validate(self._problem.reference)
        if (
            self._operator.resource_identity != self._resource_identity
            or self._operator.host_workspace_bytes != self.host_workspace_bytes
            or self._operator.device_workspace_bytes != self.device_workspace_bytes
        ):
            raise ResponseCompatibilityError(
                "response operator resource contract changed"
            )

    def apply(self, vector: typing.Any) -> typing.Any:
        return self._operator.apply_transpose(vector)


@dataclass(frozen=True)
class ImplicitVJPResult:
    """Publish only after primal, solver, true-residual and freshness gates."""

    parameter_cotangents: Mapping[str, np.ndarray]
    adjoint: np.ndarray
    primal_residual_norm: float
    adjoint_residual_norm: float
    iterations: int
    operator_actions: int
    plan_identity: str
    state_identity: str
    execution_identity: str
    solver_backend: str
    tensor_backend: str
    logical_reserved_host_bytes: int
    transpose_backend: str
    logical_reserved_device_bytes: int


@dataclass(frozen=True, init=False, eq=False, repr=False)
class BoundImplicitState:
    """Immutable numerical snapshot plus declared solver/executor ownership.

    The default executor is CPU reference-only. An injected executor provides
    plan_identity, identity, backend, workspace_bytes (all simultaneously live
    host buffers), and execute(stage, feeds) -> outputs/backend. Its native
    device resource owner must perform device admission before preparation.

    ``current_reference`` is the runtime owner's live identity check. Without
    it this object represents only an immutable tooling snapshot, not a proof
    of a current native molecular state. Each VJP still requires its reference
    identity explicitly; snapshots from another geometry cannot be substituted.
    """

    plan: ImplicitVJPPlan
    reference_identity: str
    feeds: Mapping[str, np.ndarray]
    state_identity: str
    execution_identity: str
    primal_residual_norm: float
    logical_reserved_host_bytes: int
    logical_reserved_device_bytes: int
    solver: TransposeSolver
    executor: object
    transpose_binding: ResponseTransposeBinding | None

    def __init__(
        self,
        plan: ImplicitVJPPlan,
        feeds: Mapping,
        *,
        reference_identity: str,
        solver: TransposeSolver | None = None,
        executor: typing.Any = None,
        response_operator: typing.Any = None,
        current_reference: Callable[[], str] | None = None,
        primal_atol: float = 1e-10,
        max_bytes: int = 256 << 20,
        max_device_bytes: int | None = None,
    ) -> None:
        if not isinstance(plan, ImplicitVJPPlan):
            raise TypeError("expected a generated ImplicitVJPPlan")
        if not isinstance(reference_identity, str) or not reference_identity.strip():
            raise ValueError("reference identity must be nonempty")
        if (
            type(primal_atol) not in (float, int)
            or not math.isfinite(primal_atol)
            or primal_atol <= 0
        ):
            raise ValueError("primal_atol must be finite and positive")
        if current_reference is not None and not callable(current_reference):
            raise TypeError("current_reference must be a callable live identity check")
        if not isinstance(feeds, Mapping) or set(feeds) != set(plan.spec.inputs):
            raise ValueError("implicit feeds must exactly match the residual inputs")
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "reference_identity", reference_identity)
        object.__setattr__(
            self,
            "solver",
            ResponseGMRES(plan.spec.dimension) if solver is None else solver,
        )
        object.__setattr__(
            self,
            "executor",
            ReferenceTensorExecutor(plan) if executor is None else executor,
        )
        binding = (
            None
            if response_operator is None
            else ResponseTransposeBinding(plan, response_operator)
        )
        if binding is not None:
            if binding.reference_identity != reference_identity:
                raise ResponseCompatibilityError(
                    "response operator/reference identity mismatch"
                )
            if current_reference is None:
                raise ValueError(
                    "response-operator binding requires a live current_reference callback"
                )
        object.__setattr__(self, "transpose_binding", binding)
        object.__setattr__(self, "_current_reference", current_reference)
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(
            self, "_contract", MappingProxyType(self._callback_contract())
        )
        if self.executor.plan_identity != plan.identity:
            raise ResponseCompatibilityError(
                "executor belongs to a different implicit plan"
            )
        _checked_bytes(max_bytes, "implicit host budget")
        required = (
            plan.reference_workspace_bytes
            + self._contract["solver_workspace_bytes"]
            + self._contract["executor_workspace_bytes"]
            + self._contract["transpose_host_workspace_bytes"]
        )
        _checked_bytes(required, "combined implicit host reservation")
        if required > max_bytes:
            raise ImplicitSolveError(
                "implicit simultaneous host workspace budget exceeded"
            )
        object.__setattr__(self, "logical_reserved_host_bytes", required)
        required_device = (
            self._contract["executor_device_workspace_bytes"]
            + self._contract["transpose_device_workspace_bytes"]
        )
        _checked_bytes(required_device, "combined implicit device reservation")
        if self._contract["transpose_device_workspace_bytes"]:
            if max_device_bytes is None:
                raise ImplicitSolveError(
                    "device response binding requires an explicit combined device budget"
                )
            _checked_bytes(max_device_bytes, "implicit device budget")
            if required_device > max_device_bytes:
                raise ImplicitSolveError(
                    "implicit simultaneous device workspace budget exceeded"
                )
        object.__setattr__(self, "logical_reserved_device_bytes", required_device)
        self._assert_current(reference_identity)
        arrays = {
            name: _array(feeds[name], spec.shape, name)
            for name, spec in plan.spec.inputs.items()
        }
        object.__setattr__(
            self,
            "feeds",
            MappingProxyType(
                {name: _immutable(value) for name, value in arrays.items()}
            ),
        )
        object.__setattr__(
            self,
            "state_identity",
            canonical_hash(
                {
                    "reference": reference_identity,
                    "plan": plan.identity,
                    "inputs": {
                        name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                        for name, value in self.feeds.items()
                    },
                }
            ),
        )
        object.__setattr__(
            self,
            "execution_identity",
            canonical_hash(
                {
                    "state": self.state_identity,
                    "callbacks": dict(self._contract),
                    "primal_atol": float(primal_atol),
                    "logical_host_reservation": required,
                    "logical_device_reservation": required_device,
                }
            ),
        )
        primal = self._run("primal")["value"]
        object.__setattr__(self, "primal_residual_norm", _vector_norm(primal))
        if self.primal_residual_norm > primal_atol:
            raise ImplicitSolveError(
                f"implicit primal is not converged: residual={self.primal_residual_norm:.6g}"
            )
        self._assert_current(reference_identity)

    def _callback_contract(self) -> typing.Any:
        solver, executor = self.solver, self.executor
        for obj in (solver, executor):
            for name in ("identity", "backend"):
                if (
                    not isinstance(getattr(obj, name, None), str)
                    or not getattr(obj, name).strip()
                ):
                    raise ValueError(f"implicit callback requires a nonempty {name}")
        for name in ("rtol", "atol"):
            value = getattr(solver, name, None)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"solver {name} must be finite and nonnegative")
        if solver.rtol >= 1 or max(solver.rtol, solver.atol) == 0:
            raise ValueError(
                "solver requires rtol < 1 and a nonzero convergence tolerance"
            )
        return {
            "solver": solver.identity,
            "solver_backend": solver.backend,
            "solver_workspace_bytes": _checked_bytes(
                solver.workspace_bytes, "solver workspace"
            ),
            "solver_rtol": float(solver.rtol),
            "solver_atol": float(solver.atol),
            "executor": executor.identity,
            "executor_backend": executor.backend,
            "executor_workspace_bytes": _checked_bytes(
                executor.workspace_bytes, "executor workspace"
            ),
            "executor_device_workspace_bytes": _checked_bytes(
                getattr(executor, "device_workspace_bytes", 0),
                "executor device workspace",
            ),
            "executor_plan": executor.plan_identity,
            "transpose_operator": (
                self.transpose_binding.identity
                if self.transpose_binding is not None
                else self.plan.programs["transpose"].logical_hash
            ),
            "transpose_backend": (
                self.transpose_binding.backend
                if self.transpose_binding is not None
                else executor.backend
            ),
            "transpose_host_workspace_bytes": (
                self.transpose_binding.host_workspace_bytes
                if self.transpose_binding is not None
                else 0
            ),
            "transpose_device_workspace_bytes": (
                self.transpose_binding.device_workspace_bytes
                if self.transpose_binding is not None
                else 0
            ),
        }

    def _assert_current(self, reference_identity: typing.Any) -> None:
        if reference_identity != self.reference_identity or (
            self._current_reference is not None
            and self._current_reference() != self.reference_identity
        ):
            raise ResponseCompatibilityError(
                "implicit reference identity is stale or incompatible"
            )
        if self.transpose_binding is not None:
            self.transpose_binding.assert_current(reference_identity)
        if self._callback_contract() != self._contract:
            raise ResponseCompatibilityError(
                "implicit solver/executor contract changed"
            )

    def _run(self, stage: typing.Any, extra: typing.Any = None) -> typing.Any:
        feeds = dict(self.feeds)
        if extra:
            feeds.update(extra)
        execution = self.executor.execute(stage, feeds)
        if execution.backend != self._contract["executor_backend"]:
            raise ResponseCompatibilityError(
                "implicit tensor backend changed; no fallback is allowed"
            )
        expected = self.plan.programs[stage].outputs
        if set(execution.outputs) != set(expected):
            raise ImplicitSolveError("implicit executor returned incomplete outputs")
        return {
            name: _immutable(
                _array(execution.outputs[name], node.spec.shape, f"{stage}/{name}")
            )
            for name, node in expected.items()
        }

    def vjp(
        self,
        state_cotangent: typing.Any,
        *,
        reference_identity: str,
        direct: typing.Any = None,
    ) -> ImplicitVJPResult:
        """Apply the first-order implicit rule without any solver-history tape."""
        with self._lock:
            self._assert_current(reference_identity)
            spec = self.plan.spec
            seed = _array(state_cotangent, spec.state_spec.shape, "state cotangent")
            direct = {} if direct is None else direct
            if not isinstance(direct, Mapping) or set(direct) - set(
                spec.parameter_names
            ):
                raise ValueError("direct cotangents contain unknown parameter names")
            extra = {
                PREFIX + f"direct_{name}": _immutable(
                    _array(
                        direct.get(
                            name, np.zeros(spec.inputs[name].shape, dtype=np.float64)
                        ),
                        spec.inputs[name].shape,
                        f"direct/{name}",
                    )
                )
                for name in spec.parameter_names
            }
            rhs = _immutable(
                self._run("rhs", {PREFIX + "seed": seed})["value"].reshape(-1)
            )
            owner = self

            class Operator:
                dimension = spec.dimension

                def apply(self, vector: typing.Any) -> typing.Any:
                    if owner.transpose_binding is not None:
                        return owner.transpose_binding.apply(vector)
                    return owner._run(
                        "transpose",
                        {PREFIX + "vector": vector.reshape(spec.residual_spec.shape)},
                    )["value"].reshape(-1)

            result = checked_transpose_solve(
                Operator(),
                rhs,
                solver=self.solver,
                assert_current=lambda: self._assert_current(reference_identity),
            )
            y, residual = result.solution, result.residual_norm
            extra[PREFIX + "vector"] = y.reshape(spec.residual_spec.shape)
            source = self._run("source", extra)
            adjoint = self._run(
                "adjoint", {PREFIX + "vector": extra[PREFIX + "vector"]}
            )["value"]
            self._assert_current(reference_identity)
            return ImplicitVJPResult(
                MappingProxyType(
                    {
                        name: _immutable(source[f"bar_{name}"])
                        for name in spec.parameter_names
                    }
                ),
                _immutable(adjoint),
                self.primal_residual_norm,
                residual,
                result.iterations,
                result.operator_actions,
                self.plan.identity,
                self.state_identity,
                self.execution_identity,
                self._contract["solver_backend"],
                self._contract["executor_backend"],
                self.logical_reserved_host_bytes,
                self._contract["transpose_backend"],
                self.logical_reserved_device_bytes,
            )


__all__ = [
    "BoundImplicitState",
    "CheckedAdjointResult",
    "ImplicitSolveError",
    "ImplicitVJPResult",
    "ReferenceTensorExecutor",
    "ResponseGMRES",
    "ResponseTransposeBinding",
    "TransposeSolver",
    "_array",
    "_checked_bytes",
    "_immutable",
    "checked_transpose_solve",
    "transpose_solver_contract",
]
