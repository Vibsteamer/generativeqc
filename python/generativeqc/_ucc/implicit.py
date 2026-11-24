"""Checked transpose solves and packed GMRES for native stationary response."""

from __future__ import annotations

import math
import typing
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np
from generativeqc_compiler.common.provenance import canonical_hash

from generativeqc.response_problem import ResponseCompatibilityError
from generativeqc.response_solver import (
    GMRESOptions,
    _single_workspace_bytes,
    _vector_norm,
    solve,
)


def _checked_bytes(value: typing.Any, name: typing.Any) -> typing.Any:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be a nonnegative signed-64-bit byte count")
    return value


def _array(value: typing.Any, shape: typing.Any, name: typing.Any) -> typing.Any:
    array = np.asarray(value)
    if array.dtype != np.dtype("float64") or array.shape != shape:
        raise ValueError(f"{name} must have real FP64 dtype and shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _immutable(value: typing.Any) -> typing.Any:
    # A bytes-backed snapshot cannot be made writable by setflags(), and does
    # not alias caller arrays or an executor's reusable staging buffer.
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(
        value.shape
    )


class ImplicitSolveError(RuntimeError):
    """The implicit primitive rejected a primal/adjoint/result publication."""


class TransposeSolver(Protocol):
    """Opaque solve of the supplied, already-transposed Euclidean operator."""

    identity: str
    backend: str
    workspace_bytes: int
    rtol: float
    atol: float

    def solve(self, operator: typing.Any, rhs: typing.Any) -> typing.Any:
        """Return the #179 solution/status/residual/workspace result contract."""
        ...


@dataclass(frozen=True)
class ResponseGMRES:
    """Use #179 unchanged; do not introduce another Krylov implementation."""

    dimension: int
    options: GMRESOptions = field(default_factory=GMRESOptions)
    backend: str = field(default="python-response-gmres", init=False)

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("response dimension must be a positive integer")
        if not isinstance(self.options, GMRESOptions):
            raise TypeError("response callback requires GMRESOptions")

    @property
    def workspace_bytes(self) -> typing.Any:
        return _single_workspace_bytes(self.dimension, self.options)

    @property
    def rtol(self) -> typing.Any:
        return self.options.rtol

    @property
    def atol(self) -> typing.Any:
        return self.options.atol

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(
            {
                "callback": "generativeqc-response-gmres-v1",
                "dimension": self.dimension,
                "backend": self.backend,
                "operator": "caller-supplied-transpose-in-euclidean-coordinates",
                "options": asdict(self.options),
            }
        )

    def solve(self, operator: typing.Any, rhs: typing.Any) -> typing.Any:
        if operator.dimension != self.dimension:
            raise ValueError("response callback dimension mismatch")
        return solve(operator, rhs, options=self.options)


def transpose_solver_contract(solver: TransposeSolver) -> dict:
    """Freeze an opaque solver's identity, tolerances and declared storage."""
    for name in ("identity", "backend"):
        value = getattr(solver, name, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"implicit callback requires a nonempty {name}")
    for name in ("rtol", "atol"):
        value = getattr(solver, name, None)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"solver {name} must be finite and nonnegative")
    if solver.rtol >= 1 or max(solver.rtol, solver.atol) == 0:
        raise ValueError("solver requires rtol < 1 and a nonzero convergence tolerance")
    return {
        "identity": solver.identity,
        "backend": solver.backend,
        "workspace_bytes": _checked_bytes(solver.workspace_bytes, "solver workspace"),
        "rtol": float(solver.rtol),
        "atol": float(solver.atol),
    }


@dataclass(frozen=True)
class CheckedAdjointResult:
    """A solution rechecked against the supplied physical transpose action."""

    solution: np.ndarray
    residual_norm: float
    iterations: int
    operator_actions: int
    workspace_bytes: int


def checked_transpose_solve(
    operator: typing.Any,
    rhs: typing.Any,
    *,
    solver: typing.Any,
    assert_current: typing.Any = None,
) -> typing.Any:
    """Shared first-order adjoint acceptance for generated single/block states.

    The caller supplies the already-transposed Euclidean action and performs
    its own simultaneous state/operator/solver resource admission. No equation,
    coordinate map, preconditioner or Krylov algorithm is introduced here.
    A solver's success flag cannot bypass a fresh physical residual evaluation.
    """
    contract = transpose_solver_contract(solver)
    dimension = operator.dimension
    if type(dimension) is not int or dimension < 1:
        raise ValueError("adjoint dimension must be a positive integer")
    if assert_current is not None and not callable(assert_current):
        raise TypeError("assert_current must be callable")
    rhs = _immutable(_array(rhs, (dimension,), "adjoint RHS"))
    actions = 0

    def check() -> None:
        if assert_current is not None:
            assert_current()
        if transpose_solver_contract(solver) != contract:
            raise ResponseCompatibilityError("implicit solver contract changed")

    source_engine = getattr(operator, "_krylov_engine", None)

    class CheckedKrylovEngine:
        """Preserve resident vector execution while keeping action checks outside it."""

        def __init__(self, engine: typing.Any) -> None:
            self._engine = engine

        def __getattr__(self, name: str) -> typing.Any:
            return getattr(self._engine, name)

        def apply(self, checked_operator: typing.Any, vector: typing.Any) -> typing.Any:
            del checked_operator
            nonlocal actions
            check()
            actions += 1
            value = self._engine.apply(operator, vector)
            check()
            return value

    class CheckedOperator:
        def __init__(self) -> None:
            self.dimension = dimension
            if source_engine is not None:
                self._krylov_engine = CheckedKrylovEngine(source_engine)

        def validate_current(self) -> None:
            check()

        def apply(self, vector: typing.Any) -> typing.Any:
            nonlocal actions
            check()
            vector = _immutable(_array(vector, (dimension,), "adjoint iterate"))
            actions += 1
            value = _immutable(
                _array(operator.apply(vector), (dimension,), "transpose action")
            )
            check()
            return value

    checked = CheckedOperator()
    check()
    result = solver.solve(checked, rhs)
    check()
    if result.converged is not True:
        raise ImplicitSolveError(f"implicit adjoint failed: {result.reason}")
    if (
        not math.isfinite(result.residual_norm)
        or result.residual_norm < 0
        or _checked_bytes(result.workspace_bytes, "reported solver workspace")
        > contract["workspace_bytes"]
    ):
        raise ImplicitSolveError(
            "implicit solver returned invalid residual/resource diagnostics"
        )
    if type(result.iterations) is not int or result.iterations < 0:
        raise ImplicitSolveError(
            "implicit solver returned invalid iteration diagnostics"
        )
    solution = _immutable(_array(result.solution, (dimension,), "adjoint solution"))
    residual = _vector_norm(checked.apply(solution) - rhs)
    target = max(contract["atol"], contract["rtol"] * _vector_norm(rhs))
    if not math.isfinite(target) or residual > target or result.residual_norm > target:
        raise ImplicitSolveError(
            f"implicit true adjoint residual failed: {residual:.6g} > {target:.6g}"
        )
    check()
    return CheckedAdjointResult(
        solution, residual, result.iterations, actions, result.workspace_bytes
    )
