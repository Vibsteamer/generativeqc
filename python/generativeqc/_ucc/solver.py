"""Shared bounded native CC iteration controls and DIIS proposals."""

import typing
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolverOptions:
    """CPU iteration controls with fixed scientific acceptance limits.

    Damping, level shifts and CC DIIS affect proposals only; final acceptance
    uses the original physical equations. ``max_bytes`` bounds logical numeric
    solver/interpreter storage independently of the provider's cache budget;
    it is not a Python/BLAS process-RSS cap.
    """

    max_iterations: int = 100
    energy_tolerance: float = 1e-11
    residual_tolerance: float = 1e-9
    denominator_threshold: float = 1e-10
    damping: float = 0.0
    level_shift: float = 0.0
    diis_size: int = 6
    max_bytes: int = 256 << 20

    def __post_init__(self) -> None:
        for key in ("max_iterations", "max_bytes", "diis_size"):
            v = getattr(self, key)
            if type(v) is not int or v < (0 if key == "diis_size" else 1):
                raise ValueError(f"invalid CCSD option {key}")
        if self.diis_size not in (0, *range(2, 21)):
            raise ValueError("diis_size must be zero or between 2 and 20")
        for key in (
            "energy_tolerance",
            "residual_tolerance",
            "denominator_threshold",
            "damping",
            "level_shift",
        ):
            if not np.isfinite(getattr(self, key)):
                raise ValueError(f"nonfinite CCSD option {key}")
        if (
            not 0 < self.energy_tolerance <= 1e-8
            or not 0 < self.residual_tolerance <= 1e-9
        ):
            raise ValueError(
                "CCSD tolerances must respect energy <=1e-8 and physical residual <=1e-9"
            )
        if (
            self.denominator_threshold <= 0
            or not 0 <= self.damping < 1
            or self.level_shift < 0
        ):
            raise ValueError("invalid CCSD denominator/damping/level shift")


class _DIIS:
    """Independent CC history of actual (amplitude, physical residual) pairs."""

    def __init__(self, size: typing.Any) -> None:
        self.size = size
        self.vectors, self.errors = [], []
        self.restarts = 0

    def update(self, vector: typing.Any, error: typing.Any) -> typing.Any:
        if not self.size:
            return vector
        self.vectors.append(vector.copy())
        self.errors.append(error.copy())
        self.vectors = self.vectors[-self.size :]
        self.errors = self.errors[-self.size :]
        while len(self.vectors) > 1:
            errors = np.array(self.errors)
            gram = errors @ errors.T
            scale = float(np.max(np.abs(gram)))
            if scale == 0:
                return vector
            n = len(gram)
            B = np.empty((n + 1, n + 1))
            B[:n, :n] = gram / scale
            B[n, :] = B[:, n] = -1
            B[n, n] = 0
            rhs = np.zeros(n + 1)
            rhs[n] = -1
            try:
                coefficients = np.linalg.solve(B, rhs)[:n]
                if (
                    not np.isfinite(coefficients).all()
                    or np.max(np.abs(coefficients)) > 1e6
                ):
                    raise np.linalg.LinAlgError("ill-conditioned CC DIIS weights")
                return coefficients @ np.array(self.vectors)
            except np.linalg.LinAlgError:
                self.vectors.pop(0)
                self.errors.pop(0)
                self.restarts += 1
        return vector
