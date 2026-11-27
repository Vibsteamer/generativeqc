"""Kállay et al. (T+) energy and occupied-source cotangents, DOI 10.1063/5.0057426."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    broadcast,
    constant,
    divide,
    input_tensor,
    multiply,
    reduce_sum,
    scaled_bilinear,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import numpy as np

    from generativeqc_compiler.tensor import Execution


@dataclass(frozen=True)
class TPlusResponse:
    """Energy and unit-energy cotangents in the input occupied coordinates."""

    energy: float
    mp2_cotangent: tuple[float, ...]
    f12_cotangent: tuple[float, ...]
    triples_cotangent: tuple[float, ...]

    @property
    def multipliers(self) -> tuple[float, ...]:
        """The active orbital factors q_i for the triples contraction."""
        return self.triples_cotangent


@lru_cache(maxsize=32)
def build_tplus_program(n_active: int) -> Program:
    """Build Eq. (12) and its three adjoints for packed, nonsingular orbitals.

    Owners must validate operand identity, pack structural zeros out, and
    scatter cotangents back in the same order. An empty active list has the
    exact zero response and requires no program execution.
    """
    if type(n_active) is not int or n_active <= 0:
        raise ValueError("(T+) requires a positive packed active-orbital count")
    orbital = Index("i", IndexSpace("triples_scaling_occupied", "occupied", n_active))
    spec = TensorSpec((orbital,), role="input", differentiable=True)
    mp2, f12, triples = (
        input_tensor(name, spec) for name in ("mp2", "f12_increment", "triples")
    )
    one = broadcast(constant(1), spec.indices, ())
    minus_one = broadcast(constant(-1), spec.indices, ())
    zero = broadcast(constant(0), spec.indices, ())
    # Eq. (12) has q_i = (M_i + F_i)/M_i. The common scaled arithmetic
    # avoids losing the sum near cancellation or forming M_i ** 2 in FP64.
    q = scaled_bilinear(mp2, one, f12, minus_one, mp2, one)
    return Program(
        {
            "energy": reduce_sum(multiply(q, triples), (0,)),
            "mp2_cotangent": scaled_bilinear(zero, zero, f12, triples, mp2, mp2),
            "f12_cotangent": divide(triples, mp2),
            "triples_cotangent": q,
        },
        provenance={
            "method": "CCSD(T+)",
            "equation_doi": "10.1063/5.0057426",
            "equations": (12, 13),
            "partition": "same occupied frame, unscaled signed increments",
        },
    )


def _occupied_vector(values: Iterable[Real], name: str) -> tuple[float, ...]:
    data = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in data):
        raise ValueError(f"(T+) {name} must be a real occupied vector")
    data = tuple(float(value) for value in data)
    if not all(map(isfinite, data)):
        raise ValueError(f"(T+) {name} must be finite")
    return data


def evaluate_tplus(
    mp2: Iterable[Real],
    f12_increment: Iterable[Real],
    triples: Iterable[Real],
    *,
    inactive: Iterable[bool] | None = None,
    run: Callable[[Program, Mapping[str, np.ndarray]], Execution] | None = None,
) -> TPlusResponse:
    """Evaluate the packed equation using the caller's executor or the reference.

    Each slot is an occupied orbital and must come from one identity-matched
    orbital frame. The three inputs are conventional MP2, the F12 increment
    alone, and conventional triples. Empty spin blocks are legal. A producer
    may mark a slot structurally inactive only when all three contributions
    vanish identically. A zero active MP2 value gives an undefined quotient
    response even when all three values happen to vanish at that geometry.
    """
    import numpy as np

    from generativeqc_compiler.tensor import execute

    names = ("mp2", "f12_increment", "triples")
    operands = {
        name: _occupied_vector(raw, name)
        for name, raw in zip(names, (mp2, f12_increment, triples), strict=True)
    }
    size = len(operands["mp2"])
    if any(len(data) != size for data in operands.values()):
        raise ValueError("(T+) partitions must have identical occupied dimensions")
    mask = (False,) * size if inactive is None else tuple(inactive)
    if len(mask) != size or any(
        not isinstance(flag, (bool, np.bool_)) for flag in mask
    ):
        raise ValueError(
            "(T+) requires one Boolean inactivity flag per occupied orbital"
        )
    selected = []
    for orbital, (m, f, t, structural_zero) in enumerate(
        zip(*operands.values(), mask, strict=True)
    ):
        if structural_zero:
            if any(value != 0.0 for value in (m, f, t)):
                raise ValueError(f"(T+) inactive orbital {orbital} has nonzero data")
        elif m == 0.0:
            raise ValueError(f"(T+) orbital {orbital} has a singular MP2 denominator")
        else:
            selected.append(orbital)
    if not selected:
        zeros = (0.0,) * size
        return TPlusResponse(0.0, zeros, zeros, zeros)
    feeds = {
        name: np.asarray(tuple(data[i] for i in selected), dtype=np.float64)
        for name, data in operands.items()
    }
    program = build_tplus_program(len(selected))
    outputs = (execute if run is None else run)(program, feeds).outputs
    if np.shape(outputs["energy"]) != () or not isfinite(float(outputs["energy"])):
        raise ArithmeticError("(T+) executor returned an invalid energy")

    def scatter(name: str) -> tuple[float, ...]:
        output = np.asarray(outputs[name])
        if (
            output.shape != (len(selected),)
            or output.dtype != np.float64
            or not np.isfinite(output).all()
        ):
            raise ArithmeticError(f"(T+) executor returned an invalid {name}")
        result = [0.0] * size
        for orbital, value in zip(selected, output, strict=True):
            result[orbital] = float(value)
        return tuple(result)

    return TPlusResponse(
        float(outputs["energy"]),
        scatter("mp2_cotangent"),
        scatter("f12_cotangent"),
        scatter("triples_cotangent"),
    )
