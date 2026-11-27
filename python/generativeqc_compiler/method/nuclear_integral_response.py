"""Build-only AD of a projected, externally weighted integral energy term.

The integral owner supplies value/gradient/gradient-JVP from its existing AD
authority. The method owner supplies physical D/W-derived source weights. This
bilinear envelope does not choose HF factors, spin conventions or ECP policy.
"""

from functools import cache
from typing import TYPE_CHECKING

from generativeqc_compiler.integral.expr import Graph

if TYPE_CHECKING:
    from generativeqc_compiler.tensor.program import Program


@cache
def projected_integral_program() -> tuple:
    """Value, right-direction term, left gradient and mixed weighted response.

    The projection is fixed in nuclear coordinates. Left weight motion is zero
    for a stationary nuclear partial; right weight motion is retained exactly.
    Each call represents one mathematical-center coordinate, before atom scatter.
    """
    graph = Graph()
    value, first, mixed, weight, delta_weight, projection, direction = (
        graph.variable(name)
        for name in (
            "value",
            "first",
            "mixed",
            "weight",
            "delta_weight",
            "projection",
            "direction",
        )
    )
    projected = projection * value
    left = graph.variable("left")
    right = graph.variable("right")
    first_value = graph.differentiate(projected, left, {"value": first})
    gradient = graph.differentiate(weight * projected, left, {"value": first})
    response = graph.differentiate(
        gradient, right, {"first": mixed, "weight": delta_weight}
    )
    return graph, (projected, first_value * direction, gradient, response)


def overlap_hcore_gradient(atoms: int, ao_count: int) -> "Program":
    """Contract two full AO Jacobians with fixed overlap and core weights."""
    from generativeqc_compiler.tensor.ir import add, einsum, input_tensor
    from generativeqc_compiler.tensor.program import Program
    from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec

    centers = IndexSpace("nuclei", "atom", atoms)
    axes = IndexSpace("spatial", "cartesian", 3)
    atomic = IndexSpace("molecular_ao", "ao", ao_count)
    a, x = Index("a", centers), Index("x", axes)
    i, j = Index("i", atomic), Index("j", atomic)
    matrix = TensorSpec((i, j), role="input")
    jacobian = TensorSpec((a, x, i, j), role="input")
    terms = [
        einsum(
            "axij,ij->ax",
            input_tensor("d_" + name, jacobian),
            input_tensor(name, matrix),
        )
        for name in ("overlap", "hcore")
    ]
    return Program(
        {"gradient": add(*terms)},
        provenance={
            "operation": "one-electron positive nuclear gradient from full AO weights"
        },
    )
