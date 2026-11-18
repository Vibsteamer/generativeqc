"""Bilinear robust operator contraction for density-fitted orbital-pair tiles."""

from .ir import Node, add, einsum


def robust_fitted_operator(
    fit_left: Node, raw_left: Node, fit_right: Node, raw_right: Node, metric: Node
) -> Node:
    """Robust non-Coulomb integral for two Coulomb-fitted density tiles."""
    residual = add(
        raw_right, einsum("ab,br->ar", metric, fit_right), coefficients=(1, -1)
    )
    return add(
        einsum("al,ar->lr", fit_left, residual),
        einsum("bl,br->lr", raw_left, fit_right),
    )
