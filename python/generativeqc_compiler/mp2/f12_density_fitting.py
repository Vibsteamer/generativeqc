"""Coulomb-metric robust fitting of a tile of F12 two-electron integrals."""

from ..tensor.ir import Node, add, einsum, input_tensor
from ..tensor.program import Program
from ..tensor.types import Index, IndexSpace, TensorSpec


def fitted_integral_block(
    auxiliaries: int, left_pairs: int, right_pairs: int, *, coulomb: bool = False
) -> Program:
    """Map three-center inputs to a real orbital-pair by orbital-pair block.

    Each fitted density D solves J D = G, with the Coulomb metric J and
    bare three-center Coulomb integrals G in the declared auxiliary space.
    For a general operator take O_L, O_R and K from that same operator.
    The robust block is D_L^T (O_R - K D_R) + O_L^T D_R. For Coulomb the
    right parenthesis vanishes and either cross term is the fitted block;
    the implementation uses D_L^T G_R. Each column retains the caller's
    orbital-pair ordering, including when the two tiles have different sizes.

    The program's transpose acts on all live factors. Propagate derivatives
    of D through its defining Coulomb-metric solve separately.
    """
    if type(coulomb) is not bool or any(
        type(size) is not int or size <= 0
        for size in (auxiliaries, left_pairs, right_pairs)
    ):
        raise ValueError(
            "Fitted integral blocks require positive dimensions and a boolean Coulomb flag"
        )
    aux = IndexSpace("fitting_basis", "auxiliary", auxiliaries)
    aux_left = Index("a", aux)
    aux_right = Index("b", aux)
    left_axis = Index("l", IndexSpace("left_density_pairs", "pair", left_pairs))
    right_axis = Index("r", IndexSpace("right_density_pairs", "pair", right_pairs))

    def factor(name: str, axes: tuple[Index, Index]) -> Node:
        return input_tensor(name, TensorSpec(axes, role="input", differentiable=True))

    dl = factor("fit_left", (aux_left, left_axis))
    right = factor("raw_right", (aux_left, right_axis))
    if coulomb:
        block = einsum("al,ar->lr", dl, right)
    else:
        dr = factor("fit_right", (aux_right, right_axis))
        left = factor("raw_left", (aux_right, left_axis))
        metric = factor("operator_metric", (aux_left, aux_right))
        residual = add(right, einsum("ab,br->ar", metric, dr), coefficients=(1, -1))
        block = add(einsum("al,ar->lr", dl, residual), einsum("bl,br->lr", left, dr))
    return Program(
        {"block": block},
        provenance={
            "method": "Coulomb DF" if coulomb else "robust F12 density fitting",
            "columns": "ordered orbital pairs on each side",
            "coefficients": "full Coulomb-metric solve",
        },
    )
