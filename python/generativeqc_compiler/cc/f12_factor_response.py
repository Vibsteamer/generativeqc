"""Map a gauge-invariant DF-CC Gram functional onto the raw integrals and metric."""

from ..tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    transpose,
)


def coulomb_gram_factor_response(auxiliaries: int, pairs: int) -> Program:
    """Whiten G→A=W G, with W^T W=J^-1. The consumer depends on A only through A^T A.

    For a physical gradient H=dE/dA and D=J^-1 G, Gbar=W^T H and
    Jbar=-sym(Gbar D^T)/2. This form is valid for an arbitrary orthogonal
    fitting gauge (including the inverse Cholesky), without eigenvector derivatives.
    Contract physical symmetric variations of the full-rank J only.
    """
    if any(type(value) is not int or value < 1 for value in (auxiliaries, pairs)):
        raise ValueError("positive fitting and orbital-pair sizes required")
    a = IndexSpace("fitting_basis", "auxiliary", auxiliaries)
    i, j = Index("i", a), Index("j", a)
    p = Index("p", IndexSpace("orbital_pairs", "pair", pairs))

    def data(k: str, *axes: Index) -> Node:
        return input_tensor(k, TensorSpec(axes, role="input"))

    w = data("whitener", i, j)
    g = data("raw", j, p)
    a_bar = data("whitened_bar", i, p)
    d = data("fitted", i, p)
    b = einsum("ij,ip->jp", w, a_bar)
    mixed = einsum("ip,jp->ij", b, d)
    return Program(
        {
            "whitened": einsum("ij,jp->ip", w, g),
            "raw_bar": b,
            "metric_bar": add(
                mixed, transpose(mixed, (1, 0)), coefficients=("-1/4", "-1/4")
            ),
        },
        provenance={
            "operation": "full-rank Coulomb Gram whitening and gauge-invariant metric response"
        },
    )
