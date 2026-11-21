"""Coulomb-metric robust fitting of a tile of F12 two-electron integrals."""

from ..tensor.ir import Node, add, einsum, input_tensor, slice_tensor, transpose
from ..tensor.program import Program
from ..tensor.robust_density_fit import robust_fitted_operator
from ..tensor.types import Index, IndexSpace, TensorSpec


def symmetric_coulomb_pairs(raw: Node) -> Node:
    """Project a three-center Coulomb tensor onto symmetric orbital pairs."""
    return add(raw, transpose(raw, (0, 2, 1)), coefficients=("1/2", "1/2"))


def coulomb_pair_projection(auxiliaries: int, orbitals: int) -> Program:
    aux = Index("a", IndexSpace("fitting_basis", "auxiliary", auxiliaries))
    mo = IndexSpace("ordinary_orbital", "orbital", orbitals)
    axes = (aux, Index("p", mo), Index("q", mo))
    raw = input_tensor("raw_coulomb", TensorSpec(axes, role="input"))
    return Program({"raw_coulomb": symmetric_coulomb_pairs(raw)})


def molecular_orbital_sources(
    atomic: int, orbitals: int, occupied: int, auxiliaries: int
) -> Program:
    """Transform AO sources to one shared occupied/OBS/CABS orbital frame.

    The frame coefficients include the ordinary MOs and complementary virtuals
    in that order. Their AO rows use the same combined basis as each integral.
    Coulomb transforms retain the full frame for the extended Fock. The other
    four operators need an occupied index on the left. All inputs are live in
    the reverse program, including both appearances of the shared coefficients.
    A separate metric solve supplies the fitted Coulomb coefficients.
    """
    if (
        any(type(n) is not int or n <= 0 for n in (atomic, orbitals, auxiliaries))
        or type(occupied) is not int
        or not 0 <= occupied < orbitals
        or orbitals > atomic
    ):
        raise ValueError(
            "Expected positive AO/auxiliary sizes and 0 <= occupied < orbitals <= AO"
        )
    ao = IndexSpace("combined_atomic", "ao", atomic)
    mo = IndexSpace("extended_orbital", "orbital", orbitals)
    u, v = Index("u", ao), Index("v", ao)
    p = Index("p", mo)
    a = Index("a", IndexSpace("fitting_basis", "auxiliary", auxiliaries))

    def source(name: str, axes: tuple[Index, ...]) -> Node:
        return input_tensor(name, TensorSpec(axes, role="input", differentiable=True))

    coefficients = source("coefficients", (u, p))
    occupied_coefficients = (
        slice_tensor(coefficients, ((0, atomic), (0, occupied))) if occupied else None
    )
    hcore = source("hcore_ao", (u, v))
    h_right = einsum("uv,vq->uq", hcore, coefficients)
    result = {"hcore": einsum("up,uq->pq", coefficients, h_right)}
    for operator in (
        "coulomb",
        "f12",
        "f12_squared",
        "f12_over_coulomb",
        "double_commutator",
    ):
        if not occupied and operator != "coulomb":
            continue
        raw = source("ao_" + operator, (a, u, v))
        right = einsum("auv,vq->auq", raw, coefficients)
        left = coefficients if operator == "coulomb" else occupied_coefficients
        assert left is not None
        result["raw_" + operator] = einsum("up,auq->apq", left, right)
    return Program(
        result,
        provenance={
            "operation": "AO to shared OBS/CABS frame with full source and frame adjoints"
        },
    )


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
        block = robust_fitted_operator(dl, left, dr, right, metric)
    return Program(
        {"block": block},
        provenance={
            "method": "Coulomb DF" if coulomb else "robust F12 density fitting",
            "columns": "ordered orbital pairs on each side",
            "coefficients": "full Coulomb-metric solve",
        },
    )


def coulomb_fit_response(auxiliaries: int, pairs: int) -> Program:
    """Propagate one pair tile through the full-rank metric solve J D = G.

    The caller supplies a full-rank whitening factor W with W.T W = J^{-1}
    for the same metric and three-center integrals. An adjoint seed on D yields an
    adjoint on those integrals and the symmetric full-Frobenius weight on J.
    Sum the latter over all tiles sharing J. Rank truncation requires its own
    spectral derivative and must be handled before selecting this graph.
    """
    if any(type(n) is not int or n < 1 for n in (auxiliaries, pairs)):
        raise ValueError("Coulomb fit response requires positive dimensions")
    aux = IndexSpace("fitting_basis", "auxiliary", auxiliaries)
    a, b = Index("a", aux), Index("b", aux)
    p = Index("p", IndexSpace("orbital_pairs", "pair", pairs))
    whitener = input_tensor("metric_whitener", TensorSpec((a, b), role="input"))
    raw = input_tensor("three_center", TensorSpec((b, p), role="input"))
    seed = input_tensor("fit_bar", TensorSpec((a, p), role="input"))

    def solve(rhs: Node) -> Node:
        whitened = einsum("ab,bp->ap", whitener, rhs)
        return einsum("ab,ap->bp", whitener, whitened)

    fit = solve(raw)
    raw_bar = solve(seed)
    # dD = J^{-1}(dG - dJ D). Both triangles describe the same metric.
    weight = einsum("ap,bp->ab", fit, raw_bar)
    metric_bar = add(weight, transpose(weight, (1, 0)), coefficients=("-1/2", "-1/2"))
    return Program(
        {"fit": fit, "three_center_bar": raw_bar, "metric_bar": metric_bar},
        provenance={"operation": "full-rank Coulomb-fit implicit response"},
    )
