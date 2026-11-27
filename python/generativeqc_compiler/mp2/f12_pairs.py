"""Restricted 3C(FIX) Hylleraas correction for an unordered occupied-pair tile."""

from collections.abc import Mapping
from fractions import Fraction

from ..tensor.ad_program import transpose_program
from ..tensor.ir import Node, add, input_tensor, multiply, reduce_sum
from ..tensor.program import Program
from ..tensor.types import Index, IndexSpace, TensorSpec


def restricted_pair_functional(terms: Mapping[str, Node], *, diagonal: bool) -> Node:
    """Spin-adapted 3C(FIX) energy for a pair or pair batch with two trailing virtual axes."""
    c = terms["c_direct"]
    virtual_axes = (len(c.spec.indices) - 2, len(c.spec.indices) - 1)
    d = terms["inverse_gap"]
    gd = multiply(terms["g"], d)
    cd = multiply(c, d)
    f = terms["occupied_fock_sum"]

    def dressed(suffix: str, coupling: Node) -> tuple[Node, Node]:
        linear = add(
            terms["v_" + suffix],
            reduce_sum(multiply(coupling, gd), virtual_axes),
            coefficients=(1, -1),
        )
        quadratic = add(
            terms["b_" + suffix],
            multiply(f, terms["x_" + suffix]),
            reduce_sum(multiply(coupling, cd), virtual_axes),
            coefficients=(1, -1, -1),
        )
        return linear, quadratic

    vd, bd = dressed("direct", c)
    if diagonal:
        energy = add(vd, bd, coefficients=(1, Fraction(1, 4)))
    else:
        vx, bx = dressed("exchange", terms["c_exchange"])
        # Spin singlet plus three triplet components for a spatial i < j pair.
        energy = add(
            vd,
            vx,
            bd,
            bx,
            coefficients=(
                Fraction(5, 2),
                Fraction(-1, 2),
                Fraction(7, 16),
                Fraction(1, 16),
            ),
        )
    return energy


def restricted_f12_pairs(pairs: int, virtuals: int, *, diagonal: bool) -> Program:
    """Evaluate the diagonal or off-diagonal subset of i <= j in real spatial MOs.

    V, X and B enter through their ijij and ijji elements. C(ex) is C_jiab,
    in the same virtual order as C_ijab. Each inverse positive MP2 denominator
    D_ijab belongs to that external pair, in both the linear and quadratic
    eliminations. The input Fock sum belongs to that pair too. An unordered
    pair's spin and permutation multiplicities are included exactly once.
    """
    if type(diagonal) is not bool or any(
        type(n) is not int or n < 1 for n in (pairs, virtuals)
    ):
        raise ValueError(
            "F12 pair tiles require positive integer dimensions and a boolean diagonal flag"
        )
    p = Index("p", IndexSpace("occupied_pair_tile", "pair", pairs))
    vir = IndexSpace("orbital_virtual", "virtual", virtuals)
    pab = (p, Index("a", vir), Index("b", vir))

    def feed(name: str, indices: tuple[Index, ...]) -> Node:
        return input_tensor(
            name,
            TensorSpec(
                indices,
                role="input",
                representation="restricted_spatial",
                differentiable=True,
            ),
        )

    terms = {name: feed(name, pab) for name in ("c_direct", "inverse_gap", "g")}
    terms["occupied_fock_sum"] = feed("occupied_fock_sum", (p,))
    suffixes = ("direct",) if diagonal else ("direct", "exchange")
    for suffix in suffixes:
        for key in ("v", "x", "b"):
            name = f"{key}_{suffix}"
            terms[name] = feed(name, (p,))
    if not diagonal:
        terms["c_exchange"] = feed("c_exchange", pab)
    energy = restricted_pair_functional(terms, diagonal=diagonal)
    return Program(
        {"pair_energy": energy},
        provenance={
            "method": "restricted MP2-F12/3C(FIX) pair correction",
            "units": "Hartree",
            "diagonal_pairs": diagonal,
            "reference": "real semicanonical spatial orbitals",
            "inputs": "same-frame V, X, B, C and Coulomb intermediates",
        },
    )


def restricted_f12_pair_response(
    pairs: int, virtuals: int, *, diagonal: bool
) -> Program:
    """Use the shared TensorIR reverse AD for all live intermediate weights."""
    return transpose_program(
        restricted_f12_pairs(pairs, virtuals, diagonal=diagonal), ("pair_energy",)
    ).program
