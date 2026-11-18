"""Spin-contracted all-pair restricted CC-F12 energies without building the fixed cusp tensor."""

from fractions import Fraction

from ..tensor.ir import Node, add, einsum, input_tensor, transpose
from ..tensor.program import Program
from ..tensor.types import Index, IndexSpace, TensorSpec


def all_pair_geminal_energy(occupied: int, virtual: int) -> Program:
    """Common geminal energy and the F12b correction at variable t1/t2 (Knizia 4.125–127).

    The fixed T maps a row to 3/8 times the row plus 1/8 times its transpose.
    After spin summation W = 2T - T(exchange) gives 5/8 and -1/8. Both are
    applied algebraically before contraction. E.g. C_ijab only contracts the
    matching ijab and jiab amplitudes. No all-pair-by-all-pair virtual sum
    and no occupied selector vectors are required.
    """
    if any(type(n) is not int or n < 1 for n in (occupied, virtual)):
        raise ValueError("positive occupied and virtual spaces required")
    occ = IndexSpace("occupied", "occupied", occupied)
    vir = IndexSpace("virtual", "virtual", virtual)
    m, n, i, j = (Index(s, occ) for s in "mnij")
    a, b = (Index(s, vir) for s in "ab")

    def source(name: str, *axes: Index) -> Node:
        return input_tensor(
            name,
            TensorSpec(
                axes,
                role="input",
                differentiable=True,
                representation="restricted_spatial",
            ),
        )

    def contract_occupied(row: Node) -> Node:
        return add(
            einsum("mnmn->", row),
            einsum("mnnm->", row),
            coefficients=(Fraction(5, 8), Fraction(-1, 8)),
        )

    def weighted_pairs(amplitude: Node) -> Node:
        return add(
            amplitude,
            transpose(amplitude, (1, 0, 2, 3)),
            coefficients=(Fraction(5, 8), Fraction(-1, 8)),
        )

    v, x, b_row = (source(key, m, n, i, j) for key in ("v", "x", "b"))
    f, t1 = source("foo", i, j), source("t1", i, a)
    c, t2, vv = (source(key, i, j, a, b) for key in ("c", "t2", "v_vv"))
    vo, ov = source("v_vo", m, n, a, j), source("v_ov", m, n, i, b)
    projected_x = add(
        x, transpose(x, (0, 1, 3, 2)), coefficients=(Fraction(3, 8), Fraction(1, 8))
    )
    common = add(
        contract_occupied(v),
        einsum("mnmn->", b_row),
        einsum("mnnm->", b_row),
        contract_occupied(einsum("ik,mnkj->mnij", f, projected_x)),
        contract_occupied(einsum("mnik,jk->mnij", projected_x, f)),
        einsum("ijab,ijab->", c, weighted_pairs(t2)),
        coefficients=(2, Fraction(7, 32), Fraction(1, 32), -1, -1, 1),
    )
    tau = add(t2, einsum("ia,jb->ijab", t1, t1))
    extra = add(
        einsum("mnab,mnab->", vv, weighted_pairs(tau)),
        einsum("mnmb,nb->", ov, t1),
        einsum("mnnb,mb->", ov, t1),
        einsum("ma,mnan->", t1, vo),
        einsum("na,mnam->", t1, vo),
        coefficients=(
            1,
            Fraction(5, 8),
            Fraction(-1, 8),
            Fraction(5, 8),
            Fraction(-1, 8),
        ),
    )
    return Program(
        {"f12a_geminal_energy": common, "f12b_extra_energy": extra},
        provenance={
            "operation": "restricted spin-contracted CC-F12 common and extra geminal energies",
            "reference": "Knizia Eqs. (4.125)–(4.127)",
        },
    )
