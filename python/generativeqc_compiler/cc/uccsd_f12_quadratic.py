"""Common B and occupied-Fock/X sources of a fixed-cusp UCCSD-F12 pair."""

from fractions import Fraction

from ..common.f12_spin import pair_frame_spins
from ..tensor.ir import (
    Node,
    add,
    constant,
    einsum,
    input_tensor,
    scatter_add,
    slice_tensor,
    transpose,
)
from ..tensor.program import Program
from ..tensor.robust_density_fit import robust_fitted_operator
from ..tensor.types import Index, IndexSpace, TensorSpec


def unrestricted_fixed_geminal_quadratic_pair(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    auxiliaries: int,
    *,
    channel: str,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> Program:
    """Evaluate B and the complete occupied-Fock contraction of X for one pair.

    The two frames share the OBS, CABS and auxiliary spaces. Selectors are
    unit vectors in the respective active occupied subspaces. Selected fit and
    raw columns span each full extended frame. Frozen cores remain in Q and in
    the supplied full Fock and positive exchange, but the X contraction uses
    the active occupied Fock only. The occupied Fock need not be diagonal.

    For one unique pair ij, the returned ``b_energy`` is the quadratic
    Hylleraas contribution of the fixed geminal. ``x_fock_energy`` is
    <χ_ij | f Q f | χ_(F_oo i),j + χ_i,(F_oo j)>, with the same
    spin normalization. Add ``b_energy - x_fock_energy`` to the linear V
    source before solving the fixed-geminal CC amplitude equations.
    """
    sl, sr = pair_frame_spins(left, right, channel)
    no, nv, nc = left
    nr, vr, cr = right
    obs, ext = no + nv, no + nv + nc
    fl, fr = frozen_occupied
    same_spin = sl == sr
    if type(auxiliaries) is not int or auxiliaries < 1:
        raise ValueError("A positive common auxiliary size is required")
    if obs != nr + vr or nc != cr:
        raise ValueError("Both spins must span the same OBS and CABS")
    if not (type(fl) is type(fr) is int and 0 <= fl < no and 0 <= fr < nr):
        raise ValueError("Both legs need active occupied orbitals")
    if same_spin and (fl != fr or no - fl < 2):
        raise ValueError("A same-spin pair needs two active orbitals in one frozen space")

    aux = IndexSpace("f12_fitting_basis", "auxiliary", auxiliaries)
    ls = IndexSpace("extended_" + sl, "orbital", ext, spin=sl)
    rs = IndexSpace("extended_" + sr, "orbital", ext, spin=sr)
    p, q = Index("p", ls), Index("q", rs)
    a, b = Index("a", aux), Index("b", aux)

    def field(name: str, *axes: Index, differentiable: bool = True) -> Node:
        return input_tensor(name, TensorSpec(axes, role="input", differentiable=differentiable))

    def leg(
        suffix: str, frame: IndexSpace, first: int, count: int
    ) -> tuple[Index, Node, Node, dict[str, Node]]:
        active = Index("i", frame, first, count)
        col = Index("r", frame)
        row = Index("s", frame)
        fock = field("fock_" + suffix, col, row)
        exchange = field("exchange_" + suffix, col, row)
        factors = {
            k: field(k + "_" + suffix, a, active, col)
            for k in ("fit", "raw_geminal", "raw_squared_geminal", "raw_double_commutator")
        }
        return active, fock, exchange, factors

    l, fock_l, exchange_l, factors_l = leg("left", ls, fl, no)
    r, fock_r, exchange_r, factors_r = (
        (l, fock_l, exchange_l, factors_l) if same_spin else leg("right", rs, fr, nr)
    )
    il = field("select_i", l, differentiable=False)
    ir = field("select_j", r, differentiable=False)
    s = None if same_spin else field("cross_overlap", p, q)

    def dress(matrix: Node) -> Node:
        if s is None:
            return add(matrix, transpose(matrix, (1, 0)), coefficients=("1/4", "-1/4"))
        transposed_spin = einsum("pv,uv->pu", s, matrix)
        return add(matrix, einsum("pu,uq->pq", transposed_spin, s),
                   coefficients=(Fraction(3, 8), Fraction(1, 8)))

    def columns(factors: dict[str, Node], selection: Node) -> dict[str, Node]:
        return {k: einsum("aip,i->ap", node, selection) for k, node in factors.items()}

    cl, cr = columns(factors_l, il), columns(factors_r, ir)
    mid = {op: field("metric_" + op, a, b) for op in ("geminal", "squared_geminal", "double_commutator")}

    def fitted(k: str, tile_l: dict[str, Node], tile_r: dict[str, Node]) -> Node:
        return robust_fitted_operator(tile_l["fit"], tile_l["raw_" + k],
                                      tile_r["fit"], tile_r["raw_" + k], mid[k])

    def support(axis: Index, start: int, end: int) -> Node:
        return constant([int(start <= k < end) for k in range(ext)], TensorSpec((axis,), role="constant"))

    def mask(matrix: Node, lm: Node, rm: Node) -> Node:
        return einsum("pq,p,q->pq", matrix, lm, rm)

    def strong(matrix: Node) -> Node:
        return add(mask(matrix, support(p, no, ext), support(q, nr, ext)),
                   mask(matrix, support(p, no, obs), support(q, nr, obs)), coefficients=(1, -1))

    def action(a_l: Node, a_r: Node, matrix: Node) -> Node:
        return add(einsum("pu,uq->pq", a_l, matrix), einsum("pv,qv->pq", matrix, a_r))

    def dot(x: Node, y: Node) -> Node:
        return einsum("pq,pq->", x, y)

    il_full = scatter_add(il, 0, range(fl, no), p)
    ir_full = scatter_add(ir, 0, range(fr, nr), q)
    chi = dress(einsum("p,q->pq", il_full, ir_full))
    f = dress(fitted("geminal", cl, cr))
    ff = dress(fitted("squared_geminal", cl, cr))
    uf = dress(fitted("double_commutator", cl, cr))
    qf = strong(f)
    b_energy = add(
        dot(chi, uf),
        dot(ff, action(add(fock_l, exchange_l), add(fock_r, exchange_r), chi)),
        dot(f, action(exchange_l, exchange_r, f)),
        dot(qf, action(fock_l, fock_r, qf)),
        dot(f, action(fock_l, fock_r, f)),
        coefficients=(1, 1, -1, 1, -1),
    )

    # Act on the selected pair before dressing its cusp. Thus the derivative
    # retains off-diagonal active occupied Fock entries in each spin frame.
    fl_selected = einsum("ij,j->i", slice_tensor(fock_l, ((fl, no), (fl, no))), il)
    fr_selected = einsum("ij,j->i", slice_tensor(fock_r, ((fr, nr), (fr, nr))), ir)
    xl, xr = columns(factors_l, fl_selected), columns(factors_r, fr_selected)

    def fock_dressed(k: str) -> Node:
        return dress(add(fitted(k, xl, cr), fitted(k, cl, xr)))

    x = add(dot(chi, fock_dressed("squared_geminal")),
            dot(add(f, qf, coefficients=(1, -1)), fock_dressed("geminal")),
            coefficients=(1, -1))

    # The like-spin spatial antisymmetric geminal on an ordered tensor has
    # twice the squared norm of the normalized triplet determinant convention.
    normalization = Fraction(1, 2) if same_spin else 1
    return Program(
        {"b_energy": add(b_energy, coefficients=(normalization,)),
         "x_fock_energy": add(x, coefficients=(normalization,))},
        provenance={"operation": "fixed-cusp UCCSD-F12b common B and occupied-Fock/X",
                    "channel": channel, "frozen_occupied": frozen_occupied},
    )
