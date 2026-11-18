"""UHF mixed-spin SP geminal and orbital-space Hylleraas functional."""

from fractions import Fraction

from ..tensor.ir import (
    Node,
    add,
    broadcast,
    constant,
    einsum,
    input_tensor,
    multiply,
    power,
    slice_tensor,
)
from ..tensor.program import Program
from ..tensor.types import Index, IndexSpace, TensorSpec


def opposite_spin_projected_pair(na: int, nb: int, obs: int, extended: int) -> Program:
    """Unordered alpha-i/beta-j pair in different frames spanning the same space.

    Inputs are orbital-frame Coulomb, f12, f12², Coulomb*f12 and half kinetic
    double-commutator tiles for the original external spin pair. Each tile is
    the full matrix <p_alpha q_beta|O|i_alpha j_beta>. The flipped-geminal
    column is recovered by interchanging electron coordinates and transforming
    both legs back to the physical spin frames using C_alpha.T S_AO C_beta.
    The spatial operator is invariant under electron exchange, including the
    scalar two-electron double commutator. Orbital/Fock and 3C metric sources
    must use a common CABS RI subspace and consistent geminal convention.

    Weights 3/8 (direct) and 1/8 (flipped) include the cusp amplitudes before
    the strong-orthogonality projection. There is exactly one conventional
    determinant per alpha-virtual/beta-virtual pair. All raw matrices remain
    differentiable. Nuclear response and the assembly of the source integral
    tiles are separate from this projected pair functional.
    """
    if any(type(n) is not int for n in (na, nb, obs, extended)) or not (
        0 < na < obs and 0 < nb < obs < extended
    ):
        raise ValueError("Both spins need occupied and virtual OBS and a common CABS")
    ls = IndexSpace("extended_alpha", "orbital", extended, spin="alpha")
    rs = IndexSpace("extended_beta", "orbital", extended, spin="beta")
    p, u = Index("p", ls), Index("u", ls)
    q, v = Index("q", rs), Index("v", rs)

    def field(name: str, *axes: Index) -> Node:
        return input_tensor(name, TensorSpec(axes, role="input", differentiable=True))

    i = input_tensor("select_i", TensorSpec((p,), role="input", differentiable=False))
    j = input_tensor("select_j", TensorSpec((q,), role="input", differentiable=False))
    sab = field("cross_overlap", p, q)
    fa, fb = field("fock_alpha", p, u), field("fock_beta", q, v)
    ka, kb = field("exchange_alpha", p, u), field("exchange_beta", q, v)

    def weight(column: Node) -> Node:
        # S_ab M.T S_ab; the two instances of S_ab share one source adjoint.
        flipped = einsum("pv,uv->pu", sab, column)
        flipped = einsum("pu,uq->pq", flipped, sab)
        return add(column, flipped, coefficients=(Fraction(3, 8), Fraction(1, 8)))

    def dot(a: Node, b: Node) -> Node:
        return einsum("pq,pq->", a, b)

    def action(a: Node, b: Node, t: Node) -> Node:
        return add(einsum("pu,uq->pq", a, t), einsum("pv,qv->pq", t, b))

    def mask(index: Index, start: int, end: int) -> Node:
        return constant(
            [int(start <= k < end) for k in range(extended)],
            TensorSpec((index,), role="constant"),
        )

    def rect(t: Node, l: Node, r: Node) -> Node:
        return einsum("pq,p,q->pq", t, l, r)

    w = weight(einsum("p,q->pq", i, j))
    g = field("coulomb", p, q)
    f = weight(field("geminal", p, q))
    ff = weight(field("squared_geminal", p, q))
    uf = weight(field("double_commutator", p, q))
    qf = add(
        rect(f, mask(p, na, extended), mask(q, nb, extended)),
        rect(f, mask(p, na, obs), mask(q, nb, obs)),
        coefficients=(1, -1),
    )
    pf = add(f, qf, coefficients=(1, -1))
    linear = add(
        dot(field("geminal_coulomb", p, q), w), dot(g, pf), coefficients=(1, -1)
    )
    norm = add(dot(w, ff), dot(f, pf), coefficients=(1, -1))
    # f F f = half the local kinetic double commutator, the symmetric
    # f²(F+K) term, and the negative exchange expectation. Use QFQ-F to
    # remove the parts in the complement without forming a two-particle F.
    b = add(
        dot(w, uf),
        dot(ff, action(add(fa, ka), add(fb, kb), w)),
        dot(f, action(ka, kb, f)),
        dot(qf, action(fa, fb, qf)),
        dot(f, action(fa, fb, f)),
        coefficients=(1, 1, -1, 1, -1),
    )
    ranges = ((na, obs), (nb, obs))
    coupling = slice_tensor(action(fa, fb, qf), ranges)
    gvv = slice_tensor(g, ranges)
    sum_occupied = add(einsum("p,pu,u->", i, fa, i), einsum("q,qv,v->", j, fb, j))
    a_e = slice_tensor(einsum("pp->p", fa), ((na, obs),))
    b_e = slice_tensor(einsum("qq->q", fb), ((nb, obs),))
    ab = coupling.spec.indices
    gap = add(
        broadcast(a_e, ab, (0,)),
        broadcast(b_e, ab, (1,)),
        broadcast(sum_occupied, ab, ()),
        coefficients=(1, 1, -1),
    )
    d = power(gap, -1)
    energy = add(
        linear,
        b,
        multiply(sum_occupied, norm),
        dot(multiply(coupling, d), add(gvv, coupling, coefficients=(2, 1))),
        coefficients=(2, 1, -1, -1),
    )
    conventional = add(dot(multiply(gvv, d), gvv), coefficients=(-1,))
    return Program(
        {"pair_energy": energy, "ordinary_pair_energy": conventional},
        provenance={
            "method": "UHF opposite-spin SP 3C(FIX), explicit spin-flipped geminal",
            "source_frame": "equal extended subspace, spin-dependent occupied/virtual partitions",
            "units": "Hartree; one ordered alpha/beta pair",
        },
    )
