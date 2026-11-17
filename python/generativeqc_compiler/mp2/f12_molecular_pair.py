"""Restricted 3C(FIX) contraction and response for one molecular occupied pair."""

from ..tensor.ir import (
    Node,
    add,
    broadcast,
    einsum,
    input_tensor,
    power,
    slice_tensor,
    transpose,
)
from ..tensor.program import Program
from ..tensor.types import Index, IndexSpace, TensorSpec
from .f12_density_fitting import robust_fitted_operator
from .f12_pairs import restricted_pair_functional


def _dot(left: Node, right: Node) -> Node:
    return einsum("xy,xy->", left, right)


def _fitted_blocks(
    orbital: IndexSpace, auxiliaries: int, occupied: int, obs: int, diagonal: bool
) -> dict[str, Node]:
    ri = orbital.size
    aux = Index("a", IndexSpace("fitting_basis", "auxiliary", auxiliaries))
    orb = Index("x", orbital)
    inputs: dict[str, Node] = {}

    def source(name: str, axes: tuple[Index, ...]) -> Node:
        if name not in inputs:
            inputs[name] = input_tensor(
                name, TensorSpec(axes, role="input", differentiable=True)
            )
        return inputs[name]

    def sides(name: str) -> tuple[Node, Node]:
        left = source(name + "_i", (aux, orb))
        return left, left if diagonal else source(name + "_j", (aux, orb))

    def columns(value: Node, lo: int, hi: int) -> Node:
        if (lo, hi) == (0, ri):
            return value
        return slice_tensor(value, ((0, auxiliaries), (lo, hi)))

    dl, dr = sides("fit")
    result = {}
    for name, operator, (a, b), (c, d) in (
        ("geminal", "f12", (0, ri), (0, ri)),
        ("squared_occ_ri", "f12_squared", (0, occupied), (0, ri)),
        ("squared_ri_occ", "f12_squared", (0, ri), (0, occupied)),
        ("geminal_coulomb", "f12_over_coulomb", (0, occupied), (0, occupied)),
        ("double_commutator", "double_commutator", (0, occupied), (0, occupied)),
        ("coulomb_obs", "coulomb", (0, obs), (0, obs)),
        ("coulomb_oc", "coulomb", (0, occupied), (obs, ri)),
        ("coulomb_co", "coulomb", (obs, ri), (0, occupied)),
    ):
        lo, ro = sides("raw_" + operator)
        left, right = columns(dl, a, b), columns(ro, c, d)
        if operator == "coulomb":
            result[name] = einsum("al,ar->lr", left, right)
        else:
            metric = source("metric_" + operator, (aux, Index("b", aux.space)))
            result[name] = robust_fitted_operator(
                left, columns(lo, a, b), columns(dr, c, d), right, metric
            )
    return result


def extended_df_fock(occupied: int, orbitals: int, auxiliaries: int) -> Program:
    """Build F and positive exchange from occupied-only Coulomb fits.

    D[A,i,p] solves J D = G for occupied i and all extended-frame p. Contract
    the occupied trace of D before forming Coulomb; exchange also needs only
    this occupied slice. No virtual/virtual or CABS/CABS fitted densities are
    formed. Raw G[A,p,q] and D are independent differentiable inputs here;
    the metric owner supplies the implicit response of D to J and G.
    """
    if (
        any(type(x) is not int or x < 1 for x in (occupied, orbitals, auxiliaries))
        or occupied >= orbitals
    ):
        raise ValueError(
            "Positive auxiliary, occupied and complementary dimensions are required"
        )
    axis = IndexSpace("extended_orbital", "orbital", orbitals)
    p, q = Index("p", axis), Index("q", axis)
    o = Index("i", axis, 0, occupied)
    a = Index("a", IndexSpace("fitting_basis", "auxiliary", auxiliaries))

    def source(name: str, axes: tuple[Index, ...]) -> Node:
        return input_tensor(name, TensorSpec(axes, role="input", differentiable=True))

    h, fit, raw = (
        source("hcore", (p, q)),
        source("fit", (a, o, q)),
        source("raw_coulomb_all", (a, p, q)),
    )
    tr = einsum(
        "aii->a", slice_tensor(fit, ((0, auxiliaries), (0, occupied), (0, occupied)))
    )
    coul = einsum("apq,a->pq", raw, tr, coefficient=2)
    occ_raw = slice_tensor(raw, ((0, auxiliaries), (0, occupied), (0, orbitals)))
    exchange = einsum("aip,aiq->pq", fit, occ_raw)
    return Program(
        {"fock": add(h, coul, exchange, coefficients=(1, 1, -1)), "exchange": exchange},
        provenance={
            "method": "restricted extended-frame Coulomb-fitted Fock",
            "fitted_density_domain": "occupied by full OBS/CABS frame",
        },
    )


def projected_f12_pair(
    occupied: int,
    virtuals: int,
    cabs: int,
    *,
    diagonal: bool,
    auxiliaries: int | None = None,
) -> Program:
    """Assemble V, X, B, C and the spin-adapted increment from fitted MO blocks.

    The caller supplies one ordered external pair i <= j. The matrix for the
    reversed pair is its electron-coordinate transpose. The select_i/select_j
    vectors are fixed one-hot controls in the full occupied space. Frozen
    orbitals remain in that projector; the caller schedules only active pairs.
    Extended orbital order is occupied, orbital virtual, complementary virtual.
    With auxiliaries given, the graph consumes three-center factors directly.
    All live matrices, including the Fock matrix in the gap, are differentiable.
    """
    if type(diagonal) is not bool or any(
        type(n) is not int or n < 1 for n in (occupied, virtuals, cabs)
    ):
        raise ValueError(
            "Positive orbital spaces and a boolean diagonal flag are required"
        )
    obs = occupied + virtuals
    ri = obs + cabs
    orbital = IndexSpace("extended_orbital", "orbital", ri)
    occ, vir, orb, comp, full = (
        (0, occupied),
        (occupied, obs),
        (0, obs),
        (obs, ri),
        (0, ri),
    )

    if auxiliaries is not None and (type(auxiliaries) is not int or auxiliaries < 1):
        raise ValueError("Positive auxiliary dimension is required")
    projected = (
        {}
        if auxiliaries is None
        else _fitted_blocks(orbital, auxiliaries, occupied, obs, diagonal)
    )

    def feed(name: str, *bounds: tuple[int, int], control: bool = False) -> Node:
        if name in projected:
            return projected[name]
        axes = tuple(
            Index(label, orbital, start, stop)
            for label, (start, stop) in zip("xy", bounds, strict=False)
        )
        return input_tensor(
            name, TensorSpec(axes, role="input", differentiable=not control)
        )

    def view(value: Node, rows: tuple[int, int], columns: tuple[int, int]) -> Node:
        # All ranges here use absolute coordinates of the extended orbital space.
        x, y = value.spec.indices
        return slice_tensor(
            value,
            (
                (rows[0] - x.start, rows[1] - x.start),
                (columns[0] - y.start, columns[1] - y.start),
            ),
        )

    def swapped(value: Node) -> Node:
        return transpose(value, (1, 0))

    def selected(matrix: Node, left: Node, right: Node) -> Node:
        return einsum("x,x->", einsum("xy,y->x", matrix, right), left)

    i = feed("select_i", occ, control=True)
    j = i if diagonal else feed("select_j", occ, control=True)
    fock, exchange = feed("fock", full, full), feed("exchange", full, full)
    gem = feed("geminal", full, full)
    gem_t = swapped(gem)
    f2_left = feed("squared_occ_ri", occ, full)
    f2_right = feed("squared_ri_occ", full, occ)
    fg, uf = feed("geminal_coulomb", occ, occ), feed("double_commutator", occ, occ)
    g_obs = feed("coulomb_obs", orb, orb)
    g_oc, g_co = feed("coulomb_oc", occ, comp), feed("coulomb_co", comp, occ)
    f_orb, f_oc, f_co = view(gem, orb, orb), view(gem, occ, comp), view(gem, comp, occ)
    fk = add(view(fock, occ, full), view(exchange, occ, full))
    f_vc = view(fock, vir, comp)
    coupling = add(
        einsum("aq,bq->ab", view(gem, vir, comp), f_vc),
        einsum("aq,qb->ab", f_vc, view(gem, comp, vir)),
    )
    f_oo, f_pp = view(fock, occ, occ), view(fock, orb, orb)
    occupied_sum = add(selected(f_oo, i, i), selected(f_oo, j, j))
    eps = einsum("aa->a", view(fock, vir, vir))
    vv = coupling.spec.indices
    gap = add(
        broadcast(eps, vv, (0,)),
        broadcast(eps, vv, (1,)),
        broadcast(occupied_sum, vv, ()),
        coefficients=(1, 1, -1),
    )
    terms = {
        "c_direct": coupling,
        "occupied_fock_sum": occupied_sum,
        "inverse_gap": power(gap, -1),
        "g": view(g_obs, vir, vir),
    }

    def b_left(f: Node) -> tuple[Node, ...]:
        f_vo = view(f, vir, orb)
        return (
            einsum("xs,sy->xy", f, exchange),
            einsum("xs,sy->xy", view(f, occ, full), fock),
            einsum("xs,sy->xy", view(f, comp, occ), f_oo),
            einsum("xs,sy->xy", f_vo, f_pp),
            einsum("xs,ys->xy", view(f, comp, full), view(fock, occ, full)),
            einsum("xs,sy->xy", f_vo, view(fock, orb, comp)),
        )

    def b_overlap(actions: tuple[Node, ...], other: Node) -> Node:
        blocks = (
            other,
            view(other, occ, full),
            view(other, comp, occ),
            view(other, vir, orb),
            view(other, comp, occ),
            view(other, vir, comp),
        )
        return add(
            *(_dot(x, y) for x, y in zip(actions, blocks, strict=True)),
            coefficients=(-1, -1, 1, -1, -2, -2),
        )

    left, left_t = b_left(gem), b_left(gem_t)
    targets = [("direct", i, j, gem, gem_t)]
    if not diagonal:
        terms["c_exchange"] = swapped(coupling)
        targets.append(("exchange", j, i, gem_t, gem))
    for suffix, m, n, rhs, rhs_t in targets:
        ro = view(rhs, orb, orb)
        roc, rco = view(rhs, occ, comp), view(rhs, comp, occ)
        terms["v_" + suffix] = add(
            selected(fg, m, n),
            _dot(g_obs, ro),
            _dot(g_oc, roc),
            _dot(g_co, rco),
            coefficients=(1, -1, -1, -1),
        )
        terms["x_" + suffix] = add(
            selected(view(f2_left, occ, occ), m, n),
            _dot(f_orb, ro),
            _dot(f_oc, roc),
            _dot(f_co, rco),
            coefficients=(1, -1, -1, -1),
        )
        # For the two required columns the electron-swapped companion also
        # supplies the Hermitian bra/ket partner for the exchange element.
        squared_m = einsum("x,xy->y", m, f2_left)
        squared_n = einsum("xy,y->x", f2_right, n)
        fk_m = einsum("x,xy->y", m, fk)
        fk_n = einsum("x,xy->y", n, fk)
        terms["b_" + suffix] = add(
            selected(uf, m, n),
            einsum("a,a->", squared_m, fk_n),
            einsum("a,a->", squared_n, fk_m),
            b_overlap(left, rhs),
            b_overlap(left_t, rhs_t),
        )
    return Program(
        {**terms, "pair_energy": restricted_pair_functional(terms, diagonal=diagonal)},
        provenance={
            "method": "restricted DF-MP2-F12/3C(FIX) ordered-pair projection",
            "basis": "orthonormal OBS plus CABS",
            "units": "Hartree for pair_energy",
        },
    )
