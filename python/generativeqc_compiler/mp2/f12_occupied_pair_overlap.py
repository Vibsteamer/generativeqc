"""Strong-orthogonality projected geminal rows linking distinct restricted occupied pairs."""

from ..tensor.ir import Node, add, einsum, input_tensor, reshape, slice_tensor
from ..tensor.program import Program
from ..tensor.types import Index, IndexSpace, TensorSpec
from .f12_molecular_pair import _fitted_blocks


def occupied_pair_vx(
    occupied: int,
    virtual: int,
    cabs: int,
    auxiliary: int,
    *,
    bra_diagonal: bool,
    ket_diagonal: bool,
) -> Program:
    """Compute V_mn,kl and X_mn,kl, including independent bra and ket pairs.

    V inserts Coulomb on the bra and the geminal on the ket. X inserts the
    geminal on both sides. Resolve the identity in OBS plus CABS; Q12 removes
    OBS×OBS and the two occupied×CABS exchange-related sectors. The projector
    uses all occupied orbitals, including inactive orbitals if present.
    Separate bra and ket factors share physical operator metrics. When one MO
    appears on both sides, accumulate all returned weights on that orbital.
    """
    if any(type(n) is not int or n < 1 for n in (occupied, virtual, cabs, auxiliary)):
        raise ValueError(
            "positive occupied, virtual, complementary and auxiliary sizes required"
        )
    if any(type(k) is not bool for k in (bra_diagonal, ket_diagonal)):
        raise ValueError("both pair selection flags must be booleans")
    obs = occupied + virtual
    extended = obs + cabs
    space = IndexSpace("extended_orbital", "orbital", extended)
    bra = _fitted_blocks(
        space, auxiliary, occupied, obs, bra_diagonal, pair_prefix="bra_"
    )
    ket = _fitted_blocks(
        space, auxiliary, occupied, obs, ket_diagonal, pair_prefix="ket_"
    )
    i = Index("x", space, 0, occupied)
    control = TensorSpec((i,), role="input", differentiable=False)
    k = input_tensor("select_k", control)
    l = k if ket_diagonal else input_tensor("select_l", control)

    def select(matrix: Node) -> Node:
        return einsum("x,x->", einsum("xy,y->x", matrix, l), k)

    def block(g: Node, first: tuple[int, int], second: tuple[int, int]) -> Node:
        return slice_tensor(g, (first, second))

    sectors = (
        ((0, obs), (0, obs)),
        ((0, occupied), (obs, extended)),
        ((obs, extended), (0, occupied)),
    )
    right = tuple(block(ket["geminal"], *s) for s in sectors)
    f = tuple(block(bra["geminal"], *s) for s in sectors)
    g = (bra["coulomb_obs"], bra["coulomb_oc"], bra["coulomb_co"])

    def project(direct: Node, left: tuple[Node, ...]) -> Node:
        products = tuple(
            einsum("xy,xy->", a, b) for a, b in zip(left, right, strict=True)
        )
        return add(select(direct), *products, coefficients=(1, -1, -1, -1))

    direct_square = block(bra["squared_occ_ri"], (0, occupied), (0, occupied))
    return Program(
        {"v": project(bra["geminal_coulomb"], g), "x": project(direct_square, f)},
        provenance={
            "operation": "off-diagonal occupied-pair projected V and X",
            "density_fitting": "robust real spatial geminal and operator product; Coulomb metric for all coefficient fits",
        },
    )


def all_occupied_vx(occupied: int, virtual: int, cabs: int, auxiliary: int) -> Program:
    """Complete small-molecule occupied V/X tables on shared DF sources.

    Reuse the geminal pair panels for all bra and ket pairs. The largest
    intermediate contains (occupied,occupied,extended,extended) entries.
    For larger systems the per-pair constructor limits working memory.
    """
    if any(type(n) is not int or n < 1 for n in (occupied, virtual, cabs, auxiliary)):
        raise ValueError(
            "positive occupied, virtual, complementary and auxiliary sizes required"
        )
    o = occupied
    n = o + virtual
    r = n + cabs
    aux = IndexSpace("auxiliary", "auxiliary", auxiliary)
    occ = IndexSpace("occupied", "occupied", o)
    ext = IndexSpace("extended", "orbital", r)
    p, q = Index("p", aux), Index("q", aux)
    i, j = Index("i", occ), Index("j", ext)

    def source(name: str, *axes: Index) -> Node:
        return input_tensor(name, TensorSpec(axes, role="input", differentiable=True))

    d = source("fit", p, i, j)
    cn = source("raw_coulomb", p, i, j)

    def robust(name: str, end: int) -> Node:
        a = source("raw_" + name, p, i, j)
        k = source("metric_" + name, p, q)
        ds = d
        if end < r:
            ds = slice_tensor(ds, ((0, auxiliary), (0, o), (0, end)))
            a = slice_tensor(a, ((0, auxiliary), (0, o), (0, end)))
        kd = einsum("PQ,Qjs->Pjs", k, ds)
        return add(
            einsum("Pir,Pjs->ijrs", ds, a),
            einsum("Pir,Pjs->ijrs", a, ds),
            einsum("Pir,Pjs->ijrs", ds, kd),
            coefficients=(1, 1, -1),
        )

    gf = robust("f12_over_coulomb", o)
    f2 = robust("f12_squared", o)
    g = einsum("Pir,Pjs->ijrs", d, cn)
    f = robust("f12", r)
    sectors = (((0, n), (0, n)), ((0, o), (n, r)), ((n, r), (0, o)))

    def corr(x: Node, y: Node) -> Node:
        return einsum("ijab,klab->ijkl", x, y)

    def project(direct: Node, left: Node) -> Node:
        sums = []
        for x, y in sectors:
            bounds = ((0, o), (0, o), x, y)
            sums.append(corr(slice_tensor(left, bounds), slice_tensor(f, bounds)))
        # The occupied segment of the extended-frame index is the active
        # occupied index in the full-pair product at this closed-shell endpoint.
        direct = reshape(direct, sums[0].spec.indices)
        return add(direct, *sums, coefficients=(1, -1, -1, -1))

    return Program(
        {"v": project(gf, g), "x": project(f2, f)},
        provenance={
            "operation": "full multi-occupied projected geminal V/X tables; shared robust fitting and occupied pair panels"
        },
    )
