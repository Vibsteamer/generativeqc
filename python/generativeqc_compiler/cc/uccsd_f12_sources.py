"""Spin-resolved molecular source contractions for fixed-cusp UCCSD-F12."""

from fractions import Fraction

from ..common.f12_spin import pair_frame_spins
from ..tensor.ir import Node, add, einsum, input_tensor, slice_tensor, transpose
from ..tensor.program import Program
from ..tensor.robust_density_fit import robust_fitted_operator
from ..tensor.types import Index, IndexSpace, TensorSpec
from .f12_residual import _projected_v


def unrestricted_f12_pair_sources(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    auxiliaries: int,
    *,
    channel: str,
) -> Program:
    """Build W, C and the linear occupied V source for one active UHF pair.

    Each space gives occupied, virtual and CABS counts. The two spin frames
    span the same OBS and RI spaces. The caller selects each distinct
    same-spin pair once (i<j) and each alpha/beta pair once. The robust DF
    operators use a shared auxiliary basis and Coulomb fit. ``fit_left`` and
    ``fit_right`` are the fitted occupied-by-RI columns for this pair,
    and ``fit_obs_left`` and ``raw_coulomb_obs_right`` are the OBS density
    tiles for the standard external projection. The fixed one-hot selections
    identify the occupied bra in both frames.

    W and C carry the cusp factor and have target indices in OBS×OBS and
    orbital-virtual×orbital-virtual, respectively. The same-spin tiles are
    already antisymmetric. The opposite-spin permutation signs belong to the
    block assembler. ``linear_v_energy`` is twice the occupied matrix
    element with the strong-orthogonality projector, counted once per pair.
    The common geminal energy additionally needs B and occupied-Fock/X.
    """
    spin_left, spin_right = pair_frame_spins(left, right, channel)
    no_left, nv_left, nc_left = left
    no_right, nv_right, nc_right = right
    obs = no_left + nv_left
    extended = obs + nc_left
    if obs != no_right + nv_right or nc_left != nc_right:
        raise ValueError("Both spin frames must span the same OBS and CABS spaces")
    if type(auxiliaries) is not int or auxiliaries < 1:
        raise ValueError("A positive common fitting dimension is required")
    same_spin = spin_left == spin_right
    if same_spin and no_left < 2:
        raise ValueError("A same-spin geminal requires distinct occupied orbitals")

    ls = IndexSpace("extended_" + spin_left, "orbital", extended, spin=spin_left)
    rs = IndexSpace("extended_" + spin_right, "orbital", extended, spin=spin_right)
    aux = IndexSpace("f12_fitting_basis", "auxiliary", auxiliaries)
    p, u, r = Index("p", ls), Index("u", ls), Index("r", ls, 0, obs)
    q, v, s = Index("q", rs), Index("v", rs), Index("s", rs, 0, obs)
    a, b = Index("a", aux), Index("b", aux)

    def field(name: str, *axes: Index, differentiable: bool = True) -> Node:
        return input_tensor(
            name,
            TensorSpec(axes, role="input", differentiable=differentiable),
        )

    fit_left, fit_right = field("fit_left", a, p), field("fit_right", a, q)
    raw_coulomb = field("raw_coulomb_right", a, q)
    select_i = field("select_i", p, differentiable=False)
    select_j = field("select_j", q, differentiable=False)
    cross_overlap = None if same_spin else field("cross_overlap", p, q)

    def dressed(column: Node) -> Node:
        if cross_overlap is None:
            return add(column, transpose(column, (1, 0)), coefficients=("1/4", "-1/4"))
        swapped = einsum("pv,uv->pu", cross_overlap, column)
        swapped = einsum("pu,uq->pq", swapped, cross_overlap)
        return add(column, swapped, coefficients=(Fraction(3, 8), Fraction(1, 8)))

    def fitted_dressed(name: str) -> Node:
        return dressed(robust_fitted_operator(
            fit_left,
            field("raw_" + name + "_left", a, p),
            fit_right,
            field("raw_" + name + "_right", a, q),
            field("metric_" + name, a, b),
        ))

    orb, cabs = (0, obs), (obs, extended)
    lo, ro = (0, no_left), (0, no_right)
    lv, rv = (no_left, obs), (no_right, obs)

    def view(value: Node, rows: tuple[int, int], cols: tuple[int, int]) -> Node:
        return slice_tensor(value, (rows, cols))

    geminal = fitted_dressed("geminal")
    geminal_coulomb = fitted_dressed("geminal_coulomb")
    geminal_obs = view(geminal, orb, orb)
    w_obs = _projected_v(
        view(geminal_coulomb, orb, orb),
        geminal_obs,
        field("fit_obs_left", a, r, Index("t", ls, 0, obs)),
        field("raw_coulomb_obs_right", a, s, Index("t", rs, 0, obs)),
    )
    fock_left = field("fock_left", p, u)
    fock_right = fock_left if same_spin else field("fock_right", q, v)
    coupling = add(
        einsum("ac,cb->ab", view(fock_left, lv, cabs), view(geminal, cabs, rv)),
        einsum("ad,bd->ab", view(geminal, lv, cabs), view(fock_right, rv, cabs)),
    )

    # 1-Q has exactly these three disjoint blocks. Use the actual occupied
    # counts, including core occupations, even when only active pairs are scheduled.
    coulomb = einsum("ap,aq->pq", fit_left, raw_coulomb)
    complement = (geminal_obs, view(geminal, lo, cabs), view(geminal, cabs, ro))
    blocks = (view(coulomb, orb, orb), view(coulomb, lo, cabs), view(coulomb, cabs, ro))
    v_occupied = add(
        einsum("pq,p,q->", geminal_coulomb, select_i, select_j),
        *(einsum("pq,pq->", x, y) for x, y in zip(blocks, complement, strict=True)),
        coefficients=(2, -2, -2, -2),
    )
    return Program(
        {"w_obs": w_obs, "c_vv": coupling, "linear_v_energy": v_occupied},
        provenance={
            "operation": "molecular fixed-cusp UCCSD-F12b pair W, C and linear V",
            "channel": channel,
            "external_projection": "1-P_OBS,left(1)P_OBS,right(2)",
            "occupied_projection": "(1-O1)(1-O2)-V1 V2",
        },
    )


def unrestricted_standard_v_tile(
    left_obs: int,
    right_obs: int,
    auxiliary: int,
    left: int,
    right: int,
    *,
    left_spin: str,
    right_spin: str,
) -> Program:
    """Project an ordered geminal column onto two spin-resolved external tiles.

    For a supplied geminal product |χ>, evaluate <uv|g f|χ> minus
    sum_rs (ur|vs)<rs|f|χ>. The standard external projector includes
    the full OBS of each spin, including all occupied and virtual orbitals.
    The Coulomb fit on the left and the raw Coulomb integrals on the right
    use one shared fitting basis. The free indices may address different
    occupied or virtual tiles, including for same-spin pairs.

    The caller can supply cusp-dressed direct and OBS geminal inputs because
    both enter linearly. The program adds no pair multiplicities or exchange
    terms. The occupied Hylleraas matrix requires the full strong-orthogonality
    projector and has a separate source.
    """
    if any(
        type(n) is not int or n < 1
        for n in (left_obs, right_obs, auxiliary, left, right)
    ):
        raise ValueError("positive OBS, auxiliary and free orbital dimensions required")
    if left_spin not in ("alpha", "beta") or right_spin not in ("alpha", "beta"):
        raise ValueError("external geminal legs require alpha or beta spin")
    if left_spin == right_spin and left_obs != right_obs:
        raise ValueError("same-spin geminal legs require one common OBS")

    r = Index("r", IndexSpace(f"f12_obs_{left_spin}", "orbital", left_obs, left_spin))
    s = Index(
        "s", IndexSpace(f"f12_obs_{right_spin}", "orbital", right_obs, right_spin)
    )
    p = Index("p", IndexSpace("f12_fitting_basis", "auxiliary", auxiliary))
    u = Index("u", IndexSpace("f12_left_tile", "orbital", left, left_spin))
    v = Index("v", IndexSpace("f12_right_tile", "orbital", right, right_spin))

    def source(name: str, *axes: Index) -> Node:
        return input_tensor(
            name,
            TensorSpec(
                axes, role="input", differentiable=True, representation="spin_orbital"
            ),
        )

    projected = _projected_v(
        source("geminal_coulomb", u, v),
        source("geminal_obs", r, s),
        source("fit_left", p, u, r),
        source("raw_coulomb_right", p, v, s),
    )
    return Program(
        {"v": projected},
        provenance={
            "operation": "spin-resolved standard projected CC-F12 V tile",
            "projection": "1-P_OBS,left(1)P_OBS,right(2)",
            "geminal_source": "one ordered column, optionally cusp dressed by the caller",
        },
    )
