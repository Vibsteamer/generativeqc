"""Unrestricted extended-frame sources for real MP2-F12 pair projections."""

from fractions import Fraction

from ..tensor.ir import (
    Node,
    add,
    broadcast,
    einsum,
    input_tensor,
    multiply,
    slice_tensor,
    transpose,
)
from ..tensor.program import Program
from ..tensor.types import Index, IndexSpace, TensorSpec
from .f12_molecular_pair import projected_f12_pair


def _field(name: str, *axes: Index) -> Node:
    return input_tensor(name, TensorSpec(axes, role="input", differentiable=True))


def unrestricted_extended_df_fock(
    nalpha: int, nbeta: int, orbitals: int, auxiliaries: int
) -> Program:
    """Form both UHF Fock matrices in their separate orthonormal OBS+CABS frames.

    The raw three-center factors are transformed with the corresponding spin
    orbitals. Occupied-by-extended fit = J^{-1} raw lives in a common auxiliary
    basis for both spins. The shared charge is the sum of both occupied traces.
    Exchange is spin diagonal. Empty-spin input has no occupied fit operand.
    """
    if (
        any(type(n) is not int for n in (nalpha, nbeta, orbitals, auxiliaries))
        or min(nalpha, nbeta) < 0
        or nalpha + nbeta < 1
        or max(nalpha, nbeta) >= orbitals
        or auxiliaries < 1
    ):
        raise ValueError("Need occupied electrons, nonempty virtual and fitting spaces")
    a = Index("a", IndexSpace("f12_fitting", "auxiliary", auxiliaries))
    traces = []
    fields = {}
    for spin, occupied in (("alpha", nalpha), ("beta", nbeta)):
        frame = IndexSpace("extended_" + spin, "orbital", orbitals, spin=spin)
        p, q = Index("p", frame), Index("q", frame)
        hcore = _field("hcore_" + spin, p, q)
        raw = _field("raw_coulomb_" + spin, a, p, q)
        if occupied:
            fit = _field("fit_" + spin, a, Index("i", frame, 0, occupied), q)
            occ_fit = slice_tensor(
                fit, ((0, auxiliaries), (0, occupied), (0, occupied))
            )
            traces.append(einsum("aii->a", occ_fit))
            occ_raw = slice_tensor(
                raw, ((0, auxiliaries), (0, occupied), (0, orbitals))
            )
            exchange = einsum("aip,aiq->pq", fit, occ_raw)
        else:
            exchange = add(hcore, coefficients=(0,))
        fields[spin] = (hcore, raw, exchange)

    charge = add(*traces)
    outputs = {}
    for spin, (hcore, raw, exchange) in fields.items():
        coulomb = einsum("apq,a->pq", raw, charge)
        outputs["fock_" + spin] = add(hcore, coulomb, exchange, coefficients=(1, 1, -1))
        outputs["exchange_" + spin] = exchange
    return Program(
        outputs,
        provenance={"method": "UHF occupied-fit extended DF Fock", "units": "Hartree"},
    )


def _pair_spins(
    left: tuple[int, int, int], right: tuple[int, int, int], channel: str
) -> tuple[str, str]:
    spins = {
        "alpha_alpha": ("alpha", "alpha"),
        "alpha_beta": ("alpha", "beta"),
        "beta_beta": ("beta", "beta"),
    }
    if channel not in spins or any(
        len(space) != 3 or any(type(n) is not int or n < 1 for n in space)
        for space in (left, right)
    ):
        raise ValueError("Invalid spin channel or occupied/virtual/CABS dimensions")
    if channel != "alpha_beta" and left != right:
        raise ValueError(
            "Both electrons of a same-spin pair need the same orbital frame"
        )

    return spins[channel]


def unrestricted_pair_vx(
    left: tuple[int, int, int], right: tuple[int, int, int], *, channel: str
) -> Program:
    """V and X matrix elements of a spin-conserving ordered F12 pair.

    Axes in each spin frame are occupied, orbital virtual, then CABS. Given
    external bra and ket occupied spin pairs, let f and g mean the geminal and
    Coulomb operators. V = <bra|g Q f|ket>, X = <bra|f Q f|ket> use the
    strong-orthogonality projector Q = (1-O1)(1-O2) - V1 V2, with V1,V2
    the *orbital* virtual subspaces. Subtract its three disjoint complement
    blocks from the full-space unfitted (or consistently fitted) gf and f²
    integrals. Independent left/right matrix elements also allow a caller to
    select the exchange ket for same-spin pairs. Spin factors and the 3C(FIX)
    B/C contributions belong to the subsequent Hylleraas functional.
    """
    spins = _pair_spins(left, right, channel)

    def ranges(
        space: tuple[int, int, int], spin: str, name: str
    ) -> tuple[Index, Index, Index]:
        occ, vir, cabs = space
        frame = IndexSpace("extended_" + spin, "orbital", occ + vir + cabs, spin=spin)
        return (
            Index(name, frame, 0, occ),
            Index(name, frame, 0, occ + vir),
            Index(name, frame, occ + vir, occ + vir + cabs),
        )

    lo, lb, lc = ranges(left, spins[0], "p")
    ro, rb, rc = ranges(right, spins[1], "q")
    linear = [_field("geminal_coulomb_element")]
    norm = [_field("squared_geminal_element")]
    for block, axes in (
        ("obs_obs", (lb, rb)),
        ("occ_cabs", (lo, rc)),
        ("cabs_occ", (lc, ro)),
    ):
        ket = _field("ket_geminal_" + block, *axes)
        bra_g = _field("bra_coulomb_" + block, *axes)
        bra_f = _field("bra_geminal_" + block, *axes)
        linear.append(einsum("pq,pq->", bra_g, ket))
        norm.append(einsum("pq,pq->", bra_f, ket))
    signs = (1, -1, -1, -1)
    return Program(
        {"v": add(*linear, coefficients=signs), "x": add(*norm, coefficients=signs)},
        provenance={
            "method": "spin-orbital strongly orthogonal F12 V/X",
            "channel": channel,
        },
    )


def unrestricted_pair_coupling(
    left: tuple[int, int, int], right: tuple[int, int, int], *, channel: str
) -> Program:
    """Compute C_ab = <ab|(F1+F2) Q f12|ij> and the positive MP2 energy gap.

    Use one ordered spin-conserving geminal column in its alpha/beta frames.
    Projecting a one-body Fock action onto two orbital virtuals leaves only
    the virtual×CABS and CABS×virtual geminal rectangles. Both legs of like-spin
    pairs share a physical Fock block and virtual spectrum, with a common VJP.
    Summing or antisymmetrizing upstream geminal columns remains linear.
    """
    spin_left, spin_right = _pair_spins(left, right, channel)

    def fock_block(
        size: tuple[int, int, int], spin: str
    ) -> tuple[Node, Node, Index, Index]:
        occ, vir, cabs = size
        frame = IndexSpace("extended_" + spin, "orbital", occ + vir + cabs, spin=spin)
        a = Index("a", frame, occ, occ + vir)
        c = Index("c", frame, occ + vir, occ + vir + cabs)
        return (
            _field("virtual_cabs_fock_" + spin, a, c),
            _field("virtual_energies_" + spin, a),
            a,
            c,
        )

    f_l, e_l, a, c = fock_block(left, spin_left)
    if spin_left == spin_right:
        f_r, e_r, r_vir, d = f_l, e_l, a, c
    else:
        f_r, e_r, r_vir, d = fock_block(right, spin_right)
    b = Index("b", r_vir.space, r_vir.start, r_vir.stop)
    lr = _field("geminal_virtual_cabs", a, d)
    rl = _field("geminal_cabs_virtual", c, b)
    coupling = add(einsum("ac,cb->ab", f_l, rl), einsum("ac,bc->ab", lr, f_r))
    axes = coupling.spec.indices
    occupied = add(_field("occupied_energy_i"), _field("occupied_energy_j"))
    gap = add(
        broadcast(e_l, axes, (0,)),
        broadcast(e_r, axes, (1,)),
        broadcast(occupied, axes, ()),
        coefficients=(1, 1, -1),
    )
    return Program(
        {"c": coupling, "gap": gap},
        provenance={
            "method": "spin-orbital strongly orthogonal F12 doubles coupling",
            "channel": channel,
            "convention": "ordered pair, semicanonical orbital energies, positive MP2 gap",
        },
    )


def projected_same_spin_f12_pair(
    occupied: int, virtuals: int, cabs: int, *, auxiliaries: int | None = None
) -> Program:
    """Energy correction for one unordered pair of distinct like-spin occupied orbitals.

    F and positive K belong to this spin of the unrestricted reference. The
    spatial integral projection then uses only this frame. The fixed cusp
    amplitude is 1/4 for a normalized triplet determinant. Eliminate each
    conventional doubly excited determinant once; summing over the full ordered
    virtual square introduces a factor of one half. Frozen pairs are scheduled
    by the caller. This functional requires neither spin adaptation of the entire
    molecule nor shared alpha and beta spatial orbitals.
    """
    if type(occupied) is not int or occupied < 2:
        raise ValueError("A same-spin electron pair needs two occupied orbitals")
    terms = projected_f12_pair(
        occupied, virtuals, cabs, diagonal=False, auxiliaries=auxiliaries
    ).outputs

    def difference(name: str) -> Node:
        return add(
            terms[name + "_direct"], terms[name + "_exchange"], coefficients=(1, -1)
        )

    coulomb = terms["g"]
    antisym_g = add(coulomb, transpose(coulomb, (1, 0)), coefficients=(1, -1))
    coupling = difference("c")
    scaled = multiply(coupling, terms["inverse_gap"])
    linear = add(
        difference("v"),
        einsum("ab,ab->", antisym_g, scaled),
        coefficients=(1, Fraction(-1, 2)),
    )
    shifted_b = add(
        difference("b"),
        multiply(terms["occupied_fock_sum"], difference("x")),
        coefficients=(1, -1),
    )
    quadratic = add(
        shifted_b,
        einsum("ab,ab->", coupling, scaled),
        coefficients=(1, Fraction(-1, 2)),
    )
    energy = add(linear, quadratic, coefficients=(Fraction(1, 2), Fraction(1, 16)))
    return Program(
        {
            "pair_energy": energy,
            "ordinary_pair_energy": einsum(
                "ab,ab->",
                antisym_g,
                multiply(antisym_g, terms["inverse_gap"]),
                coefficient=Fraction(-1, 2),
            ),
        },
        provenance={
            "method": "same-spin semicanonical MP2-F12/3C(FIX) increment",
            "units": "Hartree",
            "active_pair": "one unordered pair i<j; either alpha/alpha or beta/beta",
        },
    )


def opposite_spin_cusp_source(orbitals: int) -> Program:
    """Spatial geminal source of an alpha-i/beta-j determinant before applying Qf.

    In the alpha(1), beta(2) sector, the spin-projector ansatz is 1/2 of
    the symmetric spatial part plus 1/4 of the antisymmetric part. Exchange
    the spatial orbitals at fixed electron spin. S_ab = C_a.T S_AO C_b
    converts the exchanged product back into the original spin frames. Its
    orbital dependence must survive differentiation. The fixed selections are
    whole-OBS unit vectors, with only occupied external orbitals scheduled.
    """
    if type(orbitals) is not int or orbitals < 1:
        raise ValueError("Positive orbital basis size required")
    p = Index("p", IndexSpace("obs_alpha", "orbital", orbitals, spin="alpha"))
    q = Index("q", IndexSpace("obs_beta", "orbital", orbitals, spin="beta"))
    i = input_tensor(
        "select_alpha", TensorSpec((p,), role="input", differentiable=False)
    )
    j = input_tensor(
        "select_beta", TensorSpec((q,), role="input", differentiable=False)
    )
    sab = _field("cross_overlap", p, q)
    direct = einsum("p,q->pq", i, j)
    flipped = einsum("p,q->pq", einsum("pq,q->p", sab, j), einsum("pq,p->q", sab, i))
    return Program(
        {
            "reference": direct,
            "cusp_source": add(
                direct, flipped, coefficients=(Fraction(3, 8), Fraction(1, 8))
            ),
        },
        provenance={"method": "opposite-spin SP geminal source in distinct UHF frames"},
    )
