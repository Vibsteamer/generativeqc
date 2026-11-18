"""Pair-tiled restricted CC-F12 matrix elements and the CCSD-F12b-specific back coupling."""

from fractions import Fraction

from ..tensor.ir import Node, add, einsum, input_tensor, transpose
from ..tensor.program import Program
from ..tensor.types import Index, IndexSpace, TensorSpec


def standard_v_tile(obs: int, auxiliary: int, left: int, right: int) -> Program:
    """Return V_uv^mn for a fixed ordered geminal pair mn and two free orbital tiles.

    Use the standard approximation to the projector in V: 1 - P_OBS(1)P_OBS(2).
    Equivalently V_uv = <uv|g f|mn> - sum_rs (ur|vs) <rs|f|mn>. Here g=1/r12,
    the OBS sums include all occupied and virtual orbitals, and the first
    density carries J^-1 in the unsymmetrized Coulomb fit. Inputs are real.
    Free indices u and v can address independent occupied/virtual tiles.
    """
    if any(type(n) is not int or n < 1 for n in (obs, auxiliary, left, right)):
        raise ValueError(
            "positive OBS, auxiliary and free orbital tile dimensions required"
        )
    basis = IndexSpace("observable_orbitals", "orbital", obs)
    r, s = Index("r", basis), Index("s", basis)
    p = Index("p", IndexSpace("fitting_basis", "auxiliary", auxiliary))
    u = Index("u", IndexSpace("left_orbital_tile", "orbital", left))
    v = Index("v", IndexSpace("right_orbital_tile", "orbital", right))
    direct = _source("geminal_coulomb", u, v)
    f = _source("geminal_obs", r, s)
    fit = _source("fit_left", p, u, r)
    raw = _source("raw_coulomb_right", p, v, s)
    right_action = einsum("rs,pvs->prv", f, raw)
    value = add(direct, einsum("pur,prv->uv", fit, right_action), coefficients=(1, -1))
    return Program(
        {"v": value},
        provenance={
            "operation": "standard projected CC-F12 V tile",
            "geminal_pairs": "one ordered pair; no fixed amplitude or spin weights applied",
        },
    )


def f12b_geminal_back_coupling(occupied: int, virtual: int) -> Program:
    """Extra F12b minus F12a geminal residual R_mn^ij for a fixed ordered mn.

    Eq. (4.127) of G. Knizia, Explicitly correlated quantum chemistry:
    V_ab^mn (t_ij^ab + t_i^a t_j^b) + V_ib^mn t_j^b + t_i^a V_aj^mn.
    The spin-free t2 input is the connected, ordered spatial amplitude. A
    separate fixed-geminal amplitude subsequently contracts this residual
    in the energy; neither its spin weighting nor a pair degeneracy is here.
    """
    if any(type(n) is not int or n < 1 for n in (occupied, virtual)):
        raise ValueError("positive occupied and virtual amplitude spaces required")
    occ, vir = (
        IndexSpace("occupied", "occupied", occupied),
        IndexSpace("virtual", "virtual", virtual),
    )
    i, j, a, b = Index("i", occ), Index("j", occ), Index("a", vir), Index("b", vir)
    vv, ov, vo = _source("v_vv", a, b), _source("v_ov", i, b), _source("v_vo", a, j)
    t1, t2 = _source("t1", i, a), _source("t2", i, j, a, b)
    connected = einsum("ab,ijab->ij", vv, t2)
    # The disconnected product is contracted without materializing a new t1*t1 tensor.
    occupied_virtual = add(ov, einsum("ia,ab->ib", t1, vv))
    residual = add(
        connected,
        einsum("ib,jb->ij", occupied_virtual, t1),
        einsum("ia,aj->ij", t1, vo),
    )
    return Program(
        {"back_coupling": residual},
        provenance={
            "operation": "restricted CCSD-F12b extra geminal residual",
            "reference": "Knizia Eq. (4.127)",
        },
    )


def _source(name: str, *axes: Index) -> Node:
    return input_tensor(
        name,
        TensorSpec(
            axes, role="input", differentiable=True, representation="restricted_spatial"
        ),
    )


def fixed_pair_cc_residual(occupied: int, virtual: int, *, diagonal: bool) -> Program:
    """Fixed-cusp correction to ordinary spatial CC singles/doubles for one unordered pair.

    For m<n the second ordered geminal is the electron-coordinate transpose
    of the first. The caller adds doubles_mn to R[m,n], its transpose to R[n,m],
    and singles_m/singles_n to the corresponding rows. For m=n apply only
    doubles_mn and singles_m once. Free indices address active CC orbitals.
    The geminal factor has unit slope at the origin. The standard singlet and
    triplet cusp coefficients are 1/2 and 1/4, respectively.
    """
    if type(diagonal) is not bool or any(
        type(n) is not int or n < 1 for n in (occupied, virtual)
    ):
        raise ValueError("positive active spaces and a boolean diagonal flag required")
    occ = IndexSpace("occupied", "occupied", occupied)
    vir = IndexSpace("virtual", "virtual", virtual)
    k, a, b = Index("k", occ), Index("a", vir), Index("b", vir)
    t1 = _source("t1", k, a)
    vv, ov, vo = _source("v_vv", a, b), _source("v_ov", k, b), _source("v_vo", a, k)
    c = _source("fock_coupling_vv", a, b)
    # V_ak*t_kb and t_ka*V_kb from Eq. (4.123) belong to distinct ordered blocks.
    source = add(
        vv,
        c,
        einsum("ak,kb->ab", vo, t1),
        einsum("ka,kb->ab", t1, ov),
        coefficients=(1, 1, -1, -1),
    )
    selector_spec = TensorSpec(
        (k,), role="input", differentiable=False, representation="restricted_spatial"
    )
    m = input_tensor("select_m", selector_spec)
    if diagonal:
        outputs = {
            "doubles_mn": add(source, coefficients=(Fraction(1, 2),)),
            "singles_m": add(einsum("ak,k->a", vo, m), coefficients=(Fraction(1, 2),)),
        }
    else:
        n = input_tensor("select_n", selector_spec)
        # Fold the mn and nm contributions of the full ordered spatial sum.
        outputs = {
            "doubles_mn": add(
                source,
                transpose(source, (1, 0)),
                coefficients=(Fraction(3, 8), Fraction(1, 8)),
            ),
            "singles_m": add(
                einsum("ak,k->a", vo, n),
                einsum("ka,k->a", ov, n),
                coefficients=(Fraction(5, 8), Fraction(-1, 8)),
            ),
            "singles_n": add(
                einsum("ak,k->a", vo, m),
                einsum("ka,k->a", ov, m),
                coefficients=(Fraction(-1, 8), Fraction(5, 8)),
            ),
        }
    return Program(
        outputs,
        provenance={
            "operation": "CCSD-F12a/b ordinary residual additions",
            "reference": "Knizia Eqs. (3.23), (4.123),(4.124)",
            "ordered_pair_sum": "folded into one unordered spatial pair",
        },
    )


def fixed_pair_geminal_energy(occupied: int, virtual: int) -> Program:
    """Geminal terms in restricted F12a and F12b energies for one *ordered* pair.

    Sum both orders for m != n. Supply the occupied-projector V_ij (the full
    Q12 used by the ordinary F12 Hylleraas functional), ordered rows of B and X
    in the occupied-pair basis, and C_ab at the same reference. The energy uses
    connected t2 in the common C_ab term. The separate F12b back-coupling input
    includes the disconnected singles product. Add conventional CC correlation
    and the reference separately. All source tensors, including the entire
    symmetric occupied Fock block, remain differentiable.
    """
    if any(type(n) is not int or n < 1 for n in (occupied, virtual)):
        raise ValueError("positive occupied and virtual active spaces required")
    occ = IndexSpace("occupied", "occupied", occupied)
    vir = IndexSpace("virtual", "virtual", virtual)
    i, j = Index("i", occ), Index("j", occ)
    a, b = Index("a", vir), Index("b", vir)
    v = _source("v_occupied", i, j)
    x = _source("x_row", i, j)
    b_row = _source("b_row", i, j)
    g = _source("fock_occupied", i, j)
    c = _source("fock_coupling_vv", a, b)
    t2 = _source("t2", i, j, a, b)
    rb = _source("back_coupling", i, j)
    sel = TensorSpec(
        (i,), role="input", differentiable=False, representation="restricted_spatial"
    )
    m = input_tensor("select_m", sel)
    n = input_tensor("select_n", sel)

    def fixed_action(matrix: Node) -> Node:
        # Contract this row with the complete fixed-amplitude tensor T_op^ij.
        return add(
            matrix,
            transpose(matrix, (1, 0)),
            coefficients=(Fraction(3, 8), Fraction(1, 8)),
        )

    bx = fixed_action(x)
    common = add(
        v,
        fixed_action(b_row),
        einsum("ik,kj->ij", g, bx),
        einsum("ik,jk->ij", bx, g),
        einsum("ab,ijab->ij", c, t2),
        coefficients=(2, 1, -1, -1, 1),
    )

    def spin_contract(values: Node) -> Node:
        # The energy contracts (2T_mn^ij - T_nm^ij) for this fixed ordered mn.
        weighted = add(
            values,
            transpose(values, (1, 0)),
            coefficients=(Fraction(5, 8), Fraction(-1, 8)),
        )
        return einsum("i,i->", m, einsum("ij,j->i", weighted, n))

    return Program(
        {
            "f12a_geminal_energy": spin_contract(common),
            "f12b_extra_energy": spin_contract(rb),
        },
        provenance={
            "operation": "restricted ordered CC-F12 geminal energy terms",
            "reference": "Knizia Eqs. (4.125)-(4.127)",
            "pair_enumeration": "ordered; diagonal visited once",
        },
    )
