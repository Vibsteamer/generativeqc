"""Pair-tiled restricted CC-F12 matrix elements and the CCSD-F12b-specific back coupling."""

from ..tensor.ir import Node, add, einsum, input_tensor
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
