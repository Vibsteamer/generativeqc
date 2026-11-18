"""Density-fitted coupled UHF orbital Hessian for unrestricted F12 references."""

from ..tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    multiply,
)


def _field(spin: str, key: str, *axes: Index) -> Node:
    return input_tensor(
        key + "_" + spin, TensorSpec(axes, role="input", differentiable=True)
    )


def unrestricted_df_orbital_action(
    nalpha: int, nbeta: int, orbitals: int, auxiliaries: int
) -> Program:
    """Apply the real, occupied/virtual spin rotation Hessian without forming it.

    B matrices use the same Coulomb-metric whitener for both sets of spatial
    orbitals. Both spin variations contribute to J. K has the genuine same-spin
    direct and exchanged occupied/virtual contractions. Vector order is a,i.
    """
    if (
        any(type(x) is not int for x in (nalpha, nbeta, orbitals, auxiliaries))
        or min(nalpha, nbeta) < 0
        or nalpha + nbeta == 0
        or max(nalpha, nbeta) >= orbitals
        or auxiliaries < 1
    ):
        raise ValueError(
            "Need occupied electrons and nonempty virtual and fitting spaces"
        )
    aux = Index("A", IndexSpace("f12_fitting", "auxiliary", auxiliaries))
    exchange = {}
    vo = {}
    charge = []
    for spin, occ in (("alpha", nalpha), ("beta", nbeta)):
        if not occ:
            continue
        occ_space = IndexSpace("occupied_" + spin, "occupied", occ, spin=spin)
        virt_space = IndexSpace("virtual_" + spin, "virtual", orbitals - occ, spin=spin)
        i, j = Index("i", occ_space), Index("j", occ_space)
        a, b = Index("a", virt_space), Index("b", virt_space)

        vv, oo = _field(spin, "vv", aux, a, b), _field(spin, "oo", aux, i, j)
        kernel = _field(spin, "vo", aux, a, i)
        trial = _field(spin, "trial", a, i)
        one = _field(spin, "gap", a, i)
        charge.append(einsum("Aai,ai->A", kernel, trial))
        kv = einsum("Aab,bj->Aaj", vv, trial)
        ki = einsum("Abi,bj->Aij", kernel, trial)
        exchange[spin] = add(
            multiply(one, trial),
            einsum("Aaj,Aij->ai", kv, oo),
            einsum("Aaj,Aij->ai", kernel, ki),
            coefficients=(1, -1, -1),
        )
        vo[spin] = kernel
    total = add(*charge)
    return Program(
        {
            spin: add(part, einsum("Aai,A->ai", vo[spin], total), coefficients=(1, 2))
            for spin, part in exchange.items()
        },
        provenance={
            "method": "coupled canonical DF-UHF occupied-virtual response",
            "layout": "virtual-major",
        },
    )
