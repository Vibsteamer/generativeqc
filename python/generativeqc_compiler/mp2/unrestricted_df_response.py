"""Analytic spin-resolved DF-UMP2 force on a moving orthonormal orbital frame."""

from ..tensor.ir import Node, add, broadcast, divide, einsum, input_tensor, multiply, reduce_sum, transpose
from ..tensor.program import Program
from ..tensor.types import Index, IndexSpace, TensorSpec


def df_ump2_pair_response(
    channel: str,
    auxiliaries: int,
    left_occupied: int,
    left_virtuals: int,
    right_occupied: int,
    right_virtuals: int,
    directions: int,
) -> Program:
    """Return energy and separate factor-/Fock-response forces for one spin pair.

    Supply the whitened MP2Fit factors B[Q,occupied,virtual] and their total
    derivatives in a consistent auxiliary gauge, plus the physical canonical
    spectrum and the full occupied/virtual Fock derivatives in the moving
    orbital frame. For nuclear coordinates in Bohr the outputs are forces in
    Eh/Bohr. A complete unrestricted correlation term sums alpha_alpha,
    alpha_beta and beta_beta once each. Empty channels contribute zero and
    do not require a graph.
    """
    if channel not in ("alpha_alpha", "alpha_beta", "beta_beta"):
        raise ValueError("expected alpha_alpha, alpha_beta or beta_beta")
    dimensions = (auxiliaries, left_occupied, left_virtuals, right_occupied, right_virtuals, directions)
    if any(type(n) is not int or n < 1 for n in dimensions):
        raise ValueError("a DF-UMP2 response graph requires positive dimensions")
    left_spin, right_spin = channel.split("_")
    same = left_spin == right_spin
    if same and (left_occupied, left_virtuals) != (right_occupied, right_virtuals):
        raise ValueError("a same-spin pair must use the same physical orbital space")
    q = Index("Q", IndexSpace("mp2fit", "auxiliary", auxiliaries))
    x = Index("x", IndexSpace("coordinate", "batch", directions))

    def input_(name: str, *indices: Index) -> Node:
        return input_tensor(name, TensorSpec(indices, role="input"))

    def spin_inputs(spin: str, nocc: int, nvir: int, olab: str, vlab: str) -> dict[str, Node]:
        o = Index(olab, IndexSpace(spin + "_occupied", "occupied", nocc, spin=spin))
        v = Index(vlab, IndexSpace(spin + "_virtual", "virtual", nvir, spin=spin))
        return {
            "B": input_("B_" + spin, q, o, v),
            "dB": input_("dB_" + spin, x, q, o, v),
            "eo": input_("eo_" + spin, o),
            "ev": input_("ev_" + spin, v),
            "dFo": input_("dFo_" + spin, x, o, Index("k", o.space)),
            "dFv": input_("dFv_" + spin, x, v, Index("c", v.space)),
        }

    left = spin_inputs(left_spin, left_occupied, left_virtuals, "i", "a")
    right = left if same else spin_inputs(right_spin, right_occupied, right_virtuals, "j", "b")
    direct = einsum("Qia,Qjb->ijab", left["B"], right["B"])
    spectra = (left["eo"], right["eo"], left["ev"], right["ev"])
    denominator = add(
        *(broadcast(e, direct.spec.indices, (axis,)) for axis, e in enumerate(spectra)),
        coefficients=(1, 1, -1, -1),
    )
    numerator = add(direct, transpose(direct, (0, 1, 3, 2)), coefficients=(1, -1)) if same else direct
    amplitude = divide(numerator, denominator)
    weight = "1/4" if same else 1
    energy = add(reduce_sum(multiply(amplitude, numerator), (0, 1, 2, 3)), coefficients=(weight,))

    # In the full ordered same-spin sum, applying the exchange permutation to
    # the antisymmetric amplitude gives a second identical factor derivative.
    factor_weight = -1 if same else -2
    factor_force = add(
        einsum("xQia,Qia->x", left["dB"], einsum("ijab,Qjb->Qia", amplitude, right["B"])),
        einsum("xQjb,Qjb->x", right["dB"], einsum("ijab,Qia->Qjb", amplitude, left["B"])),
        coefficients=(factor_weight, factor_weight),
    )
    # -dE/dR of the denominator contains the full fitted Fock derivative.
    # Retaining the off-diagonal OO and VV terms makes it frame covariant.
    virtual_weight = "-1/4" if same else -1
    fock_force = add(
        einsum("xik,ik->x", left["dFo"], einsum("ijab,kjab->ik", amplitude, amplitude)),
        einsum("xjk,jk->x", right["dFo"], einsum("ijab,ikab->jk", amplitude, amplitude)),
        einsum("xac,ac->x", left["dFv"], einsum("ijab,ijcb->ac", amplitude, amplitude)),
        einsum("xbc,bc->x", right["dFv"], einsum("ijab,ijac->bc", amplitude, amplitude)),
        coefficients=(weight, weight, virtual_weight, virtual_weight),
    )
    return Program(
        {"energy": energy, "factor_force": factor_force, "fock_force": fock_force,
         "force": add(factor_force, fock_force)},
        provenance={"method": "ordinary unrestricted DF-MP2", "spin_pair": channel,
                    "response": "negative complete moving-frame energy derivative"},
    )
