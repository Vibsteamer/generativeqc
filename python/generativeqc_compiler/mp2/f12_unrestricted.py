"""Unrestricted extended-frame sources for real MP2-F12 pair projections."""

from ..tensor.ir import (
    Node,
    add,
    einsum,
    input_tensor,
    slice_tensor,
)
from ..tensor.program import Program
from ..tensor.types import Index, IndexSpace, TensorSpec


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
