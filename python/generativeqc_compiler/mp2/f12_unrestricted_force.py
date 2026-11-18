"""Cartesian contraction of polarized F12 pair sources and their analytic tangents."""

from ..tensor import Index, IndexSpace, Program, TensorSpec, add, einsum, input_tensor
from .f12_unrestricted import projected_same_spin_f12_pair


def polarized_pair_forces(
    occupied: int, virtuals: int, cabs: int, atoms: int
) -> Program:
    """Contract analytic nuclear tangents of all sources with their electronic VJPs.

    One active like-spin pair supplies ordinary doubles and the F12 increment.
    The reference gradient includes nuclear repulsion; the singles Fock
    multiplier enters once. Each supplied tangent includes operator, orbital
    (UHF) and metric/CABS motion as appropriate. Fixed occupied selectors are
    not differentiated. The output sign is force = -dE/dR, in Hartree/bohr.
    """
    if type(atoms) is not int or atoms < 1:
        raise ValueError("Need at least one atom")
    pair = projected_same_spin_f12_pair(occupied, virtuals, cabs)
    atom = Index("A", IndexSpace("atom", "atom", atoms))
    axis = Index("k", IndexSpace("cartesian", "cartesian", 3))
    gradients = [
        input_tensor("reference_gradient", TensorSpec((atom, axis), role="input"))
    ]
    seen = set()
    for node in pair.live_nodes:
        if node.op != "input" or not node.spec.differentiable:
            continue
        key, spec = node.attrs["name"], node.spec
        if key in seen:
            continue
        seen.add(key)
        bar = input_tensor("bar_" + key, TensorSpec(spec.indices, role="input"))
        if key == "fock":
            bar = add(
                bar,
                input_tensor(
                    "singles_fock_bar", TensorSpec(spec.indices, role="input")
                ),
            )
        tangent = input_tensor(
            "tangent_" + key, TensorSpec((atom, axis, *spec.indices), role="input")
        )
        grads = einsum("Akpq,pq->Ak", tangent, bar)
        gradients.append(grads)
    return Program(
        {"force": add(*gradients, coefficients=(-1,) * len(gradients))},
        provenance={
            "units": "Hartree/bohr",
            "method": "analytic four-center source-tangent contraction for a polarized F12 pair",
        },
    )


def unrestricted_f12_nuclear_source_action(elements: int) -> Program:
    """One Cartesian force from the total correlation adjoint and physical sources.

    The nuclear tangent uses the identical flattened source layout as the
    UMP2/F12/CABS adjoint. Drop the axis after each dot to bound device memory
    independently of molecule coordinate count. The molecular reference
    gradient includes nuclear repulsion in Hartree/bohr.
    """
    if type(elements) is not int or elements < 1:
        raise ValueError("An F12 nuclear source vector must contain positive length")
    k = Index("p", IndexSpace("spin_pair_source", "component", elements))
    bar = input_tensor("weights", TensorSpec((k,), role="input"))
    tangent = input_tensor("tangents", TensorSpec((k,), role="input"))
    dref = input_tensor("reference_gradient", TensorSpec((), role="input"))
    return Program(
        {"force": add(dref, einsum("p,p->", bar, tangent), coefficients=(-1, -1))},
        provenance={
            "method": "DF UMP2-F12 source-adjoint contraction",
            "units": "Eh/bohr",
        },
    )
