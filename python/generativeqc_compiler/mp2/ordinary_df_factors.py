"""Ordinary correlation-fit occupied/virtual factors and physical metric weights."""

from ..tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    slice_tensor,
    transpose,
)


def ordinary_df_factors(atomic: int, occupied: int, auxiliaries: int) -> Program:
    """Build W A[C_occ,C_vir] and the full-rank auxiliary metric cotangent.

    W M W.T is the identity. With the factor energy cotangent G the symmetric
    metric derivative in an orthonormal fitting gauge is
    -sym(W.T G B.T W)/2. Gauge rotation terms cancel in the spin-pair energy.
    """
    if not 0 < occupied < atomic or auxiliaries < 1:
        raise ValueError(
            "ordinary DF factors need occupied, virtual and auxiliary orbitals"
        )
    ao, aux = (
        IndexSpace("ao", "ao", atomic),
        IndexSpace("mp2fit", "auxiliary", auxiliaries),
    )
    mo = IndexSpace("obs", "orbital", atomic)
    u, v, p = Index("u", ao), Index("v", ao), Index("p", mo)
    a, b = Index("A", aux), Index("B", aux)
    c = input_tensor(
        "coefficients", TensorSpec((u, p), role="input", differentiable=True)
    )
    raw = input_tensor(
        "ao_coulomb", TensorSpec((a, u, v), role="input", differentiable=True)
    )
    w = input_tensor("whitener", TensorSpec((a, b), role="input"))
    co = slice_tensor(c, ((0, atomic), (0, occupied)))
    cv = slice_tensor(c, ((0, atomic), (occupied, atomic)))
    # Contract the shorter occupied leg first.
    mo_raw = einsum("Aiv,va->Aia", einsum("ui,Auv->Aiv", co, raw), cv)
    factors = einsum("AB,Bia->Aia", w, mo_raw)
    seed = input_tensor("factor_bar", TensorSpec(factors.spec.indices, role="input"))
    weighted = einsum("AB,Aia->Bia", w, seed)
    fitted = einsum("AB,Aia->Bia", w, factors)
    nonsym = einsum("Aia,Bia->AB", weighted, fitted)
    metric = add(nonsym, transpose(nonsym, (1, 0)), coefficients=("-1/4", "-1/4"))
    return Program({"B": factors, "metric_bar": metric})
