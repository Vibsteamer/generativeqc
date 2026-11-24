"""Transfer an invariant CABS-frame cotangent to its projector kernel."""

from ..tensor.ir import add, einsum, input_tensor, transpose
from ..tensor.program import Program
from ..tensor.types import Index, IndexSpace, TensorSpec


def full_rank_cabs_cotangent(atomic: int, obs: int) -> Program:
    """Differentiate C.T S C=I and R.T S C=0 in the complete complement.

    R is an S-orthonormal basis of the fixed orbital AO span, embedded in the
    combined AO basis. For G=bar_C the invariant moment C.T G is symmetric.
    Thus bar_S = -sym(R R.T G C.T) - C sym(C.T G) C.T / 2. This formula
    omits rotations into discarded directions and requires the full CABS rank.
    """
    if type(atomic) is not int or type(obs) is not int or not 0 < obs < atomic:
        raise ValueError("Expected a nonempty full CABS complement")
    ao = IndexSpace("combined_atomic", "ao", atomic)
    orbital = IndexSpace("orbital_basis", "orbital", obs)
    comp = IndexSpace("cabs_frame", "orbital", atomic - obs)
    a, i, r = Index("a", ao), Index("i", orbital), Index("r", comp)
    frame = input_tensor("frame", TensorSpec((a, r), role="input"))
    bar = input_tensor("frame_bar", TensorSpec((a, r), role="input"))
    reference = input_tensor("orbital_frame", TensorSpec((a, i), role="input"))
    moment = einsum("ar,as->rs", frame, bar)
    rotation = add(moment, transpose(moment, (1, 0)), coefficients=("1/2", "1/2"))
    projected = einsum("ai,ir->ar", reference, einsum("bi,br->ir", reference, bar))
    cross = einsum("ar,br->ab", projected, frame)
    normalization = einsum("ar,br->ab", einsum("as,sr->ar", frame, rotation), frame)
    weight = add(
        cross,
        transpose(cross, (1, 0)),
        normalization,
        coefficients=("-1/2", "-1/2", "-1/2"),
    )
    return Program(
        {
            "overlap_bar": weight,
            "gauge_skew": add(moment, transpose(moment, (1, 0)), coefficients=(1, -1)),
        },
        provenance={
            "operation": "full-rank CABS constraint cotangent",
            "derivative_order": 1,
        },
    )


def frame_kernel_cotangent(atomic: int, rank: int) -> Program:
    """Map bar_C to a symmetric bar_K for K = C C.T, with C.T S C = I.

    This is a first-derivative transfer, not a second differentiation of its
    formula. With D = S C, M = C.T bar_C and symmetric M, a valid representative
    is bar_K = (bar_C D.T + D bar_C.T - D M D.T) / 2. It satisfies
    2 bar_K C = bar_C and therefore preserves <bar_C,dC> for any frame motion.
    The scalar must be invariant to orthogonal rotations of the CABS columns.

    The caller must check dual_gram against I, gauge_skew against zero, and the
    reconstruction residual before passing kernel_bar to the overlap response.
    A nonzero skew cannot be silently removed from a gauge-dependent scalar.
    All arrays use the same AO row order; this graph does not select a rank.
    """
    if type(atomic) is not int or type(rank) is not int or not 0 < rank <= atomic:
        raise ValueError("Expected positive frame rank <= AO dimension")
    ao = IndexSpace("combined_atomic", "ao", atomic)
    comp = IndexSpace("cabs_frame", "orbital", rank)
    u, v, r = Index("u", ao), Index("v", ao), Index("r", comp)
    overlap = input_tensor("overlap", TensorSpec((u, v), role="input"))
    frame = input_tensor("frame", TensorSpec((u, r), role="input"))
    bar = input_tensor("frame_bar", TensorSpec((u, r), role="input"))
    dual = einsum("uv,vr->ur", overlap, frame)
    moment = einsum("ur,us->rs", frame, bar)
    moment_t = transpose(moment, (1, 0))
    symmetric = add(moment, moment_t, coefficients=("1/2", "1/2"))
    cross = einsum("ur,vr->uv", bar, dual)
    projected = einsum("ur,rs->us", dual, symmetric)
    projected = einsum("us,vs->uv", projected, dual)
    kernel_bar = add(
        cross,
        transpose(cross, (1, 0)),
        projected,
        coefficients=("1/2", "1/2", "-1/2"),
    )
    reconstructed = einsum("uv,vr->ur", kernel_bar, frame, coefficient=2)
    return Program(
        {
            "kernel_bar": kernel_bar,
            "dual_gram": einsum("ur,us->rs", frame, dual),
            "gauge_skew": add(moment, moment_t, coefficients=(1, -1)),
            "reconstruction_residual": add(reconstructed, bar, coefficients=(1, -1)),
        },
        provenance={
            "operation": "CABS frame cotangent to invariant kernel weight",
            "requires": "S-orthonormal frame and rotation-invariant scalar",
            "derivative_order": 1,
        },
    )
