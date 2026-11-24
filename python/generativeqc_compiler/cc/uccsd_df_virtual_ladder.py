"""Bounded auxiliary slices of the bare DF virtual term in the UCCSD residual."""

from dataclasses import dataclass, replace
from fractions import Fraction

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    JVPProgram,
    Node,
    Program,
    TensorSpec,
    VJPProgram,
    add,
    einsum,
    input_tensor,
    linearize,
    transpose,
    transpose_program,
)

from .uccsd_equations import _canonical_tau
from .uccsd_layout import BLOCK_NAMES, UCCSDLayout

TAU_INPUTS = ("tau_aa", "tau_ab", "tau_bb")
FACTOR_INPUTS = (
    "fit_left_alpha",
    "fit_left_beta",
    "raw_right_alpha",
    "raw_right_beta",
)


@dataclass(frozen=True)
class UCCDFVirtualLadderPrograms:
    """One auxiliary slice and its dense Frobenius tangent and transpose actions.

    The caller sums the residual and tau adjoint over auxiliary slices. The
    factor adjoints belong to the supplied slice. The caller owns the physical
    amplitude packing and any projection onto symmetric DF factors.
    """

    primal: Program
    tau_jvp: JVPProgram
    tau_vjp: VJPProgram
    factor_jvp: JVPProgram
    factor_vjp: VJPProgram


def _factor(name: str, auxiliary: IndexSpace, virtual: IndexSpace) -> Node:
    spec = TensorSpec(
        (Index("P", auxiliary), Index("a", virtual), Index("e", virtual)),
        representation="spin_orbital",
        role="parameter",
        differentiable=True,
    )
    return input_tensor(name, spec)


def _ladder(tau: Node, fit: Node, raw: Node) -> Node:
    projected = einsum("Pae,ijef->Pijaf", fit, tau)
    return einsum("Pijaf,Pbf->ijab", projected, raw)


def build_uccsd_df_virtual_ladder(layout: UCCSDLayout, auxiliary_slice: int) -> Program:
    """Return the bare term 1/2 sum_ef tau_ijef <ab||ef> for one DF slice.

    Tau contains T2 plus the disconnected singles product in canonical aa, ab
    and bb orders. The mixed exchange term has the opposite sign of the
    corresponding beta-alpha tau permutation, so it doubles the direct term.
    Same-spin exchange can act on tau before contraction. Both identities
    avoid forming any tensor with three or four virtual axes.

    The canonical MP2Fit convention uses one metric-solved left factor and
    one raw right factor. Both inputs have shape (slice, virtual, virtual).
    """
    if not isinstance(layout, UCCSDLayout):
        raise TypeError("DF virtual ladder requires UCCSDLayout")
    tau = tuple(
        input_tensor(
            name,
            replace(
                block.spec,
                symmetries=(),
                role="parameter",
                differentiable=True,
            ),
        )
        for name, block in zip(TAU_INPUTS, layout.blocks[2:], strict=True)
    )
    return _virtual_ladder_program(tau, auxiliary_slice, "tau")


def _virtual_ladder_program(
    tau: tuple[Node, ...], auxiliary_slice: int, inputs: str
) -> Program:
    auxiliary = IndexSpace("df_ladder_auxiliary", "auxiliary", auxiliary_slice)
    if auxiliary.size == 0:
        raise ValueError("DF virtual ladder requires a nonempty auxiliary slice")
    tau_aa, tau_ab, tau_bb = tau
    va, vb = (index.space for index in tau_ab.spec.indices[2:])
    fit_a, fit_b, raw_a, raw_b = (
        _factor(name, auxiliary, space)
        for name, space in zip(FACTOR_INPUTS, (va, vb, va, vb), strict=True)
    )

    def same_spin(tau: Node, fit: Node, raw: Node) -> Node:
        half = Fraction(1, 2)
        antisymmetric = add(
            tau, transpose(tau, (0, 1, 3, 2)), coefficients=(half, -half)
        )
        return _ladder(antisymmetric, fit, raw)

    return Program(
        {
            "r2aa": same_spin(tau_aa, fit_a, raw_a),
            "r2ab": _ladder(tau_ab, fit_a, raw_b),
            "r2bb": same_spin(tau_bb, fit_b, raw_b),
        },
        provenance={
            "method": "UCCSD",
            "term": "bare_df_virtual_ladder",
            "version": 1,
            **(
                {"input_domain": "physical_amplitudes"}
                if inputs == "amplitudes"
                else {}
            ),
        },
    )


def build_uccsd_df_virtual_ladder_response(
    layout: UCCSDLayout, auxiliary_slice: int
) -> UCCDFVirtualLadderPrograms:
    """Differentiate the same physical term with respect to tau and DF factors."""
    primal = build_uccsd_df_virtual_ladder(layout, auxiliary_slice)
    outputs = tuple(primal.outputs)
    return UCCDFVirtualLadderPrograms(
        primal,
        linearize(primal, TAU_INPUTS, outputs=outputs),
        transpose_program(primal, outputs, inputs=TAU_INPUTS),
        linearize(primal, FACTOR_INPUTS, outputs=outputs),
        transpose_program(primal, outputs, inputs=FACTOR_INPUTS),
    )


@dataclass(frozen=True)
class UCCDFVirtualLadderAmplitudePrograms:
    """A bare slice in the full residual convention and its physical input actions."""

    primal: Program
    residual_jvp: JVPProgram
    residual_vjp: VJPProgram
    factor_vjp: VJPProgram


def build_uccsd_df_virtual_ladder_amplitude_programs(
    layout: UCCSDLayout, auxiliary_slice: int, *, max_elements: int = 1_000_000
) -> UCCDFVirtualLadderAmplitudePrograms:
    """Link each slice to physical T1/T2, including the disconnected singles derivative."""
    if not isinstance(layout, UCCSDLayout):
        raise TypeError("DF virtual ladder requires UCCSDLayout")
    primal = _virtual_ladder_program(
        _canonical_tau(layout), auxiliary_slice, "amplitudes"
    )
    outputs = tuple(primal.outputs)
    return UCCDFVirtualLadderAmplitudePrograms(
        primal,
        linearize(primal, BLOCK_NAMES, outputs=outputs, max_elements=max_elements),
        transpose_program(
            primal, outputs, inputs=BLOCK_NAMES, max_elements=max_elements
        ),
        transpose_program(
            primal, outputs, inputs=FACTOR_INPUTS, max_elements=max_elements
        ),
    )
