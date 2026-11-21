"""Output panels of the unrestricted mixed-spin bare virtual Coulomb ladder."""

from dataclasses import dataclass, replace
from functools import cache

from generativeqc_compiler.tensor import (
    Index,
    JVPProgram,
    Program,
    TensorSpec,
    VJPProgram,
    einsum,
    input_tensor,
    linearize,
    transpose_program,
)

from .uccsd_equations import _canonical_tau
from .uccsd_layout import UCCSDLayout

MIXED_AMPLITUDES = ("t1a", "t1b", "t2ab")


@dataclass(frozen=True)
class UCCMixedVirtualPanelPrograms:
    """A native `r2ab` panel and its physical-amplitude differential actions."""

    primal: Program
    jvp: JVPProgram
    vjp: VJPProgram


@cache
def build_uccsd_uhf_mixed_virtual_panel_programs(
    layout: UCCSDLayout, alpha_width: int, *, max_elements: int = 1_000_000
) -> UCCMixedVirtualPanelPrograms:
    """Bind one independent spin-Coulomb panel to the physical CC amplitudes.

    The caller slices the alpha target in disjoint adjacent panels and stacks
    the output. The transpose takes the matching target seed and sums its
    three full-amplitude adjoints with those of the other panels. Fitting
    derivatives stay in the separate auxiliary-sliced stationary source VJP.
    """
    if not isinstance(layout, UCCSDLayout):
        raise TypeError("UHF virtual panels require UCCSDLayout")
    if type(alpha_width) is not int or not 0 < alpha_width <= layout.va:
        raise ValueError("UHF virtual panel width must fit the alpha virtual space")
    _, tau, _ = _canonical_tau(layout)
    source_alpha, source_beta = (index.space for index in tau.spec.indices[2:])
    alpha_panel = replace(source_alpha, name="virtual_alpha_panel", size=alpha_width)
    operator = input_tensor(
        "mixed_spin_coulomb_panel",
        TensorSpec(
            (
                Index("e", source_alpha),
                Index("f", source_beta),
                Index("a", alpha_panel),
                Index("b", source_beta),
            ),
            representation="spin_orbital",
            role="input",
        ),
    )
    primal = Program(
        {"r2ab": einsum("ijef,efab->ijab", tau, operator)},
        provenance={
            "method": "UCCSD",
            "term": "bare_uhf_mixed_virtual_panel",
            "version": 1,
        },
    )
    return UCCMixedVirtualPanelPrograms(
        primal,
        linearize(primal, MIXED_AMPLITUDES, max_elements=max_elements),
        transpose_program(
            primal, ("r2ab",), inputs=MIXED_AMPLITUDES, max_elements=max_elements
        ),
    )
