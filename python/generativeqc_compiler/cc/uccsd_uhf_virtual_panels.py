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
from .uccsd_rhf_virtual_pairs import _packed_same_spin_ladder, _Pairs

MIXED_AMPLITUDES = ("t1a", "t1b", "t2ab")


@dataclass(frozen=True)
class UCCVirtualLadderActions:
    """A native bare-ladder output and its physical-amplitude differential actions."""

    primal: Program
    jvp: JVPProgram
    vjp: VJPProgram


@cache
def build_uccsd_uhf_mixed_virtual_panel_programs(
    layout: UCCSDLayout, alpha_width: int, *, max_elements: int = 1_000_000
) -> UCCVirtualLadderActions:
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
    return UCCVirtualLadderActions(
        primal,
        linearize(primal, MIXED_AMPLITUDES, max_elements=max_elements),
        transpose_program(
            primal, ("r2ab",), inputs=MIXED_AMPLITUDES, max_elements=max_elements
        ),
    )


@cache
def build_uccsd_uhf_same_spin_virtual_pair_programs(
    layout: UCCSDLayout, spin: str, *, max_elements: int = 1_000_000
) -> UCCVirtualLadderActions:
    """Use only the selected spin's antisymmetric virtual pair operator.

    Each spin block is independent. Callers with fewer than two orbitals
    on either axis have an empty physical same-spin doubles block and do
    not allocate a pair operator for it.
    """
    if not isinstance(layout, UCCSDLayout):
        raise TypeError("UHF virtual pairs require UCCSDLayout")
    if spin not in ("alpha", "beta"):
        raise ValueError("UHF virtual pair spin must be alpha or beta")
    ss = "aa" if spin == "alpha" else "bb"
    tau_aa, _, tau_bb = _canonical_tau(layout)
    tau = tau_aa if spin == "alpha" else tau_bb
    nocc, _, nvirt, _ = tau.spec.shape
    if nocc < 2 or nvirt < 2:
        raise ValueError(
            "UHF same-spin virtual pairs require a nonempty physical doubles block"
        )
    occupied = _Pairs.create(nocc, "occupied_pair", diagonal=False)
    virtual = _Pairs.create(nvirt, "virtual_antisymmetric", diagonal=False)
    operator = virtual.operator("spatial_coulomb_antisymmetric")
    residual = "r2" + ss
    inputs = ("t1" + ss[0], "t2" + ss)
    primal = Program(
        {residual: _packed_same_spin_ladder(tau, occupied, virtual, operator)},
        provenance={
            "method": "UCCSD",
            "term": "bare_uhf_same_spin_virtual_pairs",
            "spin": spin,
            "version": 1,
        },
    )
    return UCCVirtualLadderActions(
        primal,
        linearize(primal, inputs, max_elements=max_elements),
        transpose_program(
            primal, (residual,), inputs=inputs, max_elements=max_elements
        ),
    )
