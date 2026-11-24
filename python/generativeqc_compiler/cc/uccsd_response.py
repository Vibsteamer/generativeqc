"""UCCSD amplitude and Hamiltonian derivatives from the shared physical DAG.

These programs differentiate fixed-amplitude energy/residual equations, never
iterations, DIIS or shifted proposal denominators. The runtime response owner
supplies the physical packed primal/dual maps and qualifies stationary states.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from generativeqc_compiler.cc.uccsd_df_virtual_ladder import (
    UCCDFVirtualLadderAmplitudePrograms,
    build_uccsd_df_virtual_ladder_amplitude_programs,
)
from generativeqc_compiler.cc.uccsd_equations import (
    build_f12_correction_program,
    build_f12_df_remainder_program,
    build_f12_program,
    build_program,
)
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDLayout
from generativeqc_compiler.cc.uccsd_rhf_virtual_pairs import (
    build_uccsd_rhf_virtual_pair_programs,
)
from generativeqc_compiler.tensor import (
    JVPProgram,
    Program,
    VJPProgram,
    add,
    einsum,
    input_tensor,
    linearize,
    transpose_program,
)
from generativeqc_compiler.tensor.iteration_reuse import (
    IterationReusePrograms,
    analyze_iteration_reuse,
    materialize_iteration_frontier,
)

RESIDUALS = tuple(name.replace("t", "r", 1) for name in BLOCK_NAMES)


@dataclass(frozen=True)
class UCCSDResponsePrograms:
    """Shared CPU/CUDA RHS, J*, J and fresh Lagrangian-stationarity actions.

    Amplitude directions expand with U. Residual cotangents lift with U/w,
    where w=4 for aa/bb doubles and w=1 otherwise. Returned amplitude bars
    pull back with U*, not ordinary amplitude packing. Physical Lambda is
    the unweighted excitation dual; its full arrays expand with U, without
    the one-quarter residual-seed factor.
    """

    primal: Program
    energy_vjp: VJPProgram
    residual_vjp: VJPProgram
    residual_jvp: JVPProgram
    stationarity_vjp: VJPProgram

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Canonical independent Fock/integral inputs, excluding amplitudes."""
        return tuple(
            sorted(
                {
                    node.attrs["name"]
                    for node in self.primal.nodes
                    if node.op == "input" and node.attrs["name"] not in BLOCK_NAMES
                }
            )
        )


def build_response_programs(
    layout: UCCSDLayout, *, max_elements: int = 1_000_000
) -> UCCSDResponsePrograms:
    """Generate exact typed derivatives for all five unrestricted blocks."""
    if not isinstance(layout, UCCSDLayout):
        raise TypeError("UCCSD response requires UCCSDLayout")
    return _response_programs(
        build_program(layout.oa, layout.ob, layout.va, layout.vb), max_elements
    )


def build_f12_response_programs(
    layout: UCCSDLayout, *, max_elements: int = 1_000_000
) -> UCCSDResponsePrograms:
    """Build fixed-source amplitude and Lambda actions for CCSD-F12b."""
    return _response_programs(build_f12_program(layout), max_elements)


@dataclass(frozen=True)
class UCCSDF12DFResponsePrograms:
    """Full F12b residual and response split between one remainder and DF slices.

    Accumulate bare residuals and amplitude actions before packing the physical
    outputs. Lambda stationarity adds the same slice transposes to the remainder
    stationarity action. Factor cotangents belong to each DF slice and share the
    metric and orbital pullback of the ordinary Coulomb sources.
    """

    remainder: UCCSDResponsePrograms
    ladder: UCCDFVirtualLadderAmplitudePrograms


def build_f12_df_response_programs(
    layout: UCCSDLayout, auxiliary_slice: int, *, max_elements: int = 1_000_000
) -> UCCSDF12DFResponsePrograms:
    """Generate the complete set of electronic response parts for bounded DF."""
    return UCCSDF12DFResponsePrograms(
        _response_programs(build_f12_df_remainder_program(layout), max_elements),
        build_uccsd_df_virtual_ladder_amplitude_programs(
            layout, auxiliary_slice, max_elements=max_elements
        ),
    )


def build_f12_df_remainder_source_vjps(
    programs: UCCSDF12DFResponsePrograms, *, max_elements: int = 1_000_000
) -> tuple[VJPProgram, VJPProgram]:
    """Keep the large three-virtual source cotangents separate from other sources."""
    return _df_remainder_source_vjps(programs.remainder, max_elements)


def _df_remainder_source_vjps(
    remainder: UCCSDResponsePrograms,
    max_elements: int,
) -> tuple[VJPProgram, VJPProgram]:
    three_virtual = tuple(
        name for name in remainder.parameter_names if name.startswith("govvv_")
    )
    remaining = tuple(
        name for name in remainder.parameter_names if name not in three_virtual
    )
    return (
        transpose_program(
            remainder.primal,
            ("correlation_energy", *RESIDUALS),
            inputs=three_virtual,
            max_elements=max_elements,
        ),
        transpose_program(
            remainder.primal,
            ("correlation_energy", *RESIDUALS),
            inputs=remaining,
            max_elements=max_elements,
        ),
    )


@dataclass(frozen=True)
class UCCRHFRepeatedLambdaPrograms:
    """The retained remainder and fixed spatial bare parts of a physical transpose."""

    remainder: IterationReusePrograms
    bare: Program


def build_f12_rhf_repeated_lambda_programs(
    layout: UCCSDLayout, *, max_retained_bytes: int, max_elements: int = 1_000_000
) -> UCCRHFRepeatedLambdaPrograms:
    """Bind the remainder's immutable frontier at the current stationary amplitudes."""
    bare = build_uccsd_rhf_virtual_pair_programs(layout, max_elements=max_elements)
    base = build_f12_df_remainder_program(layout)
    adjoint = transpose_program(
        base, RESIDUALS, inputs=BLOCK_NAMES, max_elements=max_elements
    ).program
    frozen = tuple(
        node.attrs["name"]
        for node in adjoint.live_nodes
        if node.op == "input" and not node.attrs["name"].startswith("bar_")
    )
    plan = analyze_iteration_reuse(adjoint, invariant_inputs=frozen)
    return UCCRHFRepeatedLambdaPrograms(
        materialize_iteration_frontier(
            adjoint, plan, max_retained_bytes=max_retained_bytes
        ),
        bare.residual_vjp.program,
    )


def fixed_source_lagrangian_direction(layout: UCCSDLayout) -> Program:
    """Contract all molecular F12 feed tangents with the CC Lagrangian.

    Supply the same packed-dual lift as the ordinary residual transpose.
    Differentiating just the F12 additions holds ordinary MO inputs fixed.
    """
    return _parameter_direction(
        build_f12_correction_program(layout), "fixed_source_lagrangian"
    )


def molecular_lagrangian_direction(layout: UCCSDLayout) -> Program:
    """Differentiate the ordinary and fixed-source correlation Lagrangian jointly."""
    return _parameter_direction(build_f12_program(layout), "correlation_lagrangian")


def _parameter_direction(primal: Program, output: str) -> Program:
    terms = [primal.outputs["correlation_energy"]]
    for name in RESIDUALS:
        residual = primal.outputs[name]
        seed = input_tensor(
            "bar_" + name, replace(residual.spec, role="input", differentiable=False)
        )
        axes = "ia" if name.startswith("r1") else "ijab"
        terms.append(einsum(f"{axes},{axes}->", seed, residual))
    lagrangian = Program({output: add(*terms)})
    sources = tuple(
        node.attrs["name"]
        for node in primal.live_nodes
        if node.op == "input" and node.attrs["name"] not in BLOCK_NAMES
    )
    return linearize(lagrangian, sources).program


def _response_programs(primal: Program, max_elements: int) -> UCCSDResponsePrograms:
    return UCCSDResponsePrograms(
        primal,
        transpose_program(
            primal,
            ("correlation_energy",),
            inputs=BLOCK_NAMES,
            max_elements=max_elements,
        ),
        transpose_program(
            primal, RESIDUALS, inputs=BLOCK_NAMES, max_elements=max_elements
        ),
        linearize(primal, BLOCK_NAMES, outputs=RESIDUALS, max_elements=max_elements),
        transpose_program(
            primal,
            ("correlation_energy", *RESIDUALS),
            inputs=BLOCK_NAMES,
            max_elements=max_elements,
        ),
    )


def build_parameter_vjp(
    programs: UCCSDResponsePrograms, parameter: str, *, max_elements: int = 1_000_000
) -> VJPProgram:
    """Generate one fixed-orbital correlation-Lagrangian parameter cotangent.

    Seed energy with one and residuals with the full physical-dual lift.
    Several redundant/permuted inputs can depend on one physical Hamiltonian
    perturbation; upstream consumers must accumulate all canonical bindings.
    For F12 the caller also pulls source cotangents back through the cusp,
    projector, integral and metric transformations. Reference energy,
    normal ordering and orbital/nuclear response are separate.
    """
    if not isinstance(programs, UCCSDResponsePrograms):
        raise TypeError("UCCSD parameter VJP requires response programs")
    if not isinstance(parameter, str) or parameter not in programs.parameter_names:
        raise ValueError("unknown UCCSD Fock/integral parameter or F12 source")
    return transpose_program(
        programs.primal,
        ("correlation_energy", *RESIDUALS),
        inputs=(parameter,),
        max_elements=max_elements,
    )
