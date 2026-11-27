"""Stationary actions for a supplied explicit-triples residual and scalar.

The formulation owner supplies the screened equations in physical independent
coordinates. This compiler layer constructs their derivatives with the shared
TensorIR AD. The runtime owns the primal/transpose solves and packed block
mapping. A linear residual can have a general action, including cross-block
and off-diagonal terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from generativeqc_compiler.tensor import Program, add, linearize, transpose_program
from generativeqc_compiler.tensor.ad_program import DEFAULT_MAX_ELEMENTS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from generativeqc_compiler.tensor import JVPProgram, VJPProgram


@dataclass(frozen=True)
class ExplicitTriplesStationaryPrograms:
    """Electronic actions for L(t, x, λ) = E(t, x) + λ·R(t, x).

    A unit seed gives the negative energy gradient from ``lambda_rhs``.
    Solve ``residual_vjp(λ) = lambda_rhs(1)`` at a converged primal state.
    ``parameter_vjp`` then takes scalar seed one and the converged residual
    seeds λ. Its outputs are partial derivatives of the stationary scalar
    through all selected sources, including any parameters of the action.
    """

    primal: Program
    residual_jvp: JVPProgram
    residual_vjp: VJPProgram
    lambda_rhs: VJPProgram
    energy_output: str
    amplitude_to_residual: tuple[tuple[str, str], ...]

    def parameter_vjp(
        self, parameters: Iterable[str], *, max_elements: int = DEFAULT_MAX_ELEMENTS
    ) -> VJPProgram:
        """Return the selected parent, orbital, fitting or other source cotangents."""
        parameters = tuple(parameters)
        if not parameters or set(parameters).intersection(
            amplitude for amplitude, _ in self.amplitude_to_residual
        ):
            raise ValueError(
                "select independent source parameters outside triples amplitudes"
            )
        return transpose_program(
            self.primal,
            (
                self.energy_output,
                *(residual for _, residual in self.amplitude_to_residual),
            ),
            inputs=parameters,
            max_elements=max_elements,
        )


def build_explicit_triples_stationary_programs(
    primal: Program,
    amplitude_to_residual: Mapping[str, str],
    *,
    energy_output: str = "triples_energy",
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> ExplicitTriplesStationaryPrograms:
    """Differentiate a selected triples scalar and its physical residual blocks.

    Block inputs and outputs must share real Euclidean coordinates. The
    formulation may represent permutations and metric weights internally, but
    the public block pairing must be independent and symmetry free. Other
    differentiable graph inputs remain available as source parameters.
    """
    pairs = tuple(sorted(amplitude_to_residual.items()))
    outputs = tuple(residual for _, residual in pairs)
    energy = primal.outputs.get(energy_output)
    if energy is None or energy.spec.shape or energy.spec.dtype != "float64":
        raise ValueError("explicit triples requires a scalar float64 energy output")
    if not pairs or len(set(outputs)) != len(outputs) or energy_output in outputs:
        raise ValueError("triples amplitudes require distinct residual blocks")
    inputs = {
        node.attrs["name"]: node
        for node in primal.nodes
        if node.op == "input" and node.spec.differentiable
    }
    for amplitude, residual in pairs:
        if amplitude not in inputs or residual not in primal.outputs:
            raise ValueError(f"unknown triples coordinate pair {amplitude}/{residual}")
        t_spec = inputs[amplitude].spec
        r_spec = primal.outputs[residual].spec
        if (
            t_spec.dtype != "float64"
            or r_spec.dtype != "float64"
            or t_spec.symmetries
            or r_spec.symmetries
            or t_spec.representation != r_spec.representation
            or tuple(axis.domain for axis in t_spec.indices)
            != tuple(axis.domain for axis in r_spec.indices)
        ):
            raise ValueError(
                f"incompatible triples coordinate pair {amplitude}/{residual}"
            )

    amplitudes = tuple(amplitude for amplitude, _ in pairs)
    # Retain amplitude inputs outside the scalar's support to define their zero
    # Lambda RHS blocks as well.
    negative_energy = Program(
        {energy_output: add(energy, coefficients=(-1,))},
        definitions=tuple(inputs[name] for name in amplitudes),
    )
    return ExplicitTriplesStationaryPrograms(
        primal=primal,
        residual_jvp=linearize(
            primal, amplitudes, outputs=outputs, max_elements=max_elements
        ),
        residual_vjp=transpose_program(
            primal, outputs, inputs=amplitudes, max_elements=max_elements
        ),
        lambda_rhs=transpose_program(
            negative_energy,
            (energy_output,),
            inputs=amplitudes,
            max_elements=max_elements,
        ),
        energy_output=energy_output,
        amplitude_to_residual=pairs,
    )
