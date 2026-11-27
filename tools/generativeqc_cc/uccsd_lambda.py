"""Shared native CPU/CUDA UCCSD Lambda and fixed-orbital Hamiltonian VJPs.

Krylov control and physical excitation-coordinate maps are explicitly host
owned. All physical equations and derivative contractions use the borrowed
parent's native executor; no interpreter, PySCF or restricted-response fallback
belongs to this owner. Orbital relaxation and nuclear forces are separate.
"""

from __future__ import annotations

from generativeqc._ucc.uccsd_lambda import (
    UCCSDLambdaOptions,
    UCCSDLambdaResult,
    UCCSDParameterCotangent,
    _Counter,
    _freeze,
    _PackedUCCSDLambda,
    _Resource,
    _validate_physical_multipliers,
)
from generativeqc_compiler.cc.uccsd_response import build_response_programs

from .uccsd_solver import PreparedUCCSD


class PreparedUCCSDLambda(_PackedUCCSDLambda):
    """Solve the fixed-orbital Lambda equations of a conventional UCCSD owner."""

    _parent_type = PreparedUCCSD
    _build_response_programs = staticmethod(build_response_programs)


__all__ = [
    "PreparedUCCSDLambda",
    "UCCSDLambdaOptions",
    "UCCSDLambdaResult",
    "UCCSDParameterCotangent",
    "_Counter",
    "_PackedUCCSDLambda",
    "_Resource",
    "_freeze",
    "_validate_physical_multipliers",
]
