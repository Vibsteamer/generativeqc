"""Cold stationary fixed-cusp UCCSD-F12b amplitudes and Lambda from molecular equation sources."""

from collections.abc import Iterator
from contextlib import contextmanager

from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout

from generativeqc.profiles import canonical_hash

from ._f12_cc_stationary import (
    UCCSDF12Pullback,
    unrestricted_cc_f12_stationary_pullback,
)
from ._f12_focal_cc_sources import FocalUCCEquationSources
from ._ucc.native_tensor_cpu import NativeCCTensorExecutor
from ._ucc.native_tensor_cuda import CudaCCTensorExecutor
from ._ucc.solver import SolverOptions
from ._ucc.uccsd_fixed_f12 import (
    _PreparedFixedSourceUCCSD,
    _PreparedFixedSourceUCCSDLambda,
)
from ._ucc.uccsd_lambda import UCCSDLambdaOptions, UCCSDLambdaResult
from ._ucc.uccsd_solver import UCCSDResult


def _prepare_focal_ucc(
    sources: FocalUCCEquationSources,
    options: SolverOptions | None,
    executor: NativeCCTensorExecutor | CudaCCTensorExecutor | None,
) -> _PreparedFixedSourceUCCSD:
    ref = sources.fixed.reference
    if (ref.executed_backend == "cuda") != isinstance(executor, CudaCCTensorExecutor):
        raise ValueError(
            "UCC equation sources and stationary solver must share a backend"
        )
    if len(ref.occupied) == 1:
        # RHF has one independent spatial orbital variable for both spin lines.
        oa = ob = ref.occupied[0]
        ea = eb = ref.orbital_energies[0]
    else:
        oa, ob = ref.occupied
        ea, eb = ref.orbital_energies
    layout = UCCSDLayout(oa, ob, len(ea) - oa, len(eb) - ob)
    hamiltonian_id = canonical_hash(
        {
            "surface": ref.surface_id,
            "basis": ref.basis_id,
            "jk_fit": ref.jk_fit_id,
            "mp2_fit": sources.mp2_fit_id,
            "f12_fit": sources.fixed.f12_fit_id,
        }
    )
    return _PreparedFixedSourceUCCSD(
        layout,
        sources.ordinary,
        sources.fixed.feeds,
        orbital_energies_alpha=ea,
        orbital_energies_beta=eb,
        reference_id=ref.surface_id,
        hamiltonian_id=hamiltonian_id,
        reference_energy=ref.energy,
        options=options,
        executor=executor,
    )


def _primal(cc: _PreparedFixedSourceUCCSD) -> UCCSDResult:
    state = cc.solve()
    if not state.converged or state.correlation_energy is None:
        raise RuntimeError(f"Molecular UCCSD-F12b amplitudes: {state.reason}")
    return state


def solve_focal_uccsd_f12b_energy(
    sources: FocalUCCEquationSources,
    *,
    options: SolverOptions,
    executor: NativeCCTensorExecutor | CudaCCTensorExecutor,
) -> float:
    """Return the replay-qualified correlation energy from the amplitude solve."""
    with _prepare_focal_ucc(sources, options, executor) as cc:
        state = _primal(cc)
        assert state.correlation_energy is not None
        return state.correlation_energy


@contextmanager
def _cold_state(
    sources: FocalUCCEquationSources,
    options: SolverOptions | None,
    lambda_options: UCCSDLambdaOptions | None,
    executor: NativeCCTensorExecutor | CudaCCTensorExecutor | None,
) -> Iterator[tuple[_PreparedFixedSourceUCCSDLambda, UCCSDResult, UCCSDLambdaResult]]:
    with _prepare_focal_ucc(sources, options, executor) as cc:
        state = _primal(cc)
        with _PreparedFixedSourceUCCSDLambda(
            cc, state, options=lambda_options, orbital_gap_preconditioner=True
        ) as response:
            dual = response.solve()
            response.qualify(dual)
            yield response, state, dual


def solve_focal_uccsd_f12b(
    sources: FocalUCCEquationSources,
    *,
    options: SolverOptions | None = None,
    lambda_options: UCCSDLambdaOptions | None = None,
    executor: NativeCCTensorExecutor | CudaCCTensorExecutor | None = None,
) -> tuple[UCCSDResult, UCCSDLambdaResult]:
    """Solve T and Lambda from molecular UCC equation sources in their canonical frame.

    The primal total contains the UHF reference and stationary CC correlation.
    The separate CABS singles term is supplied by its own molecular provider.
    A CUDA-source calculation requires an explicit CUDA CC executor.
    """
    with _cold_state(sources, options, lambda_options, executor) as (_, state, dual):
        return state, dual


def _weights(
    sources: FocalUCCEquationSources,
    response: _PreparedFixedSourceUCCSDLambda,
    state: UCCSDResult,
    dual: UCCSDLambdaResult,
) -> UCCSDF12Pullback:
    cc = response.prepared
    packed = response._packed_result(dual)
    weights = unrestricted_cc_f12_stationary_pullback(
        cc.layout,
        sources.ordinary,
        sources.fixed.feeds,
        state.packed_amplitudes,
        packed,
        cc.executor.execute,
    )
    if abs(weights.correlation_energy - response.correlation_energy) > 1e-11:
        raise ValueError("Molecular source weights failed the stationary energy replay")
    return weights


def solve_focal_uccsd_f12b_with_cotangents(
    sources: FocalUCCEquationSources,
    *,
    options: SolverOptions | None = None,
    lambda_options: UCCSDLambdaOptions | None = None,
    executor: NativeCCTensorExecutor | CudaCCTensorExecutor | None = None,
) -> tuple[UCCSDResult, UCCSDLambdaResult, UCCSDF12Pullback]:
    """Solve one cold stationary root and its joint VJP within the same native owner."""
    with _cold_state(sources, options, lambda_options, executor) as (
        response,
        state,
        dual,
    ):
        return state, dual, _weights(sources, response, state, dual)


def focal_ucc_source_cotangents(
    sources: FocalUCCEquationSources,
    state: UCCSDResult,
    dual: UCCSDLambdaResult,
    *,
    options: SolverOptions | None = None,
    lambda_options: UCCSDLambdaOptions | None = None,
    executor: NativeCCTensorExecutor | CudaCCTensorExecutor | None = None,
) -> UCCSDF12Pullback:
    """Return active molecular input cotangents for the same stationary T and Lambda.

    Tensor entries are gradients of the CC Lagrangian with respect to the
    canonical MO inputs. Their negative contractions with true molecular
    input derivatives give the ordinary and fixed-source correlation forces.
    """
    with (
        _prepare_focal_ucc(sources, options, executor) as cc,
        _PreparedFixedSourceUCCSDLambda(
            cc, state, options=lambda_options, orbital_gap_preconditioner=True
        ) as response,
    ):
        return _weights(sources, response, state, dual)


__all__ = [
    "CudaCCTensorExecutor",
    "NativeCCTensorExecutor",
    "SolverOptions",
    "UCCSDLambdaOptions",
    "UCCSDLambdaResult",
    "UCCSDResult",
    "focal_ucc_source_cotangents",
    "solve_focal_uccsd_f12b",
    "solve_focal_uccsd_f12b_energy",
    "solve_focal_uccsd_f12b_with_cotangents",
]
