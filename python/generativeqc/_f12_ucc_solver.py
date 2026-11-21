"""Cold stationary fixed-cusp UCCSD-F12b amplitudes and Lambda from molecular equation sources."""

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from generativeqc_compiler.cc.uccsd_df_virtual_ladder import (
    FACTOR_INPUTS,
    build_uccsd_df_virtual_ladder_amplitude_programs,
)
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.cc.uccsd_response import _df_remainder_source_vjps
from generativeqc_compiler.common.resources import checked_bytes

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
        if sources.uhf_df_ladder is not None:
            raise ValueError(
                "The independent-spin bounded ladder requires an actual UHF reference"
            )
        oa = ob = ref.occupied[0]
        ea = eb = ref.orbital_energies[0]
    else:
        oa, ob = ref.occupied
        ea, eb = ref.orbital_energies
        if sources.rhf_df_ladder is not None:
            raise ValueError(
                "The bounded spatial ladder requires an actual RHF reference"
            )
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
        rhf_df_ladder=sources.rhf_df_ladder,
        virtual_factors=sources.virtual_factors,
        uhf_df_ladder=sources.uhf_df_ladder,
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
    if cc.rhf_df_ladder is not None or cc.uhf_df_ladder is not None:
        return _bounded_weights(sources, response, dual)
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


def _bounded_weights(
    sources: FocalUCCEquationSources,
    response: _PreparedFixedSourceUCCSDLambda,
    dual: UCCSDLambdaResult,
) -> UCCSDF12Pullback:
    """Compute grouped ordinary/geminal and separate MP2Fit source cotangents."""
    cc = response.prepared
    limits, factors = cc.rhf_df_ladder or cc.uhf_df_ladder, cc.virtual_factors
    assert limits is not None and factors is not None
    if cc.rhf_df_ladder is not None:
        spin_factors = {
            name: factors[name.removesuffix("_alpha").removesuffix("_beta")]
            for name in FACTOR_INPUTS
        }
    else:
        spin_factors = factors
    extent = spin_factors["fit_left_alpha"].shape[0]
    width = min(extent, limits.auxiliary_tile)
    n_full, tail = divmod(extent, width)
    element_limit = response.options.max_bytes // 8
    ordinary_programs = tuple(
        vjp.program
        for vjp in _df_remainder_source_vjps(response.programs, element_limit)
    )
    auxiliary_programs = {
        n: build_uccsd_df_virtual_ladder_amplitude_programs(
            cc.layout, n, max_elements=element_limit
        ).factor_vjp.program
        for n in {width, tail}
        if n
    }
    resources = tuple(response._resource(graph) for graph in ordinary_programs)
    auxiliary_resources = {
        n: response._resource(graph) for n, graph in auxiliary_programs.items()
    }
    additional_work = (
        sum(resource.scalar_work for resource in resources)
        + n_full * auxiliary_resources[width].scalar_work
        + (auxiliary_resources[tail].scalar_work if tail else 0)
    )
    work = checked_bytes(
        additional_work
        + response.qualification_scalar_work
        + response._response_setup_work(),
        "bounded DF complete source work",
    )
    if work > limits.max_source_work:
        raise MemoryError("Bounded DF source semantic-work budget exceeded")
    retained = 0
    graph_peaks = []
    for graph, resource in zip(ordinary_programs, resources, strict=True):
        graph_peaks.append(retained + resource.native_bytes)
        retained += sum(
            output.spec.itemsize * output.spec.size for output in graph.outputs.values()
        )
    retained += sum(value.nbytes for value in spin_factors.values())
    graph_peaks.extend(
        retained + resource.native_bytes for resource in auxiliary_resources.values()
    )
    if response._repeated is not None:
        released = response._repeated.remainder.retained_bytes + sum(
            value.nbytes for value in cc.spatial_operators.values()
        )
    else:
        assert cc.uhf_bare is not None
        released = cc.uhf_bare.retained_host_bytes
    peak = checked_bytes(
        response.base_bytes
        + response._extra_response_bytes()
        - released
        + max(graph_peaks),
        "bounded DF complete source numeric storage",
    )
    if peak > response.options.max_bytes:
        raise MemoryError("Bounded DF source numeric storage budget exceeded")
    packed, diagnostics = response.qualify(dual)
    response.release_repeated_operators()
    seeds = response._seeds(packed)
    ordinary, fixed = {}, {}
    for graph in ordinary_programs:
        bars = cc.executor.execute(
            graph, {**response.feeds, **seeds, "bar_correlation_energy": np.array(1.0)}
        )
        for name, value in bars.items():
            source = name.removeprefix("bar_")
            weights = ordinary if source in sources.ordinary else fixed
            specification = (
                sources.ordinary if weights is ordinary else sources.fixed.feeds
            )
            if (
                value.dtype != np.float64
                or value.shape != specification[source].shape
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    "Bounded DF source cotangent failed physical shape/finite validation"
                )
            weights[source] = value
        del bars
    if ordinary.keys() != sources.ordinary.keys() or fixed.keys() != {
        name for name in cc.feeds if name in sources.fixed.feeds
    }:
        raise ValueError("Incomplete bounded DF stationary source cotangents")
    virtual = {name: np.empty_like(spin_factors[name]) for name in FACTOR_INPUTS}
    for begin in range(0, extent, width):
        stop = min(begin + width, extent)
        inputs = {
            name: np.ascontiguousarray(spin_factors[name][begin:stop])
            for name in FACTOR_INPUTS
        }
        graph = auxiliary_programs[stop - begin]
        bars = cc.executor.execute(graph, {**response.amplitudes, **inputs, **seeds})
        for name, target in virtual.items():
            value = bars["bar_" + name]
            if (
                value.dtype != np.float64
                or value.shape != target[begin:stop].shape
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    "Bounded DF factor cotangent failed physical shape/finite validation"
                )
            target[begin:stop] = value
    return UCCSDF12Pullback(
        response.correlation_energy,
        diagnostics["primal_residual_max"],
        diagnostics["stationarity_norm"],
        ordinary,
        fixed,
        virtual,
        peak,
        work,
    )


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


def replay_focal_uccsd_f12b_with_cotangents(
    sources: FocalUCCEquationSources,
    packed_amplitudes: np.ndarray,
    packed_multipliers: np.ndarray,
    correlation_energy: float,
    *,
    options: SolverOptions,
    lambda_options: UCCSDLambdaOptions,
    executor: NativeCCTensorExecutor | CudaCCTensorExecutor,
) -> tuple[UCCSDResult, UCCSDLambdaResult, UCCSDF12Pullback]:
    """Qualify archived T and Lambda against the complete current molecular equations."""
    with _prepare_focal_ucc(sources, options, executor) as cc:
        state = cc._replay_fixed_source_amplitudes(
            packed_amplitudes, correlation_energy
        )
        with _PreparedFixedSourceUCCSDLambda(
            cc, state, options=lambda_options, orbital_gap_preconditioner=True
        ) as response:
            dual = response._replay_multipliers(packed_multipliers)
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
    "replay_focal_uccsd_f12b_with_cotangents",
    "solve_focal_uccsd_f12b",
    "solve_focal_uccsd_f12b_energy",
    "solve_focal_uccsd_f12b_with_cotangents",
]
