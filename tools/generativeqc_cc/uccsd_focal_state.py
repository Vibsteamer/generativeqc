"""Cold stationary fixed-cusp UCCSD-F12b amplitudes and Lambda in a source checkout."""

from generativeqc.experimental.f12_focal import FocalUCCEquationSources
from generativeqc.profiles import canonical_hash
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout

from .native_tensor_cpu import NativeCCTensorExecutor
from .native_tensor_cuda import CudaCCTensorExecutor
from .solver import SolverOptions
from .uccsd_fixed_f12 import _PreparedFixedSourceUCCSD, _PreparedFixedSourceUCCSDLambda
from .uccsd_lambda import UCCSDLambdaOptions, UCCSDLambdaResult
from .uccsd_solver import UCCSDResult


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
    ref = sources.fixed.reference
    if (ref.executed_backend == "cuda") != isinstance(executor, CudaCCTensorExecutor):
        raise ValueError("UCC equation sources and stationary solver must share a backend")
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
    with _PreparedFixedSourceUCCSD(
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
    ) as cc:
        state = cc.solve()
        if not state.converged:
            raise RuntimeError(f"Molecular UCCSD-F12b amplitudes: {state.reason}")
        with _PreparedFixedSourceUCCSDLambda(
            cc, state, options=lambda_options, orbital_gap_preconditioner=True
        ) as response:
            dual = response.solve()
            response.qualify(dual)
    return state, dual
