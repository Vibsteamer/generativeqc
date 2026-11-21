"""Geometry-driven contributions for the experimental F12 focal provider."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from ._f12_focal_cc_sources import (
    FocalUCCEquationSources,
    FocalUCCFixedSources,
    molecular_ucc_sources,
)
from ._f12_focal_energy import (
    FocalLowOrderEnergy,
    FocalLowOrderForces,
    _check_request,
    molecular_low_order_energy,
)
from ._f12_focal_increment import FocalF12Sources, reference_f12_increment
from ._f12_focal_integrals import (
    contract_live_weights,
    focal_shell_layout,
    forward_integrals,
    metric_whitener,
)
from ._f12_focal_one_electron import FocalOneElectron, one_electron_sources
from ._f12_focal_projector import focal_cabs_projector
from ._f12_focal_reference import FocalReference, FocalReferenceSeed, df_reference
from ._f12_focal_singles import FocalCabsSingles, reference_cabs_singles
from ._f12_focal_ump2 import FocalCoulombSource, reference_df_ump2
from .elements import checked_integer

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable
    from pathlib import Path

    from ._f12_cc_stationary import UCCSDF12Pullback
    from ._f12_focal_tplus_sources import SpinOccupied
    from ._f12_ucc_solver import (
        CudaCCTensorExecutor,
        NativeCCTensorExecutor,
        SolverOptions,
        UCCSDLambdaOptions,
        UCCSDLambdaResult,
        UCCSDResult,
    )
    from ._uccsd_rhf_virtual_pairs import RHFDFLadderOptions
    from ._uccsd_uhf_virtual_ladder import UHFDFLadderOptions
    from .experimental.f12_focal import FocalEvaluation, FocalRequest


@dataclass(frozen=True)
class FocalUCCStationary:
    """One fixed-cusp UCCSD-F12b solution in the canonical molecular source frame.

    primal holds the correlation energy in Eh and physical amplitudes. dual
    holds the qualified Lambda multipliers and stationarity diagnostics.
    sources retains the same determinant and Hamiltonian for response work.
    """

    sources: FocalUCCEquationSources
    primal: UCCSDResult
    dual: UCCSDLambdaResult
    source_cotangents: UCCSDF12Pullback | None


@dataclass(frozen=True)
class FocalUCCSDResult:
    """Fixed-cusp all-active UCCSD-F12b energy (Eh) and Cartesian forces (Eh/bohr)."""

    energy: float
    forces: np.ndarray
    hf_cabs_energy: float
    hf_cabs_forces: np.ndarray
    ordinary_cc_forces: np.ndarray
    fixed_cc_forces: np.ndarray
    stationary: FocalUCCStationary


@dataclass(frozen=True)
class FocalTriplesEvaluation:
    """One triples increment with its own force and complementary-space rank."""

    energy: float
    forces: np.ndarray | None
    cabs_rank: int | None = None


@dataclass
class _Panel:
    layout: dict
    data: dict[str, np.ndarray]
    whitener: np.ndarray
    fit_id: str

    def obs(self, request: FocalRequest, n: int) -> FocalCoulombSource:
        return FocalCoulombSource(
            request.surface_id,
            request.basis.orbital.identity,
            self.fit_id,
            request.backend,
            self.data["ao_coulomb"][:, :n, :n],
            self.data["metric"],
            self.whitener,
        )

    def joint(self, request: FocalRequest, *, explicit: bool) -> FocalF12Sources:
        return FocalF12Sources(
            request.surface_id,
            request.basis.basis_id,
            self.fit_id,
            request.backend,
            request.basis.beta if explicit else None,
            self.data
            if explicit
            else {k: self.data[k] for k in ("ao_coulomb", "metric")},
            self.whitener,
        )


@dataclass(frozen=True, kw_only=True)
class MolecularFocalProvider:
    """Evaluate geometric HF+CABS and all-active RHF/UHF MP2-F12/3C(FIX) focal terms.

    Supply native F12 shell and optional separate one-electron libraries. When
    supplied, the latter also provides the real-spherical projections. CUDA
    uses an FP64 TensorIR runner, the device metric/response library, and the
    canonical-response ABI of the main native library. Each call solves one
    physical determinant on the backend in the request. Pass this callable as
    the provider of singlepoint(), or evaluate individual plan steps. Select a
    high-order evaluator explicitly for conventional post-MP2 and triples steps.
    On all-active RHF singlet and UHF roots the provider also evaluates the same-reference difference
    between CCSD-F12b and MP2-F12 correlation energies and forces.
    lambda_options controls the stationary multiplier solve of each CC role.
    Its workspace is bounded by budget_mib.
    rhf_df_ladder selects the bounded DF virtual ladder on a restricted reference.
    uhf_df_ladder selects independent virtual operators on an unrestricted reference.
    Supply reference_seed to transport a converged determinant between geometries
    in internally evaluated molecular roles. A callable can select each basis seed.
    """

    shell_library: Path
    one_electron_library: Path | None = None
    metric_library: Path | None = None
    response_library: ctypes.CDLL | None = None
    canonical_library: ctypes.CDLL | None = None
    run: Callable | None = None
    high_order: Callable[[FocalRequest], FocalEvaluation] | None = None
    # One model owner call for all triples variants on a common basis and root.
    # The owner may reuse matching CC and DF state and must return role-keyed
    # increments with exactly the requested derivative scope.
    triples_models: Callable[..., dict[str, FocalTriplesEvaluation]] | None = None
    # Availability of both energy and full analytic force for this exact request.
    qualified_tplus: Callable[[FocalRequest], bool] | None = None
    relative_cutoff: float = 1e-9
    budget_mib: int = 1536
    _cuda_cc_target: tuple[str, Path, Callable[[dict], None] | None] | None = field(
        default=None, repr=False, compare=False
    )
    lambda_options: UCCSDLambdaOptions | None = None
    rhf_df_ladder: RHFDFLadderOptions | None = None
    uhf_df_ladder: UHFDFLadderOptions | None = None

    reference_seed: (
        FocalReferenceSeed | Callable[[FocalRequest], FocalReferenceSeed] | None
    ) = None

    def _seed_options(self, request: FocalRequest) -> dict[str, FocalReferenceSeed]:
        if self.reference_seed is None:
            return {}
        value = (
            self.reference_seed(request)
            if callable(self.reference_seed)
            else self.reference_seed
        )
        if not isinstance(value, FocalReferenceSeed):
            raise TypeError(
                "reference_seed must supply a FocalReferenceSeed for each molecular role"
            )
        return {"seed": value}

    @classmethod
    def cuda(
        cls,
        *,
        shell_library: Path,
        response_library: Path,
        architecture: str,
        cache_directory: Path,
        one_electron_library: Path | None = None,
        budget_mib: int = 1536,
        relative_cutoff: float = 1e-9,
        high_order: Callable[[FocalRequest], FocalEvaluation] | None = None,
        triples_models: Callable[..., dict[str, FocalTriplesEvaluation]] | None = None,
        qualified_tplus: Callable[[FocalRequest], bool] | None = None,
        record_graph: Callable[[dict], None] | None = None,
        lambda_options: UCCSDLambdaOptions | None = None,
        rhf_df_ladder: RHFDFLadderOptions | None = None,
        uhf_df_ladder: UHFDFLadderOptions | None = None,
        reference_seed: (
            FocalReferenceSeed | Callable[[FocalRequest], FocalReferenceSeed] | None
        ) = None,
    ) -> MolecularFocalProvider:
        """Use the installed CUDA backend with compiled shell and metric/response libraries.

        For one L20 per process set architecture="sm_89". cache_directory holds
        the compiled TensorIR artifacts. The main DF and canonical response
        library uses the regular GenerativeQC CUDA library resolution.
        """
        import ctypes

        from . import _native
        from ._f12_focal_runner import focal_cuda_runner

        budget = checked_integer(
            budget_mib, "molecular focal budget in MiB", low=1, high=(2**63 - 1) >> 20
        )
        return cls(
            shell_library=shell_library,
            one_electron_library=one_electron_library,
            metric_library=response_library,
            response_library=ctypes.CDLL(str(response_library.resolve())),
            canonical_library=_native.load_library(device="cuda", device_id=0),
            run=focal_cuda_runner(
                architecture,
                cache_directory,
                max_bytes=budget << 20,
                record=record_graph,
            ),
            relative_cutoff=relative_cutoff,
            budget_mib=budget,
            high_order=high_order,
            triples_models=triples_models,
            qualified_tplus=qualified_tplus,
            lambda_options=lambda_options,
            rhf_df_ladder=rhf_df_ladder,
            uhf_df_ladder=uhf_df_ladder,
            reference_seed=reference_seed,
            _cuda_cc_target=(architecture, cache_directory, record_graph),
        )

    def low_order_energy(
        self,
        request: FocalRequest,
        *,
        expected_cabs_rank: int | None = None,
        include_hf_cabs: bool = True,
        include_occupied_partitions: bool = False,
    ) -> FocalLowOrderEnergy:
        """Return HF+CABS and MP2-F12 energies for one all-active pair request.

        A single canonical RHF or UHF determinant supplies both components.
        Only forward integrals and electronic contractions are evaluated.
        Pass the center CABS rank to track its differentiable branch at a displaced point.
        Set include_hf_cabs=False when only the pair component is required.
        Occupied partitions retain the canonical spin order for triples scaling.
        """
        return molecular_low_order_energy(
            self,
            request,
            expected_cabs_rank,
            include_hf_cabs=include_hf_cabs,
            include_occupied_partitions=include_occupied_partitions,
        )

    def tplus_scaling_forces(
        self,
        request: FocalRequest,
        reference: FocalReference,
        mp2_weights: SpinOccupied,
        f12_weights: SpinOccupied,
        *,
        expected_cabs_rank: int,
    ) -> np.ndarray:
        """Relax the MP2 and F12 orbital energy seeds supplied by a (T+) owner."""
        budget = _check_request(self, request)
        one = one_electron_sources(
            request,
            library=self.one_electron_library or self.shell_library,
            derivatives=True,
            budget_mib=budget,
        )
        return self._pair_components(
            request,
            reference,
            one,
            self._pair_panels(request),
            expected_cabs_rank=expected_cabs_rank,
            mp2_weights=mp2_weights,
            f12_weights=f12_weights,
        )[3]

    def low_order_energy_and_forces(
        self, request: FocalRequest, *, include_hf_cabs: bool = True
    ) -> FocalLowOrderForces:
        """Evaluate both low-order analytic forces and energies on one canonical root.

        Use low_order_energy with this center's CABS rank for a displaced energy.
        Both center roles share the physical determinant and the AO integral panels.
        Set include_hf_cabs=False to evaluate just the pair response on this basis.
        """
        budget = _check_request(self, request)
        reference = df_reference(
            request,
            forces=include_hf_cabs,
            budget_mib=budget,
            **self._seed_options(request),
        )
        one = one_electron_sources(
            request,
            library=self.one_electron_library or self.shell_library,
            derivatives=True,
            budget_mib=budget,
        )
        panels = self._pair_panels(request)
        spin_mp2, increment, rank, correlation_forces = self._pair_components(
            request, reference, one, panels, expected_cabs_rank=None
        )
        singles_energies = hf_cabs_forces = None
        if include_hf_cabs:
            singles, hf_cabs_forces = self._singles_components(
                replace(request, role="hf_cabs"),
                reference,
                one,
                panels[0],
                expected_cabs_rank=rank,
            )
            singles_energies = singles.spin_energies
        energies = FocalLowOrderEnergy(
            reference, rank, singles_energies, spin_mp2, increment
        )
        return FocalLowOrderForces(energies, hf_cabs_forces, correlation_forces)

    def ucc_fixed_sources(self, request: FocalRequest) -> FocalUCCFixedSources:
        """Generate W_vo, W_vv, C and the V+B-X scalar for fixed-cusp UCC.

        Supply the all-active singlet RHF or UHF pair step with its declared JKFit, F12Fit,
        OptRI and exponent. Returned MO tensors and the reference share a frame.
        """
        return molecular_ucc_sources(self, request, include_ordinary=False)

    def ucc_equation_sources(self, request: FocalRequest) -> FocalUCCEquationSources:
        """Generate all ordinary and fixed-geminal inputs for one all-active singlet RHF or UHF pair step.

        The normal-ordered Fock blocks use JKFit and the antisymmetrized
        two-electron Hamiltonian blocks use MP2Fit. The fixed sources use
        this same reference and the requested F12Fit.
        """
        return molecular_ucc_sources(self, request, include_ordinary=True)

    def ucc_stationary(
        self,
        request: FocalRequest,
        *,
        options: SolverOptions | None = None,
        lambda_options: UCCSDLambdaOptions | None = None,
        source_cotangents: bool = False,
    ) -> FocalUCCStationary:
        """Solve T and Lambda for an all-active singlet RHF or UHF molecular pair request.

        The returned primal total is the DF-HF energy plus fixed-cusp
        UCCSD-F12b correlation. Obtain CABS singles separately. The CPU
        reference uses compiled FP64 CC equations. The CUDA provider runs
        those equations on device zero with its configured target and cache.
        Set source_cotangents to differentiate the stationary correlation
        energy with respect to the active ordinary and fixed MO inputs.
        """
        return self._ucc_stationary(
            request,
            options,
            lambda_options,
            source_cotangents=source_cotangents,
            reference_forces=False,
        )

    def _cc_executor(
        self, request: FocalRequest
    ) -> tuple[NativeCCTensorExecutor | CudaCCTensorExecutor, int]:
        from ._f12_ucc_solver import CudaCCTensorExecutor, NativeCCTensorExecutor

        budget = (
            checked_integer(
                self.budget_mib,
                "molecular focal budget in MiB",
                low=1,
                high=(2**63 - 1) >> 20,
            )
            << 20
        )
        if request.backend == "cuda":
            if self._cuda_cc_target is None:
                raise ValueError(
                    "Construct a CUDA stationary provider with MolecularFocalProvider.cuda"
                )
            from ._f12_focal_runner import focal_cuda_compiler

            arch, cache, record = self._cuda_cc_target
            executor = CudaCCTensorExecutor(
                max_bytes=budget,
                compiler=focal_cuda_compiler(arch),
                cache=cache,
                record_graph=record,
            )
        elif request.backend == "cpu_reference":
            executor = NativeCCTensorExecutor(max_bytes=budget)
        else:
            raise ValueError(
                "A molecular UCC stationary solve requires an explicit backend"
            )
        return executor, budget

    def _ucc_stationary(
        self,
        request: FocalRequest,
        options: SolverOptions | None,
        lambda_options: UCCSDLambdaOptions | None,
        *,
        source_cotangents: bool,
        reference_forces: bool,
        reference: FocalReference | None = None,
        expected_cabs_rank: int | None = None,
    ) -> FocalUCCStationary:
        from ._f12_ucc_solver import (
            SolverOptions,
            UCCSDLambdaOptions,
            solve_focal_uccsd_f12b,
            solve_focal_uccsd_f12b_with_cotangents,
        )

        executor, budget = self._cc_executor(request)
        sources = molecular_ucc_sources(
            self,
            request,
            include_ordinary=True,
            reference_forces=reference_forces,
            reference=reference,
            expected_cabs_rank=expected_cabs_rank,
        )
        if options is None:
            options = SolverOptions(max_bytes=budget)
        if lambda_options is None:
            lambda_options = self.lambda_options
        if lambda_options is None:
            lambda_options = UCCSDLambdaOptions(max_bytes=budget)
        if lambda_options.max_bytes > budget:
            lambda_options = replace(lambda_options, max_bytes=budget)
        if source_cotangents:
            primal, dual, weights = solve_focal_uccsd_f12b_with_cotangents(
                sources,
                options=options,
                lambda_options=lambda_options,
                executor=executor,
            )
        else:
            primal, dual = solve_focal_uccsd_f12b(
                sources,
                options=options,
                lambda_options=lambda_options,
                executor=executor,
            )
            weights = None
        return FocalUCCStationary(sources, primal, dual, weights)

    def f12_doubles_energy(
        self,
        pair_request: FocalRequest,
        *,
        low_order: FocalLowOrderEnergy,
        expected_cabs_rank: int | None = None,
    ) -> float:
        """CCSD-F12b minus MP2-F12 on the supplied determinant and CABS rank."""
        from ._f12_ucc_solver import SolverOptions, solve_focal_uccsd_f12b_energy

        if expected_cabs_rank is not None and expected_cabs_rank != low_order.cabs_rank:
            raise ValueError("The displaced and low-order CABS ranks disagree")
        sources = molecular_ucc_sources(
            self,
            pair_request,
            include_ordinary=True,
            reference=low_order.reference,
            expected_cabs_rank=low_order.cabs_rank,
        )
        executor, budget = self._cc_executor(pair_request)
        correlation = solve_focal_uccsd_f12b_energy(
            sources,
            options=SolverOptions(max_bytes=budget),
            executor=executor,
        )
        return correlation - low_order.mp2_f12_correlation_energy

    def f12_doubles_energy_and_forces(
        self,
        pair_request: FocalRequest,
        *,
        low_order: FocalLowOrderForces,
    ) -> tuple[float, np.ndarray]:
        """Differentiate CCSD-F12b minus the supplied matched MP2-F12 energy."""
        low = low_order.energies
        solution = self._ucc_stationary(
            pair_request,
            None,
            None,
            source_cotangents=True,
            reference_forces=False,
            reference=low.reference,
            expected_cabs_rank=low.cabs_rank,
        )
        ordinary, fixed = self._ucc_correlation_forces(pair_request, solution)
        assert solution.primal.correlation_energy is not None
        return (
            solution.primal.correlation_energy - low.mp2_f12_correlation_energy,
            ordinary + fixed - low_order.mp2_f12_correlation_forces,
        )

    def triples_energy(
        self, request: FocalRequest, *, reference: FocalReference
    ) -> float:
        """Conventional triples on the supplied canonical determinant in Eh."""
        from ._f12_focal_ucc_triples import molecular_triples_on_reference

        return molecular_triples_on_reference(
            request,
            reference,
            budget_mib=self.budget_mib,
            forces=False,
        )[0]

    def triples_energy_and_forces(
        self,
        request: FocalRequest,
        *,
        reference: FocalReference,
    ) -> tuple[float, np.ndarray]:
        """Conventional triples and its analytic Cartesian forces on a shared root."""
        from ._f12_focal_ucc_triples import molecular_triples_on_reference

        energy, forces = molecular_triples_on_reference(
            request,
            reference,
            budget_mib=self.budget_mib,
            forces=True,
        )
        assert forces is not None
        return energy, forces

    def ucc_ordinary_source_forces(
        self, request: FocalRequest, solution: FocalUCCStationary
    ) -> np.ndarray:
        """Contract the 38 ordinary stationary MO source weights with nuclear derivatives (Eh/bohr)."""
        from ._f12_focal_cc_ordinary import ordinary_ucc_source_forces

        weights = solution.source_cotangents
        if weights is None:
            raise ValueError(
                "Ordinary UCC forces require a stationary solution with source_cotangents=True"
            )
        return ordinary_ucc_source_forces(
            self,
            request,
            solution.sources,
            weights.ordinary_weights,
            virtual_factor_weights=weights.virtual_factor_weights,
        )

    def _ucc_correlation_forces(
        self, request: FocalRequest, solution: FocalUCCStationary
    ) -> tuple[np.ndarray, np.ndarray]:
        from ._f12_focal_cc_fixed import fixed_ucc_source_forces

        weights = solution.source_cotangents
        assert weights is not None
        ordinary = self.ucc_ordinary_source_forces(request, solution)
        fixed = fixed_ucc_source_forces(
            self, request, solution.sources, weights.fixed_source_weights
        )
        return ordinary, fixed

    def uccsd_f12b(
        self,
        request: FocalRequest,
        *,
        options: SolverOptions | None = None,
        lambda_options: UCCSDLambdaOptions | None = None,
    ) -> FocalUCCSDResult:
        """Calculate molecular UCCSD-F12b with DF-HF, CABS singles and analytic forces.

        Supply the all-active singlet RHF or UHF pair request for one cardinal.
        One DF-HF determinant supplies the reference energy and force, CABS
        singles, ordinary CC Hamiltonian, and fixed-geminal equations.
        """
        solution = self._ucc_stationary(
            request,
            options,
            lambda_options,
            source_cotangents=True,
            reference_forces=True,
        )
        source = solution.sources
        reference = source.fixed.reference
        assert solution.primal.correlation_energy is not None
        ordinary, fixed = self._ucc_correlation_forces(request, solution)
        singles_request = replace(request, role="hf_cabs")
        one = one_electron_sources(
            singles_request,
            library=self.one_electron_library or self.shell_library,
            derivatives=True,
            budget_mib=self.budget_mib,
        )
        hf_cabs, hf_cabs_forces = self._singles(
            singles_request,
            reference,
            one,
            self._panel(singles_request, "jk_fit"),
            expected_cabs_rank=source.fixed.cabs_rank,
        )
        return FocalUCCSDResult(
            hf_cabs + solution.primal.correlation_energy,
            hf_cabs_forces + ordinary + fixed,
            hf_cabs,
            hf_cabs_forces,
            ordinary,
            fixed,
            solution,
        )

    def _panel(
        self, request: FocalRequest, role: str, *, orbital_only: bool = False
    ) -> _Panel:
        host = request.backend == "cpu_reference"
        ordinary = role == "mp2_fit" or orbital_only
        layout = focal_shell_layout(
            request,
            self.shell_library,
            host_reference=host,
            budget_mib=self.budget_mib,
            ordinary_fit=role if ordinary else None,
            joint_jk=role == "jk_fit"
            and request.role == "mp2_f12_correlation"
            and not ordinary,
            projection_library=self.one_electron_library,
        )
        raw, _ = forward_integrals(
            layout,
            self.shell_library,
            host_reference=host,
            coulomb_only=role != "f12_fit",
            budget_mib=self.budget_mib,
        )
        m = raw["metric"]
        if host:
            from scipy.linalg import solve_triangular

            whitener = solve_triangular(
                np.linalg.cholesky(m), np.eye(len(m)), lower=True
            )
        else:
            assert self.metric_library is not None
            whitener, _ = metric_whitener(
                m, self.metric_library, budget_bytes=self.budget_mib << 20
            )
        return _Panel(layout, raw, whitener, getattr(request.basis, role).identity)

    def _gradient(
        self,
        request: FocalRequest,
        panel: _Panel,
        weights: dict[str, np.ndarray],
        *,
        coulomb_only: bool,
    ) -> np.ndarray:
        row = contract_live_weights(
            panel.layout,
            weights,
            self.shell_library,
            host_reference=request.backend == "cpu_reference",
            singles_only=coulomb_only,
            budget_mib=self.budget_mib,
        )
        if not row["passes_contraction_checks"]:
            raise ArithmeticError(
                "The molecular shell nuclear derivative failed its contraction checks"
            )
        return np.asarray(row["gradient_Eh_per_bohr"])

    def __call__(self, request: FocalRequest) -> FocalEvaluation:
        from .experimental.f12_focal import FocalEvaluation, FocalTerm

        if (
            request.role in ("triples_plus", "triples_f12")
            and self.triples_models is not None
        ):
            from .experimental.f12_focal import (
                FocalMolecularResult,
                FocalPlan,
                _molecular_f12_plan,
            )

            result = _molecular_f12_plan(
                FocalPlan(((request, 1.0),)), self, forces=True
            )
            assert isinstance(result, FocalMolecularResult)
            return result.components[0]
        if request.role not in (
            "hf_cabs",
            "mp2_f12_correlation",
            "ccsd_f12b_minus_mp2_f12",
        ):
            if self.high_order is None:
                raise ValueError(f"Supply the high-order evaluator for {request.role}")
            return self.high_order(request)
        checked_integer(
            self.budget_mib,
            "molecular focal budget in MiB",
            low=1,
            high=(2**63 - 1) >> 20,
        )
        if request.backend not in ("cpu_reference", "cuda"):
            raise ValueError("A molecular focal term requires an explicit backend")
        if request.backend == "cuda" and (
            self.run is None or self.metric_library is None
        ):
            raise ValueError(
                "CUDA molecular focal evaluation requires a device metric library and FP64 graph runner"
            )
        pair = request.role != "hf_cabs"
        if (
            request.basis.cabs is None
            or not np.isfinite(self.relative_cutoff)
            or self.relative_cutoff <= 0
        ):
            raise ValueError(
                "The molecular focal term requires an OptRI basis and positive relative cutoff"
            )
        if pair and (
            request.root_id not in ("rhf", "uhf")
            or (request.root_id == "rhf" and request.multiplicity != 1)
            or request.correlation_space_id != "all-active"
            or request.basis.mp2_fit is None
            or request.basis.f12_fit is None
            or request.basis.beta is None
        ):
            raise ValueError(
                "The pair term requires all-active UHF or singlet RHF, MP2Fit, F12Fit and beta"
            )
        solution = None
        calculation = request
        if request.role == "ccsd_f12b_minus_mp2_f12":
            calculation = replace(request, role="mp2_f12_correlation")
            solution = self._ucc_stationary(
                calculation, None, None, source_cotangents=True, reference_forces=False
            )
            ref = solution.sources.fixed.reference
        else:
            ref = df_reference(
                request,
                forces=not pair,
                budget_mib=self.budget_mib,
                **self._seed_options(request),
            )
        one = one_electron_sources(
            calculation,
            library=self.one_electron_library or self.shell_library,
            budget_mib=self.budget_mib,
        )
        n = one.obs_ao_count
        if not np.allclose(
            ref.overlap, one.overlap[:n, :n], atol=2e-8, rtol=0
        ) or not np.allclose(ref.hcore, one.hcore[:n, :n], atol=2e-8, rtol=0):
            raise ValueError(
                "The canonical and combined one-electron OBS matrices disagree"
            )
        if pair:
            rank = solution.sources.fixed.cabs_rank if solution is not None else None
            energy, forces = self._pair(calculation, ref, one, expected_cabs_rank=rank)
            if solution is not None:
                ordinary, fixed = self._ucc_correlation_forces(calculation, solution)
                assert solution.primal.correlation_energy is not None
                energy = solution.primal.correlation_energy - energy
                forces = ordinary + fixed - forces
        else:
            energy, forces = self._singles(
                request, ref, one, self._panel(request, "jk_fit")
            )
        return FocalEvaluation(
            FocalTerm(
                request.surface_id,
                request.basis.series_id,
                request.basis.basis_id,
                request.basis.cardinal,
                request.role,
                energy,
                forces,
            ),
            executed_backend=request.backend,
            converged=True,
        )

    def _singles(
        self,
        request: FocalRequest,
        ref: FocalReference,
        one: FocalOneElectron,
        jk: _Panel,
        *,
        expected_cabs_rank: int | None = None,
    ) -> tuple[float, np.ndarray]:
        singles, forces = self._singles_components(
            request, ref, one, jk, expected_cabs_rank=expected_cabs_rank
        )
        return singles.hf_cabs_energy, forces

    def _singles_components(
        self,
        request: FocalRequest,
        ref: FocalReference,
        one: FocalOneElectron,
        jk: _Panel,
        *,
        expected_cabs_rank: int | None,
    ) -> tuple[FocalCabsSingles, np.ndarray]:
        projector = focal_cabs_projector(
            request,
            one,
            expected_rank=expected_cabs_rank,
            relative_cutoff=self.relative_cutoff,
            library=self.response_library,
        )
        singles = reference_cabs_singles(
            request,
            ref,
            one,
            complement=projector.frame,
            three_center=jk.data["ao_coulomb"],
            metric=jk.data["metric"],
            whitener=jk.whitener,
            relax=True,
            run=self.run,
            library=self.response_library,
            relative_cutoff=self.relative_cutoff,
            budget_mib=self.budget_mib,
        )
        w = singles.weights
        assert ref.forces is not None and w is not None
        assert one.d_overlap is not None and one.d_hcore is not None
        grad = self._gradient(
            request,
            jk,
            {"metric": w["metric"], "ao_coulomb": w["three_center"]},
            coulomb_only=True,
        )
        grad += one.nuclear_gradient(w["overlap"], w["hcore"], run=self.run)
        return singles, ref.forces - grad

    def _pair_panels(self, request: FocalRequest) -> tuple[_Panel, _Panel, _Panel]:
        basis = request.basis
        assert basis.mp2_fit is not None and basis.f12_fit is not None
        f12 = self._panel(request, "f12_fit")
        jk = (
            f12
            if f12.fit_id == basis.jk_fit.identity
            else self._panel(request, "jk_fit")
        )
        mp2 = next((p for p in (jk, f12) if p.fit_id == basis.mp2_fit.identity), None)
        if mp2 is None:
            mp2 = self._panel(request, "mp2_fit")
        return jk, mp2, f12

    def _pair(
        self,
        request: FocalRequest,
        ref: FocalReference,
        one: FocalOneElectron,
        *,
        expected_cabs_rank: int | None = None,
    ) -> tuple[float, np.ndarray]:
        spin_mp2, increment, _, forces = self._pair_components(
            request,
            ref,
            one,
            self._pair_panels(request),
            expected_cabs_rank=expected_cabs_rank,
        )
        return float(sum(spin_mp2) + increment), forces

    def _pair_components(
        self,
        request: FocalRequest,
        ref: FocalReference,
        one: FocalOneElectron,
        panels: tuple[_Panel, _Panel, _Panel],
        *,
        expected_cabs_rank: int | None,
        mp2_weights: SpinOccupied | None = None,
        f12_weights: SpinOccupied | None = None,
    ) -> tuple[tuple[float, float, float], float, int, np.ndarray]:
        n = one.obs_ao_count
        jk, mp2, f12 = panels
        ordinary = reference_df_ump2(
            request,
            ref,
            jk.obs(request, n),
            mp2.obs(request, n),
            run=self.run,
            library=self.response_library,
            budget_mib=self.budget_mib,
            occupied_weights=mp2_weights,
            canonical_library=self.canonical_library,
        )
        explicit = reference_f12_increment(
            request,
            ref,
            one,
            jk.joint(request, explicit=False),
            f12.joint(request, explicit=True),
            expected_cabs_rank=expected_cabs_rank,
            relative_cutoff=self.relative_cutoff,
            budget_mib=self.budget_mib,
            run=self.run,
            library=self.response_library,
            canonical_library=self.canonical_library,
            occupied_weights=f12_weights,
        )
        # A shared fitted operator has one nuclear derivative. Scatter the OBS
        # cotangents into the leading block of its OBS+OptRI basis, including
        # metric weights. Operator equality requires the same *basis identity*.
        w = ordinary.weights
        wk, wf = explicit.jkfit_weights, explicit.f12fit_weights
        wk["metric"] += w["metric"]
        wk["ao_coulomb"][:, :n, :n] += w["three_center"]
        mp_weights = {"metric": w["mp2_metric"], "ao_coulomb": w["mp2_three_center"]}
        if mp2 is jk or mp2 is f12:
            target = wk if mp2 is jk else wf
            target["metric"] += mp_weights["metric"]
            target["ao_coulomb"][:, :n, :n] += mp_weights["ao_coulomb"]
        if jk is f12:
            wf["metric"] += wk["metric"]
            wf["ao_coulomb"] += wk["ao_coulomb"]
        grad = self._gradient(request, f12, wf, coulomb_only=False)
        if jk is not f12:
            grad += self._gradient(request, jk, wk, coulomb_only=True)
        if mp2 is not jk and mp2 is not f12:
            grad += self._gradient(request, mp2, mp_weights, coulomb_only=True)
        assert one.d_overlap is not None and one.d_hcore is not None
        for key in ("overlap", "hcore"):
            explicit.one_electron_weights[key][:n, :n] += w[key]
        grad += one.nuclear_gradient(
            explicit.one_electron_weights["overlap"],
            explicit.one_electron_weights["hcore"],
            run=self.run,
        )
        return ordinary.spin_energies, explicit.energy, explicit.cabs_rank, -grad
