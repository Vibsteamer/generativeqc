"""Geometry-driven contributions for the experimental F12 focal provider."""

from __future__ import annotations

import os
from types import MethodType
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from . import _native
from ._f12_focal_integral_kernel import (
    DEFAULT_SLATER_FIT_NORMALIZATION,
    slater_normalization_code,
)

from ._f12_focal_cc_sources import (
    FocalUCCEquationSources,
    FocalUCCFixedSources,
    molecular_ucc_sources,
)
from ._f12_focal_cv import FocalDeltaCVResult, molecular_delta_cv_mp2_energy, molecular_delta_cv_mp2_energy_and_forces
from ._f12_focal_energy import (
    FocalLowOrderEnergy,
    FocalLowOrderForces,
    _check_request,
    molecular_low_order_energy,
)
from ._f12_focal_increment import FocalF12Sources, reference_f12_increment
from ._f12_focal_integrals import (
    contract_live_weights,
    contract_single_channel,
    focal_shell_layout,
    forward_integrals,
    forward_metrics_only,
    forward_single_ao_operator,
    metric_whitener,
)
from ._f12_focal_one_electron import FocalOneElectron, one_electron_sources
from ._f12_focal_projector import focal_cabs_projector
from ._f12_focal_reference import (
    FocalReference,
    FocalReferenceSeed,
    df_reference,
    partition_focal_reference,
)
from ._f12_focal_singles import FocalCabsSingles, reference_cabs_singles
from ._f12_focal_ump2 import FocalCoulombSource, ordinary_df_ump2_unrelaxed, reference_df_ump2
from ._f12_frozen_core import DEFAULT_PRODUCTION_CORRELATION_SPACE
from ._f12_triples_w_policy import TriplesWForceAdmission, admit_triples_w_forces
from ._f12_ram_hygiene import (
    disclose_host_assisted_stage,
    record_ram_phase,
    release_host_memory,
    should_stream_panel,
    streamed_f12_increment,
)
from .elements import checked_integer

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable
    from pathlib import Path

    from ._cuda_device import CudaDeviceIdentity
    from ._f12_cc_stationary import UCCSDF12Pullback
    from ._f12_focal_tplus_sources import (
        FocalOccupiedTriples,
        FocalTPlusWeights,
        SpinOccupied,
    )
    from ._f12_focal_ucc_triples import TriplesTelemetry
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
    """One triples increment with its own force, complementary-space rank, and solver provenance."""

    energy: float
    forces: np.ndarray | None
    cabs_rank: int | None = None
    executed_backend: str | None = None
    converged: bool | None = None
    triples_model_id: str | None = None
    telemetry: dict[str, object] = field(default_factory=dict)


_UHF_CC_BACKEND_ENV = "GENERATIVEQC_F12_UHF_CC_BACKEND"


def _occupied_scaling_executor(backend: str):
    """Disclose occupied-vector algebra only when the triples model executes it."""
    from generativeqc_compiler.tensor import execute

    def run(program, feeds):
        disclose_host_assisted_stage(
            "tplus_occupied_scaling",
            requested_backend=backend,
            detail="Occupied-orbital (T+) energy and cotangents use the host FP64 tensor executor.",
        )
        return execute(program, feeds)

    return run


def _uhf_cc_backend_override() -> Literal["native", "legacy"] | None:
    """Parse the unified UHF CCSD-F12b backend selector (native | legacy)."""
    raw = os.environ.get(_UHF_CC_BACKEND_ENV)
    if raw is None or not raw.strip():
        return None
    val = raw.strip().lower()
    if val in ("native", "legacy"):
        return val  # type: ignore[return-value]
    raise ValueError(
        f"Invalid {_UHF_CC_BACKEND_ENV}={raw!r}; expected 'native' or 'legacy'"
    )


_AO_NAME_TO_OP = {
    "ao_coulomb": "coulomb",
    "ao_f12": "f12",
    "ao_f12_squared": "f12_squared",
    "ao_f12_over_coulomb": "f12_over_coulomb",
    "ao_double_commutator": "double_commutator",
}

_METRIC_KEY_TO_OP = {
    "metric": "coulomb",
    "metric_f12": "f12",
    "metric_f12_squared": "f12_squared",
    "metric_f12_over_coulomb": "f12_over_coulomb",
    "metric_double_commutator": "double_commutator",
}


@dataclass
class _Panel:
    layout: dict
    data: dict[str, np.ndarray]
    whitener: np.ndarray
    fit_id: str
    streaming: bool = False
    library: Path | None = None
    host_reference: bool = False
    budget_mib: int = 512
    coulomb_only: bool = False
    slater_normalization: str = DEFAULT_SLATER_FIT_NORMALIZATION
    _obs_coulomb_cache: tuple[tuple, np.ndarray] | None = field(default=None, init=False)

    def load_ao(self, ao_name: str) -> np.ndarray:
        if not self.streaming:
            return self.data[ao_name]
        assert self.library is not None
        return forward_single_ao_operator(
            self.layout,
            self.library,
            _AO_NAME_TO_OP[ao_name],
            host_reference=self.host_reference,
            coulomb_only=self.coulomb_only,
            budget_mib=self.budget_mib,
            slater_normalization=self.slater_normalization,
        )

    @staticmethod
    def stream_key(loader: Callable) -> tuple | None:
        """Identify all shell evaluator inputs; opaque loaders must be replayed."""
        from ._f12_spin_ao import _digest_array_full

        if not isinstance(loader, MethodType) or loader.__func__ is not _Panel.load_ao:
            return None
        owner = loader.__self__
        if type(owner) is not _Panel or not owner.streaming or owner.library is None:
            return None
        return (
            id(owner), owner.fit_id, owner.slater_normalization, owner.host_reference,
            owner.coulomb_only, owner.budget_mib, str(owner.library.resolve()),
            tuple(
                (name, _digest_array_full(np.asarray(value)))
                for name, value in sorted(owner.layout.items())
            ),
        )

    def _cached_obs(self, key: tuple | None) -> np.ndarray | None:
        entry = self._obs_coulomb_cache
        if entry is not None and entry[0] == key:
            return entry[1].view()
        self._obs_coulomb_cache = None
        return None

    def obs_coulomb(self, n: int) -> np.ndarray:
        if not self.streaming:
            return self.data["ao_coulomb"][:, :n, :n]
        key = self.stream_key(self.load_ao)
        cached = self._cached_obs(key)
        if cached is not None and cached.shape[1] == n:
            return cached
        self._obs_coulomb_cache = None
        del cached
        full = self.load_ao("ao_coulomb")
        large = full.nbytes >= (256 << 20)
        block = full[:, :n, :n]
        # Bytes-backed storage prevents exposed OBS arrays from changing the cache.
        obs = np.frombuffer(block.tobytes(order="C"), dtype=block.dtype).reshape(block.shape)
        del block, full
        if large:
            release_host_memory()
        if key is not None:
            self._obs_coulomb_cache = (key, obs)
            return obs.view()
        return obs

    def obs(self, request: FocalRequest, n: int) -> FocalCoulombSource:
        return FocalCoulombSource(
            request.surface_id,
            request.basis.orbital.identity,
            self.fit_id,
            request.backend,
            self.obs_coulomb(n),
            self.data["metric"],
            self.whitener,
        )

    def joint(self, request: FocalRequest, *, explicit: bool) -> FocalF12Sources:
        if self.streaming:
            return FocalF12Sources(
                request.surface_id,
                request.basis.basis_id,
                self.fit_id,
                request.backend,
                request.basis.beta if explicit else None,
                self.data if explicit else {"metric": self.data["metric"]},
                self.whitener,
                ao_loader=self.load_ao,
                obs_coulomb=(
                    self._cached_obs(self.stream_key(self.load_ao))
                    if self._obs_coulomb_cache is not None else None
                ),
            )
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

    The default constructor uses the installed host-reference F12 shell. A
    separate one-electron library also supplies the real-spherical projections. CUDA
    uses an FP64 TensorIR runner, the device metric/response library, and the
    canonical-response ABI of the main native library. Each call solves one
    physical determinant on the backend in the request. Pass this callable as
    the provider of singlepoint(), or evaluate individual plan steps. Select a
    high-order evaluator explicitly for conventional post-MP2 and triples steps.
    On all-active RHF singlet and UHF roots the provider also evaluates the same-reference difference
    between CCSD-F12b and MP2-F12 correlation energies and forces.
    lambda_options controls the stationary multiplier solve of each CC role.
    Its workspace is bounded by budget_mib. For conventional (T) and (T+),
    triples_w_schedule="fp32_split3" explicitly selects nine FP32 partial GEMMs
    with FP64 panel storage/recombination. It is independent of single-pass
    triples_fp32_w and requires a supported host or spatial CUDA owner.
    Analytic triples forces run single-pass triples_fp32_w only on an NVIDIA
    L20, or as an explicit per-call diagnostic on other devices.
    rhf_df_ladder selects the bounded DF virtual ladder on a restricted reference.
    uhf_df_ladder selects independent virtual operators on an unrestricted reference.
    Supply reference_seed to transport a converged determinant between geometries
    in internally evaluated molecular roles. A callable can select each basis seed.
    """

    shell_library: Path = field(
        default_factory=lambda: _native.require_f12_library("focal_shell_i_host")
    )
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
    preconvergence: str = "auto"
    triples_fp32_w: bool = False
    triples_w_schedule: str | None = field(default=None, kw_only=True)
    slater_normalization: Literal[
        "negative_exponential", "negative_exponential_over_beta"
    ] = DEFAULT_SLATER_FIT_NORMALIZATION
    use_native_uhf_f12b: bool = True
    _cuda_cc_target: tuple[str, Path, Callable[[dict], None] | None] | None = field(
        default=None, repr=False, compare=False
    )
    _pair_source_cache: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    # Identifies the CUDA device of native triples work; None queries ordinal 0.
    _cuda_device: Callable[[], CudaDeviceIdentity] | None = field(
        default=None, repr=False, compare=False
    )
    lambda_options: UCCSDLambdaOptions | None = None
    rhf_df_ladder: RHFDFLadderOptions | None = None
    uhf_df_ladder: UHFDFLadderOptions | None = None

    reference_seed: (
        FocalReferenceSeed | Callable[[FocalRequest], FocalReferenceSeed] | None
    ) = None

    def __post_init__(self) -> None:
        from ._f12_focal_ucc_triples import _w_precision_code

        _w_precision_code(self.triples_fp32_w, self.triples_w_schedule)
        slater_normalization_code(self.slater_normalization)
        object.__setattr__(self, "_pair_source_cache", {})
        if (
            (self.triples_models is None and self.high_order is None)
            or self.triples_models is MolecularFocalProvider.default_triples_models
            or getattr(self.triples_models, "__func__", None)
            is MolecularFocalProvider.default_triples_models
        ):
            object.__setattr__(self, "triples_models", self.default_triples_models)

    def _admit_triples_w_forces(
        self, request: FocalRequest, diagnostic_fp32_single_w_forces: bool
    ) -> TriplesWForceAdmission:
        return admit_triples_w_forces(
            request.backend,
            triples_fp32_w=self.triples_fp32_w,
            triples_w_schedule=self.triples_w_schedule,
            preconvergence=self.preconvergence,
            cuda_device=self._cuda_device,
            diagnostic_fp32_single_w_forces=diagnostic_fp32_single_w_forces,
        )

    def supports_tplus(self, request: FocalRequest) -> bool:
        """Check the implemented occupied model and backend, independently of qualification."""
        if request.role != "triples_plus":
            return False
        basis = request.basis
        if (
            basis.cabs is None
            or basis.mp2_fit is None
            or basis.f12_fit is None
            or basis.beta is None
        ):
            return False
        if request.root_id == "rhf":
            return (
                request.multiplicity == 1
                and request.backend == "cuda"
                and self.run is not None
            )
        if request.root_id == "uhf":
            return (
                request.backend == "cuda" and self.run is not None
            ) or request.backend == "cpu_reference"
        return False

    def check_explicit_triples_scope(self, request: FocalRequest, *, forces: bool) -> None:
        if forces:
            raise NotImplementedError("Explicit Köhn (T)-F12 production NO-GO for 3N forces")
        if request.root_id != "rhf" or request.multiplicity != 1:
            raise NotImplementedError(
                "Explicit Köhn (T)-F12 energy requires a closed-shell RHF singlet"
            )
        if self.triples_w_schedule == "fp32_split3":
            raise NotImplementedError("Explicit Köhn (T)-F12 has no fp32_split3 W kernel")
        if any(atom.atomic_number in (35, 53) for atom in request.atoms):
            from ._f12_frozen_core import resolve_frozen_occupied

            # FfgCompleted is unstable for Br/I semicore correlation in valence OptRI.
            required = replace(request, correlation_space_id=DEFAULT_PRODUCTION_CORRELATION_SPACE)
            orbital_basis = request.basis.orbital.by_element
            electrons = sum(
                atom.atomic_number - orbital_basis[atom.atomic_number].ecp_core_electrons
                for atom in request.atoms
            ) - request.charge
            occupied = (electrons // 2, electrons // 2)
            frozen = resolve_frozen_occupied(request, occupied)[0]
            if frozen < resolve_frozen_occupied(required, occupied)[0]:
                raise NotImplementedError(
                    "Explicit FfgCompleted (T)-F12 excludes Br/I semicore correlation"
                )

    def default_triples_models(
        self,
        requests: tuple[FocalRequest, ...],
        *,
        reference: FocalReference,
        low_order: FocalLowOrderEnergy | FocalLowOrderForces | None,
        forces: bool,
        expected_cabs_rank: int | None,
    ) -> dict[str, FocalTriplesEvaluation]:
        """Evaluate the selected increments on the supplied molecular determinant."""
        by_role = {r.role: r for r in requests}
        results: dict[str, FocalTriplesEvaluation] = {}
        if "triples_f12" in by_role:
            from ._f12_focal_explicit_tf12 import spatial_rhf_explicit_tf12_energy

            req = by_role["triples_f12"]
            self.check_explicit_triples_scope(req, forces=forces)
            explicit = spatial_rhf_explicit_tf12_energy(
                self, req, reference, expected_cabs_rank=expected_cabs_rank
            )
            results[req.role] = FocalTriplesEvaluation(
                explicit.energy, None, explicit.cabs_rank,
                executed_backend=explicit.telemetry["backend"], converged=True,
                triples_model_id="kohn_extended_sp_rhf_ffg_v1",
                telemetry={**explicit.telemetry, "ccsd_f12b_energy": explicit.ccsd_f12b_energy},
            )
        conv_from_tplus: float | None = None
        if "triples_plus" in by_role:
            tplus_req = by_role["triples_plus"]
            if not self.supports_tplus(tplus_req):
                raise NotImplementedError(
                    f"Built-in (T+) is unavailable for root={tplus_req.root_id!r}, "
                    f"backend={tplus_req.backend!r}, multiplicity={tplus_req.multiplicity}"
                )
            if isinstance(low_order, FocalLowOrderEnergy) and low_order.occupied_mp2_f12 is not None:
                low_occ = low_order
            else:
                rank_hint = expected_cabs_rank
                if rank_hint is None and isinstance(low_order, FocalLowOrderForces):
                    rank_hint = low_order.energies.cabs_rank
                elif rank_hint is None and isinstance(low_order, FocalLowOrderEnergy):
                    rank_hint = low_order.cabs_rank
                pair_req = replace(tplus_req, role="mp2_f12_correlation")
                low_occ = self.low_order_energy(
                    pair_req,
                    expected_cabs_rank=rank_hint,
                    include_hf_cabs=False,
                    include_occupied_partitions=True,
                    reference=reference,
                )
            rank = expected_cabs_rank if expected_cabs_rank is not None else low_occ.cabs_rank
            if tplus_req.root_id == "rhf":
                if forces:
                    conv_from_tplus, tplus_ev, _ = (
                        self.restricted_tplus_energy_and_forces_on_reference(
                            tplus_req, low_order=low_occ, expected_cabs_rank=rank
                        )
                    )
                else:
                    conv_from_tplus, tplus_ev = self.restricted_tplus_energy_on_reference(
                        tplus_req, low_order=low_occ, expected_cabs_rank=rank
                    )
            else:
                if forces:
                    conv_from_tplus, tplus_ev, _ = (
                        self.unrestricted_tplus_energy_and_forces_on_reference(
                            tplus_req, low_order=low_occ, expected_cabs_rank=rank
                        )
                    )
                else:
                    conv_from_tplus, tplus_ev = self.unrestricted_tplus_energy_on_reference(
                        tplus_req, low_order=low_occ, expected_cabs_rank=rank
                    )
            results["triples_plus"] = tplus_ev
        if "triples" in by_role:
            conv_req = by_role["triples"]
            telemetry = {}
            if forces:
                conv_e, conv_f = self.triples_energy_and_forces(
                    conv_req, reference=reference, telemetry=telemetry
                )
                results["triples"] = FocalTriplesEvaluation(
                    conv_e,
                    conv_f,
                    None,
                    executed_backend=reference.executed_backend,
                    converged=True,
                    triples_model_id=f"conventional_{conv_req.root_id}",
                    telemetry=telemetry,
                )
            else:
                if conv_from_tplus is None:
                    conv_e = self.triples_energy(conv_req, reference=reference, telemetry=telemetry)
                else:
                    conv_e = conv_from_tplus
                    telemetry = dict(results["triples_plus"].telemetry)
                results["triples"] = FocalTriplesEvaluation(
                    conv_e,
                    None,
                    None,
                    executed_backend=reference.executed_backend,
                    converged=True,
                    triples_model_id=f"conventional_{conv_req.root_id}",
                    telemetry=telemetry,
                )
        return results

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
        architecture: str,
        cache_directory: Path,
        shell_library: Path | None = None,
        response_library: Path | None = None,
        one_electron_library: Path | None = None,
        budget_mib: int = 1536,
        relative_cutoff: float = 1e-9,
        preconvergence: str = "auto",
        triples_fp32_w: bool = False,
        triples_w_schedule: str | None = None,
        use_native_uhf_f12b: bool = True,
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
        slater_normalization: Literal[
            "negative_exponential", "negative_exponential_over_beta"
        ] = DEFAULT_SLATER_FIT_NORMALIZATION,
    ) -> MolecularFocalProvider:
        """Use the installed CUDA backend with compiled shell and metric/response libraries.

        Omitted library paths select the CUDA F12 targets in this package.
        For one L20 per process set architecture="sm_89". cache_directory holds
        the compiled TensorIR artifacts. The main DF and canonical response
        library uses the regular GenerativeQC CUDA library resolution.
        """
        import ctypes

        from ._f12_focal_runner import focal_cuda_runner

        budget = checked_integer(
            budget_mib, "molecular focal budget in MiB", low=1, high=(2**63 - 1) >> 20
        )
        slater_normalization_code(slater_normalization)
        if shell_library is None:
            shell_library = _native.require_f12_library("focal_shell_i")
        if response_library is None:
            response_library = _native.require_f12_library("focal_hf_cabs_response")
        # The main loader registers the wheel's CUDA runtime providers before
        # dlopen needs them for the response library.
        canonical_library = _native.load_library(device="cuda", device_id=0)
        return cls(
            shell_library=shell_library,
            one_electron_library=one_electron_library,
            metric_library=response_library,
            response_library=ctypes.CDLL(str(response_library.resolve())),
            canonical_library=canonical_library,
            run=focal_cuda_runner(
                architecture,
                cache_directory,
                max_bytes=budget << 20,
                record=record_graph,
            ),
            relative_cutoff=relative_cutoff,
            budget_mib=budget,
            slater_normalization=slater_normalization,
            preconvergence=preconvergence,
            triples_fp32_w=triples_fp32_w,
            triples_w_schedule=triples_w_schedule,
            use_native_uhf_f12b=use_native_uhf_f12b,
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
        reference: FocalReference | None = None,
    ) -> FocalLowOrderEnergy:
        """Return HF+CABS and MP2-F12 energies for one molecular pair request.

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
            reference=reference,
        )


    def delta_cv_mp2_energy(
        self,
        request_cv: FocalRequest,
        request_fc: FocalRequest | None = None,
    ) -> FocalDeltaCVResult:
        """Return pure conventional DF-MP2 additive core-valence energy (zero CABS, zero F12)."""
        return molecular_delta_cv_mp2_energy(self, request_cv, request_fc)

    def delta_cv_mp2_energy_and_forces(
        self,
        request_cv: FocalRequest,
        request_fc: FocalRequest | None = None,
    ) -> FocalDeltaCVResult:
        """Return pure conventional DF-MP2 additive core-valence energy and analytic 3N forces."""
        return molecular_delta_cv_mp2_energy_and_forces(self, request_cv, request_fc)


    def _cached_pair_sources(
        self, request: FocalRequest, *, derivatives: bool, budget: int
    ) -> tuple[FocalOneElectron, tuple[_Panel, _Panel, _Panel]]:
        """Bounded 1-entry reuse of read-only 1e and 3-center pair panels on identical geometry+basis."""
        b = request.basis
        req_norm = getattr(request, "slater_normalization", None)
        if b.f12_fit is not None and req_norm is not None and req_norm != self.slater_normalization:
            raise ValueError(
                f"Request slater_normalization {req_norm!r} disagrees with "
                f"provider.slater_normalization {self.slater_normalization!r}"
            )
        geom_key = tuple(
            (a.atomic_number, tuple((v if v else 0.0).hex() for v in a.position))
            for a in request.atoms
        )
        key = (
            geom_key,
            request.surface_id,
            request.charge,
            request.multiplicity,
            request.backend,
            b.basis_id,
            b.orbital.identity,
            b.jk_fit.identity,
            b.mp2_fit.identity if b.mp2_fit is not None else None,
            b.f12_fit.identity if b.f12_fit is not None else None,
            b.cabs.identity if b.cabs is not None else None,
            b.beta,
            self.slater_normalization,
            str(self.shell_library),
            str(self.one_electron_library) if self.one_electron_library is not None else None,
            int(budget),
            int(self.budget_mib),
        )
        entry = self._pair_source_cache.get("entry")
        if entry is not None and entry[0] == key:
            cached_deriv, cached_one, cached_panels = entry[1], entry[2], entry[3]
            if not derivatives or cached_deriv:
                if cached_one.surface_id != request.surface_id:
                    cached_one = replace(cached_one, surface_id=request.surface_id)
                return cached_one, cached_panels
            # Upgrading derivatives=False -> True on identical geometry+basis:
            # reuse read-only panels and release the old 1e matrices first.
            self._pair_source_cache.pop("entry", None)
            self._pair_source_cache.pop("projected_frame", None)
            del cached_one, entry
            release_host_memory()
            one = one_electron_sources(
                request,
                library=self.one_electron_library or self.shell_library,
                derivatives=derivatives,
                budget_mib=budget,
            )
            self._pair_source_cache["entry"] = (key, derivatives, one, cached_panels)
            return one, cached_panels
        if entry is not None or "projected_frame" in self._pair_source_cache:
            self._pair_source_cache.pop("entry", None)
            self._pair_source_cache.pop("projected_frame", None)
            del entry
            release_host_memory()
        one = one_electron_sources(
            request,
            library=self.one_electron_library or self.shell_library,
            derivatives=derivatives,
            budget_mib=budget,
        )
        panels = self._pair_panels(request)
        self._pair_source_cache["entry"] = (key, derivatives, one, panels)
        return one, panels

    def _evict_pair_source_cache(self) -> None:
        """Drop retained pair sources and projected frames and return host pages to the OS."""
        dropped = False
        for cache_key in ("entry", "projected_frame"):
            if cache_key in self._pair_source_cache:
                self._pair_source_cache.pop(cache_key, None)
                dropped = True
        if dropped:
            release_host_memory()

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
        one, panels = self._cached_pair_sources(request, derivatives=True, budget=budget)
        record_ram_phase("after_cached_pair_sources")
        forces = self._pair_components(
            request,
            reference,
            one,
            panels,
            expected_cabs_rank=expected_cabs_rank,
            mp2_weights=mp2_weights,
            f12_weights=f12_weights,
        )[3]
        if any(p.streaming for p in panels):
            del one, panels
            self._evict_pair_source_cache()
        return forces

    def tplus_from_occupied(
        self,
        request: FocalRequest,
        low_order: FocalLowOrderEnergy,
        triples: FocalOccupiedTriples,
        *,
        inactive: tuple[bool, ...] | None = None,
    ) -> FocalTPlusWeights:
        """Combine source partitions on one exact frame and return the three seeds.

        This step does not solve the conventional CC parent or the weighted
        triples response. Feed the MP2 and F12 seeds to tplus_scaling_forces
        and the triples seed to the same conventional triples response owner.
        """
        from generativeqc_compiler.cc.tplus import evaluate_tplus

        from ._f12_focal_tplus_sources import FocalTPlusWeights, checked_occupied_weights
        from ._f12_frozen_core import active_occupied_counts, resolve_frozen_occupied

        _check_request(self, request)
        ref, source = low_order.reference, low_order.occupied_mp2_f12
        basis = request.basis
        if (
            source is None
            or triples.reference is not ref
            or ref.surface_id != request.surface_id
            or ref.basis_id != basis.basis_id
            or ref.orbital_id != basis.orbital.identity
            or ref.jk_fit_id != basis.jk_fit.identity
            or ref.executed_backend != request.backend
            or triples.correlation_space_id != request.correlation_space_id
            or basis.mp2_fit is None
            or triples.correlation_fit_id != basis.mp2_fit.identity
        ):
            raise ValueError("(T+) sources require the same determinant, basis and correlation fit")
        spins = ref._pair_spin_indices(request)
        full_counts = tuple(ref.occupied[index] for index in spins)
        frozen = resolve_frozen_occupied(request, full_counts)
        counts = active_occupied_counts(full_counts, frozen)
        mp2 = checked_occupied_weights(source.mp2, counts)
        f12 = checked_occupied_weights(source.f12_increment, counts)
        if not np.allclose(
            [sum(np.sum(block) for block in parts) for parts in (mp2, f12)],
            [sum(low_order.ordinary_mp2_spin_energies), low_order.f12_increment],
            atol=1e-10, rtol=1e-10,
        ):
            raise ValueError("(T+) occupied partitions disagree with their unscaled energies")
        values = triples.values
        if request.backend == "cuda":
            from generativeqc_compiler.tensor import execute as _cpu_tensor_execute

            if self.run is None or self.run is _cpu_tensor_execute:
                raise RuntimeError("(T+) scalar requires the FP64 CUDA executor")

        if request.root_id == "rhf":
            if (
                not isinstance(values, np.ndarray)
                or values.dtype != np.float64
                or values.shape != (counts[0],)
            ):
                raise ValueError("RHF (T+) requires one spatial occupied triples vector")
            scaled = evaluate_tplus(
                mp2[0] + mp2[1],
                f12[0] + f12[1],
                values,
                inactive=inactive,
                run=_occupied_scaling_executor(request.backend),
            )
            mp2_seed = np.array(scaled.mp2_cotangent)
            f12_seed = np.array(scaled.f12_cotangent)
            return FocalTPlusWeights(
                scaled.energy, (mp2_seed, mp2_seed.copy()),
                (f12_seed, f12_seed.copy()), np.array(scaled.triples_cotangent),
            )
        if not isinstance(values, tuple):
            raise ValueError("UHF (T+) requires separate alpha and beta triples vectors")
        values = checked_occupied_weights(values, counts)
        scaled = evaluate_tplus(
            np.concatenate(mp2),
            np.concatenate(f12),
            np.concatenate(values),
            inactive=inactive,
            run=_occupied_scaling_executor(request.backend),
        )

        def split(source):
            weights = np.array(source)
            return weights[:counts[0]], weights[counts[0]:]

        return FocalTPlusWeights(
            scaled.energy, split(scaled.mp2_cotangent), split(scaled.f12_cotangent),
            split(scaled.triples_cotangent),
        )

    def restricted_tplus_energy_on_reference(
        self,
        request: FocalRequest,
        *,
        low_order: FocalLowOrderEnergy,
        expected_cabs_rank: int | None = None,
    ) -> tuple[float, FocalTriplesEvaluation]:
        """Evaluate both conventional and scaled (T) energies on one CUDA RHF frame.

        The caller supplies the retained low-order energies and occupied MP2-F12
        partitions. The two scalars share the native spatial CC solve and the
        conventional occupied triples contraction. Supports common or split
        JK/correlation fits, scalar ECP, frozen-core and mixed preconvergence.
        """
        from ._f12_focal_ucc_triples import molecular_occupied_triples_on_reference

        if request.role != "triples_plus" or request.root_id != "rhf":
            raise ValueError("Restricted (T+) requires the RHF scaled triples role")
        if request.backend != "cuda" or self.run is None:
            raise NotImplementedError("Restricted (T+) energy requires the CUDA executor")
        pair_request = replace(request, role="mp2_f12_correlation")
        _check_request(self, pair_request)
        if expected_cabs_rank is not None and expected_cabs_rank != low_order.cabs_rank:
            raise ValueError("The displaced (T+) CABS rank differs from the center")
        if low_order.occupied_mp2_f12 is None:
            raise ValueError("Restricted (T+) requires the low-order occupied partitions")
        extra: dict[str, object] = {}
        if self.preconvergence != "auto":
            extra["preconvergence"] = self.preconvergence
        if self.triples_fp32_w:
            extra["triples_fp32_w"] = self.triples_fp32_w
        if self.triples_w_schedule is not None:
            extra["triples_w_schedule"] = self.triples_w_schedule
        conventional, occupied = molecular_occupied_triples_on_reference(
            request,
            low_order.reference,
            budget_mib=self.budget_mib,
            **extra,
        )
        scaled = self.tplus_from_occupied(pair_request, low_order, occupied)
        return conventional, FocalTriplesEvaluation(
            scaled.energy,
            None,
            low_order.cabs_rank,
            executed_backend=occupied.reference.executed_backend,
            converged=True,
            telemetry=dict(occupied.telemetry),
            triples_model_id="tplus_rhf_occupied_v2",
        )

    def unrestricted_tplus_energy_on_reference(
        self,
        request: FocalRequest,
        *,
        low_order: FocalLowOrderEnergy,
        expected_cabs_rank: int | None = None,
    ) -> tuple[float, FocalTriplesEvaluation]:
        """Evaluate both conventional and scaled (T+) energies on one UHF frame.

        The caller supplies the retained low-order energies and occupied MP2-F12
        partitions. The two scalars share the native spin-blocked DF-UCCSD(T) v2
        solve and occupied triples partitions across alpha and beta channels.
        """
        from ._f12_focal_ucc_triples import molecular_occupied_triples_on_reference

        if request.role != "triples_plus" or request.root_id != "uhf":
            raise ValueError("Unrestricted (T+) requires the UHF scaled triples role")
        if request.backend == "cuda" and self.run is None:
            raise NotImplementedError("CUDA unrestricted (T+) energy requires the CUDA executor")
        pair_request = replace(request, role="mp2_f12_correlation")
        _check_request(self, pair_request)
        if expected_cabs_rank is not None and expected_cabs_rank != low_order.cabs_rank:
            raise ValueError("The displaced (T+) CABS rank differs from the center")
        if low_order.occupied_mp2_f12 is None:
            raise ValueError("Unrestricted (T+) requires the low-order occupied partitions")
        extra: dict[str, object] = {}
        if self.preconvergence != "auto":
            extra["preconvergence"] = self.preconvergence
        if self.triples_fp32_w:
            extra["triples_fp32_w"] = self.triples_fp32_w
        if self.triples_w_schedule is not None:
            extra["triples_w_schedule"] = self.triples_w_schedule
        conventional, occupied = molecular_occupied_triples_on_reference(
            request,
            low_order.reference,
            budget_mib=self.budget_mib,
            **extra,
        )
        scaled = self.tplus_from_occupied(pair_request, low_order, occupied)
        return conventional, FocalTriplesEvaluation(
            scaled.energy,
            None,
            low_order.cabs_rank,
            executed_backend=occupied.reference.executed_backend,
            converged=True,
            telemetry=dict(occupied.telemetry),
            triples_model_id="tplus_uhf_occupied_v2",
        )

    def restricted_tplus_energy_and_forces_on_reference(
        self,
        request: FocalRequest,
        *,
        low_order: FocalLowOrderEnergy,
        expected_cabs_rank: int | None = None,
        inactive: tuple[bool, ...] | None = None,
        diagnostic_fp32_single_w_forces: bool = False,
    ) -> tuple[float, FocalTriplesEvaluation, TriplesTelemetry]:
        """Evaluate conventional (T) energy plus full analytic (T+) energy & 3N forces.

        Uses one spatial CUDA DF-RCCSD + weighted DF-(T) pullback + Delta-Lambda solve
        for sum_i w_i * d(e_i^(T))/dR, combined with tplus_scaling_forces for
        sum_i e_i^(T) * dw_i/dR. Supports split-fit, scalar ECP, frozen-core and
        mixed preconvergence. Single-pass FP32 W runs only on an NVIDIA L20 unless
        diagnostic_fp32_single_w_forces=True, which tags the result as ineligible
        for production focal assembly. The telemetry records the executed W
        schedule, the preconvergence mode and the device.
        """
        from generativeqc_compiler.cc.tplus import evaluate_tplus

        from ._f12_focal_tplus_sources import FocalOccupiedTriples
        from ._f12_focal_ucc_triples import _record_frozen_core, spatial_rhf_triples_v2
        from ._f12_frozen_core import active_occupied_counts, resolve_frozen_occupied

        if request.role != "triples_plus" or request.root_id != "rhf":
            raise ValueError("Restricted (T+) forces require the RHF scaled triples role")
        if request.backend != "cuda" or self.run is None:
            raise NotImplementedError("Restricted (T+) forces require the CUDA executor")
        admission = self._admit_triples_w_forces(request, diagnostic_fp32_single_w_forces)
        pair_request = replace(request, role="mp2_f12_correlation")
        _check_request(self, pair_request)
        if expected_cabs_rank is not None and expected_cabs_rank != low_order.cabs_rank:
            raise ValueError("The (T+) CABS rank differs from the center")
        source = low_order.occupied_mp2_f12
        if source is None:
            raise ValueError("Restricted (T+) forces require low-order occupied partitions")
        ref = low_order.reference
        o_full = ref.occupied[0]
        frozen = resolve_frozen_occupied(request, (o_full, o_full))
        o_act = active_occupied_counts((o_full, o_full), frozen)[0]
        mp2_spatial = source.mp2[0] + source.mp2[1]
        f12_spatial = source.f12_increment[0] + source.f12_increment[1]
        if mp2_spatial.shape != (o_act,) or f12_spatial.shape != (o_act,):
            raise ValueError("Occupied MP2/F12 partition shape disagrees with active occupied count")
        # Fixed (T+) weights depend only on occupied MP2 and F12 energies, independent of e_i^(T).
        weight_probe = evaluate_tplus(
            mp2_spatial, f12_spatial, np.ones(o_act, dtype=np.float64), inactive=inactive,
            run=_occupied_scaling_executor(request.backend),
        )
        occ_weights = np.asarray(weight_probe.triples_cotangent, dtype=np.float64)

        # Evict any low-order pair caches before the C++ triples spike so host arenas never stack.
        self._evict_pair_source_cache()
        release_host_memory()
        record_ram_phase("before_triples")

        spatial_res = spatial_rhf_triples_v2(
            replace(request, role="triples"),
            ref,
            budget_mib=self.budget_mib,
            forces=True,
            occupied_partition=True,
            occupied_weights=occ_weights,
            preconvergence=self.preconvergence,
            triples_fp32_w=self.triples_fp32_w,
            triples_w_schedule=self.triples_w_schedule,
            provider=self,
        )
        record_ram_phase("after_triples")
        release_host_memory()
        record_ram_phase("after_trim")
        telemetry = dict(spatial_res.telemetry)
        admission.record(telemetry)
        assert spatial_res.occupied is not None and spatial_res.forces is not None
        assert request.basis.mp2_fit is not None
        occupied_obj = FocalOccupiedTriples(
            ref, request.correlation_space_id, request.basis.mp2_fit.identity, spatial_res.occupied
        )
        scaled = self.tplus_from_occupied(
            pair_request, low_order, occupied_obj, inactive=inactive
        )
        if abs(scaled.energy - spatial_res.energy) > 1e-10 * (1.0 + abs(scaled.energy)):
            raise RuntimeError(
                f"Weighted spatial (T+) energy {spatial_res.energy} disagrees with "
                f"occupied partition sum {scaled.energy}"
            )
        scaling_forces = self.tplus_scaling_forces(
            pair_request,
            ref,
            scaled.mp2,
            scaled.f12_increment,
            expected_cabs_rank=low_order.cabs_rank,
        )
        total_forces = spatial_res.forces + scaling_forces
        conventional_energy = float(np.sum(spatial_res.occupied))
        _record_frozen_core(telemetry, request, ref)
        return (
            conventional_energy,
            FocalTriplesEvaluation(
                scaled.energy,
                total_forces,
                low_order.cabs_rank,
                executed_backend=ref.executed_backend,
                converged=True,
                telemetry=dict(telemetry),
                triples_model_id="tplus_rhf_occupied_v2",
            ),
            telemetry,
        )

    def unrestricted_tplus_energy_and_forces_on_reference(
        self,
        request: FocalRequest,
        *,
        low_order: FocalLowOrderEnergy,
        expected_cabs_rank: int | None = None,
        inactive: tuple[bool, ...] | None = None,
        diagnostic_fp32_single_w_forces: bool = False,
    ) -> tuple[float, FocalTriplesEvaluation, TriplesTelemetry]:
        """Evaluate conventional (T) energy plus full analytic (T+) energy & 3N forces on UHF.

        Uses one spin-blocked DF-UCCSD + 4-channel weighted DF-(T) pullback + Delta-Lambda
        solve for sum_i w_i * d(e_i^(T))/dR across alpha and beta channels, combined with
        tplus_scaling_forces for sum_i e_i^(T) * dw_i/dR. Supports split-fit, scalar ECP,
        frozen-core and mixed preconvergence. Single-pass FP32 W runs only on an NVIDIA
        L20 unless diagnostic_fp32_single_w_forces=True, which tags the result as
        ineligible for production focal assembly. The telemetry records the executed
        W schedule, the preconvergence mode and the device.
        """
        from generativeqc_compiler.cc.tplus import evaluate_tplus

        from ._f12_focal_tplus_sources import FocalOccupiedTriples
        from ._f12_focal_ucc_triples import _record_frozen_core, spin_blocked_uhf_triples_v2
        from ._f12_frozen_core import active_occupied_counts, resolve_frozen_occupied

        if request.role != "triples_plus" or request.root_id != "uhf":
            raise ValueError("Unrestricted (T+) forces require the UHF scaled triples role")
        if request.backend == "cuda" and self.run is None:
            raise NotImplementedError("CUDA unrestricted (T+) forces require the CUDA executor")
        admission = self._admit_triples_w_forces(request, diagnostic_fp32_single_w_forces)
        pair_request = replace(request, role="mp2_f12_correlation")
        _check_request(self, pair_request)
        if expected_cabs_rank is not None and expected_cabs_rank != low_order.cabs_rank:
            raise ValueError("The (T+) CABS rank differs from the center")
        source = low_order.occupied_mp2_f12
        if source is None:
            raise ValueError("Unrestricted (T+) forces require low-order occupied partitions")
        ref = low_order.reference
        full_counts = (ref.occupied[0], ref.occupied[1])
        frozen = resolve_frozen_occupied(request, full_counts)
        oa_act, ob_act = active_occupied_counts(full_counts, frozen)
        mp2_a, mp2_b = source.mp2
        f12_a, f12_b = source.f12_increment
        if (
            mp2_a.shape != (oa_act,)
            or mp2_b.shape != (ob_act,)
            or f12_a.shape != (oa_act,)
            or f12_b.shape != (ob_act,)
        ):
            raise ValueError(
                "Occupied MP2/F12 partition shape disagrees with active occupied counts"
            )
        # Fixed (T+) weights depend only on occupied MP2 and F12 energies, independent of e_i^(T).
        weight_probe = evaluate_tplus(
            np.concatenate((mp2_a, mp2_b)),
            np.concatenate((f12_a, f12_b)),
            np.ones(oa_act + ob_act, dtype=np.float64),
            inactive=inactive,
            run=_occupied_scaling_executor(request.backend),
        )
        all_weights = np.asarray(weight_probe.triples_cotangent, dtype=np.float64)
        occ_weights = (all_weights[:oa_act], all_weights[oa_act:])

        # Evict any low-order pair caches before the C++ triples spike so host arenas never stack.
        self._evict_pair_source_cache()
        release_host_memory()
        record_ram_phase("before_triples")

        uhf_res = spin_blocked_uhf_triples_v2(
            replace(request, role="triples"),
            ref,
            budget_mib=self.budget_mib,
            forces=True,
            occupied_partition=True,
            occupied_weights=occ_weights,
            preconvergence=self.preconvergence,
            triples_fp32_w=self.triples_fp32_w,
            triples_w_schedule=self.triples_w_schedule,
            provider=self,
        )
        record_ram_phase("after_triples")
        release_host_memory()
        record_ram_phase("after_trim")
        telemetry = dict(uhf_res.telemetry)
        admission.record(telemetry)
        assert (
            uhf_res.occupied_alpha is not None
            and uhf_res.occupied_beta is not None
            and uhf_res.forces is not None
        )
        assert request.basis.mp2_fit is not None
        occupied_obj = FocalOccupiedTriples(
            ref,
            request.correlation_space_id,
            request.basis.mp2_fit.identity,
            (uhf_res.occupied_alpha, uhf_res.occupied_beta),
        )
        scaled = self.tplus_from_occupied(
            pair_request, low_order, occupied_obj, inactive=inactive
        )
        if abs(scaled.energy - uhf_res.energy) > 1e-10 * (1.0 + abs(scaled.energy)):
            raise RuntimeError(
                f"Weighted UHF (T+) energy {uhf_res.energy} disagrees with "
                f"occupied partition sum {scaled.energy}"
            )
        scaling_forces = self.tplus_scaling_forces(
            pair_request,
            ref,
            scaled.mp2,
            scaled.f12_increment,
            expected_cabs_rank=low_order.cabs_rank,
        )
        total_forces = uhf_res.forces + scaling_forces
        conventional_energy = float(
            np.sum(uhf_res.occupied_alpha) + np.sum(uhf_res.occupied_beta)
        )
        _record_frozen_core(telemetry, request, ref)
        return (
            conventional_energy,
            FocalTriplesEvaluation(
                scaled.energy,
                total_forces,
                low_order.cabs_rank,
                executed_backend=ref.executed_backend,
                converged=True,
                telemetry=dict(telemetry),
                triples_model_id="tplus_uhf_occupied_v2",
            ),
            telemetry,
        )

    def low_order_energy_and_forces(
        self, request: FocalRequest, *, include_hf_cabs: bool = True
    ) -> FocalLowOrderForces:
        """Evaluate both low-order analytic forces and energies on one canonical root.

        Use low_order_energy with this center's CABS rank for a displaced energy.
        Both center roles share the physical determinant and the AO integral panels.
        Set include_hf_cabs=False to evaluate just the pair response on this basis.
        """
        from ._f12_focal_reference import df_reference as resolve_df_reference

        budget = _check_request(self, request)
        reference = resolve_df_reference(
            request,
            forces=include_hf_cabs,
            budget_mib=budget,
            **self._seed_options(request),
        )
        reference = partition_focal_reference(request, reference, budget)
        one, panels = self._cached_pair_sources(request, derivatives=True, budget=budget)
        spin_mp2, increment, rank, correlation_forces = self._pair_components(
            request, reference, one, panels, expected_cabs_rank=None
        )
        singles_energies = hf_cabs_forces = None
        if include_hf_cabs:
            import gc

            gc.collect()
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

    def _spatial_rhf_ccsdf12b_available(
        self,
        request: FocalRequest,
        reference: FocalReference | None,
        *,
        use_native: bool | None = None,
    ) -> bool:
        if use_native is False:
            return False
        return (
            request.backend == "cuda"
            and request.root_id == "rhf"
            and request.multiplicity == 1
            and reference is not None
            and reference._native_owner is not None
            and getattr(
                reference._native_owner.library,
                "generativeqc_focal_rcc_f12b_v1",
                None,
            )
            is not None
        )

    def _use_native_uhf_f12b(
        self,
        pair_request: FocalRequest,
        reference: FocalReference,
        *,
        use_native: bool | None = None,
    ) -> bool:
        if use_native is False:
            return False
        if use_native is None:
            override = _uhf_cc_backend_override()
            if override == "legacy":
                return False
            if override is None and not self.use_native_uhf_f12b:
                return False
        return (
            pair_request.root_id == "uhf"
            and pair_request.backend in ("cpu_reference", "cuda")
            and pair_request.basis.mp2_fit is not None
            and reference._native_owner is not None
            and getattr(
                reference._native_owner.library,
                "generativeqc_focal_ucc_f12b_v1",
                None,
            )
            is not None
        )

    def f12_doubles_energy(
        self,
        pair_request: FocalRequest,
        *,
        low_order: FocalLowOrderEnergy,
        expected_cabs_rank: int | None = None,
        use_native: bool | None = None,
    ) -> float:
        """CCSD-F12b minus MP2-F12 on the supplied determinant and CABS rank."""
        if expected_cabs_rank is not None and expected_cabs_rank != low_order.cabs_rank:
            raise ValueError("The displaced and low-order CABS ranks disagree")
        if self._spatial_rhf_ccsdf12b_available(
            pair_request, low_order.reference, use_native=use_native
        ):
            from ._f12_focal_rcc_f12b import spatial_rhf_ccsdf12b_v1

            res = spatial_rhf_ccsdf12b_v1(
                self,
                pair_request,
                low_order.reference,
                forces=False,
                expected_cabs_rank=low_order.cabs_rank,
            )
            self._pair_source_cache["last_ccsdf12b_telemetry"] = res.telemetry
            return res.correlation_energy - low_order.mp2_f12_correlation_energy
        if self._use_native_uhf_f12b(
            pair_request, low_order.reference, use_native=use_native
        ):
            from ._f12_focal_ucc_f12b import spin_blocked_uhf_f12b_v1

            fixed = molecular_ucc_sources(
                self,
                pair_request,
                include_ordinary=False,
                reference=low_order.reference,
                expected_cabs_rank=low_order.cabs_rank,
            )
            solved = spin_blocked_uhf_f12b_v1(
                self,
                pair_request,
                fixed,
                forces=False,
                preconvergence=self.preconvergence,
            )
            return solved.ccsd_f12b_energy - low_order.mp2_f12_correlation_energy

        from ._f12_ucc_solver import SolverOptions, solve_focal_uccsd_f12b_energy

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
        use_native: bool | None = None,
    ) -> tuple[float, np.ndarray]:
        """Differentiate CCSD-F12b minus the supplied matched MP2-F12 energy."""
        low = low_order.energies
        if self._spatial_rhf_ccsdf12b_available(
            pair_request, low.reference, use_native=use_native
        ):
            if self.lambda_options is not None:
                from ._f12_ucc_solver import UCCSDLambdaOptions

                if self.lambda_options != UCCSDLambdaOptions():
                    raise NotImplementedError(
                        "Native spatial RHF CCSD-F12b C ABI does not support custom "
                        "provider.lambda_options; pass use_native=False or omit non-default lambda_options"
                    )
            from ._f12_focal_rcc_f12b import spatial_rhf_ccsdf12b_v1

            res = spatial_rhf_ccsdf12b_v1(
                self,
                pair_request,
                low.reference,
                forces=True,
                expected_cabs_rank=low.cabs_rank,
            )
            self._pair_source_cache["last_ccsdf12b_telemetry"] = res.telemetry
            assert res.ordinary_forces is not None and res.fixed_forces is not None
            return (
                res.correlation_energy - low.mp2_f12_correlation_energy,
                res.ordinary_forces
                + res.fixed_forces
                - low_order.mp2_f12_correlation_forces,
            )
        if self._use_native_uhf_f12b(
            pair_request, low.reference, use_native=use_native
        ):
            from ._f12_focal_ucc_f12b import spin_blocked_uhf_f12b_v1

            fixed = molecular_ucc_sources(
                self,
                pair_request,
                include_ordinary=False,
                reference=low.reference,
                expected_cabs_rank=low.cabs_rank,
            )
            solved = spin_blocked_uhf_f12b_v1(
                self,
                pair_request,
                fixed,
                forces=True,
                preconvergence=self.preconvergence,
            )
            assert solved.forces is not None
            return (
                solved.ccsd_f12b_energy - low.mp2_f12_correlation_energy,
                solved.forces - low_order.mp2_f12_correlation_forces,
            )
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
        self, request: FocalRequest, *, reference: FocalReference, telemetry: TriplesTelemetry | None = None
    ) -> float:
        """Conventional triples on the supplied canonical determinant in Eh."""
        from ._f12_focal_ucc_triples import molecular_triples_on_reference

        extra: dict[str, object] = {}
        if telemetry is not None:
            extra["telemetry"] = telemetry
        if self.preconvergence != "auto":
            extra["preconvergence"] = self.preconvergence
        if self.triples_fp32_w:
            extra["triples_fp32_w"] = self.triples_fp32_w
        if self.triples_w_schedule is not None:
            extra["triples_w_schedule"] = self.triples_w_schedule
        return molecular_triples_on_reference(
            request,
            reference,
            budget_mib=self.budget_mib,
            forces=False,
            provider=self,
            **extra,
        )[0]

    def triples_energy_and_forces(
        self,
        request: FocalRequest,
        *,
        reference: FocalReference,
        telemetry: TriplesTelemetry | None = None,
        diagnostic_fp32_single_w_forces: bool = False,
    ) -> tuple[float, np.ndarray]:
        """Conventional triples and its analytic Cartesian forces on a shared root.

        Single-pass FP32 W runs only on an NVIDIA L20 unless
        diagnostic_fp32_single_w_forces=True, which tags the result as ineligible
        for production focal assembly. Either single-pass use requires a telemetry
        mapping. A supplied mapping records the executed W schedule, the
        preconvergence mode and the device.
        """
        from ._f12_focal_ucc_triples import molecular_triples_on_reference

        admission = self._admit_triples_w_forces(request, diagnostic_fp32_single_w_forces)
        if admission.single_pass and telemetry is None:
            raise ValueError(
                "A single-pass FP32 W force evaluation must record its provenance in telemetry"
            )
        extra: dict[str, object] = {}
        if telemetry is not None:
            extra["telemetry"] = telemetry
        if self.preconvergence != "auto":
            extra["preconvergence"] = self.preconvergence
        if self.triples_fp32_w:
            extra["triples_fp32_w"] = self.triples_fp32_w
        if self.triples_w_schedule is not None:
            extra["triples_w_schedule"] = self.triples_w_schedule
        energy, forces = molecular_triples_on_reference(
            request,
            reference,
            budget_mib=self.budget_mib,
            forces=True,
            provider=self,
            **extra,
        )
        assert forces is not None
        if telemetry is not None:
            admission.record(telemetry)
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
        self,
        request: FocalRequest,
        role: str,
        *,
        orbital_only: bool = False,
        allow_streaming: bool = False,
    ) -> _Panel:
        if role == "f12_fit" and request.slater_normalization != self.slater_normalization:
            raise ValueError("F12 panel request and provider Slater normalization must match")
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
        coulomb_only = role != "f12_fit"
        n_ext, n_aux = map(int, layout["ao_sizes"])
        ao_cart_total = int(layout["ao_offsets"][-1])
        streaming = should_stream_panel(
            allow_streaming=allow_streaming,
            ordinary=ordinary,
            coulomb_only=coulomb_only,
            n_ext=n_ext,
            n_aux=n_aux,
            ao_cart_total=ao_cart_total,
        )
        if streaming:
            raw = forward_metrics_only(
                layout,
                self.shell_library,
                host_reference=host,
                coulomb_only=coulomb_only,
                budget_mib=self.budget_mib,
                slater_normalization=self.slater_normalization,
            )
        elif coulomb_only and ao_cart_total > 128_000_000:
            raw = forward_metrics_only(
                layout,
                self.shell_library,
                host_reference=host,
                coulomb_only=True,
                budget_mib=self.budget_mib,
            )
            raw["ao_coulomb"] = forward_single_ao_operator(
                layout,
                self.shell_library,
                "coulomb",
                host_reference=host,
                coulomb_only=True,
                budget_mib=self.budget_mib,
            )
        else:
            raw, _ = forward_integrals(
                layout,
                self.shell_library,
                host_reference=host,
                coulomb_only=coulomb_only,
                budget_mib=self.budget_mib,
                slater_normalization=self.slater_normalization,
            )
        m = raw["metric"]
        if host:
            from scipy.linalg import solve_triangular

            whitener = solve_triangular(
                np.linalg.cholesky(m), np.eye(len(m)), lower=True
            )
        else:
            assert self.metric_library is not None
            try:
                whitener, _ = metric_whitener(
                    m, self.metric_library, budget_bytes=self.budget_mib << 20
                )
            except RuntimeError:
                if not ordinary:
                    raise
                # Diffuse core-valence RI bases (e.g. aug-cc-pwCVTZ-RIFIT on ionic pairs)
                # can remain strictly SPD at min/max ~ 5e-11..9.9e-11; verify full-rank whitener accuracy.
                whitener, _ = metric_whitener(
                    m, self.metric_library, budget_bytes=self.budget_mib << 20, cutoff=1e-12
                )
                if not np.allclose(whitener @ m @ whitener.T, np.eye(len(m)), atol=2e-7, rtol=0):
                    raise
        return _Panel(
            layout,
            raw,
            whitener,
            getattr(request.basis, role).identity,
            streaming=streaming,
            library=self.shell_library if streaming else None,
            host_reference=host,
            budget_mib=self.budget_mib,
            coulomb_only=coulomb_only,
            slater_normalization=self.slater_normalization,
        )

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
            slater_normalization=self.slater_normalization,
        )
        if not row["passes_contraction_checks"]:
            raise ArithmeticError(
                "The molecular shell nuclear derivative failed its contraction checks"
            )
        return np.asarray(row["gradient_Eh_per_bohr"])

    def __call__(self, request: FocalRequest) -> FocalEvaluation:
        from .experimental.f12_focal import FocalEvaluation, FocalTerm

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
            or request.basis.mp2_fit is None
            or request.basis.f12_fit is None
            or request.basis.beta is None
        ):
            raise ValueError(
                "The pair term requires UHF or singlet RHF, MP2Fit, F12Fit and beta"
            )
        req_norm = getattr(request, "slater_normalization", None)
        if (
            request.basis.f12_fit is not None
            and req_norm is not None
            and req_norm != self.slater_normalization
        ):
            raise ValueError(
                f"Request slater_normalization {req_norm!r} disagrees with "
                f"provider.slater_normalization {self.slater_normalization!r}"
            )
        override = _uhf_cc_backend_override()
        use_native_uhf = (
            override == "native" if override is not None else self.use_native_uhf_f12b
        )
        if (
            request.role == "ccsd_f12b_minus_mp2_f12"
            and request.root_id == "uhf"
            and use_native_uhf
        ):
            calculation = replace(request, role="mp2_f12_correlation")
            low = self.low_order_energy_and_forces(
                calculation, include_hf_cabs=False
            )
            energy, forces = self.f12_doubles_energy_and_forces(
                calculation, low_order=low
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
        solution = None
        calculation = request
        if request.role == "ccsd_f12b_minus_mp2_f12":
            calculation = replace(request, role="mp2_f12_correlation")
            if (
                request.backend == "cuda"
                and request.root_id == "rhf"
                and request.multiplicity == 1
                and self.canonical_library is not None
                and getattr(
                    self.canonical_library, "generativeqc_focal_rcc_f12b_v1", None
                )
                is not None
            ):
                low_ef = self.low_order_energy_and_forces(
                    calculation, include_hf_cabs=False
                )
                energy, forces = self.f12_doubles_energy_and_forces(
                    calculation, low_order=low_ef
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
            if pair:
                ref = partition_focal_reference(calculation, ref, self.budget_mib)
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
        if projector.rank == len(one.overlap) - one.obs_ao_count:
            disclose_host_assisted_stage(
                "full_rank_cabs_overlap_bar",
                requested_backend=request.backend,
                detail="Ozaki-4 compensated host SciPy/NumPy full-rank CABS overlap cotangent",
            )
        if jk.streaming:
            obs_jk = jk.obs_coulomb(one.obs_ao_count)
            singles = reference_cabs_singles(
                request,
                ref,
                one,
                complement=projector.frame,
                three_center=None,
                three_center_loader=lambda: jk.load_ao("ao_coulomb"),
                obs_three_center=obs_jk,
                metric=jk.data["metric"],
                whitener=jk.whitener,
                relax=True,
                run=self.run,
                library=self.response_library,
                relative_cutoff=self.relative_cutoff,
                budget_mib=self.budget_mib,
            )
            del obs_jk
            jk._obs_coulomb_cache = None
        else:
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
        grad += one.nuclear_gradient(
            w["overlap"], w["hcore"], run=self.run, shell_library=self.shell_library
        )
        return singles, ref.forces - grad

    def _pair_panels(
        self, request: FocalRequest, *, allow_streaming: bool = True
    ) -> tuple[_Panel, _Panel, _Panel]:
        basis = request.basis
        assert basis.mp2_fit is not None and basis.f12_fit is not None
        f12 = self._panel(request, "f12_fit", allow_streaming=allow_streaming)
        jk = (
            f12
            if f12.fit_id == basis.jk_fit.identity
            else self._panel(request, "jk_fit", allow_streaming=allow_streaming)
        )
        mp2 = next((p for p in (jk, f12) if p.fit_id == basis.mp2_fit.identity), None)
        if mp2 is None:
            mp2 = self._panel(request, "mp2_fit", allow_streaming=False)
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
        ref = partition_focal_reference(request, ref, self.budget_mib)
        n = one.obs_ao_count
        jk, mp2, f12 = panels
        ordinary = ordinary_df_ump2_unrelaxed(
            request,
            ref,
            mp2.obs(request, n),
            run=self.run,
            budget_mib=self.budget_mib,
            occupied_weights=mp2_weights,
        )
        spin_energies = ordinary.spin_energies
        extra_obs_fock_seeds = ordinary.fock_seeds
        extra_obs_cbar = ordinary.obs_coefficients
        mp_weights: dict[str, np.ndarray] = {
            "metric": ordinary.mp2_metric,
            "ao_coulomb": ordinary.mp2_three_center,
        }
        del ordinary
        if mp2.streaming and mp2 is not jk:
            mp2._obs_coulomb_cache = None
            release_host_memory()
        record_ram_phase("after_ordinary_mp2")
        host_ref = request.backend == "cpu_reference"
        if f12.streaming:
            # Pre-cache compact [Q_jk, obs, obs] slice before any [Q, ext, ext] AO cotangent
            # is live so relaxed_f12_source_weights never reloads the full [Q_jk, ext, ext] panel.
            if jk.streaming:
                jk.obs_coulomb(n)
            # Stream each F12 [Q, ext, ext] AO cotangent directly into its [natoms, 3]
            # nuclear gradient and immediately free it (eliminates 5-panel + 5-bar host OOM).
            streamed_grad = np.zeros((len(request.atoms), 3), dtype=np.float64)
            if mp2 is not jk and mp2 is not f12:
                streamed_grad += self._gradient(request, mp2, mp_weights, coulomb_only=True)
                mp_weights.clear()
                release_host_memory()
            add_mp2_in_f12_sink = mp2 is f12 and mp2 is not jk

            def _f12_ao_sink(ao_name: str, bar_ao_mat: np.ndarray) -> None:
                if ao_name == "ao_coulomb" and add_mp2_in_f12_sink:
                    mp2_ao = mp_weights.pop("ao_coulomb")
                    bar_ao_mat[:, :n, :n] += mp2_ao
                    del mp2_ao
                streamed_grad[:] += contract_single_channel(
                    f12.layout,
                    "ao",
                    _AO_NAME_TO_OP[ao_name],
                    bar_ao_mat,
                    self.shell_library,
                    host_reference=host_ref,
                    budget_mib=self.budget_mib,
                    slater_normalization=self.slater_normalization,
                )
                release_host_memory()

            increment_fn = streamed_f12_increment if jk is not f12 else reference_f12_increment
            stream_extra = (
                {"shell_library": self.shell_library}
                if increment_fn is streamed_f12_increment
                else {}
            )
            explicit = increment_fn(
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
                ao_bar_sink=_f12_ao_sink,
                extra_obs_fock_seeds=extra_obs_fock_seeds,
                extra_obs_cbar=extra_obs_cbar,
                **stream_extra,
            )
            del extra_obs_fock_seeds, extra_obs_cbar
            record_ram_phase("after_f12_increment")
            energy, cabs_rank = explicit.energy, explicit.cabs_rank
            one_e_weights = explicit.one_electron_weights
            wk, wf = explicit.jkfit_weights, explicit.f12fit_weights
            del explicit
            if mp2 is jk:
                wk["metric"] += mp_weights.pop("metric")
                wk["ao_coulomb"][:, :n, :n] += mp_weights.pop("ao_coulomb")
            elif mp2 is f12:
                wf["metric"] += mp_weights.pop("metric")
            mp_weights.clear()
            grad = streamed_grad
            for mkey, op in _METRIC_KEY_TO_OP.items():
                grad += contract_single_channel(
                    f12.layout,
                    "metric",
                    op,
                    wf[mkey],
                    self.shell_library,
                    host_reference=host_ref,
                    budget_mib=self.budget_mib,
                    slater_normalization=self.slater_normalization,
                )
            del wf
            grad += self._gradient(request, jk, wk, coulomb_only=True)
            del wk
            release_host_memory()
            record_ram_phase("after_jk_grad")
        else:
            if (
                expected_cabs_rank is not None
                and expected_cabs_rank == len(one.overlap) - n
            ):
                disclose_host_assisted_stage(
                    "full_rank_cabs_overlap_bar",
                    requested_backend=request.backend,
                    detail="Ozaki-4 compensated host SciPy/NumPy full-rank CABS overlap cotangent",
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
                extra_obs_fock_seeds=extra_obs_fock_seeds,
                extra_obs_cbar=extra_obs_cbar,
                preprojected_cache=self._pair_source_cache,
            )
            if (
                expected_cabs_rank is None
                and explicit.cabs_rank == len(one.overlap) - n
            ):
                disclose_host_assisted_stage(
                    "full_rank_cabs_overlap_bar",
                    requested_backend=request.backend,
                    detail="Ozaki-4 compensated host SciPy/NumPy full-rank CABS overlap cotangent",
                )
            del extra_obs_fock_seeds, extra_obs_cbar
            record_ram_phase("after_f12_increment")
            energy, cabs_rank = explicit.energy, explicit.cabs_rank
            one_e_weights = explicit.one_electron_weights
            # A shared fitted operator has one nuclear derivative. Scatter the OBS
            # cotangents into the leading block of its OBS+OptRI basis, including
            # metric weights. Operator equality requires the same *basis identity*.
            wk, wf = explicit.jkfit_weights, explicit.f12fit_weights
            del explicit
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
        grad += one.nuclear_gradient(
            one_e_weights["overlap"],
            one_e_weights["hcore"],
            run=self.run,
            shell_library=self.shell_library,
        )
        del one_e_weights
        record_ram_phase("after_1e_grad")
        return spin_energies, energy, cabs_rank, -grad
