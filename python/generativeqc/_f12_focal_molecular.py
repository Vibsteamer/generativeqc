"""Geometry-driven contributions for the experimental F12 focal provider."""

from __future__ import annotations

from types import MethodType
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from . import _native
from ._f12_focal_integral_kernel import (
    DEFAULT_SLATER_FIT_NORMALIZATION,
    slater_normalization_code,
)

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
from ._f12_ram_hygiene import disclose_host_assisted_stage, record_ram_phase, release_host_memory, should_stream_panel
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


_AO_NAME_TO_OP = {
    "ao_coulomb": "coulomb",
    "ao_f12": "f12",
    "ao_f12_squared": "f12_squared",
    "ao_f12_over_coulomb": "f12_over_coulomb",
    "ao_double_commutator": "double_commutator",
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


@dataclass(frozen=True, kw_only=True)
class MolecularFocalProvider:
    """Evaluate molecular focal contributions with a native shell service.

    Evaluate HF+CABS energies and analytic Cartesian forces on the requested root.
    Supply reference_seed to transport a converged determinant between geometries.
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
    relative_cutoff: float = 1e-9
    budget_mib: int = 1536
    slater_normalization: Literal[
        "negative_exponential", "negative_exponential_over_beta"
    ] = DEFAULT_SLATER_FIT_NORMALIZATION

    reference_seed: (
        FocalReferenceSeed | Callable[[FocalRequest], FocalReferenceSeed] | None
    ) = None

    def __post_init__(self) -> None:
        slater_normalization_code(self.slater_normalization)


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
        high_order: Callable[[FocalRequest], FocalEvaluation] | None = None,
        record_graph: Callable[[dict], None] | None = None,
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
            high_order=high_order,
            reference_seed=reference_seed,
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

        if request.role != "hf_cabs":
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
        if (
            request.basis.cabs is None
            or not np.isfinite(self.relative_cutoff)
            or self.relative_cutoff <= 0
        ):
            raise ValueError(
                "The molecular focal term requires an OptRI basis and positive relative cutoff"
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
        calculation = request
        ref = df_reference(
            request,
            forces=True,
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
