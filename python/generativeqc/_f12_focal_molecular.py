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
from ._f12_focal_reference import (
    FocalReference,
    FocalReferenceSeed,
    df_reference,
    partition_focal_reference,
)
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


@dataclass(frozen=True, kw_only=True)
class MolecularFocalProvider:
    """Evaluate molecular focal contributions with a native shell service.

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


    def _panel(
        self,
        request: FocalRequest,
        role: str,
        *,
        orbital_only: bool = False,
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
        ao_cart_total = int(layout["ao_offsets"][-1])
        if coulomb_only and ao_cart_total > 128_000_000:
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
            host_reference=host,
            budget_mib=self.budget_mib,
            coulomb_only=coulomb_only,
            slater_normalization=self.slater_normalization,
        )
