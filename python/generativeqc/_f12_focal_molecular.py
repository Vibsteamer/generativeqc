"""Geometry-driven contributions for the experimental F12 focal provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._f12_focal_increment import FocalF12Sources, reference_f12_increment
from ._f12_focal_integrals import (
    contract_live_weights,
    focal_shell_layout,
    forward_integrals,
    metric_whitener,
)
from ._f12_focal_one_electron import FocalOneElectron, one_electron_sources
from ._f12_focal_projector import focal_cabs_projector
from ._f12_focal_reference import FocalReference, df_reference
from ._f12_focal_singles import reference_cabs_singles
from ._f12_focal_ump2 import FocalCoulombSource, reference_df_ump2
from .elements import checked_integer

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable
    from pathlib import Path

    from .experimental.f12_focal import FocalEvaluation, FocalRequest


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
    """Evaluate geometric HF+CABS and all-active UHF MP2-F12/3C(FIX) focal terms.

    Supply native F12 shell and optional separate one-electron libraries. CUDA
    uses an FP64 TensorIR runner, the device metric/response library, and the
    canonical-response ABI of the main native library. Each call solves one
    physical determinant on the backend in the request. Pass this callable as
    the provider of singlepoint(), or evaluate individual plan steps. Select a
    high-order evaluator explicitly for post-MP2 and triples steps.
    """

    shell_library: Path
    one_electron_library: Path | None = None
    metric_library: Path | None = None
    response_library: ctypes.CDLL | None = None
    canonical_library: ctypes.CDLL | None = None
    run: Callable | None = None
    high_order: Callable[[FocalRequest], FocalEvaluation] | None = None
    relative_cutoff: float = 1e-9
    budget_mib: int = 1536

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
        record_graph: Callable[[dict], None] | None = None,
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
        )

    def _panel(self, request: FocalRequest, role: str) -> _Panel:
        host = request.backend == "cpu_reference"
        ordinary = role == "mp2_fit"
        layout = focal_shell_layout(
            request,
            self.shell_library,
            host_reference=host,
            budget_mib=self.budget_mib,
            ordinary_fit=role if ordinary else None,
            joint_jk=role == "jk_fit" and request.role == "mp2_f12_correlation",
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

        if request.role not in ("hf_cabs", "mp2_f12_correlation"):
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
        pair = request.role == "mp2_f12_correlation"
        if (
            request.basis.cabs is None
            or not np.isfinite(self.relative_cutoff)
            or self.relative_cutoff <= 0
        ):
            raise ValueError(
                "The molecular focal term requires an OptRI basis and positive relative cutoff"
            )
        if pair and (
            request.root_id != "uhf"
            or request.correlation_space_id != "all-active"
            or request.basis.mp2_fit is None
            or request.basis.f12_fit is None
            or request.basis.beta is None
        ):
            raise ValueError(
                "The pair term requires all-active UHF, MP2Fit, F12Fit and beta"
            )
        ref = df_reference(request, forces=not pair, budget_mib=self.budget_mib)
        one = one_electron_sources(
            request,
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
            energy, forces = self._pair(request, ref, one)
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
    ) -> tuple[float, np.ndarray]:
        projector = focal_cabs_projector(
            request,
            one,
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
        grad += np.einsum("axij,ij->ax", one.d_overlap, w["overlap"])
        grad += np.einsum("axij,ij->ax", one.d_hcore, w["hcore"])
        return singles.hf_cabs_energy, ref.forces - grad

    def _pair(
        self, request: FocalRequest, ref: FocalReference, one: FocalOneElectron
    ) -> tuple[float, np.ndarray]:
        basis, n = request.basis, one.obs_ao_count
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
        ordinary = reference_df_ump2(
            request,
            ref,
            jk.obs(request, n),
            mp2.obs(request, n),
            run=self.run,
            library=self.response_library,
            budget_mib=self.budget_mib,
        )
        explicit = reference_f12_increment(
            request,
            ref,
            one,
            jk.joint(request, explicit=False),
            f12.joint(request, explicit=True),
            relative_cutoff=self.relative_cutoff,
            budget_mib=self.budget_mib,
            run=self.run,
            library=self.response_library,
            canonical_library=self.canonical_library,
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
        for derivative, key in ((one.d_overlap, "overlap"), (one.d_hcore, "hcore")):
            bar = explicit.one_electron_weights[key]
            bar[:n, :n] += w[key]
            grad += np.einsum("axij,ij->ax", derivative, bar)
        return float(sum(ordinary.spin_energies) + explicit.energy), -grad
