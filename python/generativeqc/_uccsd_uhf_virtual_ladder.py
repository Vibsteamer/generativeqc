"""Admit and apply all three independent-spin bare virtual ladder sectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.cc.uccsd_df_virtual_ladder import FACTOR_INPUTS
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES
from generativeqc_compiler.cc.uccsd_uhf_virtual_panels import (
    build_uccsd_uhf_mixed_virtual_panel_programs,
    build_uccsd_uhf_same_spin_virtual_pair_programs,
)
from generativeqc_compiler.common.resources import checked_bytes
from generativeqc_compiler.mp2.f12_density_fitting import fitted_integral_block

from ._uccsd_rhf_virtual_pairs import (
    prepare_spatial_virtual_pair_operators,
    spatial_virtual_pair_storage,
)
from ._uccsd_uhf_virtual_panels import (
    prepare_uhf_mixed_virtual_panels,
    uhf_mixed_virtual_panel_storage,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
    from generativeqc_compiler.cc.uccsd_uhf_virtual_panels import (
        UCCVirtualLadderActions,
    )
    from generativeqc_compiler.tensor import Program


@dataclass(frozen=True)
class UHFVirtualGraphResources:
    """Planner-provided native graph resources, excluding caller-owned arrays."""

    host_bytes: int
    device_bytes: int
    scalar_work: int

    def __post_init__(self) -> None:
        for name in ("host_bytes", "device_bytes", "scalar_work"):
            checked_bytes(getattr(self, name), name)


@dataclass(frozen=True, kw_only=True)
class UHFVirtualLadderLimits:
    """Remaining per-domain limits after reserving the other CC stages."""

    max_host_bytes: int
    max_device_bytes: int
    max_setup_work: int
    max_action_work: int
    pair_rows: int = 256
    mixed_alpha_width: int = 16

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if checked_bytes(getattr(self, name), name) == 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, kw_only=True)
class UHFDFLadderOptions:
    """Explicit UHF electronic work allowances for the molecular CC owner."""

    max_primal_work: int
    max_source_work: int
    pair_rows: int = 256
    mixed_alpha_width: int = 16
    auxiliary_tile: int = 8

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if checked_bytes(getattr(self, name), name) == 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class UHFVirtualLadderResourcePlan:
    retained: int
    host: int
    device: int
    setup_work: int
    primal_work: int
    adjoint_work: int


@dataclass(frozen=True)
class _Sector:
    name: str
    actions: UCCVirtualLadderActions
    operator_name: str
    operator: np.ndarray
    alpha: slice | None = None


@dataclass(frozen=True)
class UHFBareVirtualLadder:
    """One canonical determinant's admitted bare virtual operator and transpose."""

    layout: UCCSDLayout
    sectors: tuple[_Sector, ...]
    run: Callable
    retained_host_bytes: int
    planned_host_bytes: int
    planned_device_bytes: int
    setup_work: int
    primal_work: int
    adjoint_work: int

    def _execute(
        self, program: Program, values: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        feeds = {
            node.attrs["name"]: np.ascontiguousarray(values[node.attrs["name"]])
            for node in program.live_nodes
            if node.op == "input"
        }
        result = self.run(program, feeds).outputs
        for name, node in program.outputs.items():
            value = result[name]
            if (
                value.dtype != np.float64
                or value.shape != node.spec.shape
                or not np.isfinite(value).all()
            ):
                raise ValueError("Invalid bare UHF virtual ladder native output")
        return result

    def residual(self, amplitudes: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return the complete bare contribution in canonical full doubles order."""
        result = {
            "r2" + name[2:]: np.zeros_like(amplitudes[name]) for name in BLOCK_NAMES[2:]
        }
        for sector in self.sectors:
            values = {**amplitudes, sector.operator_name: sector.operator}
            target = result[sector.name]
            contribution = self._execute(sector.actions.primal, values)[sector.name]
            if sector.alpha is None:
                np.copyto(target, contribution, casting="no")
            else:
                target[:, :, sector.alpha, :] = contribution
        return result

    def amplitude_bars(
        self, amplitudes: Mapping[str, np.ndarray], seeds: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Sum full amplitude adjoints before the physical packed dual map."""
        result = {
            "bar_" + name: np.zeros_like(amplitudes[name]) for name in BLOCK_NAMES
        }
        for sector in self.sectors:
            name = "bar_" + sector.name
            seed = seeds[name]
            if sector.alpha is not None:
                seed = seed[:, :, sector.alpha, :]
            values = {**amplitudes, name: seed, sector.operator_name: sector.operator}
            for key, value in self._execute(sector.actions.vjp.program, values).items():
                result[key] += value
        return result


def prepare_uhf_virtual_ladder(
    layout: UCCSDLayout,
    factors: Mapping[str, np.ndarray],
    run: Callable,
    graph_resources: Callable[[Program], UHFVirtualGraphResources],
    limits: UHFVirtualLadderLimits,
    *,
    admit: Callable[[UHFVirtualLadderResourcePlan], None] | None = None,
) -> UHFBareVirtualLadder:
    """Plan all setup and residual/transpose graphs before staging any operator.

    The budget is for added bare-ladder allocations. The stationary caller
    must retain and charge the four factors, physical T/Lambda arrays and
    other equation stages. A supplied backend runs every admitted graph.
    """
    if set(factors) != set(FACTOR_INPUTS):
        raise ValueError("UHF virtual ladder requires all four independent DF factors")
    auxiliary = factors["fit_left_alpha"].shape[0]
    for spin, extent in (("alpha", layout.va), ("beta", layout.vb)):
        for side in ("fit_left", "raw_right"):
            value = factors[f"{side}_{spin}"]
            if value.shape != (auxiliary, extent, extent) or value.dtype != np.float64:
                raise ValueError(
                    "UHF fitting factors do not match their canonical spin frames"
                )

    def fitting_rows(
        total: int, tile: int, right: int, multiplier: int = 1
    ) -> tuple[tuple[Program, int], ...]:
        full, tail = divmod(total, tile)
        return tuple(
            (
                fitted_integral_block(
                    auxiliary, width * multiplier, right, coulomb=True
                ),
                repeat,
            )
            for width, repeat in ((tile, full), (tail, 1))
            if width
        )

    def whole_cost(graphs: tuple[tuple[Program, int], ...]) -> UHFVirtualGraphResources:
        resources = tuple((graph_resources(graph), repeat) for graph, repeat in graphs)
        return UHFVirtualGraphResources(
            max((r.host_bytes for r, _ in resources), default=0),
            max((r.device_bytes for r, _ in resources), default=0),
            sum(r.scalar_work * count for r, count in resources),
        )

    fit, repeated = [], []
    same_spins = []
    for spin, occ, extent in (
        ("alpha", layout.oa, layout.va),
        ("beta", layout.ob, layout.vb),
    ):
        if occ > 1 and extent > 1:
            pairs = extent * (extent + 1) // 2
            tile = min(pairs, limits.pair_rows)
            storage = spatial_virtual_pair_storage(
                extent, auxiliary, tile, include_symmetric=False
            )
            setup = whole_cost(fitting_rows(pairs, tile, pairs))
            actions = build_uccsd_uhf_same_spin_virtual_pair_programs(layout, spin)
            fit.append((*storage, setup))
            repeated.append((actions, 1))
            same_spins.append((spin, tile, storage[0], actions))

    mixed = {}
    width = min(limits.mixed_alpha_width, layout.va)
    if all((layout.oa, layout.ob, layout.va, layout.vb)):
        storage = uhf_mixed_virtual_panel_storage(
            layout.va, layout.vb, auxiliary, width
        )
        setup = whole_cost(fitting_rows(layout.va, width, layout.vb**2, layout.va))
        fit.append((*storage, setup))
        full, tail = divmod(layout.va, width)
        for size, repeat in ((width, full), (tail, 1)):
            if size:
                mixed[size] = build_uccsd_uhf_mixed_virtual_panel_programs(layout, size)
                repeated.append((mixed[size], repeat))

    host, device, setup_work, retained = 0, 0, 0, 0
    for preparation, keeps, resource in fit:
        host = max(host, retained + preparation + resource.host_bytes)
        device = max(device, resource.device_bytes)
        setup_work += resource.scalar_work
        retained += keeps
    primal = whole_cost(tuple((a.primal, n) for a, n in repeated))
    adjoint = whole_cost(tuple((a.vjp.program, n) for a, n in repeated))
    # Detached native outputs and one contiguous mixed seed coexist with the
    # returned amplitude/residual arrays around the operator application.
    full_bytes = sum(block.full_layout.storage_bytes(8) for block in layout.blocks)
    host = max(
        host, retained + 3 * full_bytes + max(primal.host_bytes, adjoint.host_bytes)
    )
    device = max(device, primal.device_bytes, adjoint.device_bytes)
    if host > limits.max_host_bytes or device > limits.max_device_bytes:
        raise MemoryError(
            "UHF bare virtual ladder exceeds its host or device allowance"
        )
    if (
        setup_work > limits.max_setup_work
        or max(primal.scalar_work, adjoint.scalar_work) > limits.max_action_work
    ):
        raise MemoryError(
            "UHF bare virtual ladder exceeds its setup or action work allowance"
        )

    if admit is not None:
        admit(
            UHFVirtualLadderResourcePlan(
                retained,
                host,
                device,
                setup_work,
                primal.scalar_work,
                adjoint.scalar_work,
            )
        )
    sectors = []
    pair_name = "spatial_coulomb_antisymmetric"
    for spin, tile, cap, actions in same_spins:
        operator = prepare_spatial_virtual_pair_operators(
            factors["fit_left_" + spin],
            factors["raw_right_" + spin],
            run,
            row_tile=tile,
            max_bytes=cap,
            include_symmetric=False,
        )[pair_name]
        sectors.append(
            _Sector("r2aa" if spin == "alpha" else "r2bb", actions, pair_name, operator)
        )
    if mixed:
        panels = prepare_uhf_mixed_virtual_panels(
            factors["fit_left_alpha"],
            factors["raw_right_beta"],
            run,
            width=width,
            max_host_bytes=limits.max_host_bytes,
        )
        for panel in panels:
            alpha = slice(panel.alpha_start, panel.alpha_stop)
            sectors.append(
                _Sector(
                    "r2ab",
                    mixed[alpha.stop - alpha.start],
                    "mixed_spin_coulomb_panel",
                    panel.operator,
                    alpha,
                )
            )
    return UHFBareVirtualLadder(
        layout,
        tuple(sectors),
        run,
        retained,
        host,
        device,
        setup_work,
        primal.scalar_work,
        adjoint.scalar_work,
    )
