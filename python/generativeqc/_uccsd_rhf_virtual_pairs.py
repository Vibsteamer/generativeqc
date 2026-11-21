"""Prepare fixed spatial virtual Coulomb matrices for repeated RHF CC actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDLayout
from generativeqc_compiler.cc.uccsd_response import (
    UCCRHFRepeatedLambdaPrograms,
    build_f12_rhf_repeated_lambda_programs,
)
from generativeqc_compiler.cc.uccsd_rhf_virtual_pairs import virtual_pair_coordinates
from generativeqc_compiler.common.resources import checked_bytes
from generativeqc_compiler.mp2.f12_density_fitting import fitted_integral_block

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from generativeqc_compiler.tensor import Program


@dataclass(frozen=True, kw_only=True)
class RHFDFLadderOptions:
    """Explicit logical work budgets for the bounded RHF molecular CC route."""

    max_primal_work: int
    max_source_work: int
    row_tile: int = 256
    auxiliary_tile: int = 8

    def __post_init__(self) -> None:
        for name in (
            "max_primal_work",
            "max_source_work",
            "row_tile",
            "auxiliary_tile",
        ):
            if checked_bytes(getattr(self, name), name) <= 0:
                raise ValueError(f"RHF DF ladder option {name} must be positive")


def rhf_virtual_pair_storage(
    virtual: int, auxiliary: int, row_tile: int
) -> tuple[int, int]:
    """Logical added host preparation and retained operator bytes."""
    n_pair = virtual * (virtual + 1) // 2
    n_strict = n_pair - virtual
    width = min(row_tile, n_pair)
    preparation = checked_bytes(
        8
        * max(
            2 * auxiliary * n_pair + 2 * n_pair,
            n_pair**2
            + auxiliary * (n_pair + 3 * width)
            + 5 * width * n_pair
            + 8 * n_pair,
            n_pair**2 + max(5 * auxiliary * virtual, 5 * width * n_pair) + 8 * n_pair,
            2 * n_pair**2 + n_strict**2 + virtual**2 + 12 * n_pair,
        ),
        "RHF virtual pair preparation storage",
    )
    return preparation, 8 * (n_pair**2 + n_strict**2)


def prepare_rhf_virtual_pair_operators(
    fit: np.ndarray,
    raw: np.ndarray,
    run: Callable,
    *,
    row_tile: int,
    max_bytes: int,
) -> dict[str, np.ndarray]:
    """Prepare the restricted operator once from canonical MP2Fit factors.

    Each factor is a real whole-spin virtual block. The fitted factor must be
    the metric solve of the raw factor. The staged Coulomb matrix uses unique
    chemists' orbital pairs. Separate output matrices fold the two ordered
    virtual sources into symmetric and antisymmetric representatives.
    max_bytes bounds added host storage. The caller also admits the fitting
    graph through its own executor and accounts for the retained inputs.
    """
    if (
        fit.ndim != 3
        or fit.shape != raw.shape
        or fit.dtype != np.float64
        or raw.dtype != np.float64
        or fit.shape[1] != fit.shape[2]
        or fit.shape[1] < 2
        or not fit.shape[0]
    ):
        raise ValueError("RHF virtual operator requires matched FP64 fitting blocks")
    if type(row_tile) is not int or row_tile < 1:
        raise ValueError("RHF virtual Coulomb row tile must be positive")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("RHF virtual Coulomb memory budget must be positive")

    auxiliary, virtual, _ = fit.shape
    n_pair = virtual * (virtual + 1) // 2
    n_strict = n_pair - virtual
    width = min(row_tile, n_pair)
    required, _ = rhf_virtual_pair_storage(virtual, auxiliary, row_tile)
    if required > max_bytes:
        raise MemoryError("RHF virtual pair operators exceed the host numeric budget")
    for array in (fit, raw):
        for row in range(virtual):
            if not np.allclose(array[:, row, :], array[:, :, row], rtol=0, atol=1e-10):
                raise ValueError("Virtual factors lack the RHF within-pair symmetry")
    pairs = virtual_pair_coordinates(virtual, diagonal=True)
    strict = virtual_pair_coordinates(virtual, diagonal=False)
    first, second = map(np.asarray, zip(*pairs, strict=True))
    right = np.ascontiguousarray(raw[:, first, second])
    coulomb = np.empty((n_pair, n_pair), dtype=np.float64)
    graphs = {
        n: fitted_integral_block(auxiliary, n, n_pair, coulomb=True)
        for n in {width, n_pair % width}
        if n
    }
    for start in range(0, n_pair, width):
        stop = min(start + width, n_pair)
        left = np.ascontiguousarray(fit[:, first[start:stop], second[start:stop]])
        result = run(
            graphs[stop - start], {"fit_left": left, "raw_right": right}
        ).outputs["block"]
        if (
            result.dtype != np.float64
            or result.shape != (stop - start, n_pair)
            or not np.isfinite(result).all()
        ):
            raise ValueError("Invalid prepared spatial Coulomb row block")
        coulomb[start:stop] = result
    del left, right, result
    for start in range(0, n_pair, width):
        stop = min(start + width, n_pair)
        if not np.allclose(
            coulomb[start:stop], coulomb[:, start:stop].T, rtol=0, atol=1e-10
        ):
            raise ValueError(
                "Prepared MP2Fit Coulomb matrix lacks pair-exchange symmetry"
            )

    lookup = np.empty((virtual, virtual), dtype=np.int64)
    lookup[first, second] = lookup[second, first] = np.arange(n_pair)
    strict_first, strict_second = map(np.asarray, zip(*strict, strict=True))
    symmetric = np.empty_like(coulomb)
    antisymmetric = np.empty((n_strict, n_strict), dtype=np.float64)
    for row, (e, f) in enumerate(pairs):
        symmetric[row] = coulomb[lookup[first, e], lookup[second, f]]
        if e != f:
            symmetric[row] += coulomb[lookup[first, f], lookup[second, e]]
    for row, (e, f) in enumerate(strict):
        antisymmetric[row] = (
            coulomb[lookup[strict_first, e], lookup[strict_second, f]]
            - coulomb[lookup[strict_first, f], lookup[strict_second, e]]
        )
    symmetric.setflags(write=False)
    antisymmetric.setflags(write=False)
    return {
        "spatial_coulomb_symmetric": symmetric,
        "spatial_coulomb_antisymmetric": antisymmetric,
    }


def prepare_rhf_f12_lambda_action(
    layout: UCCSDLayout,
    packed_amplitudes: np.ndarray,
    equation_feeds: Mapping[str, np.ndarray],
    operators: Mapping[str, np.ndarray],
    run: Callable,
    *,
    max_retained_bytes: int,
    programs: UCCRHFRepeatedLambdaPrograms | None = None,
) -> _RHFVirtualLambdaAction:
    """Prepare an exact physical Λ transpose for one fixed-reference epoch.

    The caller owns the remaining immutable equation feeds and spatial
    operators. This coordinator retains the derived remainder frontier and
    the unpacked amplitudes. Complete native executor/work admission and
    state and Lambda qualification belong to the higher stationary owner.
    """
    if programs is None:
        programs = build_f12_rhf_repeated_lambda_programs(
            layout, max_retained_bytes=max_retained_bytes
        )
    if programs.remainder.retained_bytes > max_retained_bytes:
        raise MemoryError("RHF Lambda frontier retained storage exceeds the budget")
    amplitudes = layout.unpack(packed_amplitudes)

    def bind(graph: Program, values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        feeds = {}
        for node in graph.live_nodes:
            if node.op != "input":
                continue
            name = node.attrs["name"]
            if name.startswith("bar_"):
                continue
            value = values[name]
            if value.dtype != np.float64 or value.shape != node.spec.shape:
                raise ValueError("RHF Lambda epoch received an invalid FP64 binding")
            if name not in amplitudes and value.flags.writeable:
                raise ValueError("RHF Lambda epoch requires immutable static inputs")
            feeds[name] = value
        return feeds

    preparation = bind(programs.remainder.prepare, {**equation_feeds, **amplitudes})
    retained = run(programs.remainder.prepare, preparation).outputs
    if retained.keys() != programs.remainder.prepare.outputs.keys():
        raise ValueError("RHF Lambda preparation returned incomplete retained inputs")
    for value in retained.values():
        value.setflags(write=False)
    replay = bind(
        programs.remainder.replay, {**equation_feeds, **amplitudes, **retained}
    )
    bare = bind(programs.bare, {**operators, **amplitudes})

    return _RHFVirtualLambdaAction(layout, run, programs, replay, bare)


@dataclass(frozen=True)
class _RHFVirtualLambdaAction:
    layout: UCCSDLayout
    run: Callable
    programs: UCCRHFRepeatedLambdaPrograms
    replay: dict[str, np.ndarray]
    bare: dict[str, np.ndarray]

    def bars(self, seeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        remainder = self.run(
            self.programs.remainder.replay, {**self.replay, **seeds}
        ).outputs
        ladder = self.run(self.programs.bare, {**self.bare, **seeds}).outputs
        return {
            "bar_" + name: remainder["bar_" + name] + ladder["bar_" + name]
            for name in BLOCK_NAMES
        }

    def __call__(self, dual: np.ndarray) -> np.ndarray:
        seeds = {
            "bar_" + name.replace("t", "r", 1): value
            for name, value in self.layout.unpack_dual(dual).items()
        }
        result = self.bars(seeds)
        return self.layout.pack_dual(
            {name: result["bar_" + name] for name in BLOCK_NAMES}
        )
