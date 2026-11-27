"""Stage exact independent-spin virtual Coulomb panels for bounded CC actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.common.resources import checked_bytes
from generativeqc_compiler.mp2.f12_density_fitting import fitted_integral_block

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class UHFMixedVirtualPanel:
    """One read-only alpha-target interval of the live unlike-spin operator."""

    alpha_start: int
    alpha_stop: int
    operator: np.ndarray


def uhf_mixed_virtual_panel_storage(
    virtual_alpha: int, virtual_beta: int, auxiliary: int, width: int
) -> tuple[int, int]:
    """Return added host staging and retained operator bytes for disjoint panels.

    The host preparation accounts for the simultaneous native output and
    its reordered owner copy. Native forward planning, the caller-owned DF
    factors and repeated amplitude/adjoint transfers have separate budgets.
    """
    for name, number in (
        ("virtual_alpha", virtual_alpha),
        ("virtual_beta", virtual_beta),
        ("auxiliary", auxiliary),
        ("width", width),
    ):
        if checked_bytes(number, "UHF mixed virtual " + name) == 0:
            raise ValueError(f"UHF mixed virtual {name} must be positive")
    if width > virtual_alpha:
        raise ValueError("UHF mixed virtual width must fit the alpha virtual space")
    elements_per_alpha = virtual_alpha * virtual_beta**2
    retained = checked_bytes(
        8 * virtual_alpha * elements_per_alpha,
        "UHF mixed virtual retained host storage",
    )
    tile = width * elements_per_alpha
    # Count the alternate physical ordering and any contiguous copies of input factors.
    factors = auxiliary * (width * virtual_alpha + virtual_beta**2)
    scratch = 8 * (tile + factors) + max(tile, factors)
    return checked_bytes(
        retained + scratch, "UHF mixed virtual host preparation"
    ), retained


def prepare_uhf_mixed_virtual_panels(
    fit_alpha: np.ndarray,
    raw_beta: np.ndarray,
    run: Callable,
    *,
    width: int,
    max_host_bytes: int,
) -> tuple[UHFMixedVirtualPanel, ...]:
    """Reuse retained host panels with one live device operator per CC action.

    This path uses both independent spin frames without assuming a common
    virtual space or projecting either factor onto a pair symmetry. The
    supplied fitting runner must admit each native graph before execution.
    """
    if (
        any(
            np_array.ndim != 3
            or np_array.dtype != np.float64
            or len(np_array) == 0
            or np_array.shape[1] != np_array.shape[2]
            for np_array in (fit_alpha, raw_beta)
        )
        or fit_alpha.shape[0] != raw_beta.shape[0]
    ):
        raise ValueError(
            "UHF mixed virtual panels require square FP64 factors with a shared auxiliary axis"
        )
    auxiliary, virtual_alpha, _ = fit_alpha.shape
    virtual_beta = raw_beta.shape[1]
    required, _ = uhf_mixed_virtual_panel_storage(
        virtual_alpha, virtual_beta, auxiliary, width
    )
    if required > checked_bytes(max_host_bytes, "UHF mixed virtual max_host_bytes"):
        raise MemoryError(
            "UHF mixed virtual panels exceed the admitted host numeric budget"
        )
    if not np.isfinite(raw_beta).all():
        raise ValueError("UHF mixed virtual panels require finite fitting factors")
    right = np.ascontiguousarray(raw_beta).reshape(auxiliary, virtual_beta**2)
    graphs = {
        count: fitted_integral_block(
            auxiliary, count * virtual_alpha, virtual_beta**2, coulomb=True
        )
        for count in {width, virtual_alpha % width}
        if count
    }
    panels = []
    for start in range(0, virtual_alpha, width):
        stop = min(start + width, virtual_alpha)
        count = stop - start
        left = np.ascontiguousarray(fit_alpha[:, start:stop, :]).reshape(
            auxiliary, count * virtual_alpha
        )
        if not np.isfinite(left).all():
            raise ValueError("UHF mixed virtual panels require finite fitting factors")
        block = run(graphs[count], {"fit_left": left, "raw_right": right}).outputs[
            "block"
        ]
        if (
            block.dtype != np.float64
            or block.shape != (count * virtual_alpha, virtual_beta**2)
            or not np.isfinite(block).all()
        ):
            raise ValueError(
                "UHF mixed virtual fitting returned an invalid Coulomb panel"
            )
        operator = np.ascontiguousarray(
            block.reshape(count, virtual_alpha, virtual_beta, virtual_beta).transpose(
                1, 3, 0, 2
            )
        )
        operator.setflags(write=False)
        panels.append(UHFMixedVirtualPanel(start, stop, operator))
        del left, block
    return tuple(panels)
