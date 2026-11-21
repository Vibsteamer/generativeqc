"""Stationary unrestricted CC cotangents in the canonical molecular spin frame."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np
from generativeqc_compiler.cc.uccsd_equations import build_f12_program
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.tensor import transpose_program


@dataclass(frozen=True)
class UCCSDF12Pullback:
    correlation_energy: float
    amplitude_residual_max: float
    lambda_stationarity_norm: float
    ordinary_weights: dict[str, np.ndarray]
    fixed_source_weights: dict[str, np.ndarray]
    virtual_factor_weights: dict[str, np.ndarray] | None = None
    numeric_bound_bytes: int | None = None
    planned_scalar_work: int | None = None


def unrestricted_cc_f12_stationary_pullback(
    layout: UCCSDLayout,
    ordinary: dict[str, np.ndarray],
    fixed: dict[str, np.ndarray],
    packed_amplitudes: np.ndarray,
    packed_multipliers: np.ndarray,
    evaluate: Callable,
) -> UCCSDF12Pullback:
    """Compute all active ordinary and geminal adjoints from one physical CC root.

    The evaluator returns named arrays, as in CudaCCTensorExecutor.execute.
    A single Lagrangian transpose computes amplitude stationarity and source
    weights together; the weights can be reused for every molecular direction.
    """
    t = layout.unpack(packed_amplitudes)
    dual = layout.unpack_dual(packed_multipliers)
    cc = build_f12_program(layout)
    inputs = {**ordinary, **fixed, **t}
    values = evaluate(cc, inputs)
    corr = float(values["correlation_energy"])
    primal_max = float(
        np.max(
            [
                float(np.max(np.abs(v), initial=0))
                for k, v in values.items()
                if k != "correlation_energy"
            ]
        )
    )
    if not isfinite(corr) or not isfinite(primal_max) or primal_max > 1e-9:
        raise ValueError(
            "Molecular F12 amplitudes failed fresh physical residual check"
        )
    adjoint = transpose_program(cc, tuple(cc.outputs)).program
    seeds = {"bar_" + k.replace("t", "r", 1): v for k, v in dual.items()}
    bars = evaluate(
        adjoint, {**inputs, **seeds, "bar_correlation_energy": np.array(1.0)}
    )
    t_bars = {b.name: bars["bar_" + b.name] for b in layout.blocks}
    lambda_norm = float(np.linalg.norm(layout.pack_dual(t_bars)))
    lambda_max = max(
        b.full_metric_weight * float(np.max(np.abs(t_bars[b.name]), initial=0))
        for b in layout.blocks
    )
    if (
        not isfinite(lambda_norm)
        or not isfinite(lambda_max)
        or max(lambda_norm, lambda_max) > 1e-9
    ):
        raise ValueError("Molecular Lambda failed fresh Lagrangian stationarity check")
    ordinary_bars = {k: bars["bar_" + k] for k in ordinary if "bar_" + k in bars}
    fixed_bars = {k: bars["bar_" + k] for k in fixed if "bar_" + k in bars}
    return UCCSDF12Pullback(corr, primal_max, lambda_norm, ordinary_bars, fixed_bars)
