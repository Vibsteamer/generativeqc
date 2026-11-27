"""Physical extended spin Fock and exchange with independently attributed JK fitting."""

import ctypes
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from ._f12_cabs_fock import physical_spin_operators

FockResponse = tuple[
    dict[str, np.ndarray],
    Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], None],
]


def extended_fock_response(
    space: tuple[int, int, int, int, int],
    ao: Mapping[str, np.ndarray],
    jkfit: Mapping[str, np.ndarray] | None,
    run: Callable,
    *,
    cuda: bool = False,
    library: ctypes.CDLL | None = None,
    native_library: Path | str | ctypes.CDLL | None = None,
    host_reference: bool | None = None,
) -> FockResponse:
    panel = ao if jkfit is None else jkfit
    frames = [ao["coefficients_alpha"], ao["coefficients_beta"]]
    if "ao_coulomb" in panel:
        raw_coulomb = panel["ao_coulomb"]
        raw_loader = None
    else:
        raw_coulomb = None
        loader = panel["__ao_loader__"]
        raw_loader = lambda: loader("ao_coulomb")
    fock, physical = physical_spin_operators(
        ao["hcore_ao"],
        raw_coulomb,
        panel["metric_whitener"],
        frames,
        (space[0], space[1]),
        run,
        cuda,
        library,
        raw_loader=raw_loader,
        native_library=native_library,
        host_reference=host_reference,
    )
    for matrix in fock.values():
        if not np.isfinite(matrix).all() or not np.allclose(
            matrix, matrix.T, atol=2e-8, rtol=0
        ):
            raise ValueError("Non-Hermitian extended CC Fock or exchange matrix")

    def pullback(seed: dict[str, np.ndarray], bars: dict[str, np.ndarray]) -> None:
        coefficients, core, raw, metric = physical(seed)
        bars["coefficients_alpha"] += coefficients[0]
        bars["coefficients_beta"] += coefficients[1]
        bars["hcore_ao"] += core
        if jkfit is None:
            if "ao_coulomb" in bars:
                bars["ao_coulomb"] += raw
            else:
                bars["ao_coulomb"] = raw
            bars["metric/coulomb"] += metric
        else:
            if "jkfit/ao_coulomb" in bars:
                bars["jkfit/ao_coulomb"] += raw
            else:
                bars["jkfit/ao_coulomb"] = raw
            if "jkfit/metric/coulomb" in bars:
                bars["jkfit/metric/coulomb"] += metric
            else:
                bars["jkfit/metric/coulomb"] = metric

    return fock, pullback
