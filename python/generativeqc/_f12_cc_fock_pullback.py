"""Physical extended spin Fock and exchange with independently attributed JK fitting."""

import ctypes
from collections.abc import Callable, Mapping

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
) -> FockResponse:
    panel = ao if jkfit is None else jkfit
    frames = [ao["coefficients_alpha"], ao["coefficients_beta"]]
    fock, physical = physical_spin_operators(
        ao["hcore_ao"],
        panel["ao_coulomb"],
        panel["metric_whitener"],
        frames,
        (space[0], space[1]),
        run,
        cuda,
        library,
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
            bars["ao_coulomb"] += raw
            bars["metric/coulomb"] += metric
        else:
            bars["jkfit/ao_coulomb"] = raw
            bars["jkfit/metric/coulomb"] = metric

    return fock, pullback
