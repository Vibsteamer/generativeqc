"""Physical extended spin Fock response with independently attributed JK fitting."""

from collections.abc import Callable, Mapping

import numpy as np

from ._f12_spin_ao import _orbital_sources
from ._f12_spin_fock import spin_fock, spin_fock_pullback


def extended_fock_response(
    space: tuple[int, int, int, int, int],
    ao: Mapping[str, np.ndarray],
    jkfit: Mapping[str, np.ndarray] | None,
    run: Callable,
) -> tuple[
    dict[str, np.ndarray],
    Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], None],
]:
    shared = ("coefficients_alpha", "coefficients_beta", "hcore_ao")
    fit = ("ao_coulomb", "metric/coulomb", "metric_whitener")
    panel = {k: ao[k] for k in shared}
    panel.update({k: (ao if jkfit is None else jkfit)[k] for k in fit})
    qjk = len(panel["metric/coulomb"])
    frame = (*space[:4], qjk)
    fspace = (*space[:2], space[3], qjk)
    data, _ = _orbital_sources(frame, panel, run, None, projected=None, role="fock")
    inputs = {
        k.removeprefix("source/"): v for k, v in data.items() if k.startswith("source/")
    }
    fock = spin_fock(fspace, inputs, run)

    def pullback(seed: dict[str, np.ndarray], bars: dict[str, np.ndarray]) -> None:
        transformed = spin_fock_pullback(fspace, inputs, seed, run)
        _, physical = _orbital_sources(
            frame,
            panel,
            run,
            {"source/" + k: v for k, v in transformed.items()},
            projected=data,
            role="fock",
        )
        assert physical is not None
        for key in shared:
            bars[key] += physical[key]
        for key in ("ao_coulomb", "metric/coulomb"):
            if jkfit is None:
                bars[key] += physical[key]
            else:
                bars["jkfit/" + key] = physical[key]

    return fock, pullback
