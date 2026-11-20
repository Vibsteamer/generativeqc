"""Molecular fixed-cusp UCCSD-F12b energy and Cartesian forces in a common DF frame."""

from collections.abc import Callable
from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np
from generativeqc._f12_cabs import cabs_singles
from generativeqc._f12_cc_spin_sources import (
    unrestricted_cc_f12_fixed_source_feeds,
    unrestricted_cc_f12_fixed_source_tangent,
)
from generativeqc._f12_spin_force import spin_source_force
from generativeqc_compiler.cc.uccsd_equations import build_f12_program
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_extended_df_fock
from generativeqc_compiler.tensor import linearize, transpose_program

from .uccsd_df_direction import (
    unrestricted_cc_df_hamiltonian,
    unrestricted_cc_df_hamiltonian_tangent,
)


@dataclass(frozen=True)
class UCCSDF12CartesianResult:
    energy: float
    reference_energy: float
    correlation_energy: float
    cabs_singles_energy: float | None
    forces: np.ndarray
    reference_forces: np.ndarray
    ordinary_cc_forces: np.ndarray
    fixed_source_forces: np.ndarray
    cabs_singles_forces: np.ndarray | None
    amplitude_residual_max: float
    lambda_stationarity_norm: float


def unrestricted_cc_f12_energy_forces(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    whitener: np.ndarray,
    packed_amplitudes: np.ndarray,
    packed_multipliers: np.ndarray,
    run: Callable,
    *,
    natoms: int,
    coordinate_tangent: Callable[
        [int, int], tuple[dict[str, np.ndarray], np.ndarray, float]
    ],
    reference_energy: float,
    include_cabs_singles: bool,
    frozen_occupied: tuple[int, int] = (0, 0),
    singles: Callable = cabs_singles,
    block_elements: int = 262144,
) -> UCCSDF12CartesianResult:
    """Check one CC root, then contract its stationary Lagrangian for every nucleus.

    The packed T and Lambda use the same two moving spin frames as the supplied
    molecular DF sources. Each coordinate callback supplies the complete
    projected source tangent, physical Coulomb metric tangent and total UHF
    reference derivative in Eh/bohr (including internuclear repulsion).
    The callback is requested separately for every atom and Cartesian axis.
    CABS singles uses both full occupied spaces and is added once when enabled.
    """
    if type(natoms) is not int or natoms < 1 or not isfinite(reference_energy):
        raise ValueError("Expected a positive atom count and a finite reference energy")
    if type(include_cabs_singles) is not bool:
        raise TypeError("Expected an explicit boolean CABS singles selection")
    oa, ob, obs, ext, aux = space
    fa, fb = frozen_occupied
    layout = UCCSDLayout(oa - fa, ob - fb, obs - oa, obs - ob)
    t = layout.unpack(packed_amplitudes)
    dual = layout.unpack_dual(packed_multipliers)
    source = {
        k.removeprefix("source/"): v for k, v in data.items() if k.startswith("source/")
    }
    fg = unrestricted_extended_df_fock(oa, ob, ext, aux)
    fock = run(fg, source).outputs
    ordinary = unrestricted_cc_df_hamiltonian(
        space, data, fock, whitener, run, frozen_occupied=frozen_occupied
    )
    fixed = unrestricted_cc_f12_fixed_source_feeds(
        space, data, fock, whitener, run, frozen_occupied=frozen_occupied
    )
    cc = build_f12_program(layout)
    inputs = {**ordinary, **fixed, **t}
    values = run(cc, inputs).outputs
    corr = float(values["correlation_energy"])
    primal_max = max(
        float(np.max(np.abs(v), initial=0))
        for k, v in values.items()
        if k != "correlation_energy"
    )
    if not isfinite(corr) or not isfinite(primal_max) or primal_max > 1e-9:
        raise ValueError(
            "Molecular F12 amplitudes failed fresh physical residual check"
        )
    adjoint = transpose_program(cc, tuple(cc.outputs)).program
    seeds = {"bar_" + k.replace("t", "r", 1): v for k, v in dual.items()}
    bars = run(
        adjoint, {**inputs, **seeds, "bar_correlation_energy": np.array(1.0)}
    ).outputs
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
    del bars, values, inputs, ordinary, fixed, t_bars

    singles_energies = []
    singles_bars = {}
    if include_cabs_singles:
        for spin, occ in (("alpha", oa), ("beta", ob)):
            key = "fock_" + spin
            matrix = fock[key]
            energy, bar = singles(
                (matrix + matrix.T) / 2, occ, spin_weight=1, pullback=True
            )
            if bar is None or not isfinite(energy):
                raise ValueError(
                    "Missing finite spin CABS singles energy or Fock derivative"
                )
            singles_energies.append(energy)
            singles_bars[key] = (bar + bar.T) / 2

    dfg = linearize(fg, None).program
    ref_force, ordinary_force, fixed_force = (np.empty((natoms, 3)) for _ in range(3))
    singles_force = np.empty((natoms, 3)) if include_cabs_singles else None
    for atom, axis in np.ndindex(natoms, 3):
        ddata, dmetric, ref_grad = coordinate_tangent(atom, axis)
        if not isfinite(ref_grad):
            raise ValueError("Expected finite Cartesian reference derivative")
        dfock = {
            k.removeprefix("d_"): v
            for k, v in run(
                dfg, {**source, **{"d_" + k: ddata["source/" + k] for k in source}}
            ).outputs.items()
        }
        args = {
            "data_tangent": ddata,
            "fock_tangent": dfock,
            "coulomb_metric_tangent": dmetric,
            "frozen_occupied": frozen_occupied,
        }
        dord = unrestricted_cc_df_hamiltonian_tangent(
            space, data, whitener, run, **args
        )
        ordinary_force[atom, axis] = spin_source_force(
            ordinary_bars, dord, 0.0, run, block_elements=block_elements
        )
        del dord
        dfixed = unrestricted_cc_f12_fixed_source_tangent(
            space, data, fock, whitener, run, **args
        )
        fixed_force[atom, axis] = spin_source_force(
            fixed_bars, dfixed, 0.0, run, block_elements=block_elements
        )
        ref_force[atom, axis] = -ref_grad
        if singles_force is not None:
            singles_force[atom, axis] = spin_source_force(
                singles_bars, dfock, 0.0, run, block_elements=block_elements
            )
    forces = ref_force + ordinary_force + fixed_force
    if singles_force is not None:
        forces += singles_force
    return UCCSDF12CartesianResult(
        fsum((reference_energy, corr, *singles_energies)),
        reference_energy,
        corr,
        fsum(singles_energies) if include_cabs_singles else None,
        immutable(forces),
        immutable(ref_force),
        immutable(ordinary_force),
        immutable(fixed_force),
        immutable(singles_force) if singles_force is not None else None,
        primal_max,
        lambda_norm,
    )
