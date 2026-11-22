"""Nuclear and orbital pullback of the unrestricted fixed-geminal scalar V + B − X."""

from collections.abc import Callable, Mapping
from typing import NamedTuple

import numpy as np
from generativeqc_compiler.cc.uccsd_f12_quadratic import (
    unrestricted_fixed_geminal_quadratic_pair,
)
from generativeqc_compiler.cc.uccsd_f12_sources import unrestricted_f12_pair_sources
from generativeqc_compiler.tensor import Program, transpose_program

from generativeqc._f12_cc_fock_pullback import FockResponse, extended_fock_response
from generativeqc._f12_cc_spin_sources import _quadratic_pair_fields
from generativeqc._f12_spin_ao import _orbital_sources
from generativeqc._f12_spin_pairs import occupied_pair_blocks


class UnrestrictedCCScalarResponse(NamedTuple):
    linear_v: float
    b_energy: float
    x_fock_energy: float
    weights: dict[str, np.ndarray]


def _with_transpose(primal: Program) -> Program:
    reverse = transpose_program(primal, tuple(primal.outputs)).program
    return Program({**primal.outputs, **reverse.outputs})


def unrestricted_cc_scalar_ao_weights(
    space: tuple[int, int, int, int, int],
    ao: dict[str, np.ndarray],
    scalar_weight: float | np.ndarray,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    jkfit: Mapping[str, np.ndarray] | None = None,
    fock_response: FockResponse | None = None,
) -> UnrestrictedCCScalarResponse:
    """Return each unweighted energy and the weighted derivative of their signed sum.

    Both spin frames use the full occupied/virtual/CABS space. The same fitted
    Coulomb, positive exchange and five robust integral channels generate the
    scalar and its adjoint. Frozen pairs are excluded from V, B and F_oo X.
    Frozen orbitals still contribute to the physical Fock and the projector.
    """
    oa, ob, obs, ext, aux = space
    weight = np.asarray(scalar_weight)
    if weight.shape or weight.dtype.kind not in "fiu" or not np.isfinite(weight):
        raise ValueError("A finite stationary scalar cotangent is required")
    if not np.allclose(
        ao["metric_whitener"] @ ao["metric/coulomb"] @ ao["metric_whitener"].T,
        np.eye(aux),
        atol=2e-7,
        rtol=0,
    ):
        raise ValueError("The full-rank whitener and Coulomb metric must match")
    occupied = dict(zip(("a", "b"), (oa, ob), strict=True))
    frozen = dict(zip(("a", "b"), frozen_occupied, strict=True))
    active = {s: slice(frozen[s], no) for s, no in occupied.items()}
    data, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="pairs", occupied_coulomb=True
    )
    fock, fock_pullback = fock_response or extended_fock_response(space, ao, jkfit, run)
    dbar = {k: np.zeros_like(v) for k, v in data.items() if k != "space"}
    fbar = {k: np.zeros_like(v) for k, v in fock.items()}
    e_v = e_b = e_x = 0.0
    eye = np.eye(ext)
    seed = weight.astype(np.float64)
    for left, right, nl, nr, _, pairs in occupied_pair_blocks(
        space, frozen_occupied=frozen_occupied
    ):
        ls, rs = (nl, obs - nl, ext - obs), (nr, obs - nr, ext - obs)
        channel = left + "_" + right
        vgraph = unrestricted_f12_pair_sources(ls, rs, aux, channel=channel)
        linear = _with_transpose(
            Program({"linear_v_energy": vgraph.outputs["linear_v_energy"]})
        )
        quadratic = _with_transpose(
            unrestricted_fixed_geminal_quadratic_pair(
                ls,
                rs,
                aux,
                channel=channel,
                frozen_occupied=(frozen[left[0]], frozen[right[0]]),
            )
        )
        qfields = _quadratic_pair_fields(data, fock, left, right, active)
        qbars = {
            "metric_" + op: dbar["metric/" + op]
            for op in ("geminal", "squared_geminal", "double_commutator")
        }
        if left != right:
            qbars["cross_overlap"] = dbar["cross_overlap"]
        # The operand builder packs possibly noncontiguous active tiles. Keep
        # explicit target views for accumulating into the full spin arrays.
        legs = (
            (("left", left),) if left == right else (("left", left), ("right", right))
        )
        for side, spin in legs:
            qbars["fock_" + side] = fbar["fock_" + spin]
            qbars["exchange_" + side] = fbar["exchange_" + spin]
            qbars["fit_" + side] = dbar["source/fit_" + spin][:, active[spin[0]]]
            for op in ("geminal", "squared_geminal", "double_commutator"):
                qbars[f"raw_{op}_{side}"] = dbar[f"raw/{op}/{spin}"][:, active[spin[0]]]
        qfields.update(bar_b_energy=seed, bar_x_fock_energy=np.asarray(-seed))
        for i, j in pairs:
            selects = {"select_i": eye[i], "select_j": eye[j]}
            linear_slots = {
                "fit_left": ("source/fit_" + left, i),
                "fit_right": ("source/fit_" + right, j),
                "raw_coulomb_right": ("source/raw_coulomb_" + right, j),
            }
            for op in ("geminal", "geminal_coulomb"):
                linear_slots["metric_" + op] = ("metric/" + op, None)
                linear_slots["raw_" + op + "_left"] = (f"raw/{op}/{left}", i)
                linear_slots["raw_" + op + "_right"] = (f"raw/{op}/{right}", j)
            if left != right:
                linear_slots["cross_overlap"] = ("cross_overlap", None)
            values = {
                k: np.ascontiguousarray(data[n] if pos is None else data[n][:, pos])
                for k, (n, pos) in linear_slots.items()
            }
            vr = run(linear, {**values, **selects, "bar_linear_v_energy": seed}).outputs
            e_v += float(vr["linear_v_energy"])
            for k, (n, pos) in linear_slots.items():
                if pos is None:
                    dbar[n] += vr["bar_" + k]
                else:
                    dbar[n][:, pos] += vr["bar_" + k]
            qr = run(
                quadratic,
                {
                    **qfields,
                    "select_i": eye[i, active[left[0]]],
                    "select_j": eye[j, active[right[0]]],
                },
            ).outputs
            e_b += float(qr["b_energy"])
            e_x += float(qr["x_fock_energy"])
            for k, target in qbars.items():
                target[...] += qr["bar_" + k]
    _, physical = _orbital_sources(
        space, ao, run, dbar, projected=data, role="pairs", occupied_coulomb=True
    )
    assert physical is not None
    fock_pullback(fbar, physical)
    return UnrestrictedCCScalarResponse(e_v, e_b, e_x, physical)
