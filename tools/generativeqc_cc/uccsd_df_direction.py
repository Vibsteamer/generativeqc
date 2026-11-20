"""Ordinary canonical Hamiltonian directions on a moving unrestricted DF frame."""

from collections.abc import Callable, Mapping
from math import prod

import numpy as np
from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.cc.uccsd_f12_sources import unrestricted_cc_obs_fit_direction
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES
from generativeqc_compiler.mp2.f12_density_fitting import (
    coulomb_fit_response,
    fitted_integral_block,
)
from generativeqc_compiler.tensor import Program, linearize

from .uccsd_solver import _antisymmetrized_integral_requests


def unrestricted_cc_df_hamiltonian_tangent(
    space: tuple[int, int, int, int, int],
    data: Mapping[str, np.ndarray],
    whitener: np.ndarray,
    run: Callable,
    *,
    data_tangent: Mapping[str, np.ndarray],
    fock_tangent: Mapping[str, np.ndarray],
    coulomb_metric_tangent: np.ndarray,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> dict[str, np.ndarray]:
    """Build the ordinary dF and antisymmetrized dERI inputs for one direction.

    Raw Coulomb columns and their total derivatives use the same moving MO
    frames as the F12 sources. Fock tangents include both occupied spin
    densities. Frozen orbitals remain in the supplied Fock and are removed
    from the output occupied legs. Each Coulomb tile contracts one solved
    density and one raw density, including both derivatives and the metric.
    """
    oa, ob, obs, ext, aux = space
    fa, fb = frozen_occupied
    if not (0 <= fa <= oa < obs and 0 <= fb <= ob < obs <= ext and aux > 0):
        raise ValueError("Invalid UCC occupied, frozen, OBS or fitting dimensions")
    ordinary = build_program(oa - fa, ob - fb, obs - oa, obs - ob)
    shapes = {
        node.attrs["name"]: node.spec.shape for node in ordinary.live_nodes
        if node.op == "input" and node.attrs["name"] not in BLOCK_NAMES
    }
    spins = {"a": "alpha", "b": "beta"}
    domains = {
        (spin, region): sl
        for spin, occupied, frozen in (("a", oa, fa), ("b", ob, fb))
        for region, sl in (("o", slice(frozen, occupied)), ("v", slice(occupied, obs)))
    }
    solve = coulomb_fit_response(aux, obs * obs)
    fit_graph = Program({"fit": solve.outputs["fit"]})
    dfit_graph = unrestricted_cc_obs_fit_direction(aux, obs)
    factors = {}
    for spin in spins.values():
        fock = fock_tangent["fock_" + spin]
        if fock.shape != (ext, ext) or fock.dtype != np.float64 or not np.isfinite(fock).all():
            raise ValueError("Expected a finite FP64 extended spin Fock tangent")
        name = "source/raw_coulomb_" + spin
        raw = np.ascontiguousarray(data[name][:, :obs, :obs]).reshape(aux, -1)
        draw = np.ascontiguousarray(data_tangent[name][:, :obs, :obs]).reshape(aux, -1)
        fit = run(
            fit_graph, {"metric_whitener": whitener, "three_center": raw}
        ).outputs["fit"]
        dfit = run(
            dfit_graph,
            {
                "metric_whitener": whitener,
                "d_metric": coulomb_metric_tangent,
                "fit": fit,
                "d_three_center": draw,
            },
        ).outputs["d_fit"]
        fields = {
            "fit_left": fit,
            "raw_right": raw,
            "d_fit_left": dfit,
            "d_raw_right": draw,
        }
        factors[spin] = {k: v.reshape(aux, obs, obs) for k, v in fields.items()}

    kernels = {}
    result = {}
    for name, shape in shapes.items():
        sections, labels = name[1:].split("_")
        slices = tuple(domains[s, d] for s, d in zip(labels, sections, strict=True))
        if name.startswith("f"):
            result[name] = np.ascontiguousarray(fock_tangent["fock_" + spins[labels[0]]][slices])
            continue
        value = np.zeros(shape, dtype=np.float64)
        if value.size:
            slots = tuple(tuple(range(sl.start, sl.stop)) for sl in slices)
            owners = tuple(spins[s] for s in labels)
            for sign, request, axes in _antisymmetrized_integral_requests(slots, owners):
                nl, nr = prod(request.shape[:2]), prod(request.shape[2:])
                if (nl, nr) not in kernels:
                    graph = fitted_integral_block(aux, nl, nr, coulomb=True)
                    kernels[nl, nr] = linearize(graph, None).program
                rows = tuple(slice(s[0], s[-1] + 1) for s in request.slots)
                fields = {}
                for key, side in (("fit_left", 0), ("raw_right", 2)):
                    for prefix in ("", "d_"):
                        field = prefix + key
                        arr = factors[request.spins[side]][field]
                        tile = arr[:, rows[side], rows[side + 1]]
                        fields[field] = np.ascontiguousarray(tile).reshape(aux, -1)
                chemists = run(kernels[nl, nr], fields).outputs["d_block"]
                value += sign * chemists.reshape(request.shape).transpose(axes)
        result[name] = value
    return result
