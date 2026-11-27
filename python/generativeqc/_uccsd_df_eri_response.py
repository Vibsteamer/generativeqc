"""Pull the ordinary antisymmetrized unrestricted CC ERIs back to their MP2Fit and MO inputs."""

from collections.abc import Callable, Mapping
from math import prod

import numpy as np
from generativeqc_compiler.cc.uccsd_coulomb_blocks import (
    antisymmetrized_integral_requests,
)
from generativeqc_compiler.cc.uccsd_df_virtual_ladder import FACTOR_INPUTS
from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.mp2.f12_density_fitting import (
    coulomb_fit_response,
    fitted_integral_block,
    molecular_orbital_sources,
    symmetric_coulomb_pairs,
)
from generativeqc_compiler.tensor import Program, transpose_program


def unrestricted_cc_eri_pullback(
    occupied: tuple[int, int],
    weights: Mapping[str, np.ndarray],
    coefficients: Mapping[str, np.ndarray],
    three_center: np.ndarray,
    metric: np.ndarray,
    whitener: np.ndarray,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    virtual_factor_weights: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Transpose all live antisymmetrized ordinary ERI blocks of the UCC Lagrangian.

    The fitted Coulomb tile is (M^-1 G_left).T @ G_right. Each positive
    direct and negative exchange contribution uses the same orbital slices,
    canonical chemists' order and permutations as the ordinary UCC builder.
    The returned four energy cotangents act on the shared AO three-center and
    metric integrals and on the complete ordinary MO columns in each spin.
    Frozen orbitals occur in the UHF reference but are omitted from active ERI
    occupied legs. Only one reconstructed four-index tile is needed at a time.
    A bounded virtual ladder supplies its four accumulated fit/raw factor
    cotangents in place of the dense four-virtual ERI block cotangents.
    """
    spins = {"a": "alpha", "b": "beta"}
    nao, nmo = coefficients["alpha"].shape
    aux = len(metric)
    oa, ob = occupied
    fa, fb = frozen_occupied
    if not (0 <= fa <= oa < nmo and 0 <= fb <= ob < nmo and nmo <= nao and aux):
        raise ValueError("Invalid active UCC occupied or orbital space")
    if (
        coefficients["beta"].shape != (nao, nmo)
        or three_center.shape != (aux, nao, nao)
        or metric.shape != (aux, aux)
        or whitener.shape != metric.shape
        or any(
            x.dtype != np.float64
            for x in (*coefficients.values(), three_center, metric, whitener)
        )
    ):
        raise ValueError("Inconsistent FP64 molecular orbital and fitting arrays")
    if not np.allclose(whitener @ metric @ whitener.T, np.eye(aux), atol=2e-7, rtol=0):
        raise ValueError("The Coulomb whitener must be full rank for the same metric")
    cc = build_program(oa - fa, ob - fb, nmo - oa, nmo - ob)
    needed = {
        node.attrs["name"]: node.spec.shape
        for node in cc.live_nodes
        if node.op == "input"
        and node.attrs["name"].startswith("g")
        and (
            virtual_factor_weights is None
            or not node.attrs["name"].startswith("gvvvv_")
        )
    }
    if set(weights) != set(needed):
        raise ValueError(
            "Expected every live ordinary antisymmetrized two-electron weight"
        )
    if virtual_factor_weights is not None:
        extents = (nmo - oa, nmo - ob) * 2
        if set(virtual_factor_weights) != set(FACTOR_INPUTS) or any(
            value.shape != (aux, extent, extent)
            or value.dtype != np.float64
            or not np.isfinite(value).all()
            for value, extent in zip(
                (virtual_factor_weights[name] for name in FACTOR_INPUTS),
                extents,
                strict=True,
            )
        ):
            raise ValueError("Invalid MP2Fit virtual factor cotangents")
    domains = {
        (label, kind): tuple(range(begin, end))
        for label, no, nf in zip(spins, occupied, frozen_occupied, strict=True)
        for kind, begin, end in (("o", nf, no), ("v", no, nmo))
    }
    # The full OBS projection and implicit solve are reused by every signed block.
    mo_source = molecular_orbital_sources(nao, nmo, 0, aux).outputs["raw_coulomb"]
    if virtual_factor_weights is not None:
        # Transposing this graph applies the same self-adjoint pair projection
        # to the combined weight from the bare factors and the metric solve.
        mo_source = symmetric_coulomb_pairs(mo_source)
    project = Program({"raw_coulomb": mo_source})
    project_rev = transpose_program(project, ("raw_coulomb",)).program
    solve = coulomb_fit_response(aux, nmo * nmo)
    fit_graph = Program({"fit": solve.outputs["fit"]})
    fit_rev = Program({k: solve.outputs[k] for k in ("metric_bar", "three_center_bar")})
    raw, fit, raw_bar, fit_bar = {}, {}, {}, {}
    for spin, no in zip(spins.values(), occupied, strict=True):
        raw[spin] = run(
            project, {"coefficients": coefficients[spin], "ao_coulomb": three_center}
        ).outputs["raw_coulomb"]
        fit[spin] = (
            run(
                fit_graph,
                {
                    "metric_whitener": whitener,
                    "three_center": raw[spin].reshape(aux, -1),
                },
            )
            .outputs["fit"]
            .reshape(aux, nmo, nmo)
        )
        raw_bar[spin] = np.zeros_like(raw[spin])
        fit_bar[spin] = np.zeros_like(raw[spin])
        if virtual_factor_weights is not None:
            raw_bar[spin][:, no:, no:] = virtual_factor_weights["raw_right_" + spin]
            fit_bar[spin][:, no:, no:] = virtual_factor_weights["fit_left_" + spin]
    reverse = {}
    for name, shape in needed.items():
        bar = weights[name]
        if bar.shape != shape or bar.dtype != np.float64:
            raise ValueError(f"Inconsistent FP64 cotangent for {name}")
        if not bar.size:
            continue
        parts, labels = name[1:].split("_")
        slots = tuple(
            domains[lab, kind] for lab, kind in zip(labels, parts, strict=True)
        )
        channels = tuple(spins[lab] for lab in labels)
        for sign, request, axes in antisymmetrized_integral_requests(slots, channels):
            rows = tuple(slice(r[0], r[-1] + 1) for r in request.slots)
            left = (slice(None), *rows[:2])
            right = (slice(None), *rows[2:])
            ls, rs = request.spins[0], request.spins[2]
            nl, nr = prod(request.shape[:2]), prod(request.shape[2:])
            if (nl, nr) not in reverse:
                tile = fitted_integral_block(aux, nl, nr, coulomb=True)
                reverse[nl, nr] = transpose_program(tile, ("block",)).program
            seed = bar.transpose(tuple(np.argsort(axes))).reshape(nl, nr)
            fields = {
                "fit_left": np.ascontiguousarray(fit[ls][left]).reshape(aux, nl),
                "raw_right": np.ascontiguousarray(raw[rs][right]).reshape(aux, nr),
                "bar_block": np.ascontiguousarray(sign * seed),
            }
            result = run(reverse[nl, nr], fields).outputs
            fit_bar[ls][left] += result["bar_fit_left"].reshape(aux, *request.shape[:2])
            raw_bar[rs][right] += result["bar_raw_right"].reshape(
                aux, *request.shape[2:]
            )
    physical = {
        "metric/coulomb": np.zeros_like(metric),
        "ao_coulomb": np.zeros_like(three_center),
    }
    for spin in spins.values():
        solved = run(
            fit_rev,
            {
                "metric_whitener": whitener,
                "three_center": raw[spin].reshape(aux, -1),
                "fit_bar": fit_bar[spin].reshape(aux, -1),
            },
        ).outputs
        physical["metric/coulomb"] += solved["metric_bar"]
        raw_bar[spin] += solved["three_center_bar"].reshape(aux, nmo, nmo)
        bars = run(
            project_rev,
            {
                "coefficients": coefficients[spin],
                "ao_coulomb": three_center,
                "bar_raw_coulomb": raw_bar[spin],
            },
        ).outputs
        physical["ao_coulomb"] += bars["bar_ao_coulomb"]
        physical["coefficients_" + spin] = bars["bar_coefficients"]
    return physical
