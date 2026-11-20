"""CPU reference quadrature for semilocal ECP projectors in combined OBS/OptRI bases."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import cache
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.integral.ecp_grid import harmonic_roots, radial_map_roots
from generativeqc_compiler.integral.ecp_policy import (
    DERIVATIVE_ABS_TOLERANCE,
    EXTENDED_POLAR_POINTS,
    EXTENDED_RADIAL_POINTS,
    MATRIX_ABS_TOLERANCE,
    MAX_CUDA_POLAR_POINTS,
    MAX_CUDA_RADIAL_POINTS,
)
from generativeqc_compiler.integral.ecp_projector import radial_roots
from generativeqc_compiler.integral.shell_spec import cartesian_components

if TYPE_CHECKING:
    from ._f12_focal_shells import FocalShells
    from .basis import ElementBasis


def projector_terms(nuclei: list[ElementBasis]) -> dict:
    """Decode the highest BSE channel as local and all lower channels as differences."""
    fields = {
        "ecp_type",
        "angular_momentum",
        "r_exponents",
        "gaussian_exponents",
        "coefficients",
    }
    terms = {}
    for center, element in enumerate(nuclei):
        if not element.ecp_data:
            continue
        records = json.loads(element.ecp_data)
        channels = [item["angular_momentum"][0] for item in records]
        if len(set(channels)) != len(channels):
            raise ValueError("duplicate scalar ECP angular channel")
        if max(channels) > 4:
            raise NotImplementedError("focal ECP quadrature has projectors through f")
        terms[center] = []
        for channel, item in zip(channels, records, strict=True):
            if set(item) != fields or len(item["coefficients"]) != 1:
                raise NotImplementedError("unsupported scalar ECP parameter layout")
            for power, alpha, coeff in zip(
                item["r_exponents"],
                item["gaussian_exponents"],
                item["coefficients"][0],
                strict=True,
            ):
                if float(coeff) == 0:
                    continue
                if channel == max(channels):
                    raise NotImplementedError(
                        "focal projector quadrature requires a zero residual local channel"
                    )
                if power > 4:
                    raise NotImplementedError(
                        "focal ECP quadrature supports radial powers through four"
                    )
                terms[center].append((channel, power, float(alpha), float(coeff)))
    return terms


@cache
def _powers(l: int) -> tuple:
    return tuple(
        tuple(component.count(axis) for axis in "xyz")
        for component in cartesian_components(l)
    )


@cache
def _nodes(count: int) -> tuple:
    return np.polynomial.legendre.leggauss(count)


def _ring_moments(
    displacement: np.ndarray, polar: np.ndarray, weights: np.ndarray, maximum_l: int
) -> list:
    # Rotate the integration axis to the center separation. Gaussian exponentials
    # then have no azimuth dependence, even in a bent molecule. A through-i AO
    # gradient times an f projector has Fourier order at most ten.
    norm = np.linalg.norm(displacement)
    axis = displacement / norm if norm else np.array([0.0, 0.0, 1.0])
    first = np.cross(axis, np.eye(3)[np.argmin(abs(axis))])
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    phi = np.arange(24) * (2 * np.pi / 24)
    equator = np.cos(phi)[:, None] * first + np.sin(phi)[:, None] * second
    u = polar[:, None, None] * axis + np.sqrt(1 - polar**2)[:, None, None] * equator
    variables = dict(zip("xyz", u.transpose(2, 0, 1), strict=True))
    graph, roots = harmonic_roots()
    y = np.stack(
        [
            np.broadcast_to(a, u.shape[:-1])
            for a in evaluate_array_graph(graph, roots, variables)
        ],
        axis=-1,
    )
    y *= weights[:, None, None] * (2 * np.pi / len(phi))
    moments = []
    xyz = u.transpose(2, 0, 1)
    for degree in range(maximum_l + 1):
        polynomials = np.stack(
            [xyz[0] ** i * xyz[1] ** j * xyz[2] ** k for i, j, k in _powers(degree)],
            axis=1,
        )
        moments.append(polynomials @ y)
    return moments


def _polynomial_jets(
    l: int, transform: np.ndarray, delta: np.ndarray, derivatives: bool
) -> list:
    """Expand P(delta+r*u), its gradient and x*P for the Gaussian product rule."""
    count = 7 if derivatives else 1
    coeffs = [
        np.zeros((len(_powers(k)), count, 2 * l + 1))
        for k in range(l + 1 + int(derivatives))
    ]

    def accumulate(power: tuple, column: int, polynomial: np.ndarray) -> None:
        for q in product(*(range(p + 1) for p in power)):
            k = sum(q)
            weight = math.prod(
                math.comb(p, t) * d ** (p - t)
                for p, t, d in zip(power, q, delta, strict=True)
            )
            if weight:
                coeffs[k][_powers(k).index(q), column] += weight * polynomial

    for powers, c in zip(_powers(l), transform.T, strict=True):
        accumulate(powers, 0, c)
        if derivatives:
            for axis in range(3):
                new = list(powers)
                if new[axis]:
                    new[axis] -= 1
                    accumulate(tuple(new), 1 + axis, powers[axis] * c)
                new[axis] = powers[axis] + 1
                accumulate(tuple(new), 4 + axis, c)
    return coeffs


def _project_radial_tile(
    shells: FocalShells,
    transforms: dict,
    xyz: np.ndarray,
    center: int,
    r: np.ndarray,
    polar: np.ndarray,
    wz: np.ndarray,
    shell_groups: dict,
    shell_starts: np.ndarray,
    derivatives: bool,
) -> np.ndarray:
    """Project AO values and lab-frame electron gradients on one radial tile."""
    jets = 4 if derivatives else 1
    n, nchan, npolar = sum(shells.ao_counts), 16, len(polar)
    max_l = int(max(shells.angular))
    panel = np.empty((jets, n, len(r), nchan))
    for atom in np.unique(shells.atom):
        delta = xyz[center] - xyz[atom]
        distance = np.linalg.norm(delta)
        # |delta+r*u|^2, avoiding cancellation near a displaced AO center.
        r2 = (r[:, None] + distance * polar) ** 2 + distance**2 * (1 - polar**2)
        rings = _ring_moments(delta, polar, wz, max_l + int(derivatives))
        for (owner, l), indices in shell_groups.items():
            if owner != atom:
                continue
            coefficients = _polynomial_jets(int(l), transforms[l], delta, derivatives)
            angles = [
                (mom.swapaxes(1, 2) @ c.reshape(len(c), -1)).reshape(npolar, -1)
                for mom, c in zip(rings[: len(coefficients)], coefficients, strict=True)
            ]
            for shell in indices:
                low, high = shells.prim_offsets[shell : shell + 2]
                gaussian = np.zeros_like(r2)
                grad = np.zeros_like(r2)
                for a, c in zip(
                    shells.exponents[low:high],
                    shells.radial[low:high],
                    strict=True,
                ):
                    g = c * np.exp(-a * r2)
                    gaussian += g
                    if derivatives:
                        grad -= 2 * a * g
                target = panel[:, shell_starts[shell] : shell_starts[shell + 1]]
                target.fill(0.0)
                for k, angle in enumerate(angles):
                    polynomials = angle.reshape(npolar, nchan, -1, 2 * l + 1)
                    for channel in range(jets):
                        coef = polynomials[:, :, channel, :].reshape(npolar, -1)
                        val = gaussian @ coef
                        if channel:
                            val += grad @ polynomials[:, :, channel + 3, :].reshape(
                                npolar, -1
                            )
                        target[channel] += (
                            ((r**k)[:, None] * val)
                            .reshape(len(r), nchan, -1)
                            .transpose(2, 0, 1)
                        )
    return panel


def _grid(
    shells: FocalShells,
    transforms: dict,
    xyz: np.ndarray,
    terms: dict,
    grid: tuple[int, int],
    derivatives: bool,
    budget: int,
) -> np.ndarray:
    nr, npolar = grid
    n = sum(shells.ao_counts)
    jets = 4 if derivatives else 1
    rows = 1 + 3 * len(xyz) * int(derivatives)
    max_l = int(max(shells.angular))
    nchan = 16
    # Includes both output panels used by the refinement gate, AO products,
    # the largest translated shell polynomial table, and its contraction.
    fixed = 8 * (
        2 * rows * n * n
        + (jets + 2) * n * n
        + 2 * (max_l + 2) * npolar * nchan * 7 * (2 * max_l + 1)
        + 2 * npolar * 24 * (nchan + 36)
        + 4 * npolar * 120 * nchan
        + 20_000
    )
    per_radius = 8 * (
        nchan * n * (jets + 1) + 4 * nchan * 7 * (2 * max_l + 1) + 8 * npolar + nchan
    )
    radial_tile = min(nr, (budget - fixed) // per_radius)
    if radial_tile < 1:
        raise MemoryError("combined scalar ECP projection exceeds the numeric budget")
    output = np.zeros((rows, n, n))
    ao_atom = np.repeat(shells.atom, 2 * shells.angular + 1)
    shell_starts = np.r_[0, np.cumsum(2 * shells.angular + 1)]
    shell_groups = defaultdict(list)
    for i, key in enumerate(zip(shells.atom, shells.angular, strict=True)):
        shell_groups[key].append(i)
    graph, roots = radial_map_roots()
    t, wt = _nodes(nr)
    radii, dr = evaluate_array_graph(graph, roots, {"z": t, "weight": wt})
    polar, wz = _nodes(npolar)
    for center, records in terms.items():
        channels = np.zeros((nr, 4))
        for l, power, alpha, c in records:
            graph, root = radial_roots(power)
            channels[:, l] += evaluate_array_graph(
                graph,
                root,
                {"r": radii, "weight": dr, "alpha": alpha, "coefficient": c},
            )[0]
        radial_weights = np.repeat(channels, [1, 3, 5, 7], axis=1)
        for begin in range(0, nr, radial_tile):
            stop = min(nr, begin + radial_tile)
            r = radii[begin:stop]
            panel = _project_radial_tile(
                shells,
                transforms,
                xyz,
                center,
                r,
                polar,
                wz,
                shell_groups,
                shell_starts,
                derivatives,
            )
            weighted = (panel[0] * radial_weights[begin:stop]).reshape(n, -1)
            output[0] += panel[0].reshape(n, -1) @ weighted.T
            if derivatives:
                for d in range(3):
                    left = panel[d + 1].reshape(n, -1) @ weighted.T
                    for atom in np.unique(ao_atom):
                        if atom == center:
                            continue
                        mask = np.flatnonzero(ao_atom == atom)
                        # The moving projector sees the opposite displacement
                        # to the basis; either or both indices can share this atom.
                        for owner, sign in ((atom, -1.0), (center, 1.0)):
                            part = output[1 + 3 * owner + d]
                            part[mask, :] += sign * left[mask, :]
                            part[:, mask] += sign * left[mask, :].T
            del panel, weighted
    return output


def add_projector_residual(
    hcore: np.ndarray,
    shells: FocalShells,
    transforms: dict,
    xyz: np.ndarray,
    terms: dict,
    derivatives: bool,
    budget: int,
) -> None:
    """Add converged residual matrices and all physical-atom derivatives exactly once."""
    grids = (
        (EXTENDED_RADIAL_POINTS, EXTENDED_POLAR_POINTS),
        (MAX_CUDA_RADIAL_POINTS, MAX_CUDA_POLAR_POINTS),
    )
    coarse = _grid(shells, transforms, xyz, terms, grids[0], derivatives, budget)
    fine = _grid(shells, transforms, xyz, terms, grids[1], derivatives, budget)
    if (
        not np.isfinite(fine).all()
        or not np.allclose(coarse[0], fine[0], rtol=0, atol=MATRIX_ABS_TOLERANCE)
        or any(
            not np.allclose(a, b, rtol=0, atol=DERIVATIVE_ABS_TOLERANCE)
            for a, b in zip(coarse[1:], fine[1:], strict=True)
        )
    ):
        raise ArithmeticError("focal scalar ECP quadrature grid pair did not converge")
    hcore += fine
