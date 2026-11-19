"""Combine independently calculated, same-surface F12 energy and analytic-force increments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .. import _native
from .._cbs_plan import native_cbs_plan
from ..elements import checked_integer

Role = Literal[
    "hf_cabs", "mp2_f12_correlation", "post_mp2", "ccsd_minus_mp2", "triples"
]


@dataclass(frozen=True)
class FocalTerm:
    """One center from a fixed protocol, energy in Eh and force in Eh/bohr.

    surface_id identifies ordered nuclei, coordinates, Q/M, electronic root,
    pseudopotentials and the correlated-electron convention. series_id identifies
    a basis family and auxiliary policy across cardinal numbers. basis_id
    identifies the actual orbital/auxiliary files used for this term.
    """

    surface_id: str
    series_id: str
    basis_id: str
    cardinal: int
    role: Role
    energy: float
    forces: np.ndarray


@dataclass(frozen=True)
class FocalSeries:
    """A finite endpoint or a pair with a fixed exponent and decay law."""

    upper: FocalTerm
    lower: FocalTerm | None = None
    power: float | None = None
    extrapolation: Literal["inverse_power", "exponential"] = "inverse_power"


@dataclass(frozen=True)
class FocalResult:
    """The composed energy and forces and their actual per-role input weights."""

    energy: float
    forces: np.ndarray
    weights: tuple[tuple[Role, str, int, float], ...]


def combine(
    hf_cabs: FocalSeries,
    correlation: FocalSeries,
    post_mp2: FocalTerm | FocalSeries,
    *,
    triples: FocalTerm | FocalSeries | None = None,
) -> FocalResult:
    """Apply fixed coefficients to both E and -dE/dR after the electronic solves.

    Correlation includes ordinary MP2 correlation plus the explicit F12 pair term.
    Supply a combined post_mp2 increment, or supply a ccsd_minus_mp2
    increment together with separate conventional triples. Each increment can
    have its own basis series. CCSD and MP2 must be differenced at matching
    basis and correlation space for each cardinal in their increment.
    Exponential extrapolation is available for the HF plus CABS series.
    All energy and force terms must describe the same ordered geometry.
    """
    high = post_mp2 if isinstance(post_mp2, FocalSeries) else FocalSeries(post_mp2)
    roles: list[tuple[FocalSeries, Role]] = [
        (hf_cabs, "hf_cabs"),
        (correlation, "mp2_f12_correlation"),
        (high, "post_mp2" if triples is None else "ccsd_minus_mp2"),
    ]
    if triples is not None:
        triple_series = (
            triples if isinstance(triples, FocalSeries) else FocalSeries(triples)
        )
        roles.append((triple_series, "triples"))
    weighted = []
    for series, role in roles:
        upper, lower = series.upper, series.lower
        if series.extrapolation not in ("inverse_power", "exponential"):
            raise ValueError("unknown extrapolation law")
        if series.extrapolation == "exponential" and (
            role != "hf_cabs" or lower is None
        ):
            raise ValueError(
                "exponential extrapolation requires an HF plus CABS cardinal pair"
            )
        if lower is None:
            if series.power is not None:
                raise ValueError("a finite endpoint has no extrapolation power")
            weighted.append((role, upper, 1.0))
            continue
        if (
            isinstance(series.power, bool)
            or not isinstance(series.power, (int, float))
            or not math.isfinite(series.power)
            or series.power <= 0
        ):
            raise ValueError("a cardinal pair requires a finite positive power")
        if lower.series_id != upper.series_id or lower.basis_id == upper.basis_id:
            raise ValueError(
                "a cardinal pair requires distinct bases from the same series"
            )
        lo = checked_integer(lower.cardinal, "lower cardinal", low=2, high=100)
        hi = checked_integer(upper.cardinal, "upper cardinal", low=2, high=100)
        if lo >= hi:
            raise ValueError("cardinal numbers must increase")
        if series.extrapolation == "exponential":
            weights = native_cbs_plan(
                _native.load_library(), lo, hi, 1.0, series.power, "exponential"
            ).hf_weights
        else:
            weights = native_cbs_plan(
                _native.load_library(), lo, hi, series.power, 1.0, "highest"
            ).mp2_weights
        weighted.extend(((role, lower, weights[0]), (role, upper, weights[1])))

    shape = np.shape(high.upper.forces)
    if len(shape) != 2 or shape[0] == 0 or shape[1] != 3:
        raise ValueError("Cartesian forces require shape (atoms, 3)")
    energies = []
    force_terms = []
    inputs = []
    for role, term, coefficient in weighted:
        if term.role != role or term.surface_id != high.upper.surface_id:
            raise ValueError(
                "focal terms must have the requested role and identical surface identity"
            )
        if not all(
            isinstance(v, str) and v.strip()
            for v in (term.surface_id, term.basis_id, term.series_id)
        ):
            raise ValueError("each term requires a surface, basis and series identity")
        checked_integer(term.cardinal, "cardinal", low=2, high=100)
        value = float(term.energy)
        arr = np.asarray(term.forces)
        if (
            arr.shape != shape
            or not np.issubdtype(arr.dtype, np.floating)
            or not np.isfinite(arr).all()
            or not math.isfinite(value)
        ):
            raise ValueError(
                "energy and Cartesian forces must be finite, with matching shape"
            )
        energies.append(value * coefficient)
        force_terms.append(np.asarray(arr, dtype=np.float64) * coefficient)
        inputs.append((term.role, term.basis_id, term.cardinal, coefficient))
    f = np.sum(force_terms, axis=0)
    e = math.fsum(energies)
    if not math.isfinite(e) or not np.isfinite(f).all():
        raise ValueError("combined energy or force is not finite")
    return FocalResult(e, f, tuple(inputs))
