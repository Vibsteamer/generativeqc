"""Connect the molecular F12 reverse graph to the DF-RHF reference that owns its orbitals."""

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from generativeqc_compiler.method.canonical_frame_response import (
    orbital_rotation_cotangent,
)

from ._canonical_frame import canonical_frame_seed


class F12ReferenceResponse(NamedTuple):
    increment_energy: float
    reference_energy: float
    reference_forces: np.ndarray
    explicit_weights: dict
    diagnostics: dict


def orbital_source_rows(
    native_to_source: np.ndarray, obs_first: np.ndarray, nobs: int
) -> np.ndarray:
    """Return the combined-source row of each public orbital AO, in native order."""
    public, ordered = np.asarray(native_to_source), np.asarray(obs_first)
    if public.ndim != 1 or not 0 < nobs <= public.size:
        raise ValueError("invalid combined and orbital dimensions")
    all_rows = np.arange(len(public))
    for perm in (public, ordered):
        if (
            perm.dtype.kind not in "iu"
            or perm.shape != all_rows.shape
            or not np.array_equal(np.sort(perm), all_rows)
        ):
            raise ValueError("both orbital layouts must be bijective integer AO maps")
    return public[np.isin(public, ordered[:nobs])]


def with_reference_orbitals(
    frame: np.ndarray, rows: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Replace the OBS columns while retaining every complementary column."""
    rows = np.asarray(rows)
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise ValueError("invalid orbital AO row indices")
    n = rows.size
    if (
        frame.ndim != 2
        or not 0 < n <= frame.shape[1] <= frame.shape[0]
        or np.any(rows < 0)
        or np.any(rows >= frame.shape[0])
        or np.unique(rows).size != n
        or coefficients.shape != (n, n)
        or frame.dtype != np.float64
        or coefficients.dtype != np.float64
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError(
            "invalid combined frame, orbital AO rows or reference coefficients"
        )
    result = frame.copy()
    result[:, :n] = 0
    result[rows, :n] = coefficients
    return result


def molecular_reference_response(
    session: Any,
    data: dict,
    native_to_source: np.ndarray,
    evaluate_sources: Callable,
    run: Callable,
    *,
    canonical: Callable = canonical_frame_seed,
    maximum_bytes: int = 43 << 30,
) -> F12ReferenceResponse:
    """Compute the F12 and singles scalar and propagate OBS motion through this SCF.

    The reverse graph recomputes the extended Fock from AO inputs. The SCF
    Fock receives only the same-space canonical multipliers. The session's
    forces include HF, nuclear repulsion and the response to the OBS source.
    The returned explicit AO adjoints still require nuclear contraction.
    """
    occ, nobs, extended = map(int, data["space"])
    frame = data["coefficients"]
    rows = orbital_source_rows(native_to_source, data["ao_permutation"], nobs)
    if (
        (session.n, session.occupied) != (nobs, occ)
        or not 0 < occ < nobs <= extended
        or frame.shape != (len(native_to_source), extended)
    ):
        raise ValueError(
            "molecular sources and reference have different orbital spaces"
        )
    state = session.snapshot()
    ix = np.ix_(rows, rows)
    for name, source in (("overlap", "overlap"), ("hcore", "hcore_ao")):
        matrix = data[source]
        if matrix.shape != (frame.shape[0], frame.shape[0]):
            raise ValueError(f"invalid combined {source} dimensions")
        reference = np.asarray(state[name])
        if (
            reference.shape != (nobs, nobs)
            or not np.allclose(reference, matrix[ix], rtol=0, atol=2e-8)
            or not np.isfinite(reference).all()
        ):
            raise ValueError(f"reference {name} disagrees with the molecular AO source")
    c = state["coefficients"]
    inputs = {**data, "coefficients": with_reference_orbitals(frame, rows, c)}
    energy, weights, checks = evaluate_sources(inputs, pullback=True)
    if not np.isfinite(energy) or not np.isfinite(state["energy"]):
        raise ValueError("nonfinite molecular energy")
    weights = dict(weights)
    combined = weights.pop("coefficients")
    if combined.shape != (frame.shape[0], nobs):
        raise ValueError("the reverse graph must consume the complementary frame first")
    bar_c = np.ascontiguousarray(combined[rows])
    if not np.isfinite(bar_c).all():
        raise ValueError("nonfinite OBS orbital adjoint")
    rotation = run(
        orbital_rotation_cotangent(nobs, nobs), {"coefficients": c, "frame_bar": bar_c}
    ).outputs["rotation_gradient"]
    fock_seed, allocated = canonical(
        rotation, state["energies"], occ, maximum_bytes=maximum_bytes
    )
    forces, stats = session.response(fock_seed, bar_c, maximum_bytes=maximum_bytes)
    diagnostics = {**stats, "canonical_bytes": allocated}
    if "cabs_response" in checks:
        diagnostics["cabs_response"] = checks["cabs_response"]
    return F12ReferenceResponse(
        float(energy), float(state["energy"]), forces, weights, diagnostics
    )
