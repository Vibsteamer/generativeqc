"""Dense CPU reference of the restricted CABS-singles stationary nuclear sources."""

from collections.abc import Callable
import numpy as np
from scipy.linalg import solve_sylvester, solve_triangular

from benchmarks.f12_ao_response import evaluate


def restricted_singles(
    fock: np.ndarray, occupied: int, *, pullback: bool
) -> tuple[float, np.ndarray | None]:
    """Solve the Sylvester equation for occupied-to-combined-virtual singles.

    In a real orthonormal frame the variational energy is -2 <F_vo,T> with
    F_vv T - T F_oo = F_vo. Its stationary derivative does not require
    differentiating eigenvectors within either block.
    """
    if fock.ndim != 2 or fock.shape[0] != fock.shape[1] or not 0 < occupied < len(fock):
        raise ValueError(
            "A square Fock matrix and nonempty occupied and virtual spaces are required"
        )
    if not np.allclose(fock, fock.T, atol=1e-8, rtol=0):
        raise ValueError("The physical restricted Fock matrix must be symmetric")
    f = 0.5 * (fock + fock.T)
    o = slice(occupied)
    v = slice(occupied, None)
    if np.linalg.eigvalsh(f[v, v])[0] - np.linalg.eigvalsh(f[o, o])[-1] <= 1e-12:
        raise ValueError(
            "The occupied and combined virtual subspaces lack a positive singles gap"
        )
    t = solve_sylvester(f[v, v], -f[o, o], f[v, o])
    res = f[v, v] @ t - t @ f[o, o] - f[v, o]
    if np.max(abs(res)) > 1e-10 * max(1.0, np.max(abs(f[v, o]))):
        raise ValueError("CABS singles linear equation did not converge")
    e = -2.0 * float(np.vdot(f[v, o], t))
    if not pullback:
        return e, None
    bar = np.empty_like(f)
    bar[o, o] = -2 * t.T @ t
    bar[v, v] = 2 * t @ t.T
    bar[v, o] = -2 * t
    bar[o, v] = -2 * t.T
    return e, bar


def restricted_singles_ao_weights(data: dict, run: Callable) -> tuple[float, dict]:
    """Differentiate only the CABS-singles increment to its physical full-rank AO sources."""
    j = data["metric"]
    whitener = solve_triangular(np.linalg.cholesky(j), np.eye(len(j)), lower=True)
    energy, bars, _ = evaluate(
        {**data, "metric_whitener": whitener},
        run,
        pullback=True,
        singles=restricted_singles,
        singles_only=True,
    )
    return energy, {
        "hcore": bars["hcore_ao"],
        "coefficients": bars["coefficients"],
        "ao": {"coulomb": bars["ao_coulomb"]},
        "metric": {"coulomb": bars["metric"]},
    }
