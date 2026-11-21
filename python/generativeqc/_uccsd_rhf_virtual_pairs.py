"""Prepare fixed spatial virtual Coulomb matrices for repeated RHF CC actions."""

from collections.abc import Callable

import numpy as np
from generativeqc_compiler.cc.uccsd_rhf_virtual_pairs import virtual_pair_coordinates
from generativeqc_compiler.common.resources import checked_bytes
from generativeqc_compiler.mp2.f12_density_fitting import fitted_integral_block


def prepare_rhf_virtual_pair_operators(
    fit: np.ndarray,
    raw: np.ndarray,
    run: Callable,
    *,
    row_tile: int,
    max_bytes: int,
) -> dict[str, np.ndarray]:
    """Prepare the restricted operator once from canonical MP2Fit factors.

    Each factor is a real whole-spin virtual block. The fitted factor must be
    the metric solve of the raw factor. The staged Coulomb matrix uses unique
    chemists' orbital pairs. Separate output matrices fold the two ordered
    virtual sources into symmetric and antisymmetric representatives.
    max_bytes bounds added host storage. The caller also admits the fitting
    graph through its own executor and accounts for the retained inputs.
    """
    if (
        fit.ndim != 3
        or fit.shape != raw.shape
        or fit.dtype != np.float64
        or raw.dtype != np.float64
        or fit.shape[1] != fit.shape[2]
        or fit.shape[1] < 2
        or not fit.shape[0]
    ):
        raise ValueError("RHF virtual operator requires matched FP64 fitting blocks")
    if type(row_tile) is not int or row_tile < 1:
        raise ValueError("RHF virtual Coulomb row tile must be positive")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("RHF virtual Coulomb memory budget must be positive")

    auxiliary, virtual, _ = fit.shape
    n_pair = virtual * (virtual + 1) // 2
    n_strict = n_pair - virtual
    width = min(row_tile, n_pair)
    # Include the fitting staging and symmetry-check operands in the first
    # phase and the reusable operators plus source matrix in the second.
    required = checked_bytes(
        8
        * max(
            2 * auxiliary * n_pair + 2 * n_pair,
            n_pair**2
            + auxiliary * (n_pair + 3 * width)
            + 5 * width * n_pair
            + 8 * n_pair,
            n_pair**2 + max(5 * auxiliary * virtual, 5 * width * n_pair) + 8 * n_pair,
            2 * n_pair**2 + n_strict**2 + virtual**2 + 12 * n_pair,
        ),
        "RHF virtual pair preparation storage",
    )
    if required > max_bytes:
        raise MemoryError("RHF virtual pair operators exceed the host numeric budget")
    for array in (fit, raw):
        for row in range(virtual):
            if not np.allclose(array[:, row, :], array[:, :, row], rtol=0, atol=1e-10):
                raise ValueError("Virtual factors lack the RHF within-pair symmetry")
    pairs = virtual_pair_coordinates(virtual, diagonal=True)
    strict = virtual_pair_coordinates(virtual, diagonal=False)
    first, second = map(np.asarray, zip(*pairs, strict=True))
    right = np.ascontiguousarray(raw[:, first, second])
    coulomb = np.empty((n_pair, n_pair), dtype=np.float64)
    graphs = {
        n: fitted_integral_block(auxiliary, n, n_pair, coulomb=True)
        for n in {width, n_pair % width}
        if n
    }
    for start in range(0, n_pair, width):
        stop = min(start + width, n_pair)
        left = np.ascontiguousarray(fit[:, first[start:stop], second[start:stop]])
        result = run(
            graphs[stop - start], {"fit_left": left, "raw_right": right}
        ).outputs["block"]
        if (
            result.dtype != np.float64
            or result.shape != (stop - start, n_pair)
            or not np.isfinite(result).all()
        ):
            raise ValueError("Invalid prepared spatial Coulomb row block")
        coulomb[start:stop] = result
    del left, right, result
    for start in range(0, n_pair, width):
        stop = min(start + width, n_pair)
        if not np.allclose(
            coulomb[start:stop], coulomb[:, start:stop].T, rtol=0, atol=1e-10
        ):
            raise ValueError(
                "Prepared MP2Fit Coulomb matrix lacks pair-exchange symmetry"
            )

    lookup = np.empty((virtual, virtual), dtype=np.int64)
    lookup[first, second] = lookup[second, first] = np.arange(n_pair)
    strict_first, strict_second = map(np.asarray, zip(*strict, strict=True))
    symmetric = np.empty_like(coulomb)
    antisymmetric = np.empty((n_strict, n_strict), dtype=np.float64)
    for row, (e, f) in enumerate(pairs):
        symmetric[row] = coulomb[lookup[first, e], lookup[second, f]]
        if e != f:
            symmetric[row] += coulomb[lookup[first, f], lookup[second, e]]
    for row, (e, f) in enumerate(strict):
        antisymmetric[row] = (
            coulomb[lookup[strict_first, e], lookup[strict_second, f]]
            - coulomb[lookup[strict_first, f], lookup[strict_second, e]]
        )
    symmetric.setflags(write=False)
    antisymmetric.setflags(write=False)
    return {
        "spatial_coulomb_symmetric": symmetric,
        "spatial_coulomb_antisymmetric": antisymmetric,
    }
