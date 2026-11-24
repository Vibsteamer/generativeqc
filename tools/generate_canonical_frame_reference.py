"""Independent generalized-eigenvector finite differences for canonical response."""

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


def reference_case(occupied: int, virtuals: int, degenerate: bool) -> dict:
    """Tiny externally prescribed Fock problem; no SCF or molecular-force claim."""
    n = occupied + virtuals
    rng = np.random.default_rng(9837 + n)
    a = rng.normal(size=(n, n)) / 4
    overlap = a.T @ a + np.eye(n)
    q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    c = np.linalg.solve(np.linalg.cholesky(overlap).T, q)
    invc = np.linalg.inv(c)
    eps = np.r_[np.linspace(-2, -1, occupied), np.linspace(0.3, 2, virtuals)]
    weights = rng.normal(size=n) / 4
    if degenerate:
        eps[-2] = eps[-1]
        weights[-2] = weights[-1]
    h = rng.normal(size=(n, n)) / 5
    h = (h + h.T) / 2
    fock = invc.T @ np.diag(eps) @ invc
    dm = rng.normal(size=(n, n)) / 50
    dm = (dm + dm.T) / 2
    dm[:occupied, occupied:] = 0
    dm[occupied:, :occupied] = 0
    direction = invc.T @ dm @ invc
    bar = 2 * h @ (c * weights)

    def value(matrix: np.ndarray) -> float:
        _, moved = eigh(matrix, overlap)
        return float(np.sum(weights * np.diag(moved.T @ h @ moved)))

    steps = np.array([2e-4, 1e-4])
    values = np.array(
        [[value(fock + sign * step * direction) for sign in (1, -1)] for step in steps]
    )
    return {
        "coefficients": c,
        "frame_bar": bar,
        "energies": eps,
        "occupied": np.array(occupied),
        "rotation_gradient": c.T @ bar,
        "fock_direction_mo": dm,
        "steps": steps,
        "values": values,
        "slopes": (values[:, 0] - values[:, 1]) / (2 * steps),
        "degenerate": np.array(degenerate),
    }


def all_cases() -> dict[str, dict]:
    return {
        f"o{o}-v{v}-deg{int(degenerate)}": reference_case(o, v, degenerate)
        for o, v in ((1, 3), (2, 4), (3, 4))
        for degenerate in (False, True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the original canonical reference identity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{
            name + "/" + k: v
            for name, row in all_cases().items()
            for k, v in row.items()
        },
    )


if __name__ == "__main__":
    main()
