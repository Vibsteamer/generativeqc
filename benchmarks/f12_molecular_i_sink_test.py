"""Build and replay independent sparse i/i and i/i/i force controls."""

import argparse
import json
from itertools import product
from math import pi, prod, sqrt
from pathlib import Path

import numpy as np

from benchmarks.f12_molecular_sink_test import _OPCODES, bind, channel_name, contract, sha256


def component_index(powers: tuple[int, int, int]) -> int:
    _, y, z = powers
    return (y + z) * (y + z + 1) // 2 + z


def sparse_block(ncenter: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Cover maximum axial power as well as nontrivial transverse components.
    components = (
        ((6, 0, 0), (6, 0, 0), (6, 0, 0)),
        ((1, 3, 2), (2, 1, 3), (2, 2, 2)),
        ((0, 0, 6), (0, 2, 4), (2, 4, 0)),
    )
    powers = np.asarray(components, dtype=np.int32)
    if ncenter == 2:
        powers[:, 1] = 0
    weights = np.zeros(28**ncenter)
    slots = (0, 2) if ncenter == 2 else (0, 1, 2)
    indices = tuple(
        np.asarray([component_index(tuple(row[slot])) for row in powers])
        for slot in slots
    )
    sampled = np.array([0.51, -0.34, 0.27])
    weights[np.ravel_multi_index(indices, (28,) * ncenter)] = sampled
    if np.count_nonzero(weights) != len(sampled):
        raise ValueError("sparse controls must address distinct Cartesian components")
    return powers, weights, sampled


def oracle(
    exponents: np.ndarray,
    centers: np.ndarray,
    powers: np.ndarray,
    weights: np.ndarray,
    fit: np.ndarray,
) -> dict[str, float]:
    """Integrate the original Cartesian Gaussian functions, without Boys moments."""
    from scipy.integrate import quad_vec

    # The second slot of a metric is a unit function, with zero exponent and
    # no radial factor. All other supplied Gaussian exponents are positive.
    a, b, q = exponents
    p = a + b
    origin = (a * centers[0] + b * centers[1]) / p
    distance = origin - centers[2]
    decay = a * b / p * np.sum((centers[0] - centers[1]) ** 2)
    nodes, gh = np.polynomial.hermite.hermgauss(12)
    gh = gh[:, None] * gh[None, :] / pi

    def gaussian(zeta: np.ndarray, *, squared_distance: bool = False) -> np.ndarray:
        determinant = p * q + zeta * (p + q)
        m1 = origin - (q * zeta / determinant)[:, None] * distance
        m2 = centers[2] + (p * zeta / determinant)[:, None] * distance
        x = np.sqrt((q + zeta) / determinant)[:, None, None] * nodes[None, :, None]
        y = (
            (zeta / (q + zeta))[:, None, None] * x
            + (1 / np.sqrt(q + zeta))[:, None, None] * nodes[None, None, :]
        )
        u = m1[:, None, :, None, None] + x[:, None, None]
        v = m2[:, None, :, None, None] + y[:, None, None]
        polynomial = 1.0
        for point, center, pw in zip(
            (u, u, v), centers, powers.transpose(1, 0, 2), strict=True
        ):
            polynomial = polynomial * (
                (point - center[None, None, :, None, None])
                ** pw[None, :, :, None, None]
            )
        moments = np.sum(polynomial * gh, axis=(-2, -1))
        if squared_distance:
            weighted_moments = np.sum(polynomial * (u - v) ** 2 * gh, axis=(-2, -1))
            combined = sum(
                weighted_moments[:, :, axis]
                * moments[:, :, (axis + 1) % 3]
                * moments[:, :, (axis + 2) % 3]
                for axis in range(3)
            )
        else:
            combined = np.prod(moments, axis=2)
        prefactor = (
            pi**3
            / determinant**1.5
            * np.exp(-decay - (p * q * zeta / determinant) * np.dot(distance, distance))
        )
        return prefactor * (combined @ weights)

    # 1/r = 2/sqrt(pi) integral_0^infinity exp(-t^2 r^2) dt.
    z, w = fit.T
    coulomb = quad_vec(
        lambda t: gaussian(np.r_[0.0, z] + t * t),
        0,
        np.inf,
        epsabs=3e-12,
        epsrel=3e-12,
    )[0] * (2 / sqrt(pi))
    zpair = (z[:, None] + z[None, :]).ravel()
    wpair = (w[:, None] * w[None, :]).ravel()
    return {
        "coulomb": coulomb[0],
        "f12": gaussian(z) @ w,
        "f12_over_coulomb": coulomb[1:] @ w,
        "f12_squared": gaussian(zpair) @ wpair,
        "double_commutator": gaussian(zpair, squared_distance=True)
        @ (wpair * (4 * z[:, None] * z[None, :]).ravel()),
    }


def make_fixture(
    opt_ri: Path, mp2_fit: Path, output: Path, *, symbol: str, beta: float
) -> None:
    """Use diffuse i exponents from pinned PP-F12 tables and independent Psi4 fits."""
    import psi4
    from pyscf import gto

    def diffuse_i(path: Path) -> float:
        rows = gto.basis.load(str(path), symbol)
        return min(primitive[0] for row in rows if row[0] == 6 for primitive in row[1:])

    opt_exp, fit_exp = diffuse_i(opt_ri), diffuse_i(mp2_fit)
    exps = np.array([opt_exp, opt_exp, fit_exp])
    # Place the three source centers independently to exercise all Cartesian
    # responses. These are integral controls, not an ECP molecular solve.
    centers = np.array([[0.19, -0.41, 0.27], [0.72, 0.35, -0.06], [-0.25, 0.3, 0.85]])
    radial = (2 * exps / pi) ** 0.75 * (4 * exps) ** 3 / sqrt(prod(range(1, 12, 2)))
    psi4.core.be_quiet()
    psi4.set_num_threads(1)
    hydrogen = psi4.geometry("0 2\nH 0 0 0\nsymmetry c1")
    fit_basis = psi4.core.BasisSet.build(hydrogen, "BASIS", "sto-3g", quiet=True)
    fit = np.asarray(psi4.core.MintsHelper(fit_basis).f12_cgtg(beta))
    manifest = {
        "case": f"{symbol} PP-F12 i-shell Coulomb and geminal controls",
        "source": "Gauss–Hermite direct integration of the coupled original Gaussian functions",
        "inputs": {
            "OptRI_sha256": sha256(opt_ri),
            "MP2Fit_sha256": sha256(mp2_fit),
            "psi4_version_for_fit": psi4.__version__,
        },
        "elements": [symbol] * 3,
        "geometry_bohr": centers.tolist(),
        "options": {"f12_beta": beta},
        "channels": {},
    }
    arrays = {
        "angular": np.full(3, 6, dtype=np.int32),
        "atom": np.arange(3, dtype=np.int32),
        "prim_offsets": np.arange(4, dtype=np.int32),
        "exponents": exps,
        "radial": radial,
        "centers": centers,
    }
    for group, slots in (("metric", (0, 2)), ("ao", (0, 1, 2))):
        powers, weights, sampled = sparse_block(len(slots))
        fixed = sampled * np.prod(radial[list(slots)])
        reference_exponents = exps.copy()
        if len(slots) == 2:
            reference_exponents[1] = 0.0

        def values(positions: np.ndarray) -> dict[str, float]:
            return oracle(reference_exponents, positions, powers, fixed, fit)

        direct = values(centers)
        gradients = {operator: np.zeros((3, 3)) for operator in _OPCODES}
        fd_disagreement = {operator: 0.0 for operator in _OPCODES}
        for slot, axis in product(slots, range(3)):
            derivatives = []
            for step in (2e-4, 1e-4):
                displaced = []
                for sign in (-1, 1):
                    moved = centers.copy()
                    moved[slot, axis] += sign * step
                    displaced.append(values(moved))
                derivatives.append(
                    {k: (displaced[1][k] - displaced[0][k]) / (2 * step) for k in direct}
                )
            for k in direct:
                coarse, fine = (d[k] for d in derivatives)
                fd_disagreement[k] = max(fd_disagreement[k], abs(coarse - fine))
                gradients[k][slot, axis] = (4 * fine - coarse) / 3
        arrays[group + "_blocks"] = np.array([slots], dtype=np.int32)
        arrays[group + "_offsets"] = np.array([0, len(weights)], dtype=np.int32)
        for op in direct:
            label = channel_name(group, op)
            arrays["weights_" + label] = weights
            manifest["channels"][label] = {
                "weighted_value_Eh": float(direct[op]),
                "gradient_Eh_per_bohr": gradients[op].tolist(),
                "shell_blocks": 1,
                "cartesian_weights": len(weights),
                "max_fd_step_disagreement_Eh_per_bohr": fd_disagreement[op],
            }
    np.savez_compressed(output, **arrays, manifest=json.dumps(manifest))


def replay(fixture: Path, library: Path, *, host: bool) -> dict:
    """Reuse the original contraction ABI and numerical gates on the new domain."""
    functions = {name: bind(library, name, host) for name in ("coulomb", "geminal")}
    with np.load(fixture, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        rows = dict(
            contract(data, manifest, functions, group, op, 64 << 20, host)
            for group, op in product(("metric", "ao"), _OPCODES)
        )
    return {
        "case": manifest["case"],
        "backend": "host_reference_fp64" if host else "cuda_fp64",
        "fixture_sha256": sha256(fixture),
        "library_sha256": sha256(library),
        "passed": all(row["passed"] for row in rows.values()),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture_parser = sub.add_parser("fixture")
    for file in ("opt-ri", "mp2-fit", "output"):
        fixture_parser.add_argument("--" + file, required=True, type=Path)
    fixture_parser.add_argument("--symbol", choices=("Br", "I"), required=True)
    fixture_parser.add_argument("--beta", type=float, default=1.0)
    replay_parser = sub.add_parser("replay")
    for file in ("fixture", "library", "output"):
        replay_parser.add_argument("--" + file, required=True, type=Path)
    replay_parser.add_argument("--host-reference", action="store_true")
    args = parser.parse_args()
    if args.command == "fixture":
        make_fixture(args.opt_ri, args.mp2_fit, args.output, symbol=args.symbol, beta=args.beta)
    else:
        result = replay(args.fixture, args.library, host=args.host_reference)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        raise SystemExit(0 if result["passed"] else 1)
