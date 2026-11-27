"""Validate robust F12 density fitting on archived molecular integrals and a shaped response."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc_compiler.mp2.f12_density_fitting import fitted_integral_block
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks._f12_cuda import cuda_runner

FIXTURES = Path(__file__).resolve().parents[1] / "tests/reference_data/f12"


def molecular_case(case: str, run: Callable) -> dict[str, Any]:
    """Recover independently exported Psi4 V_iiii, X_iiii and G_iiab from three-center tiles."""
    with np.load(FIXTURES / f"psi4_{case}_robust_df.npz", allow_pickle=False) as data:
        occ, obs, ri, i = map(int, data["space"])
        fit = data["fit"]
        auxiliary = len(fit)
        occ_slice, orb_slice, cabs = slice(0, occ), slice(0, obs), slice(obs, ri)
        self_pair = slice(i, i + 1)

        def block(operator: str, left: slice, right: slice) -> np.ndarray:
            coulomb = operator == "coulomb"
            raw = data[f"{operator}/raw"]
            linear_tile = raw.shape[1] == ri
            feeds = {
                "fit_left": fit[:, left],
                "raw_right": raw[:, right] if linear_tile else raw,
            }
            if not coulomb:
                feeds.update(
                    fit_right=fit[:, right],
                    raw_left=raw[:, left] if linear_tile else raw,
                    operator_metric=data[f"{operator}/metric"],
                )
            plan = fitted_integral_block(
                auxiliary,
                fit[:, left].shape[1],
                fit[:, right].shape[1],
                coulomb=coulomb,
            )
            return run(
                plan, {name: np.ascontiguousarray(v) for name, v in feeds.items()}
            ).outputs["block"]

        f_orb = block("f12", orb_slice, orb_slice)
        f_mq = block("f12", occ_slice, cabs)
        g_orb = block("coulomb", orb_slice, orb_slice)
        g_mq = block("coulomb", occ_slice, cabs)
        # These outer contractions only reconstruct independent Psi4 observables for the test.
        v = (
            block("f12_over_coulomb", self_pair, self_pair).item()
            - 2 * np.vdot(g_mq, f_mq)
            - np.vdot(g_orb, f_orb)
        )
        x = (
            block("f12_squared", self_pair, self_pair).item()
            - 2 * np.vdot(f_mq, f_mq)
            - np.vdot(f_orb, f_orb)
        )
        v_err = float(v - data["v"])
        x_err = float(x - data["x"])
        g_err = float(np.max(abs(g_orb[occ:, occ:] - data["g"])))
        np.testing.assert_allclose([v_err, x_err, g_err], 0.0, atol=3e-11, rtol=0.0)
        return {
            "case": case,
            "auxiliary_functions": auxiliary,
            "orbital": i,
            "occupied": occ,
            "obs": obs,
            "ri": ri,
            "V": float(v),
            "X": float(x),
            "V_error": v_err,
            "X_error": x_err,
            "maximum_G_error": g_err,
        }


def response_case(run: Callable) -> dict[str, Any]:
    """Check quadratic fitting error and all independent cotangents on a rectangular block."""
    rng = np.random.default_rng(11037)
    aux, rest, nl, nr = 7, 4, 5, 3
    full = rng.normal(size=(aux + rest, aux + rest))
    full = 0.5 * (full + full.T)
    left, right = rng.normal(size=(aux + rest, nl)), rng.normal(size=(aux + rest, nr))
    feeds = {
        "fit_left": left[:aux],
        "fit_right": right[:aux],
        "raw_left": full[:aux] @ left,
        "raw_right": full[:aux] @ right,
        "operator_metric": full[:aux, :aux],
    }
    feeds = {k: np.ascontiguousarray(v) for k, v in feeds.items()}
    weights = rng.normal(size=(nl, nr))
    plan = fitted_integral_block(aux, nl, nr)
    actual = run(plan, feeds).outputs["block"]
    exact = left.T @ full @ right
    residual = left[aux:].T @ full[aux:, aux:] @ right[aux:]
    identity_error = float(np.max(abs(exact - actual - residual)))
    np.testing.assert_allclose(exact - actual, residual, atol=5e-12, rtol=0)
    adjoint = transpose_program(plan, ("block",)).program
    bars = run(adjoint, {**feeds, "bar_block": weights}).outputs
    directions = {k: rng.normal(size=v.shape) for k, v in feeds.items()}
    for k, v in directions.items():
        if k == "operator_metric":
            directions[k] = 0.5 * (v + v.T)
        directions[k] /= max(1.0, np.linalg.norm(directions[k]))
    assert set(bars) == {"bar_" + k for k in feeds}

    def weighted(t: float) -> float:
        perturbed = {k: v + t * directions[k] for k, v in feeds.items()}
        dl, dr = perturbed["fit_left"], perturbed["fit_right"]
        ol, ore = perturbed["raw_left"], perturbed["raw_right"]
        ab = perturbed["operator_metric"]
        # Explicit three-term reference, independently ordered from the native graph.
        return float(np.vdot(weights, ol.T @ dr + dl.T @ ore - (dl.T @ ab) @ dr))

    derivative = sum(float(np.vdot(bars["bar_" + k], v)) for k, v in directions.items())
    errors = {
        str(h): abs((weighted(h) - weighted(-h)) / (2 * h) - derivative)
        for h in (3e-4, 1e-4)
    }
    np.testing.assert_allclose(list(errors.values()), 0.0, atol=2e-8, rtol=0)
    return {
        "case": "rectangular_robust_projection",
        "fitting_residual_identity_error": identity_error,
        "directional_errors": errors,
        "cotangents": sorted(bars),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, metrics = cuda_runner(args.arch, args.cache)
    rows = []
    for case in ("h2", "water", "response"):
        result = response_case(run) if case == "response" else molecular_case(case, run)
        rows.append(result)
        args.output.write_text(
            json.dumps({"rows": rows, "programs": metrics}, indent=2, allow_nan=False)
            + "\n"
        )
        print(result, flush=True)


if __name__ == "__main__":
    main()
