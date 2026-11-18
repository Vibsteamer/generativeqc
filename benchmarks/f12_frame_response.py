"""Qualify the CABS-frame cotangent and its native projector/overlap composition."""

import argparse
import ctypes
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc_compiler.common.compiler_cache import resolve_compiler_cache
from generativeqc_compiler.mp2.f12_frame_response import frame_kernel_cotangent

from benchmarks._f12_cuda import cuda_runner


def bridge(
    data: dict, run: Callable, *, metric_gram: Callable | None = None
) -> tuple[dict, dict]:
    frame, bar = data["frame"], data["frame_bar"]
    result = run(frame_kernel_cotangent(*frame.shape), data).outputs
    gram = result["dual_gram"]
    if metric_gram is not None and frame.shape[1] > 1:
        # Test the physical invariant at the precision used to normalize this frame.
        gram = metric_gram(data["overlap"], frame)
    moment = frame.T @ bar
    residuals = {
        "metric_dual": float(np.max(np.abs(gram - np.eye(frame.shape[1])))),
        "gauge_relative": float(
            np.linalg.norm(result["gauge_skew"]) / max(1.0, np.linalg.norm(moment))
        ),
        "reconstruction_relative": float(
            np.linalg.norm(result["reconstruction_residual"])
            / max(1.0, np.linalg.norm(bar))
        ),
        "symmetric_kernel_weight": float(
            np.max(np.abs(result["kernel_bar"] - result["kernel_bar"].T))
        ),
    }
    if not all(np.isfinite(v) and v < 2e-9 for v in residuals.values()):
        raise ValueError(f"Invalid invariant frame cotangent: {residuals}")
    return result, residuals


def molecular_case(name: str, data: dict, run: Callable) -> dict:
    start = time.monotonic()
    result, residuals = bridge(data, run)
    rng = np.random.default_rng(4591)
    frame, bar = data["frame"], data["frame_bar"]
    rotation, _ = np.linalg.qr(rng.normal(size=(frame.shape[1], frame.shape[1])))
    moved, _ = bridge(
        {**data, "frame": frame @ rotation, "frame_bar": bar @ rotation}, run
    )
    rotation_error = float(
        np.linalg.norm(moved["kernel_bar"] - result["kernel_bar"])
        / max(1.0, np.linalg.norm(result["kernel_bar"]))
    )
    if rotation_error > 2e-9:
        raise ValueError("CABS kernel weight depends on the chosen column gauge")
    direction = rng.normal(size=frame.shape)
    direction /= np.linalg.norm(direction)
    slope = float(np.vdot(bar, direction))
    errors = {}
    for h in (2e-4, 1e-4):
        plus, minus = frame + h * direction, frame - h * direction
        fd = float(
            np.vdot(result["kernel_bar"], plus @ plus.T - minus @ minus.T) / (2 * h)
        )
        errors[str(h)] = abs(fd - slope)
    if max(errors.values()) > 2e-8:
        raise ValueError(f"{name} frame-to-kernel differential failed: {errors}")
    return {
        "case": name,
        "scope": "molecular F12 coefficient cotangent transfer; supplied central frame and weight",
        "invariants": residuals,
        "rotation_relative_error": rotation_error,
        "two_step_errors": errors,
        "wall_including_compilation_s": time.monotonic() - start,
    }


def reference_frame(overlap: np.ndarray, obs: int, cutoff: float) -> np.ndarray:
    """Independent Cholesky/QR/SVD complement; native uses Schur/eigenspace algebra."""
    features = np.linalg.cholesky(overlap).T
    qo, _ = np.linalg.qr(features[:, :obs])
    aux = features[:, obs:]
    left, singular, _ = np.linalg.svd(aux - qo @ (qo.T @ aux), full_matrices=False)
    return np.linalg.solve(features, left[:, singular**2 > cutoff * singular[0] ** 2])


def synthetic_case(partial_rank: bool) -> tuple[np.ndarray, int, float, np.ndarray]:
    rng = np.random.default_rng(3209)
    a = np.array([[1.2, 0.12], [0.12, 1.7]])
    cross = rng.normal(size=(2, 4)) / 3
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    values = [0.015 if partial_rank else 0.2, 0.55, 0.55, 1.2]
    t = (q * values) @ q.T
    s = np.block([[a, cross], [cross.T, t + cross.T @ np.linalg.solve(a, cross)]])
    h = rng.normal(size=s.shape)
    h = (h + h.T) / 2
    return s, 2, 0.05 if partial_rank else 1e-9, h


def composition_case(partial_rank: bool, run: Callable, projector: Callable) -> dict:
    start = time.monotonic()
    s, obs, cutoff, hamiltonian = synthetic_case(partial_rank)
    primal = projector(s, obs, relative_cutoff=cutoff)
    frame = primal.frame
    independent = reference_frame(s, obs, cutoff)
    rank = frame.shape[1]
    k = frame @ frame.T
    np.testing.assert_allclose(k, independent @ independent.T, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(frame.T @ s @ frame, np.eye(rank), atol=2e-12)
    np.testing.assert_allclose((s @ frame)[:obs], 0.0, atol=2e-12)

    def observable(kernel: np.ndarray) -> float:
        return float(np.vdot(hamiltonian, kernel) + 0.2 * np.vdot(kernel, kernel))

    bar = 2 * hamiltonian @ frame + 0.8 * k @ frame
    transferred, residuals = bridge(
        {"overlap": s, "frame": frame, "frame_bar": bar}, run
    )
    arguments = {"relative_cutoff": cutoff, "expected_rank": rank}
    native = projector(s, obs, cotangent=transferred["kernel_bar"], **arguments)
    np.testing.assert_allclose(native.kernel, k, atol=2e-12, rtol=2e-12)
    rng = np.random.default_rng(7319)
    d = rng.normal(size=s.shape)
    d = (d + d.T) / 2
    d /= np.linalg.norm(d)
    slope = float(np.vdot(native.overlap_bar, d))
    errors = []
    for step in (1e-4, 5e-5):
        reference_values, native_values = [], []
        for sign in (1, -1):
            moved_s = s + sign * step * d
            c = reference_frame(moved_s, obs, cutoff)
            if c.shape[1] != rank:
                raise ValueError("Reference CABS rank changed")
            reference_values.append(observable(c @ c.T))
            moved = projector(moved_s, obs, **arguments)
            native_values.append(observable(moved.frame @ moved.frame.T))
        fd = (native_values[0] - native_values[1]) / (2 * step)
        reference_fd = (reference_values[0] - reference_values[1]) / (2 * step)
        row = {
            "step": step,
            "native_fd_error": abs(slope - fd),
            "reference_fd_error": abs(slope - reference_fd),
            "value_error": max(
                abs(a - b) for a, b in zip(native_values, reference_values, strict=True)
            ),
        }
        if (
            row["native_fd_error"] > 4e-7
            or row["reference_fd_error"] > 4e-7
            or row["value_error"] > 2e-10
        ):
            raise ValueError(f"Composed overlap response failed: {row}")
        errors.append(row)
    return {
        "case": "partial_rank" if partial_rank else "full_rank",
        "scope": "GPU forward frame, frame-to-kernel cotangent and CABS overlap response",
        "rank": rank,
        "retained_degeneracy": True,
        "invariants": residuals,
        "slope": slope,
        "steps": errors,
        "wall_including_compilation_s": time.monotonic() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--component-library", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from generativeqc._f12_cabs import CABSProjector, cabs_projector

    library = ctypes.CDLL(str(args.component_library))
    launcher = resolve_compiler_cache()
    before = subprocess.check_output([str(launcher.path), "--show-stats"], text=True)
    run, programs = cuda_runner(args.arch, args.cache)
    rows = []
    with np.load(args.fixture, allow_pickle=False) as f:
        for name in ("h2", "water"):
            data = {k: f[name + "/" + k] for k in ("overlap", "frame", "frame_bar")}
            rows.append(molecular_case(name, data, run))

    def projector(s: np.ndarray, obs: int, **kwargs: Any) -> CABSProjector:
        return cabs_projector(s, obs, library=library, **kwargs)

    for partial_rank in (False, True):
        rows.append(composition_case(partial_rank, run, projector))
    after = subprocess.check_output([str(launcher.path), "--show-stats"], text=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "CABS coefficient cotangent to projector and overlap, not a molecular nuclear force",
                "rows": rows,
                "programs": programs,
                "compiler_cache": {
                    "path": str(launcher.path),
                    "version": launcher.version,
                    "before": before,
                    "after": after,
                },
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
