"""Qualify occupied-only fitting against full fitting and its metric response."""

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from generativeqc_compiler.common.compiler_cache import resolve_compiler_cache
from generativeqc_compiler.mp2.f12_density_fitting import coulomb_fit_response
from generativeqc_compiler.mp2.f12_molecular_pair import extended_df_fock
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks._f12_cuda import cuda_runner


def evaluate(data: dict, run: Callable) -> dict:
    """Reference factors, values and displaced slopes are external fixtures."""
    raw, hcore, whitener = (data[k] for k in ("raw", "hcore", "whitener"))
    naux, norb, _ = raw.shape
    occupied = int(data["occupied"].item())
    columns = occupied * norb
    fgraph = extended_df_fock(occupied, norb, naux)
    fit_graph = coulomb_fit_response(naux, columns)
    fits = {
        "metric_whitener": whitener,
        "three_center": np.ascontiguousarray(raw[:, :occupied].reshape(naux, -1)),
        "fit_bar": np.zeros((naux, columns)),
    }
    timings = []
    for _ in range(3):
        start = time.monotonic()
        solved = run(fit_graph, fits).outputs
        feeds = {
            "fit": solved["fit"].reshape(naux, occupied, norb),
            "raw_coulomb_all": raw,
            "hcore": hcore,
        }
        answer = run(fgraph, feeds).outputs
        timings.append(time.monotonic() - start)
    errors = {}
    for key in ("fock", "exchange"):
        errors[key] = float(np.max(np.abs(answer[key] - data["reference_" + key])))
        if not np.isfinite(errors[key]) or errors[key] > 5e-10:
            raise ValueError(
                f"Occupied-fit {key} differs from full-fit reference: {errors[key]}"
            )
    fit_error = float(np.max(np.abs(feeds["fit"] - data["reference_fit"])))
    np.testing.assert_allclose(
        feeds["fit"], data["reference_fit"], atol=5e-10, rtol=1e-9
    )
    seeds = {"bar_" + key: data["seed_" + key] for key in answer}
    back = run(
        transpose_program(fgraph, tuple(answer)).program, {**feeds, **seeds}
    ).outputs
    implicit = run(
        fit_graph,
        {**fits, "fit_bar": np.ascontiguousarray(back["bar_fit"].reshape(naux, -1))},
    ).outputs
    raw_bar = back["bar_raw_coulomb_all"].copy()
    raw_bar[:, :occupied] += implicit["three_center_bar"].reshape(naux, occupied, norb)
    slope = float(
        np.vdot(raw_bar, data["draw"])
        + np.vdot(back["bar_hcore"], data["dhcore"])
        + np.vdot(implicit["metric_bar"], data["dmetric"])
    )
    fd_errors = np.abs(slope - data["reference_slopes"])
    if not np.all(np.isfinite(fd_errors)) or np.max(fd_errors) > 2e-7:
        raise ValueError(f"Occupied-fit metric/source response failed: {fd_errors}")
    # Measure the same fitted-density primitive at both pair domains. This is
    # fitting-only timing, not the speedup of a complete molecular F12 endpoint.
    full_graph = coulomb_fit_response(naux, norb * norb)
    full_input = {
        "metric_whitener": whitener,
        "three_center": raw.reshape(naux, -1),
        "fit_bar": np.zeros((naux, norb * norb)),
    }
    full_times, occupied_times = [], []
    for _ in range(3):
        start = time.monotonic()
        full = run(full_graph, full_input).outputs
        full_times.append(time.monotonic() - start)
        start = time.monotonic()
        small = run(fit_graph, fits).outputs
        occupied_times.append(time.monotonic() - start)
    difference = float(
        np.max(
            np.abs(
                full["fit"].reshape(naux, norb, norb)[:, :occupied]
                - small["fit"].reshape(naux, occupied, norb)
            )
        )
    )
    if difference > 5e-10:
        raise ValueError(f"Fitting pair-domain parity failed: {difference}")
    return {
        "occupied": occupied,
        "orbitals": norb,
        "auxiliaries": naux,
        "reference_value_errors": errors,
        "reference_fit_max_error": fit_error,
        "slope": slope,
        "steps": data["steps"].tolist(),
        "two_step_errors": fd_errors.tolist(),
        "fit_domain_parity_error": difference,
        "fitted_columns_old": norb * norb,
        "fitted_columns_new": columns,
        "fit_output_bytes_old": naux * norb * norb * 8,
        "fit_output_bytes_new": naux * columns * 8,
        "fixed_source_fock_seconds": timings,
        "fitting_only_full_seconds": full_times,
        "fitting_only_occupied_seconds": occupied_times,
        "timing_scope": "supplied source arrays; includes execution allocation/transfers; first execution includes JIT; no SCF or integral generation",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--arch", required=True)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    launcher = resolve_compiler_cache()
    before = subprocess.check_output([str(launcher.path), "--show-stats"], text=True)
    run, programs = cuda_runner(args.arch, args.cache, max_bytes=2 << 30)
    rows = []
    with np.load(args.fixture, allow_pickle=False) as fixture:
        for name in ("h2", "water"):
            prefix = name + "/"
            data = {
                k.removeprefix(prefix): fixture[k]
                for k in fixture.files
                if k.startswith(prefix)
            }
            rows.append({"case": name, **evaluate(data, run)})
    after = subprocess.check_output([str(launcher.path), "--show-stats"], text=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "occupied-only Coulomb fit, extended Fock and implicit metric/source response; not complete molecular E/F",
                "rows": rows,
                "programs": programs,
                "compiler_cache": {
                    "path": str(launcher.path),
                    "version": launcher.version,
                    "before": before,
                    "after": after,
                    "statistics_scope": "shared launcher diagnostics; private JIT workers do not imply shared-server hits",
                },
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
