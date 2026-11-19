"""Carry molecular 3C(FIX) adjoints back to AO integrals and orbital coefficients."""

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
from generativeqc_compiler.common.compiler_cache import resolve_compiler_cache
from generativeqc_compiler.mp2.f12_density_fitting import (
    coulomb_fit_response,
    molecular_orbital_sources,
)
from generativeqc_compiler.tensor.ad_program import transpose_program
from generativeqc_compiler.tensor.program import Program

from benchmarks._f12_cuda import cuda_runner
from benchmarks.f12_molecular_pairs import factor_inputs, fock_plan, pair_plan

source_plan = lru_cache(maxsize=8)(molecular_orbital_sources)
fit_plan = lru_cache(maxsize=8)(coulomb_fit_response)
OPERATORS = ("f12", "f12_squared", "f12_over_coulomb", "double_commutator")


def check_sources(data: dict, name: str, run: Callable) -> dict:
    """Qualify the AO/frame map independently of an ill-conditioned metric solve."""
    started = time.monotonic()
    occupied, _, orbitals = map(int, data["space"])
    coefficients = data["coefficients"]
    graph = source_plan(len(coefficients), orbitals, occupied, len(data["metric"]))
    inputs = {
        key: value
        for key, value in data.items()
        if key.startswith("ao_") or key in ("coefficients", "hcore_ao")
    }
    actual = run(graph, inputs).outputs
    reference = {"hcore": coefficients.T @ data["hcore_ao"] @ coefficients}
    for op in ("coulomb", *OPERATORS):
        left = coefficients if op == "coulomb" else coefficients[:, :occupied]
        reference["raw_" + op] = np.stack(
            [left.T @ matrix @ coefficients for matrix in data["ao_" + op]]
        )
    errors = {}
    for key, value in actual.items():
        errors[key] = float(np.max(np.abs(value - reference[key])))
        np.testing.assert_allclose(value, reference[key], atol=5e-10, rtol=2e-10)
        external = data[key + "_reference"]
        view = value[:, :occupied] if key == "raw_coulomb" else value
        errors[key + "_psi4"] = float(np.max(np.abs(view - external)))
        np.testing.assert_allclose(view, external, atol=5e-10, rtol=2e-10)
    rng = np.random.default_rng(1729)
    seeds = {}
    directions = {}
    for key, value in actual.items():
        seed = rng.normal(size=value.shape)
        seeds["bar_" + key] = seed / np.linalg.norm(seed)
    for key, value in inputs.items():
        direction = rng.normal(size=value.shape)
        directions[key] = direction / np.linalg.norm(direction)
    backward = transpose_program(graph, tuple(graph.outputs)).program
    bars = run(backward, {**inputs, **seeds}).outputs
    if set(bars) != {"bar_" + key for key in inputs}:
        raise ValueError("Incomplete AO/frame adjoints")
    slope = sum(float(np.vdot(bars["bar_" + k], d)) for k, d in directions.items())
    fd_errors = {}
    for h in (2e-4, 1e-4):
        values = []
        for step in (h, -h):
            moved = {
                key: value + step * directions[key] for key, value in inputs.items()
            }
            outputs = run(graph, moved).outputs
            values.append(
                sum(float(np.vdot(seeds["bar_" + k], v)) for k, v in outputs.items())
            )
        fd_errors[str(h)] = abs(slope - (values[0] - values[1]) / (2 * h))
    if max(fd_errors.values()) > 2e-8:
        raise ValueError(f"{name} AO/frame adjoints failed: {fd_errors}")
    return {
        "case": name,
        "scope": "one-electron and all five three-center AO maps and their shared orbital-frame adjoints",
        "max_intermediate_errors": errors,
        "two_step_errors": fd_errors,
        "differentiated_sources": sorted(inputs),
        "wall_including_compilation_s": time.monotonic() - started,
    }


def evaluate(
    data: dict,
    run: Callable,
    *,
    pullback: bool,
    singles: Callable | None = None,
    singles_only: bool = False,
    record_energy: Callable[[float, dict[str, float]], None] | None = None,
    physical_fock: Callable | None = None,
) -> tuple[float, dict, dict]:
    """Evaluate the F12 increment, optionally composing a native singles action.

    The explicit callback receives the shared extended Fock/occupied count and
    pullback flag. Its Fock cotangent is summed with the F12 pair sources before
    the shared Fock/fit/AO reverse pass. No additional HF or conventional MP2
    energy is included here, and no implicit reference-program fallback exists.
    """
    occ, obs, ri = map(int, data["space"])
    if singles_only and singles is None:
        raise ValueError("CABS-only response requires a singles functional")
    operators = () if singles_only else OPERATORS
    aux = len(data["metric"])
    plan = source_plan(len(data["coefficients"]), ri, occ, aux)
    if singles_only:
        plan = Program(
            {k: plan.outputs[k] for k in ("hcore", "raw_coulomb")},
            provenance=plan.provenance,
        )
    source = {
        k: data[k]
        for k in (
            "coefficients",
            "hcore_ao",
            *("ao_" + op for op in ("coulomb", *operators)),
        )
    }
    mapped = run(plan, source).outputs
    raw = np.ascontiguousarray(mapped["raw_coulomb"][:, :occ].reshape(aux, -1))
    fitter = fit_plan(*raw.shape)
    fit_inputs = {
        "metric_whitener": data["metric_whitener"],
        "three_center": raw,
        "fit_bar": np.zeros_like(raw),
    }
    fitted = run(fitter, fit_inputs).outputs["fit"].reshape(aux, occ, ri)
    mol = {
        **mapped,
        "space": data["space"],
        "fit": fitted,
        "raw_coulomb_all": mapped["raw_coulomb"],
        "raw_occupied": np.ascontiguousarray(mapped["raw_coulomb"][:, :occ]),
        **{"metric_" + op: data["metric_" + op] for op in operators},
    }
    mol["raw_coulomb"] = mol["raw_occupied"]
    fock = fock_plan(occ, ri, aux)
    fock_inputs = {k: mol[k] for k in ("hcore", "fit", "raw_coulomb_all")}
    fock_out = run(fock, fock_inputs).outputs
    if physical_fock is not None:
        # Reassociate C.T (h + 2 J) C - K in the AO basis before both frame
        # transforms. The TensorIR pullback below differentiates the same
        # contraction on physical Hermitian AO displacements.
        fock_out["fock"] = physical_fock(
            data["hcore_ao"],
            data["ao_coulomb"],
            data["coefficients"],
            fitted,
            fock_out["exchange"],
        )
    fock_bars = {"bar_" + k: np.zeros_like(v) for k, v in fock_out.items()}
    pair_bars = {
        k: np.zeros_like(mol[k])
        for k in (
            "fit",
            "raw_coulomb",
            *("raw_" + o for o in operators),
            *("metric_" + o for o in operators),
        )
    }
    total = 0.0
    pair_sum = 0.0
    singles_energy = 0.0
    if singles is not None:
        singles_energy, singles_bar = singles(fock_out["fock"], occ, pullback=pullback)
        total += float(singles_energy)
        if pullback:
            if singles_bar is None or singles_bar.shape != fock_out["fock"].shape:
                raise ValueError("CABS singles must supply the complete Fock cotangent")
            fock_bars["bar_fock"] += singles_bar
    pairs = () if singles_only else zip(*np.triu_indices(occ), strict=True)
    for first, second in pairs:
        i, j = int(first), int(second)
        pair = pair_plan(occ, obs - occ, ri - obs, diagonal=i == j, auxiliaries=aux)
        feeds = factor_inputs(mol, i, j, fock_out)
        pair_energy = float(run(pair, feeds).outputs["pair_energy"])
        total += pair_energy
        pair_sum += pair_energy
        if not pullback:
            continue
        bars = run(
            transpose_program(pair, ("pair_energy",)).program,
            {**feeds, "bar_pair_energy": np.array(1.0)},
        ).outputs
        for key, value in bars.items():
            name = key.removeprefix("bar_")
            if key in fock_bars:
                fock_bars[key] += value
            elif name.startswith("metric_"):
                pair_bars[name] += value
            else:
                column = i if name.endswith("_i") else j
                pair_bars[name[:-2]][:, column] += value
    components = {
        "f12_pair_energy_Eh": pair_sum,
        "cabs_singles_energy_Eh": float(singles_energy),
    }
    if record_energy is not None:
        record_energy(total, components)
    intermediates = {
        **mapped,
        "fit": fitted,
        **fock_out,
        "energy_components": components,
    }
    if not pullback:
        return total, {}, intermediates
    fock_back = run(
        transpose_program(fock, ("fock", "exchange")).program,
        {**fock_inputs, **fock_bars},
    ).outputs
    fit_bar = fock_back["bar_fit"] + pair_bars["fit"]
    fit_back = run(
        fitter,
        {**fit_inputs, "fit_bar": np.ascontiguousarray(fit_bar.reshape(aux, -1))},
    ).outputs
    raw_bar = fock_back["bar_raw_coulomb_all"].copy()
    raw_bar[:, :occ] += (
        fit_back["three_center_bar"].reshape(aux, occ, ri) + pair_bars["raw_coulomb"]
    )
    seeds = {
        "bar_hcore": fock_back["bar_hcore"],
        "bar_raw_coulomb": raw_bar,
        **{"bar_raw_" + op: pair_bars["raw_" + op] for op in operators},
    }
    source_back = run(
        transpose_program(plan, tuple(plan.outputs)).program, {**source, **seeds}
    ).outputs
    result = {key.removeprefix("bar_"): value for key, value in source_back.items()}
    result["metric"] = fit_back["metric_bar"]
    result.update({"metric_" + op: pair_bars["metric_" + op] for op in operators})
    return total, result, intermediates


def check(data: dict, name: str, run: Callable) -> dict:
    started = time.monotonic()
    energy, bars, intermediate = evaluate(data, run, pullback=True)
    error = energy - float(data["energy"])
    if abs(error) > 5e-11:
        raise ValueError(
            f"{name} AO F12 correction differs from independent Psi4: {error}"
        )
    occupied = int(data["space"][0])
    maxima = {}
    for key in (
        "hcore",
        "fit",
        "fock",
        "exchange",
        "raw_coulomb",
        *("raw_" + op for op in OPERATORS),
    ):
        target = data[key] if key in ("fock", "exchange") else data[key + "_reference"]
        value = (
            intermediate[key][:, :occupied]
            if key == "raw_coulomb"
            else intermediate[key]
        )
        maxima[key] = float(np.max(np.abs(value - target)))
        np.testing.assert_allclose(
            value, target, atol=5e-10, rtol=2e-10, err_msg=name + " " + key
        )
    rng = np.random.default_rng(461087)
    directions = {}
    for key, gradient in sorted(bars.items()):
        if key == "metric":
            continue
        delta = rng.normal(size=gradient.shape)
        if key.startswith("ao_"):
            delta = (delta + delta.swapaxes(1, 2)) / 2
        elif key != "coefficients":
            delta = (delta + delta.T) / 2
        directions[key] = delta / max(1.0, np.linalg.norm(delta))
    slope = sum(float(np.vdot(bars[key], delta)) for key, delta in directions.items())
    errors = {}
    for h in (3e-4, 1e-4):
        energies = []
        for step in (h, -h):
            moved = {
                **data,
                **{key: data[key] + step * delta for key, delta in directions.items()},
            }
            energies.append(evaluate(moved, run, pullback=False)[0])
        errors[str(h)] = abs((energies[0] - energies[1]) / (2 * h) - slope)
    if max(errors.values()) > 2e-8:
        raise ValueError(f"{name} full AO source/frame adjoint failed: {errors}")
    return {
        "case": name,
        "F12_increment_Eh": energy,
        "error_Eh": error,
        "max_intermediate_errors": maxima,
        "source_direction_slope": slope,
        "two_step_errors": errors,
        "differentiated_sources": sorted(bars),
        "metric_response_frobenius_norm": float(np.linalg.norm(bars["metric"])),
        "direction_scope": "AO integrals, common orbital coefficients and operator metrics; Coulomb metric held fixed in this directional test",
        "wall_including_compilation_s": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--scope", choices=("sources", "energy-chain"), default="sources"
    )
    args = parser.parse_args()
    launcher = resolve_compiler_cache()
    cache_before = subprocess.check_output(
        [str(launcher.path), "--show-stats"], text=True
    )
    # The explicit AO reference panel retains all five operators and their bars.
    # This bounded qualification allocation is not a production integral schedule.
    run, programs = cuda_runner(args.arch, args.cache, max_bytes=2 << 30)
    rows = []
    with np.load(args.fixture, allow_pickle=False) as packed:
        for name in ("h2", "water"):
            prefix = name + "/"
            data = {
                key.removeprefix(prefix): packed[key]
                for key in packed.files
                if key.startswith(prefix)
            }
            verify = check_sources if args.scope == "sources" else check
            rows.append(verify(data, name, run))
    cache_after = subprocess.check_output(
        [str(launcher.path), "--show-stats"], text=True
    )
    args.output.write_text(
        json.dumps(
            {
                "scope": args.scope,
                "rows": rows,
                "programs": programs,
                "compiler_cache": {
                    "path": str(launcher.path),
                    "version": launcher.version,
                    "before": cache_before,
                    "after": cache_after,
                },
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
