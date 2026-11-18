"""Qualify self-consistent closed-shell two-electron CC-F12 equations with matched fixed molecular sources."""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.cc.f12_residual import (
    f12b_geminal_back_coupling,
    fixed_pair_cc_residual,
    fixed_pair_geminal_energy,
)
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks._f12_cuda import cuda_runner


def qualify(source: dict, reference: dict, run: Callable) -> dict:
    """Use one occupied spatial orbital. The dense conventional tensors are diagnostic-sized."""
    obs = source["h"].shape[0]
    if (
        not 2 <= obs <= 24
        or source["f"].shape != (obs, obs)
        or source["eri"].shape != (obs,) * 4
    ):
        raise ValueError("bounded two-electron full-basis integral dimensions required")
    n = obs - 1
    upper = np.triu_indices(n)
    o, v = slice(0, 1), slice(1, None)
    slices = {"o": o, "v": v}
    hamiltonian = {
        key: source["eri"][tuple(slices[x] for x in key)].copy()
        for key in ("oooo", "oovv", "ovov", "ovvo", "ovoo", "ovvv", "vvvv")
    }
    hamiltonian.update(
        foo=source["f"][o, o].copy(),
        fov=source["f"][o, v].copy(),
        fvv=source["f"][v, v].copy(),
    )
    conventional = build_ccsd_program(1, n, form="optimized", diagnostics=False)
    fixed = fixed_pair_cc_residual(1, n, diagonal=True)
    back = f12b_geminal_back_coupling(1, n)
    energy = fixed_pair_geminal_energy(1, n)
    sel = np.ones(1)
    z = source["vext"]
    f12v = {"v_vv": z[v, v].copy(), "v_ov": z[o, v].copy(), "v_vo": z[v, o].copy()}
    t1 = np.zeros((1, n))
    t2 = np.zeros((1, 1, n, n))
    gaps = source["f"][0, 0] - np.diag(source["f"])[1:]
    if np.any(gaps >= 0):
        raise ValueError("negative occupied–virtual denominators required")
    d1 = gaps[None, :]
    d2 = (gaps[:, None] + gaps[None, :])[None, None, :, :]
    start = time.monotonic()
    history = []
    for _ in range(72):
        cq = {**hamiltonian, "t1": t1, "t2": t2}
        fq = {**f12v, "t1": t1, "fock_coupling_vv": source["c"], "select_m": sel}
        co = run(conventional, cq).outputs
        fr = run(fixed, fq).outputs
        r1 = co["singles_residual"] + fr["singles_m"]
        r2 = co["doubles_residual"] + fr["doubles_mn"]
        error = max(float(np.max(abs(r1))), float(np.max(abs(r2))))
        history.append(error)
        if not np.isfinite(error):
            raise FloatingPointError("non-finite CC residual")
        if error < 1e-10:
            break
        t1 = t1 + r1 / d1
        t2 = t2 + 0.5 * (r2 + r2.transpose(1, 0, 3, 2)) / d2
    else:
        raise ValueError(
            f"F12-adjusted CC amplitudes failed to converge: {history[-1]}"
        )

    bq = {**f12v, "t1": t1, "t2": t2}
    rb = run(back, bq).outputs["back_coupling"]
    eq = {
        "select_m": sel,
        "select_n": sel,
        "v_occupied": source["v"],
        "x_row": source["x"],
        "b_row": source["b"],
        "fock_occupied": source["f"][o, o].copy(),
        "fock_coupling_vv": source["c"],
        "t2": t2,
        "back_coupling": rb,
    }
    e = run(energy, eq).outputs
    components = np.array(
        [co["correlation_energy"], e["f12a_geminal_energy"], e["f12b_extra_energy"]]
    )
    packed = np.r_[t1.ravel(), t2[0, 0][upper]]
    lam = reference["lambda_source"]
    q2 = np.zeros_like(t2)
    q2[0, 0][upper] = lam[n:]
    # Split symmetric-pair adjoints by their actual output coordinates; fold input duals at the end.
    cv = run(
        transpose_program(
            conventional,
            ("correlation_energy", "singles_residual", "doubles_residual"),
            inputs=("t1", "t2"),
        ).program,
        {
            **cq,
            "bar_correlation_energy": np.array(1.0),
            "bar_singles_residual": lam[:n][None, :],
            "bar_doubles_residual": q2,
        },
    ).outputs
    fv = run(
        transpose_program(fixed, ("singles_m", "doubles_mn"), inputs=("t1",)).program,
        {**fq, "bar_singles_m": lam[:n], "bar_doubles_mn": q2[0, 0]},
    ).outputs
    ev = run(
        transpose_program(
            energy,
            ("f12a_geminal_energy", "f12b_extra_energy"),
            inputs=("t2", "back_coupling"),
        ).program,
        {
            **eq,
            "bar_f12a_geminal_energy": np.array(1.0),
            "bar_f12b_extra_energy": np.array(1.0),
        },
    ).outputs
    bv = run(
        transpose_program(back, ("back_coupling",), inputs=("t1", "t2")).program,
        {**bq, "bar_back_coupling": ev["bar_back_coupling"]},
    ).outputs
    bar1 = cv["bar_t1"][0] + fv["bar_t1"][0] + bv["bar_t1"][0]
    bar2 = (cv["bar_t2"] + ev["bar_t2"] + bv["bar_t2"])[0, 0]
    packed2 = bar2[upper].copy()
    off = upper[0] != upper[1]
    packed2[off] += bar2[(upper[1][off], upper[0][off])]
    errors = {
        "max_amplitude_vs_oracle": float(
            np.max(abs(packed - reference["oracle_amplitudes"]))
        ),
        "max_component_energy_Eh": float(
            np.max(abs(components - reference["components_Eh"]))
        ),
        "max_lambda_equation_residual": float(np.max(abs(np.r_[bar1, packed2]))),
    }
    return {
        "passed": bool(history[-1] < 1e-10 and max(errors.values()) < 2e-9),
        "errors": errors,
        "ordered_component_energies_Eh": components.tolist(),
        "components": ["ordinary_correlation", "F12a_common", "F12b_extra"],
        "residual_history": history,
        "iterations": len(history),
        "compute_compile_seconds": time.monotonic() - start,
        "active_occupied": 1,
        "virtuals": n,
        "independent_amplitudes": packed.size,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for label in ("source", "reference", "cache", "output"):
        p.add_argument("--" + label, type=Path, required=True)
    p.add_argument("--arch", required=True)
    a = p.parse_args()
    run, metrics = cuda_runner(a.arch, a.cache)
    with (
        np.load(a.source, allow_pickle=False) as data,
        np.load(a.reference, allow_pickle=False) as ref,
    ):
        out = qualify(dict(data), dict(ref), run)
    a.output.write_text(
        json.dumps(
            {"rows": {"h2": out}, "programs": metrics}, indent=2, allow_nan=False
        )
        + "\n"
    )
    if not out["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
