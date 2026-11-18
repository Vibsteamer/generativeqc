"""Matrix-free CC-F12 amplitude response for a small, multi-occupied restricted molecule."""

from collections.abc import Callable

import numpy as np
from generativeqc_compiler.cc.f12_geminal_energy import all_pair_geminal_energy
from generativeqc_compiler.cc.f12_residual import all_pair_cc_residual
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks.f12_restricted_multipair_cc import (
    _graphs,
    conventional_inputs,
    fixed_residual_inputs,
    geminal_energy_inputs,
)


def prepare_lagrangian_gradient(
    src: dict, t1: np.ndarray, t2: np.ndarray, run: Callable
) -> Callable:
    """Return the gradient of E + lambda1:R1 + lambda2:R2 in symmetric spatial amplitudes.

    Lambda doubles use the full Frobenius contraction. Store half of the
    dual of an independent off-diagonal pair on each equivalent entry.
    """
    o, v = t1.shape
    if not 1 <= o <= 8 or v < 1 or o + v > 48 or t2.shape != (o, o, v, v):
        raise ValueError("bounded spatial pair amplitude dimensions required")
    cc, *_ = _graphs(o, v)
    energy = all_pair_geminal_energy(o, v)
    conv = transpose_program(
        cc,
        ("correlation_energy", "singles_residual", "doubles_residual"),
        inputs=("t1", "t2"),
    ).program
    fixed = transpose_program(
        all_pair_cc_residual(o, v), ("singles", "doubles"), inputs=("t1",)
    ).program
    egrad = transpose_program(
        energy, tuple(energy.outputs), inputs=("t1", "t2")
    ).program
    ham = conventional_inputs(src, o)
    cvfeed = {**ham, "t1": t1, "t2": t2}
    fvfeed = {**fixed_residual_inputs(src, o), "t1": t1}
    unit = np.array(1.0)
    sources = geminal_energy_inputs(src, t1, t2)
    energy_bars = run(
        egrad, {**sources, **{"bar_" + k: unit for k in energy.outputs}}
    ).outputs
    ge1, ge2 = energy_bars["bar_t1"], energy_bars["bar_t2"]

    def gradient(l1: np.ndarray, l2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The source derivatives of the geminal energy are independent of Lambda.
        c = run(
            conv,
            {
                **cvfeed,
                "bar_correlation_energy": unit,
                "bar_singles_residual": l1,
                "bar_doubles_residual": l2,
            },
        ).outputs
        r = run(fixed, {**fvfeed, "bar_singles": l1, "bar_doubles": l2}).outputs
        s = c["bar_t1"] + ge1 + r["bar_t1"]
        t = c["bar_t2"] + ge2
        return s, 0.5 * (t + t.transpose(1, 0, 3, 2))

    return gradient


def solve_lambda(gradient: Callable, fock: np.ndarray, occupied: int) -> dict:
    gaps = np.diag(fock)[:occupied, None] - np.diag(fock)[None, occupied:]
    if not gaps.size or np.any(gaps >= 0):
        raise ValueError("negative occupied–virtual gaps required")
    d2 = gaps[:, None, :, None] + gaps[None, :, None, :]
    l1 = np.zeros_like(gaps)
    l2 = np.zeros_like(d2)
    history = []
    for _ in range(100):
        r1, r2 = gradient(l1, l2)
        err = max(float(np.max(abs(r1))), float(np.max(abs(r2))))
        if not np.isfinite(err):
            raise FloatingPointError("nonfinite CC-F12 Lambda residual")
        history.append(err)
        if err < 1e-10:
            break
        l1 += r1 / gaps
        l2 += r2 / d2
    else:
        raise ArithmeticError("multi-occupied CC-F12 Lambda failed to converge")
    return {"lambda1": l1, "lambda2": l2, "residual_history": history}


def relaxed_source_weights(
    src: dict,
    t1: np.ndarray,
    t2: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    run: Callable,
) -> dict[str, np.ndarray]:
    """Differentiate the stationary Lagrangian with respect to the ordinary and geminal sources."""
    o, v = t1.shape
    cc, *_ = _graphs(o, v)
    energy = all_pair_geminal_energy(o, v)
    cor = all_pair_cc_residual(o, v)
    ham = conventional_inputs(src, o)
    all_fixed = fixed_residual_inputs(src, o)
    gen = transpose_program(
        cc,
        ("correlation_energy", "singles_residual", "doubles_residual"),
        inputs=tuple(ham),
    ).program
    fg = transpose_program(cor, ("singles", "doubles"), inputs=tuple(all_fixed)).program
    gem = geminal_energy_inputs(src, t1, t2)
    eg = transpose_program(
        energy,
        tuple(energy.outputs),
        inputs=tuple(k for k in gem if k not in ("t1", "t2")),
    ).program
    one = np.array(1.0)
    normal = run(
        gen,
        {
            **ham,
            "t1": t1,
            "t2": t2,
            "bar_correlation_energy": one,
            "bar_singles_residual": l1,
            "bar_doubles_residual": l2,
        },
    ).outputs
    fixed = run(
        fg, {**all_fixed, "t1": t1, "bar_singles": l1, "bar_doubles": l2}
    ).outputs
    result = {
        k: np.zeros_like(src[k]) for k in ("eri", "f", "external", "c", "v", "x", "b")
    }
    rs = {"o": slice(0, o), "v": slice(o, None)}
    for key in ham:
        if key.startswith("f"):
            result["f"][tuple(rs[k] for k in key[1:])] += normal["bar_" + key]
        else:
            result["eri"][tuple(rs[k] for k in key)] += normal["bar_" + key]
    ebar = run(eg, {**gem, **{"bar_" + k: one for k in energy.outputs}}).outputs
    result["f"][:o, :o] += ebar["bar_foo"]
    for k in ("v", "x", "b"):
        result[k] += ebar["bar_" + k]
    result["c"] += fixed["bar_c"] + ebar["bar_c"]
    for name, block in (
        ("v_vv", (rs["v"], rs["v"])),
        ("v_vo", (rs["v"], rs["o"])),
        ("v_ov", (rs["o"], rs["v"])),
    ):
        result["external"][(slice(None), slice(None), *block)] = (
            fixed["bar_" + name] + ebar["bar_" + name]
        )
    return result
