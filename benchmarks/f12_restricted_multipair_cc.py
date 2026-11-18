"""Evaluate restricted fixed-geminal CC equations for all occupied spatial pairs in a small molecule."""

import argparse
import json
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.cc.f12_geminal_energy import all_pair_geminal_energy
from generativeqc_compiler.cc.f12_residual import (
    all_pair_cc_residual,
    f12b_geminal_back_coupling,
    fixed_pair_cc_residual,
    fixed_pair_geminal_energy,
)
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks._f12_cuda import cuda_runner
from benchmarks.f12_restricted_df_action import RestrictedDFAction


def conventional_inputs(src: dict, o: int) -> dict:
    blocks = {"o": slice(0, o), "v": slice(o, None)}
    ham = {
        key: src["eri"][tuple(blocks[k] for k in key)]
        for key in ("oooo", "oovv", "ovov", "ovvo", "ovoo", "ovvv", "vvvv")
    }
    occ, vir = blocks["o"], blocks["v"]
    ham.update(foo=src["f"][occ, occ], fov=src["f"][occ, vir], fvv=src["f"][vir, vir])
    return ham


def fixed_residual_inputs(src: dict, o: int) -> dict:
    z = src["external"]
    return {
        "v_vv": z[:, :, o:, o:],
        "v_vo": z[:, :, o:, :o],
        "v_ov": z[:, :, :o, o:],
        "c": src["c"],
    }


def geminal_energy_inputs(src: dict, t1: np.ndarray, t2: np.ndarray) -> dict:
    o = len(t1)
    return {
        **fixed_residual_inputs(src, o),
        **{k: src[k] for k in ("v", "x", "b")},
        "foo": src["f"][:o, :o],
        "t1": t1,
        "t2": t2,
    }


@lru_cache(maxsize=4)
def _graphs(occupied: int, virtual: int) -> tuple:
    return (
        build_ccsd_program(occupied, virtual, form="optimized", diagnostics=False),
        {
            p: fixed_pair_cc_residual(occupied, virtual, diagonal=p)
            for p in (False, True)
        },
        f12b_geminal_back_coupling(occupied, virtual),
        fixed_pair_geminal_energy(occupied, virtual),
    )


def fixed_source_cc_equations(
    src: dict,
    t1: np.ndarray,
    t2: np.ndarray,
    run: Callable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordered energy and folded symmetric R1/R2; real spin-restricted orbitals.

    The caller supplies an explicitly matched conventional Hamiltonian and
    geminal matrix rows in the same fixed orbital frame. The diagnostic
    materializes the ordinary four-orbital integral tensor. The molecular
    H2 driver separately handles orbital, integral and Lambda response.
    """
    o, u = t1.shape
    n = o + u
    if not 1 <= o <= 8 or not 1 <= u or n > 48:
        raise ValueError(
            "bounded molecular qualification requires 1–8 occupied and at most 48 total orbitals"
        )
    if (
        t2.shape != (o, o, u, u)
        or src["f"].shape != (n, n)
        or src["eri"].shape != (n,) * 4
    ):
        raise ValueError("inconsistent CC Hamiltonian or amplitude dimensions")
    if (
        any(src[k].shape != (o,) * 4 for k in ("v", "x", "b"))
        or src["c"].shape != (o, o, u, u)
        or src["external"].shape != (o, o, n, n)
    ):
        raise ValueError("inconsistent ordered geminal pair sources")
    cc, cor, back, energy = _graphs(o, u)
    rs = {"o": slice(0, o), "v": slice(o, None)}
    oi, vi = rs["o"], rs["v"]
    ham = conventional_inputs(src, o)
    reg = run(cc, {**ham, "t1": t1, "t2": t2}).outputs
    r1 = reg["singles_residual"].copy()
    r2 = reg["doubles_residual"].copy()
    e = np.array([float(reg["correlation_energy"]), 0.0, 0.0])
    eye = np.eye(o)
    for m, norb in np.ndindex(o, o):
        z = src["external"][m, norb]
        vf = {"v_vv": z[vi, vi], "v_vo": z[vi, oi], "v_ov": z[oi, vi]}
        sel = {"select_m": eye[m], "select_n": eye[norb]}
        c = src["c"][m, norb]
        if m <= norb:
            diag = m == norb
            rhs = run(
                cor[diag],
                {
                    **vf,
                    "t1": t1,
                    "fock_coupling_vv": c,
                    **({"select_m": eye[m]} if diag else sel),
                },
            ).outputs
            r1[m] += rhs["singles_m"]
            r2[m, norb] += rhs["doubles_mn"]
            if not diag:
                r1[norb] += rhs["singles_n"]
                r2[norb, m] += rhs["doubles_mn"].T
        rb = run(back, {**vf, "t1": t1, "t2": t2}).outputs["back_coupling"]
        ef = run(
            energy,
            {
                **sel,
                "v_occupied": src["v"][m, norb],
                "x_row": src["x"][m, norb],
                "b_row": src["b"][m, norb],
                "fock_occupied": ham["foo"],
                "fock_coupling_vv": c,
                "t2": t2,
                "back_coupling": rb,
            },
        ).outputs
        e[1] += ef["f12a_geminal_energy"]
        e[2] += ef["f12b_extra_energy"]
    return e, r1, r2


class DenseRestrictedAction:
    """Four-index diagnostic for cross-checking the factorized Hamiltonian actions."""

    def __init__(self, src: dict, o: int, run: Callable) -> None:
        self.run = run
        self.f = src["f"]
        self.eri = src["eri"]
        self.o = o
        self.ham = conventional_inputs(src, o)
        self.cc = _graphs(o, len(self.f) - o)[0]
        self.derivatives = {}

    def evaluate(self, t1: np.ndarray, t2: np.ndarray) -> dict:
        return self.run(self.cc, {**self.ham, "t1": t1, "t2": t2}).outputs

    def adjoint(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        l1: np.ndarray,
        l2: np.ndarray,
        *,
        amplitudes: bool,
    ) -> dict:
        if amplitudes not in self.derivatives:
            keys = ("t1", "t2") if amplitudes else tuple(self.ham)
            self.derivatives[amplitudes] = transpose_program(
                self.cc, tuple(self.cc.outputs), inputs=keys
            ).program
        out = self.run(
            self.derivatives[amplitudes],
            {
                **self.ham,
                "t1": t1,
                "t2": t2,
                "bar_correlation_energy": np.array(1.0),
                "bar_singles_residual": l1,
                "bar_doubles_residual": l2,
            },
        ).outputs
        if amplitudes:
            return {name: out["bar_" + name] for name in ("t1", "t2")}
        ans = {"f": np.zeros_like(self.f), "eri": np.zeros_like(self.eri)}
        rs = {"o": slice(0, self.o), "v": slice(self.o, None)}
        for k in self.ham:
            target = "f" if k.startswith("f") else "eri"
            axes = k[1:] if target == "f" else k
            ans[target][tuple(rs[a] for a in axes)] += out["bar_" + k]
        return ans


def solve_all_pair_cc(
    src: dict, run: Callable, *, ordinary: RestrictedDFAction | None = None
) -> dict:
    """Converge ordinary and fixed-cusp residuals jointly from zero, then evaluate ordered energies."""
    o, _, v, _ = src["c"].shape
    if min(o, v) < 1 or (ordinary is None and (o > 8 or o + v > 48)):
        raise ValueError("bounded molecular qualification dimensions required")
    ordinary = DenseRestrictedAction(src, o, run) if ordinary is None else ordinary
    correction = all_pair_cc_residual(o, v)
    z = fixed_residual_inputs(src, o)
    gaps = np.diag(src["f"])[:o, None] - np.diag(src["f"])[None, o:]
    if np.any(gaps >= 0):
        raise ValueError("occupied-virtual gaps must be negative")
    gaps2 = gaps[:, None, :, None] + gaps[None, :, None, :]
    t1 = np.zeros((o, v))
    t2 = np.zeros((o, o, v, v))
    errors = []
    for _ in range(100):
        r = ordinary.evaluate(t1, t2)
        w = run(correction, {**z, "t1": t1}).outputs
        one = r["singles_residual"] + w["singles"]
        two = r["doubles_residual"] + w["doubles"]
        err = max(float(np.max(abs(one))), float(np.max(abs(two))))
        errors.append(err)
        if not np.isfinite(err):
            raise FloatingPointError("nonfinite complete CC-F12 residual")
        if err < 1e-10:
            break
        t1 += one / gaps
        t2 += 0.5 * (two + two.transpose(1, 0, 3, 2)) / gaps2
    else:
        raise ArithmeticError("multi-occupied CC-F12 failed to converge")
    energy = run(
        all_pair_geminal_energy(o, v), geminal_energy_inputs(src, t1, t2)
    ).outputs
    energies = np.array(
        [
            float(r["correlation_energy"]),
            float(energy["f12a_geminal_energy"]),
            float(energy["f12b_extra_energy"]),
        ]
    )
    return {
        "t1": t1,
        "t2": t2,
        "component_energies_Eh": energies,
        "residual_history": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arch", required=True)
    args = parser.parse_args()
    run, programs = cuda_runner(args.arch, args.cache, max_bytes=1024 << 20)
    with np.load(args.source, allow_pickle=False) as archive:
        sources = dict(archive)
    begin = time.monotonic()
    result = solve_all_pair_cc(sources, run)
    with np.load(args.reference, allow_pickle=False) as ref:
        errors = {
            key: float(np.max(abs(result[key] - ref[source])))
            for key, source in (
                ("t1", "t1"),
                ("t2", "t2"),
                ("component_energies_Eh", "components"),
            )
        }
    occ, v = result["t1"].shape
    row = {
        "passed": max(errors.values()) <= 2e-9
        and result["residual_history"][-1] < 1e-10,
        "errors": errors,
        "component_energies_Eh": result["component_energies_Eh"].tolist(),
        "residual_history": result["residual_history"],
        "occupied": occ,
        "virtual": v,
        "unordered_pairs": occ * (occ + 1) // 2,
        "ordinary_and_geminal_seconds": time.monotonic() - begin,
    }
    args.output.write_text(
        json.dumps(
            {"rows": {"water": row}, "programs": programs}, indent=2, allow_nan=False
        )
        + "\n"
    )
    if not row["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
