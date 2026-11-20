"""Physical AO DF Fock and its full-rank metric/occupied-orbital adjoint."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.mp2.f12_density_fitting import coulomb_fit_response
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    transpose_program,
)

if TYPE_CHECKING:
    from generativeqc_compiler.tensor import Execution, Node

Runner = Callable[[Program, Mapping[str, np.ndarray]], "Execution"]
Sources = Mapping[str, np.ndarray]
FockBars = tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]


def _field(name: str, *indices: Index) -> Node:
    return input_tensor(name, TensorSpec(indices, role="input", differentiable=True))


def _run(run: Runner, program: Program, inputs: Sources) -> Sources:
    names = {n.attrs["name"] for n in program.live_nodes if n.op == "input"}
    return run(program, {name: inputs[name] for name in names}).outputs


def _transpose(
    run: Runner, program: Program, inputs: Sources, seeds: Sources
) -> Sources:
    reversed_program = transpose_program(program, tuple(seeds)).program
    return _run(
        run, reversed_program, {**inputs, **{"bar_" + k: v for k, v in seeds.items()}}
    )


def _project(
    matrix: np.ndarray, frame: np.ndarray, cuda: bool, library: ctypes.CDLL | None
) -> np.ndarray:
    # Validate the physical AO matrix before discarding the last-bit difference
    # between its two representations. The final MO check sees both independent
    # triangles from the congruence of this same symmetric physical operator.
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix, matrix.T, atol=1e-11, rtol=1e-12
    ):
        raise ValueError("physical AO DF Fock is asymmetric")
    matrix = np.ascontiguousarray(0.5 * matrix + 0.5 * matrix.T)
    if not cuda:
        wide = frame.astype(np.longdouble)
        return np.asarray(
            wide.T @ (matrix.astype(np.longdouble) @ wide), dtype=np.float64
        )
    if library is None:
        from . import _native

        library = _native.load_library(device="cuda", device_id=0)
    f = library.generativeqc_f12_cabs_congruence_cuda_v1
    p = ctypes.POINTER(ctypes.c_double)
    f.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        p,
        p,
        p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    f.restype = ctypes.c_int
    frame = np.ascontiguousarray(frame)
    result = np.empty((frame.shape[1], frame.shape[1]))
    error = ctypes.create_string_buffer(512)
    if f(
        0,
        *frame.shape,
        matrix.ctypes.data_as(p),
        frame.ctypes.data_as(p),
        result.ctypes.data_as(p),
        error,
        len(error),
    ):
        raise RuntimeError(error.value.decode(errors="replace"))
    return result


def physical_fock(
    h: np.ndarray,
    raw: np.ndarray,
    whitener: np.ndarray,
    frames: Sequence[np.ndarray],
    occupied: tuple[int, int],
    run: Runner,
    cuda: bool,
    library: ctypes.CDLL | None,
) -> tuple[list[np.ndarray], Callable[[Sequence[np.ndarray]], FockBars]]:
    """Build F_AO before projection. Return extended matrices and their pullback.

    A_Q is the physical AO three-center panel. For each occupied spin frame O,
    R[Q,i,m] = A[Q,m,n] O[n,i], D = M^-1 R and B = W R. Then K = B.T B,
    rho[Q] = sum_s,i,m D[Q,i,m] O[m,i], and J = sum_Q rho[Q] A[Q].
    Differentiate K as R.T D with the implicit metric solve. This is exactly
    the Gram derivative for an arbitrary full-rank W satisfying W.T W=M^-1.
    """
    if not np.allclose(h, h.T, rtol=1e-12, atol=1e-11):
        raise ValueError("physical AO core Hamiltonian is asymmetric")
    for tile in np.array_split(raw, max(1, (len(raw) + 3) // 4)):
        if not np.allclose(tile, tile.swapaxes(1, 2), rtol=1e-12, atol=1e-11):
            raise ValueError("physical AO three-center Coulomb integral is asymmetric")
    n, ext = frames[0].shape
    q = len(whitener)
    ao = IndexSpace("joint_atomic", "ao", n)
    aux = IndexSpace("jk_fit", "auxiliary", q)
    mo = IndexSpace("extended_orbital", "orbital", ext)
    m, v = Index("m", ao), Index("n", ao)
    a, b = Index("a", aux), Index("b", aux)
    p = Index("p", mo)
    A, H, W = _field("panel", a, m, v), _field("core", m, v), _field("whitener", a, b)
    inputs = {"core": h, "panel": raw, "whitener": whitener}
    charges, exchange, mixed, records = [], {}, {}, {}
    for spin, (c, o) in enumerate(zip(frames, occupied, strict=True)):
        if not o:
            continue
        i = Index("i", IndexSpace(f"occ{spin}", "orbital", o))
        name = f"occ{spin}"
        O = _field(name, m, i)
        rname, dname = f"raw{spin}", f"fit{spin}"
        R, D = _field(rname, b, i, m), _field(dname, b, i, m)
        half = Program({rname: einsum("amn,ni->aim", A, O)})
        inputs[name] = np.ascontiguousarray(c[:, :o])
        inputs[rname] = _run(run, half, inputs)[rname]
        fit = coulomb_fit_response(q, o * n)
        solve_inputs = {
            "metric_whitener": whitener,
            "three_center": inputs[rname].reshape(q, -1),
        }
        inputs[dname] = _run(run, Program({"fit": fit.outputs["fit"]}), solve_inputs)[
            "fit"
        ].reshape(q, o, n)
        B = einsum("ab,bim->aim", W, R)
        exchange[spin] = einsum("aim,ain->mn", B, B)
        mixed[spin] = einsum("aim,ain->mn", D, R)
        charges.append(einsum("aim,mi->a", D, O))
        records[spin] = half, fit, solve_inputs, name, rname, dname
    rho = charges[0] if len(charges) == 1 else add(*charges)
    g = add(H, einsum("a,amn->mn", rho, A))

    def expressions(k: dict[int, Node]) -> dict[str, Node]:
        return {
            f"ao{s}": add(g, k[s], coefficients=(1, -1)) if s in k else g
            for s in range(2)
        }

    primal = _run(run, Program(expressions(exchange)), inputs)
    mo_focks = [
        _project(primal[f"ao{s}"], c, cuda, library) for s, c in enumerate(frames)
    ]

    def pullback(seeds: Sequence[np.ndarray]) -> FockBars:
        C, F = _field("coeff", m, p), _field("matrix", m, v)
        transformed = Program(
            {"result": einsum("mp,mq->pq", C, einsum("mn,nq->mq", F, C))}
        )
        ao_seeds, coeff_bars = {}, []
        for s, c in enumerate(frames):
            mat = primal[f"ao{s}"]
            bar = _transpose(
                run,
                transformed,
                {"coeff": c, "matrix": (mat + mat.T) * 0.5},
                {"result": seeds[s]},
            )
            ao_seeds[f"ao{s}"] = (bar["bar_matrix"] + bar["bar_matrix"].T) * 0.5
            coeff_bars.append(bar["bar_coeff"])
        bars = _transpose(run, Program(expressions(mixed)), inputs, ao_seeds)
        metric_bar = np.zeros_like(whitener)
        raw_bar = bars["bar_panel"]
        for s, (half, fit, solve_inputs, name, rname, dname) in records.items():
            solved = _run(
                run,
                Program(
                    {k: fit.outputs[k] for k in ("metric_bar", "three_center_bar")}
                ),
                {**solve_inputs, "fit_bar": bars["bar_" + dname].reshape(q, -1)},
            )
            metric_bar += solved["metric_bar"]
            seed = bars["bar_" + rname] + solved["three_center_bar"].reshape(
                inputs[rname].shape
            )
            half_bars = _transpose(run, half, inputs, {rname: seed})
            raw_bar += half_bars["bar_panel"]
            coeff_bars[s][:, : occupied[s]] += (
                bars["bar_" + name] + half_bars["bar_" + name]
            )
        return coeff_bars, bars["bar_core"], raw_bar, metric_bar

    return mo_focks, pullback


def obs_sources(
    raw: np.ndarray, frames: Sequence[np.ndarray], obs: int, run: Runner
) -> dict[str, np.ndarray]:
    """Canonical OBS-only DF factors for the coupled occupied/virtual response."""
    from generativeqc_compiler.mp2.f12_density_fitting import molecular_orbital_sources

    program = molecular_orbital_sources(obs, obs, 0, len(raw))
    graph = Program({"raw_coulomb": program.outputs["raw_coulomb"]})
    panel = np.ascontiguousarray(raw[:, :obs, :obs])
    return {
        "raw_coulomb_" + name: _run(
            run,
            graph,
            {"ao_coulomb": panel, "coefficients": np.ascontiguousarray(c[:obs, :obs])},
        )["raw_coulomb"]
        for name, c in zip(("alpha", "beta"), frames, strict=True)
    }
