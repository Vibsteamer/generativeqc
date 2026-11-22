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
    """Build both extended Fock matrices and their occupied/metric pullback."""
    values, adjoint = physical_spin_operators(
        h, raw, whitener, frames, occupied, run, cuda, library, include_exchange=False
    )
    names = ("fock_alpha", "fock_beta")

    def pullback(seeds: Sequence[np.ndarray]) -> FockBars:
        return adjoint(dict(zip(names, seeds, strict=True)))

    return [values[name] for name in names], pullback


def physical_spin_operators(
    h: np.ndarray,
    raw: np.ndarray,
    whitener: np.ndarray,
    frames: Sequence[np.ndarray],
    occupied: tuple[int, int],
    run: Runner,
    cuda: bool,
    library: ctypes.CDLL | None,
    *,
    include_exchange: bool = True,
) -> tuple[dict[str, np.ndarray], Callable[[Sources], FockBars]]:
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
    from ._f12_spin_fock import _tile_groups, default_auxiliary_batch_size
    from ._f12_spin_pairs import batch_execute

    width = default_auxiliary_batch_size(q, n)
    groups = _tile_groups(q, width)
    ao = IndexSpace("joint_atomic", "ao", n)
    aux = IndexSpace("jk_fit", "auxiliary", q)
    mo = IndexSpace("extended_orbital", "orbital", ext)
    m, v = Index("m", ao), Index("n", ao)
    a, b = Index("a", aux), Index("b", aux)
    p = Index("p", mo)
    W = _field("whitener", a, b)
    inputs = {"whitener": whitener}
    charges, exchange, mixed, records = [], {}, {}, {}
    for spin, (c, o) in enumerate(zip(frames, occupied, strict=True)):
        if not o:
            continue
        i = Index("i", IndexSpace(f"occ{spin}", "orbital", o))
        name = f"occ{spin}"
        O = _field(name, m, i)
        rname, dname = f"raw{spin}", f"fit{spin}"
        R, D = _field(rname, b, i, m), _field(dname, b, i, m)
        occ_mat = np.ascontiguousarray(c[:, :o])
        inputs[name] = occ_mat
        raw_s = np.empty((q, o, n), dtype=np.float64)
        for tile_len, slices in groups:
            a_t = Index("a", IndexSpace("jk_fit", "auxiliary", tile_len))
            half_t = Program({rname: einsum("amn,ni->aim", _field("panel", a_t, m, v), O)})
            with batch_execute(run, half_t) as exec_one:
                for rows in slices:
                    raw_s[rows] = exec_one(
                        {"panel": np.ascontiguousarray(raw[rows]), name: occ_mat}
                    ).outputs[rname]
        inputs[rname] = raw_s
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
        records[spin] = (o, occ_mat, fit, solve_inputs, name, rname, dname)
    rho = charges[0] if len(charges) == 1 else add(*charges)

    spins = ("alpha", "beta")
    slots = [(f"fock_{name}", s) for s, name in enumerate(spins)]
    if include_exchange:
        slots.extend((f"exchange_{name}", s) for s, name in enumerate(spins))

    core_fwd = _run(
        run,
        Program({"rho": rho, **{f"k{s}": exchange[s] for s in exchange}}),
        inputs,
    )
    rho_val = core_fwd["rho"]
    j_mat = np.zeros((n, n), dtype=np.float64)
    for tile_len, slices in groups:
        a_t = Index("a", IndexSpace("jk_fit", "auxiliary", tile_len))
        j_prog = Program(
            {"j": einsum("a,amn->mn", _field("rho", a_t), _field("panel", a_t, m, v))}
        )
        with batch_execute(run, j_prog) as exec_one:
            for rows in slices:
                j_mat += exec_one(
                    {
                        "rho": np.ascontiguousarray(rho_val[rows]),
                        "panel": np.ascontiguousarray(raw[rows]),
                    }
                ).outputs["j"]
    g_val = h + j_mat
    primal = {}
    for s, name in enumerate(spins):
        primal["fock_" + name] = g_val - core_fwd[f"k{s}"] if s in exchange else g_val
        if include_exchange:
            primal["exchange_" + name] = (
                core_fwd[f"k{s}"] if s in exchange else np.zeros_like(h)
            )
    matrices = {
        name: _project(primal[name], frames[s], cuda, library) for name, s in slots
    }

    def pullback(seeds: Sources) -> FockBars:
        C, F = _field("coeff", m, p), _field("matrix", m, v)
        transformed = Program(
            {"result": einsum("mp,mq->pq", C, einsum("mn,nq->mq", F, C))}
        )
        if not seeds.keys() <= matrices.keys():
            raise ValueError("Unknown extended spin operator cotangent")
        ao_seeds = {}
        coeff_bars = [np.zeros_like(c) for c in frames]
        for name, s in slots:
            mat = primal[name]
            if name not in seeds:
                ao_seeds[name] = np.zeros_like(mat)
                continue
            bar = _transpose(
                run,
                transformed,
                {"coeff": frames[s], "matrix": (mat + mat.T) * 0.5},
                {"result": seeds[name]},
            )
            ao_seeds[name] = (bar["bar_matrix"] + bar["bar_matrix"].T) * 0.5
            coeff_bars[s] += bar["bar_coeff"]
        bar_g = ao_seeds["fock_alpha"] + ao_seeds["fock_beta"]
        bar_core = bar_g.copy()
        bar_rho = np.zeros(q, dtype=np.float64)
        raw_bar = np.zeros_like(raw)
        bar_g_c = np.ascontiguousarray(bar_g)
        for tile_len, slices in groups:
            a_t = Index("a", IndexSpace("jk_fit", "auxiliary", tile_len))
            j_prog = Program(
                {"j": einsum("a,amn->mn", _field("rho", a_t), _field("panel", a_t, m, v))}
            )
            j_rev = transpose_program(j_prog, ("j",)).program
            with batch_execute(run, j_rev) as exec_one:
                for rows in slices:
                    jr = exec_one(
                        {
                            "rho": np.ascontiguousarray(rho_val[rows]),
                            "panel": np.ascontiguousarray(raw[rows]),
                            "bar_j": bar_g_c,
                        }
                    ).outputs
                    bar_rho[rows] += jr["bar_rho"]
                    raw_bar[rows] += jr["bar_panel"]
        core_seeds = {"rho": bar_rho}
        for s, name in enumerate(spins):
            if s in mixed:
                k_seed = -ao_seeds["fock_" + name]
                if include_exchange:
                    k_seed = k_seed + ao_seeds["exchange_" + name]
                core_seeds[f"k{s}"] = k_seed
        bars = _transpose(
            run,
            Program({"rho": rho, **{f"k{s}": mixed[s] for s in mixed}}),
            inputs,
            core_seeds,
        )
        metric_bar = np.zeros_like(whitener)
        for s, (o, occ_mat, fit, solve_inputs, name, rname, dname) in records.items():
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
            coeff_bars[s][:, :o] += bars["bar_" + name]
            i_idx = Index("i", IndexSpace(f"occ{s}", "orbital", o))
            O_node = _field(name, m, i_idx)
            for tile_len, slices in groups:
                a_t = Index("a", IndexSpace("jk_fit", "auxiliary", tile_len))
                half_t = Program(
                    {rname: einsum("amn,ni->aim", _field("panel", a_t, m, v), O_node)}
                )
                half_rev = transpose_program(half_t, (rname,)).program
                with batch_execute(run, half_rev) as exec_one:
                    for rows in slices:
                        hb = exec_one(
                            {
                                "panel": np.ascontiguousarray(raw[rows]),
                                name: occ_mat,
                                "bar_" + rname: np.ascontiguousarray(seed[rows]),
                            }
                        ).outputs
                        raw_bar[rows] += hb["bar_panel"]
                        coeff_bars[s][:, :o] += hb["bar_" + name]
        return coeff_bars, bar_core, raw_bar, metric_bar

    return matrices, pullback


def obs_sources(
    raw: np.ndarray, frames: Sequence[np.ndarray], obs: int, run: Runner
) -> dict[str, np.ndarray]:
    """Canonical OBS-only DF factors for the coupled occupied/virtual response."""
    from ._f12_spin_ao import project_ao_operator_tiled

    panel = np.ascontiguousarray(raw[:, :obs, :obs])
    return {
        "raw_coulomb_" + name: project_ao_operator_tiled(
            obs,
            obs,
            0,
            len(raw),
            "raw_coulomb",
            "ao_coulomb",
            np.ascontiguousarray(c[:obs, :obs]),
            panel,
            run,
        )
        for name, c in zip(("alpha", "beta"), frames, strict=True)
    }
