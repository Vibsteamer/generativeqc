"""Physical AO DF Fock and its full-rank metric/occupied-orbital adjoint."""

from __future__ import annotations

import ctypes
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
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

from ._f12_ext_fock_native import ExtFockNative, resolve_ext_fock_native

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
    if library is None or not hasattr(
        library, "generativeqc_f12_cabs_congruence_cuda_v1"
    ):
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


def _validate_raw_panel(
    raw_panel: np.ndarray, q: int, n: int, *, fingerprint: bool = False
) -> bytes | None:
    if not isinstance(raw_panel, np.ndarray) or raw_panel.ndim != 3:
        raise ValueError(
            f"physical AO three-center Coulomb integral must be a 3D float64 array of shape {(q, n, n)}"
        )
    if raw_panel.dtype != np.float64:
        raise ValueError(
            f"physical AO three-center Coulomb integral must have dtype float64, got {raw_panel.dtype}"
        )
    if raw_panel.shape != (q, n, n):
        raise ValueError(
            f"physical AO three-center Coulomb integral shape {raw_panel.shape} disagrees with expected {(q, n, n)}"
        )
    digest = hashlib.sha256() if fingerprint else None
    tile_q = _native_tile_rows(q, n)
    for q0 in range(0, q, tile_q):
        tile = raw_panel[q0 : min(q, q0 + tile_q)]
        if not np.isfinite(tile).all():
            raise ValueError(
                "physical AO three-center Coulomb integral contains non-finite values"
            )
        if not np.allclose(tile, tile.swapaxes(1, 2), rtol=1e-12, atol=1e-11):
            raise ValueError("physical AO three-center Coulomb integral is asymmetric")
        if digest is not None:
            digest.update(memoryview(np.ascontiguousarray(tile)).cast("B"))
    return digest.digest() if digest is not None else None


def _validate_operator_inputs(
    h: np.ndarray,
    whitener: np.ndarray,
    frames: Sequence[np.ndarray],
    occupied: tuple[int, int],
) -> tuple[int, int, int, int]:
    if len(frames) != 2 or len(occupied) != 2:
        raise ValueError("physical_spin_operators requires two spin frames and occupied counts")
    for idx, frame in enumerate(frames):
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 2
            or frame.dtype != np.float64
            or frame.shape[0] < 1
            or frame.shape[1] < 1
            or not np.isfinite(frame).all()
        ):
            raise ValueError(f"spin frame {idx} must be a nonempty finite 2D float64 array")
    n = int(frames[0].shape[0])
    if int(frames[1].shape[0]) != n:
        raise ValueError("spin frames must share the same AO row dimension")
    oa_raw, ob_raw = occupied
    if (
        isinstance(oa_raw, bool)
        or isinstance(ob_raw, bool)
        or not isinstance(oa_raw, (int, np.integer))
        or not isinstance(ob_raw, (int, np.integer))
    ):
        raise ValueError("occupied spin counts must be integers")
    oa, ob = int(oa_raw), int(ob_raw)
    if oa < 0 or ob < 0 or oa > frames[0].shape[1] or ob > frames[1].shape[1]:
        raise ValueError("occupied spin counts exceed their frame orbital dimensions")
    if (
        not isinstance(h, np.ndarray)
        or h.ndim != 2
        or h.shape != (n, n)
        or h.dtype != np.float64
        or not np.isfinite(h).all()
    ):
        raise ValueError(
            f"physical AO core Hamiltonian must be a finite float64 array of shape {(n, n)}"
        )
    if not np.allclose(h, h.T, rtol=1e-12, atol=1e-11):
        raise ValueError("physical AO core Hamiltonian is asymmetric")
    if (
        not isinstance(whitener, np.ndarray)
        or whitener.ndim != 2
        or whitener.shape[0] != whitener.shape[1]
        or whitener.shape[0] < 1
        or whitener.dtype != np.float64
        or not np.isfinite(whitener).all()
    ):
        raise ValueError("whitener must be a nonempty square finite float64 matrix")
    q = int(whitener.shape[0])
    return n, q, oa, ob


def physical_fock(
    h: np.ndarray,
    raw: np.ndarray | None,
    whitener: np.ndarray,
    frames: Sequence[np.ndarray],
    occupied: tuple[int, int],
    run: Runner,
    cuda: bool,
    library: ctypes.CDLL | None,
    *,
    raw_loader: Callable[[], np.ndarray] | None = None,
    raw_loader_exclusive: bool = False,
    native_library: Path | str | ctypes.CDLL | None = None,
    host_reference: bool | None = None,
) -> tuple[list[np.ndarray], Callable[[Sequence[np.ndarray]], FockBars]]:
    """Build both extended Fock matrices and their occupied/metric pullback."""
    values, adjoint = physical_spin_operators(
        h,
        raw,
        whitener,
        frames,
        occupied,
        run,
        cuda,
        library,
        include_exchange=False,
        raw_loader=raw_loader,
        raw_loader_exclusive=raw_loader_exclusive,
        native_library=native_library,
        host_reference=host_reference,
    )
    names = ("fock_alpha", "fock_beta")

    def pullback(seeds: Sequence[np.ndarray]) -> FockBars:
        return adjoint(dict(zip(names, seeds, strict=True)))

    return [values[name] for name in names], pullback


def _native_tile_rows(q: int, n: int) -> int:
    bytes_per_q = max(1, n * n * 8)
    return max(1, min(q, (64 * 1024 * 1024) // bytes_per_q))


def _physical_spin_operators_native(
    h: np.ndarray,
    raw_local: np.ndarray,
    whitener: np.ndarray,
    frames: Sequence[np.ndarray],
    occupied: tuple[int, int],
    cuda: bool,
    library: ctypes.CDLL | None,
    *,
    include_exchange: bool,
    reload_raw: Callable[[], np.ndarray],
    raw_loader_exclusive: bool,
    native: ExtFockNative,
) -> tuple[dict[str, np.ndarray], Callable[[Sources], FockBars]]:
    n = frames[0].shape[0]
    q = len(whitener)
    oa, ob = occupied
    same_occ_frame = (
        oa == ob
        and oa > 0
        and np.array_equal(frames[0][:, :oa], frames[1][:, :ob])
    )
    occ_a = np.ascontiguousarray(frames[0][:, :oa], dtype=np.float64) if oa > 0 else None
    occ_b = (
        occ_a
        if same_occ_frame
        else (
            np.ascontiguousarray(frames[1][:, :ob], dtype=np.float64) if ob > 0 else None
        )
    )

    is_contig_f64 = raw_local.flags.c_contiguous and raw_local.dtype == np.float64
    tile_q = _native_tile_rows(q, n)

    def _run_half_fwd(occ: np.ndarray) -> np.ndarray:
        if is_contig_f64:
            return native.half_fwd(raw_local, occ)
        out = np.empty((q, occ.shape[1], n), dtype=np.float64)
        for q0 in range(0, q, tile_q):
            q1 = min(q, q0 + tile_q)
            out[q0:q1] = native.half_fwd(raw_local[q0:q1], occ)
        return out

    R_a = _run_half_fwd(occ_a) if oa > 0 and occ_a is not None else None
    if ob == 0 or occ_b is None:
        R_b = None
    elif same_occ_frame:
        R_b = R_a
    else:
        R_b = _run_half_fwd(occ_b)

    if oa == 0 and ob == 0:
        D_a = D_b = None
        K_a = np.zeros((n, n), dtype=np.float64)
        K_b = np.zeros((n, n), dtype=np.float64)
        rho_val = np.zeros(q, dtype=np.float64)
    else:
        D_a, D_b, K_a, K_b, rho_val = native.fwd(
            whitener,
            occ_a if oa > 0 else np.empty((n, 0), dtype=np.float64),
            occ_b if ob > 0 else np.empty((n, 0), dtype=np.float64),
            R_a,
            R_b,
            same_occ_frame=same_occ_frame,
        )

    j_mat = np.zeros((n, n), dtype=np.float64)
    if oa > 0 or ob > 0:
        if is_contig_f64:
            native.j_fwd(raw_local, rho_val, j_mat)
        else:
            for q0 in range(0, q, tile_q):
                q1 = min(q, q0 + tile_q)
                native.j_fwd(raw_local[q0:q1], rho_val[q0:q1], j_mat)

    del raw_local
    g_val = h + j_mat
    spins = ("alpha", "beta")
    slots = [(f"fock_{name}", s) for s, name in enumerate(spins)]
    if include_exchange:
        slots.extend((f"exchange_{name}", s) for s, name in enumerate(spins))
    k_by_spin = (K_a, K_b)
    primal: dict[str, np.ndarray] = {}
    for s, name in enumerate(spins):
        primal["fock_" + name] = g_val - k_by_spin[s] if occupied[s] > 0 else g_val
        if include_exchange:
            primal["exchange_" + name] = (
                k_by_spin[s] if occupied[s] > 0 else np.zeros_like(h)
            )
    matrices = {
        name: _project(primal[name], frames[s], cuda, library) for name, s in slots
    }

    def pullback(seeds: Sources) -> FockBars:
        raw_pb = reload_raw()
        if not seeds.keys() <= matrices.keys():
            raise ValueError("Unknown extended spin operator cotangent")

        ao_seeds: dict[str, np.ndarray] = {}
        coeff_bars = [np.zeros_like(c, dtype=np.float64, order="C") for c in frames]
        for name, s in slots:
            if name not in seeds:
                ao_seeds[name] = np.zeros((n, n), dtype=np.float64)
                continue
            ao_seeds[name] = native.congruence_rev(
                frames[s], primal[name], seeds[name], coeff_bars[s]
            )

        bar_g = np.ascontiguousarray(
            ao_seeds["fock_alpha"] + ao_seeds["fock_beta"], dtype=np.float64
        )
        bar_core = bar_g.copy()
        pb_contig_f64 = raw_pb.flags.c_contiguous and raw_pb.dtype == np.float64

        if pb_contig_f64:
            bar_rho = native.j_rev(raw_pb, bar_g)
        else:
            bar_rho = np.empty(q, dtype=np.float64)
            for q0 in range(0, q, tile_q):
                q1 = min(q, q0 + tile_q)
                bar_rho[q0:q1] = native.j_rev(raw_pb[q0:q1], bar_g)

        k_seed_a = None
        if oa > 0:
            k_seed_a = -ao_seeds["fock_alpha"]
            if include_exchange:
                k_seed_a = k_seed_a + ao_seeds["exchange_alpha"]
        k_seed_b = None
        if ob > 0:
            k_seed_b = -ao_seeds["fock_beta"]
            if include_exchange:
                k_seed_b = k_seed_b + ao_seeds["exchange_beta"]

        bar_occ_a = np.zeros((n, oa), dtype=np.float64) if oa > 0 else None
        bar_occ_b = np.zeros((n, ob), dtype=np.float64) if ob > 0 else None
        bar_R_a, bar_R_b, metric_bar = native.rev(
            whitener,
            n,
            occ_a,
            occ_b,
            R_a,
            R_b,
            D_a,
            D_b,
            bar_rho,
            k_seed_a,
            k_seed_b,
            bar_occ_a,
            bar_occ_b,
        )

        inplace_raw = raw_loader_exclusive and raw_pb.flags.writeable
        if inplace_raw:
            if pb_contig_f64:
                native.half_rev(
                    raw_pb,
                    occ_a,
                    occ_b,
                    bar_R_a,
                    bar_R_b,
                    raw_pb,
                    bar_occ_a,
                    bar_occ_b,
                    rho_tile=rho_val,
                    bar_g=bar_g,
                    add_to_bar_raw=False,
                )
            else:
                for q0 in range(0, q, tile_q):
                    q1 = min(q, q0 + tile_q)
                    tile = np.ascontiguousarray(raw_pb[q0:q1], dtype=np.float64)
                    native.half_rev(
                        tile,
                        occ_a,
                        occ_b,
                        bar_R_a[q0:q1] if bar_R_a is not None else None,
                        bar_R_b[q0:q1] if bar_R_b is not None else None,
                        tile,
                        bar_occ_a,
                        bar_occ_b,
                        rho_tile=rho_val[q0:q1],
                        bar_g=bar_g,
                        add_to_bar_raw=False,
                    )
                    raw_pb[q0:q1] = tile
            raw_out = raw_pb
        else:
            raw_bar = np.empty((q, n, n), dtype=np.float64)
            if pb_contig_f64:
                native.half_rev(
                    raw_pb,
                    occ_a,
                    occ_b,
                    bar_R_a,
                    bar_R_b,
                    raw_bar,
                    bar_occ_a,
                    bar_occ_b,
                    rho_tile=rho_val,
                    bar_g=bar_g,
                    add_to_bar_raw=False,
                )
            else:
                for q0 in range(0, q, tile_q):
                    q1 = min(q, q0 + tile_q)
                    native.half_rev(
                        raw_pb[q0:q1],
                        occ_a,
                        occ_b,
                        bar_R_a[q0:q1] if bar_R_a is not None else None,
                        bar_R_b[q0:q1] if bar_R_b is not None else None,
                        raw_bar[q0:q1],
                        bar_occ_a,
                        bar_occ_b,
                        rho_tile=rho_val[q0:q1],
                        bar_g=bar_g,
                        add_to_bar_raw=False,
                    )
            del raw_pb
            raw_out = raw_bar

        if oa > 0 and bar_occ_a is not None:
            coeff_bars[0][:, :oa] += bar_occ_a
        if ob > 0 and bar_occ_b is not None:
            coeff_bars[1][:, :ob] += bar_occ_b
        return coeff_bars, bar_core, raw_out, metric_bar

    return matrices, pullback


def physical_spin_operators(
    h: np.ndarray,
    raw: np.ndarray | None,
    whitener: np.ndarray,
    frames: Sequence[np.ndarray],
    occupied: tuple[int, int],
    run: Runner,
    cuda: bool,
    library: ctypes.CDLL | None,
    *,
    include_exchange: bool = True,
    raw_loader: Callable[[], np.ndarray] | None = None,
    raw_loader_exclusive: bool = False,
    native_library: Path | str | ctypes.CDLL | None = None,
    host_reference: bool | None = None,
) -> tuple[dict[str, np.ndarray], Callable[[Sources], FockBars]]:
    """Build F_AO before projection. Return extended matrices and their pullback.

    A_Q is the physical AO three-center panel. For each occupied spin frame O,
    R[Q,i,m] = A[Q,m,n] O[n,i], D = M^-1 R and B = W R. Then K = B.T B,
    rho[Q] = sum_s,i,m D[Q,i,m] O[m,i], and J = sum_Q rho[Q] A[Q].
    Differentiate K as R.T D with the implicit metric solve. This is exactly
    the Gram derivative for an arbitrary full-rank W satisfying W.T W=M^-1.
    """
    if type(raw_loader_exclusive) is not bool:
        raise TypeError("raw_loader_exclusive must be Boolean")
    n, q, _, _ = _validate_operator_inputs(h, whitener, frames, occupied)
    if raw is None and raw_loader is None:
        raise ValueError("physical_spin_operators requires raw or raw_loader")
    raw_local = raw_loader() if raw is None else raw
    if raw_local is None:
        raise ValueError("physical_spin_operators raw_loader returned None")
    streamed = raw is None
    raw_loader_exclusive = streamed and raw_loader_exclusive
    forward_digest = _validate_raw_panel(raw_local, q, n, fingerprint=streamed)

    def reload_raw() -> np.ndarray:
        panel = raw if raw is not None else raw_loader()
        if panel is None:
            raise ValueError("physical_spin_operators raw_loader returned None")
        digest = _validate_raw_panel(panel, q, n, fingerprint=streamed)
        if digest != forward_digest:
            raise ValueError("streamed AO Coulomb integral changed since the forward evaluation")
        return panel

    native = resolve_ext_fock_native(
        cuda=cuda,
        library=library,
        native_library=native_library,
        host_reference=host_reference,
    )
    if native is not None:
        return _physical_spin_operators_native(
            h,
            raw_local,
            whitener,
            frames,
            occupied,
            cuda,
            library,
            include_exchange=include_exchange,
            reload_raw=reload_raw,
            raw_loader_exclusive=raw_loader_exclusive,
            native=native,
        )
    if frames[0].shape[1] != frames[1].shape[1]:
        raise ValueError("TensorIR physical_spin_operators requires matching spin frame column counts")
    ext = frames[0].shape[1]
    from contextlib import ExitStack

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
    same_occ_frame = (
        occupied[0] == occupied[1]
        and occupied[0] > 0
        and np.array_equal(frames[0][:, : occupied[0]], frames[1][:, : occupied[1]])
    )
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
        if spin == 1 and same_occ_frame:
            raw_s = inputs["raw0"]
        else:
            raw_s = np.empty((q, o, n), dtype=np.float64)
            for tile_len, slices in groups:
                a_t = Index("a", IndexSpace("jk_fit", "auxiliary", tile_len))
                half_t = Program({rname: einsum("amn,ni->aim", _field("panel", a_t, m, v), O)})
                with batch_execute(run, half_t) as exec_one:
                    for rows in slices:
                        raw_s[rows] = exec_one(
                            {"panel": np.ascontiguousarray(raw_local[rows]), name: occ_mat}
                        ).outputs[rname]
        inputs[rname] = raw_s
        fit = coulomb_fit_response(q, o * n)
        solve_inputs = {
            "metric_whitener": whitener,
            "three_center": inputs[rname].reshape(q, -1),
        }
        if spin == 1 and same_occ_frame:
            inputs[dname] = inputs["fit0"]
        else:
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
                        "panel": np.ascontiguousarray(raw_local[rows]),
                    }
                ).outputs["j"]
    # Release streamed [Q, ext, ext] JK panel during intermediate MO pullbacks; reload on pullback.
    del raw_local
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
        raw_pb = reload_raw()
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
        inplace_raw = raw_loader_exclusive and raw_pb.flags.writeable
        raw_bar = None if inplace_raw else np.zeros_like(raw_pb)
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
                            "panel": np.ascontiguousarray(raw_pb[rows]),
                            "bar_j": bar_g_c,
                        }
                    ).outputs
                    bar_rho[rows] += jr["bar_rho"]
                    if raw_bar is not None:
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
        seed_by_spin: dict[int, np.ndarray] = {}
        for s, (o, occ_mat, fit, solve_inputs, name, rname, dname) in records.items():
            solved = _run(
                run,
                Program(
                    {k: fit.outputs[k] for k in ("metric_bar", "three_center_bar")}
                ),
                {**solve_inputs, "fit_bar": bars["bar_" + dname].reshape(q, -1)},
            )
            metric_bar += solved["metric_bar"]
            seed_by_spin[s] = bars["bar_" + rname] + solved["three_center_bar"].reshape(
                inputs[rname].shape
            )
            coeff_bars[s][:, :o] += bars["bar_" + name]
        if inplace_raw:
            for tile_len, slices in groups:
                a_t = Index("a", IndexSpace("jk_fit", "auxiliary", tile_len))
                with ExitStack() as stack:
                    exec_h = {}
                    for s, (o, _occ_mat, _fit, _si, name, rname, _dn) in records.items():
                        i_idx = Index("i", IndexSpace(f"occ{s}", "orbital", o))
                        O_node = _field(name, m, i_idx)
                        half_t = Program(
                            {rname: einsum("amn,ni->aim", _field("panel", a_t, m, v), O_node)}
                        )
                        half_rev = transpose_program(half_t, (rname,)).program
                        exec_h[s] = stack.enter_context(batch_execute(run, half_rev))
                    for rows in slices:
                        panel_tile = np.ascontiguousarray(raw_pb[rows])
                        bar_tile = np.einsum("a,mn->amn", rho_val[rows], bar_g_c)
                        for s, (o, occ_mat, _fit, _si, name, rname, _dn) in records.items():
                            hb = exec_h[s](
                                {
                                    "panel": panel_tile,
                                    name: occ_mat,
                                    "bar_" + rname: np.ascontiguousarray(seed_by_spin[s][rows]),
                                }
                            ).outputs
                            bar_tile += hb["bar_panel"]
                            coeff_bars[s][:, :o] += hb["bar_" + name]
                        raw_pb[rows] = bar_tile
            return coeff_bars, bar_core, raw_pb, metric_bar
        assert raw_bar is not None
        for s, (o, occ_mat, _fit, _si, name, rname, _dn) in records.items():
            seed = seed_by_spin[s]
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
                                "panel": np.ascontiguousarray(raw_pb[rows]),
                                name: occ_mat,
                                "bar_" + rname: np.ascontiguousarray(seed[rows]),
                            }
                        ).outputs
                        raw_bar[rows] += hb["bar_panel"]
                        coeff_bars[s][:, :o] += hb["bar_" + name]
        del raw_pb
        return coeff_bars, bar_core, raw_bar, metric_bar

    return matrices, pullback


def obs_sources(
    raw: np.ndarray, frames: Sequence[np.ndarray], obs: int, run: Runner
) -> dict[str, np.ndarray]:
    """Canonical OBS-only DF factors for the coupled occupied/virtual response."""
    from ._f12_spin_ao import project_ao_operator_tiled

    panel = np.ascontiguousarray(raw[:, :obs, :obs])
    c_a = np.ascontiguousarray(frames[0][:obs, :obs])
    c_b = np.ascontiguousarray(frames[1][:obs, :obs])
    proj_a = project_ao_operator_tiled(
        obs,
        obs,
        0,
        len(raw),
        "raw_coulomb",
        "ao_coulomb",
        c_a,
        panel,
        run,
    )
    proj_b = (
        proj_a
        if np.array_equal(c_a, c_b)
        else project_ao_operator_tiled(
            obs,
            obs,
            0,
            len(raw),
            "raw_coulomb",
            "ao_coulomb",
            c_b,
            panel,
            run,
        )
    )
    return {"raw_coulomb_alpha": proj_a, "raw_coulomb_beta": proj_b}
