"""Host-RAM hygiene, streaming thresholds, phase diagnostics, and streamed JK-Fock F12 pullback."""

from __future__ import annotations

import ctypes
import gc
import math
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_increment import FocalF12Increment, FocalF12Sources
    from ._f12_focal_one_electron import FocalOneElectron
    from ._f12_focal_reference import FocalReference
    from ._f12_focal_tplus_sources import SpinOccupied
    from .experimental.f12_focal import FocalRequest

# Stream dense [Q, ext, ext] AO panels whenever a single panel reaches 512 MiB
# or the 5-operator F12 panel set reaches 1.5 GiB (or Cartesian count > 128M).
SINGLE_PANEL_STREAM_BYTES = 512 << 20
F12_TOTAL_PANEL_STREAM_BYTES = 1536 << 20
AO_CART_STREAM_THRESHOLD = 128_000_000

STRICT_GPU_ONLY_ENV = "GENERATIVEQC_F12_STRICT_GPU_ONLY"

_LIBC_MALLOC_TRIM: Callable[[int], int] | None | bool = False
_ACTIVE_RAM_TRACES: ContextVar[tuple[list[dict[str, float | str]], ...]] = ContextVar(
    "_ACTIVE_RAM_TRACES", default=()
)
_ACTIVE_HYBRID_TRACES: ContextVar[tuple[list[dict[str, object]], ...]] = ContextVar(
    "_ACTIVE_HYBRID_TRACES", default=()
)


def _resolve_malloc_trim() -> Callable[[int], int] | None:
    global _LIBC_MALLOC_TRIM
    if _LIBC_MALLOC_TRIM is not False:
        return _LIBC_MALLOC_TRIM  # type: ignore[return-value]
    fn: Callable[[int], int] | None = None
    try:
        libc = ctypes.CDLL("libc.so.6")
        raw = getattr(libc, "malloc_trim", None)
        if raw is not None:
            raw.argtypes = [ctypes.c_size_t]
            raw.restype = ctypes.c_int
            fn = raw
    except OSError:
        fn = None
    _LIBC_MALLOC_TRIM = fn
    return fn


def host_memory_mib() -> dict[str, float]:
    """Read Linux process VmRSS and VmHWM in MiB from /proc/self/status."""
    rss_mib = 0.0
    hwm_mib = 0.0
    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                rss_mib = float(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                hwm_mib = float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return {"vm_rss_mib": rss_mib, "vm_hwm_mib": hwm_mib}


def release_host_memory() -> bool:
    """Collect Python cycles and trim retained glibc heap arenas back to the OS."""
    gc.collect()
    trim = _resolve_malloc_trim()
    if trim is None:
        return False
    try:
        return bool(trim(0))
    except Exception:
        return False


def should_stream_panel(
    *,
    allow_streaming: bool,
    ordinary: bool,
    coulomb_only: bool,
    n_ext: int,
    n_aux: int,
    ao_cart_total: int,
    env_stream: str | None = None,
) -> bool:
    """Decide whether a 3-center F12/JK fitting panel must stream on demand."""
    if not allow_streaming or ordinary:
        return False
    flag = os.environ.get("GENERATIVEQC_F12_STREAM_PANELS") if env_stream is None else env_stream
    if flag == "1":
        return True
    if flag == "0":
        return False
    single_panel_bytes = int(n_aux) * int(n_ext) * int(n_ext) * 8
    est_panel_bytes = (1 if coulomb_only else 5) * single_panel_bytes
    return (
        int(ao_cart_total) > AO_CART_STREAM_THRESHOLD
        or single_panel_bytes >= SINGLE_PANEL_STREAM_BYTES
        or est_panel_bytes >= F12_TOTAL_PANEL_STREAM_BYTES
    )


def strict_gpu_only_enabled() -> bool:
    """Return True when GENERATIVEQC_F12_STRICT_GPU_ONLY=1 forbids host-assisted GPU stages."""
    return os.environ.get(STRICT_GPU_ONLY_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@contextmanager
def capture_ram_trace() -> Iterator[list[dict[str, float | str]]]:
    """Capture per-phase host VmRSS/VmHWM checkpoints emitted by F12 routines.

    Uses a ContextVar so concurrent threads/tasks and interleaved exits never
    corrupt another trace's bucket.
    """
    records: list[dict[str, float | str]] = []
    _ACTIVE_RAM_TRACES.set((*_ACTIVE_RAM_TRACES.get(), records))
    try:
        yield records
    finally:
        cur = _ACTIVE_RAM_TRACES.get()
        _ACTIVE_RAM_TRACES.set(tuple(bucket for bucket in cur if bucket is not records))


@contextmanager
def capture_hybrid_phases() -> Iterator[list[dict[str, object]]]:
    """Capture structured hybrid/host-assisted phase disclosures emitted by F12 routines."""
    records: list[dict[str, object]] = []
    _ACTIVE_HYBRID_TRACES.set((*_ACTIVE_HYBRID_TRACES.get(), records))
    try:
        yield records
    finally:
        cur = _ACTIVE_HYBRID_TRACES.get()
        _ACTIVE_HYBRID_TRACES.set(tuple(bucket for bucket in cur if bucket is not records))


def disclose_host_assisted_stage(
    stage: str,
    *,
    requested_backend: str,
    detail: str,
    scientific_contraction: bool = True,
) -> None:
    """Record a host-assisted scientific stage and enforce GENERATIVEQC_F12_STRICT_GPU_ONLY=1."""
    entry: dict[str, object] = {
        "stage": stage,
        "requested_backend": requested_backend,
        "execution_mode": "hybrid_host_assisted"
        if requested_backend == "cuda"
        else "host",
        "scientific_contraction": bool(scientific_contraction),
        "detail": detail,
    }
    for bucket in _ACTIVE_HYBRID_TRACES.get():
        bucket.append(entry)
    if (
        requested_backend == "cuda"
        and scientific_contraction
        and strict_gpu_only_enabled()
    ):
        raise NotImplementedError(
            f"{STRICT_GPU_ONLY_ENV}=1 forbids host-contracted scientific stage "
            f"{stage!r} on backend='cuda': {detail}"
        )


def record_ram_phase(phase: str) -> None:
    """Record a host VmRSS/VmHWM checkpoint when tracing is enabled."""
    active = _ACTIVE_RAM_TRACES.get()
    print_trace = os.environ.get("GENERATIVEQC_F12_RAM_TRACE") == "1"
    if not active and not print_trace:
        return
    mem = host_memory_mib()
    entry: dict[str, float | str] = {
        "phase": phase,
        "vm_rss_mib": mem["vm_rss_mib"],
        "vm_hwm_mib": mem["vm_hwm_mib"],
    }
    for bucket in active:
        bucket.append(entry)
    if print_trace:
        print(
            f"[f12-ram] {phase}: VmRSS={mem['vm_rss_mib']:.1f} MiB "
            f"({mem['vm_rss_mib'] / 1024.0:.2f} GiB), "
            f"VmHWM={mem['vm_hwm_mib']:.1f} MiB ({mem['vm_hwm_mib'] / 1024.0:.2f} GiB)",
            flush=True,
        )


def _congruence_program(nao: int, ext: int) -> Program:
    from generativeqc_compiler.tensor import Index, IndexSpace, Program, TensorSpec, einsum, input_tensor

    ao_sp = IndexSpace("ao", "orbital", nao)
    mo_sp = IndexSpace("mo", "orbital", ext)
    m, n = Index("m", ao_sp), Index("n", ao_sp)
    p, q = Index("p", mo_sp), Index("q", mo_sp)
    mat = input_tensor("mat", TensorSpec((m, n), role="input", differentiable=True))
    coeff = input_tensor("coefficients", TensorSpec((m, p), role="input", differentiable=True))
    half = einsum("mn,nq->mq", mat, coeff)
    out = einsum("mp,mq->pq", coeff, half)
    return Program({"out": out})


def _j_ao_program(tile_len: int, nao: int) -> Program:
    from generativeqc_compiler.tensor import Index, IndexSpace, Program, TensorSpec, einsum, input_tensor

    a_sp = IndexSpace("aux", "auxiliary", tile_len)
    ao_sp = IndexSpace("ao", "orbital", nao)
    a = Index("a", a_sp)
    m, n = Index("m", ao_sp), Index("n", ao_sp)
    charge = input_tensor("charge", TensorSpec((a,), role="input"))
    ao = input_tensor("ao", TensorSpec((a, m, n), role="input"))
    return Program({"j_ao": einsum("a,amn->mn", charge, ao)})


def _bar_charge_program(tile_len: int, nao: int) -> Program:
    from generativeqc_compiler.tensor import Index, IndexSpace, Program, TensorSpec, einsum, input_tensor

    a_sp = IndexSpace("aux", "auxiliary", tile_len)
    ao_sp = IndexSpace("ao", "orbital", nao)
    a = Index("a", a_sp)
    m, n = Index("m", ao_sp), Index("n", ao_sp)
    ao = input_tensor("ao", TensorSpec((a, m, n), role="input"))
    m_j = input_tensor("m_j", TensorSpec((m, n), role="input"))
    return Program({"bar_charge": einsum("amn,mn->a", ao, m_j)})


def _exchange_forward_program(tile_len: int, occ: int, ext: int) -> Program:
    from generativeqc_compiler.tensor import Index, IndexSpace, Program, TensorSpec, einsum, input_tensor

    a_sp = IndexSpace("aux", "auxiliary", tile_len)
    o_sp = IndexSpace("occ", "occupied", occ)
    e_sp = IndexSpace("ext", "orbital", ext)
    a = Index("a", a_sp)
    i = Index("i", o_sp)
    p, q = Index("p", e_sp), Index("q", e_sp)
    fit = input_tensor("fit", TensorSpec((a, i, p), role="input"))
    raw = input_tensor("raw", TensorSpec((a, i, q), role="input"))
    return Program({"exchange": einsum("aip,aiq->pq", fit, raw)})


def _exchange_pullback_program(tile_len: int, occ: int, ext: int) -> Program:
    from generativeqc_compiler.tensor import Index, IndexSpace, Program, TensorSpec, einsum, input_tensor

    a_sp = IndexSpace("aux", "auxiliary", tile_len)
    o_sp = IndexSpace("occ", "occupied", occ)
    e_sp = IndexSpace("ext", "orbital", ext)
    a = Index("a", a_sp)
    i = Index("i", o_sp)
    p, q = Index("p", e_sp), Index("q", e_sp)
    fit = input_tensor("fit", TensorSpec((a, i, p), role="input"))
    raw = input_tensor("raw", TensorSpec((a, i, q), role="input"))
    b_k = input_tensor("b_k", TensorSpec((p, q), role="input"))
    return Program(
        {
            "bar_fit": einsum("aiq,pq->aip", raw, b_k),
            "bar_raw": einsum("aip,pq->aiq", fit, b_k),
        }
    )


def streamed_f12_increment(
    request: FocalRequest,
    reference: FocalReference,
    one: FocalOneElectron,
    jk: FocalF12Sources,
    f12: FocalF12Sources,
    *,
    expected_cabs_rank: int | None = None,
    relative_cutoff: float = 1e-9,
    budget_mib: int = 1536,
    run: Callable | None = None,
    library: ctypes.CDLL | None = None,
    canonical_library: ctypes.CDLL | None = None,
    occupied_weights: SpinOccupied | None = None,
    ao_bar_sink: Callable[[str, np.ndarray], None] | None = None,
    extra_obs_fock_seeds: Mapping[str, np.ndarray] | None = None,
    extra_obs_cbar: np.ndarray | None = None,
    shell_library: Path | str | ctypes.CDLL | None = None,
) -> FocalF12Increment:
    """Memory-bounded F12 increment with factorized JK-Fock forward/reverse.

    Avoids materializing full [q_jk, ext, ext] MO Coulomb projections and full
    [q_jk, ext, ext] Fock-pullback cotangents by keeping only occupied-row
    [q_jk, occ, ext] factors plus 2D [nao, nao] Coulomb matrices between phases,
    and capturing the [q_jk, obs, obs] slice during JK forward so CPHF relaxation
    never reloads the full [q_jk, ext, ext] AO panel.
    """
    from functools import partial

    from generativeqc_compiler.mp2.f12_density_fitting import coulomb_fit_response
    from generativeqc_compiler.tensor import Program, execute, transpose_program

    from ._f12_ao_mo_native import (
        resolve_ao_mo_ops,
        run_ao_mo_bilin_fwd,
        run_ao_mo_bilin_rev,
        run_coulomb_fit_fwd,
        run_coulomb_fit_rev,
    )
    from ._f12_ext_fock_native import resolve_ext_fock_native
    from ._f12_focal_frame import canonical_pair_frame
    from ._f12_focal_increment import FocalF12Increment, relaxed_f12_source_weights
    from ._f12_focal_tplus_sources import checked_occupied_weights
    from ._f12_focal_reference import partition_focal_reference
    from ._f12_frozen_core import resolve_frozen_occupied
    from ._f12_mp2_pair_native import (
        bind_mp2_f12_pair_response,
        run_mp2_f12_pair_response_native,
    )
    from ._f12_spin_ao import (
        _orbital_sources,
        project_ao_operator_pullback_tiled,
        project_ao_operator_tiled,
    )
    from ._f12_spin_fock import _tile_groups, default_auxiliary_batch_size
    from ._f12_spin_pairs import (
        _F12_METRIC_OPS,
        _is_rhf_spatial_identity_pair_data,
        _shared_contiguous_operands,
        batch_execute,
        is_spin_symmetric_pair_data,
        occupied_pair_blocks,
        pair_operands,
        pullback_operator_residuals,
    )
    from .elements import checked_integer

    cuda = request.backend == "cuda"
    if (
        request.role != "mp2_f12_correlation"
        or request.root_id not in ("rhf", "uhf")
        or request.backend not in ("cuda", "cpu_reference")
        or (cuda and run is None)
    ):
        raise ValueError(
            "selected F12 requires a canonical RHF or UHF pair request with its backend runner"
        )
    budget = checked_integer(
        budget_mib, "F12 increment budget in MiB", low=1, high=(2**63 - 1) >> 20
    )
    executor = run if run is not None else partial(execute, max_bytes=budget << 20)

    def checked(g: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        out = executor(g, inputs)
        if out.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError(
                "F12 electronic graph differs from its requested backend"
            )
        return out

    if hasattr(executor, "batch"):

        @contextmanager
        def checked_batch(g: Program):
            with executor.batch(g) as exec_one:

                def wrapped(inputs: Mapping[str, np.ndarray]) -> Execution:
                    out = exec_one(inputs)
                    if out.backend != (
                        "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
                    ):
                        raise RuntimeError(
                            "F12 electronic graph differs from its requested backend"
                        )
                    return out

                yield wrapped

        checked.batch = checked_batch  # type: ignore[attr-defined]

    reference = partition_focal_reference(request, reference, budget_mib)
    space, ao, panel, proj = canonical_pair_frame(
        request,
        reference,
        one,
        jk,
        f12,
        expected_cabs_rank=expected_cabs_rank,
        relative_cutoff=relative_cutoff,
        library=library,
    )
    oa, ob, obs, ext, _ = space
    nao = len(one.overlap)
    qjk = len(jk.whitener)
    frozen = resolve_frozen_occupied(request, (oa, ob))
    fa, fb = frozen
    f12_weights_full = None
    active_weights: tuple[np.ndarray, np.ndarray] | None = None
    if occupied_weights is not None:
        wa_in, wb_in = occupied_weights
        wa_arr = np.asarray(wa_in, dtype=np.float64)
        wb_arr = np.asarray(wb_in, dtype=np.float64)
        if fa or fb:
            if len(wa_arr) == oa - fa and len(wb_arr) == ob - fb:
                f12_weights_full = (
                    np.concatenate((np.zeros(fa), wa_arr)),
                    np.concatenate((np.zeros(fb), wb_arr)),
                )
            else:
                f12_weights_full = (wa_arr, wb_arr)
        else:
            f12_weights_full = (wa_arr, wb_arr)
        checked_occupied_weights(f12_weights_full, (oa, ob))
        active_weights = (
            np.ascontiguousarray(f12_weights_full[0][fa:], dtype=np.float64),
            np.ascontiguousarray(f12_weights_full[1][fb:], dtype=np.float64),
        )

    resolved_shell_lib = shell_library
    if resolved_shell_lib is None:
        for container in (ao, panel):
            loader = container.get("__ao_loader__") if isinstance(container, Mapping) else None
            owner_lib = getattr(getattr(loader, "__self__", None), "library", None)
            if owner_lib is not None:
                resolved_shell_lib = owner_lib
                break
    aomo_ops = resolve_ao_mo_ops(
        checked, ao, shell_library=resolved_shell_lib, host_reference=not cuda
    )
    ext_fock_ops = resolve_ext_fock_native(
        cuda=cuda,
        library=library,
        native_library=resolved_shell_lib,
        host_reference=not cuda,
    )
    pair_resp_fn = bind_mp2_f12_pair_response(
        host_reference=not cuda,
        shell_library=resolved_shell_lib,
    )

    # --- Phase 1: Factorized JKFit extended-DF-Fock forward ---
    # Retains only occupied-row [qjk, occ, ext/nao] factors + 2D [nao, nao] matrices,
    # never storing [qjk, ext, ext] MO tensors.
    c_alpha = np.ascontiguousarray(ao["coefficients_alpha"])
    c_beta = np.ascontiguousarray(ao["coefficients_beta"])
    same_spin = oa == ob and np.array_equal(c_alpha, c_beta)
    occupied = {"alpha": oa, "beta": ob}
    coeffs_by_spin = {"alpha": c_alpha, "beta": c_beta}
    jk_whitener = np.ascontiguousarray(panel["metric_whitener"])
    jk_streaming = "__ao_loader__" in panel
    ao_coul_jk = (
        panel["__ao_loader__"]("ao_coulomb") if jk_streaming else panel["ao_coulomb"]
    )
    if jk_streaming and "__obs_coulomb__" not in panel:
        panel["__obs_coulomb__"] = ao_coul_jk[:, :obs, :obs].copy()

    hcore_ao = np.ascontiguousarray(ao["hcore_ao"])
    fock: dict[str, np.ndarray] = {}
    used_ext_fock_aot = ext_fock_ops is not None and aomo_ops is not None
    ext_tile_q = max(1, min(qjk, (32 << 20) // max(1, nao * nao * 8)))

    # Variables populated by either the AOT ext-fock branch or the TensorIR fallback.
    occ_a = occ_b = None
    R_a = R_b = D_a = D_b = None
    rho_val: np.ndarray | None = None
    primal_fock_a = primal_fock_b = primal_exch_a = primal_exch_b = None
    jk_occ_raw: dict[str, np.ndarray] = {}
    jk_fit: dict[str, np.ndarray] = {}
    charge = np.zeros(qjk, dtype=np.float64)
    j_ao = np.zeros((nao, nao), dtype=np.float64)
    ao_width = default_auxiliary_batch_size(qjk, nao)
    cong_fwd = None

    if used_ext_fock_aot:
        assert ext_fock_ops is not None and aomo_ops is not None
        same_occ_frame = (
            oa == ob and oa > 0 and np.array_equal(c_alpha[:, :oa], c_beta[:, :ob])
        )
        occ_a = np.ascontiguousarray(c_alpha[:, :oa], dtype=np.float64) if oa > 0 else None
        occ_b = (
            occ_a
            if same_occ_frame
            else (np.ascontiguousarray(c_beta[:, :ob], dtype=np.float64) if ob > 0 else None)
        )
        jk_contig_f64 = ao_coul_jk.flags.c_contiguous and ao_coul_jk.dtype == np.float64

        def _ext_half_fwd(occ_mat: np.ndarray) -> np.ndarray:
            assert ext_fock_ops is not None
            if jk_contig_f64:
                return ext_fock_ops.half_fwd(ao_coul_jk, occ_mat)
            out = np.empty((qjk, occ_mat.shape[1], nao), dtype=np.float64)
            for q0 in range(0, qjk, ext_tile_q):
                q1 = min(qjk, q0 + ext_tile_q)
                out[q0:q1] = ext_fock_ops.half_fwd(ao_coul_jk[q0:q1], occ_mat)
            return out

        R_a = _ext_half_fwd(occ_a) if oa > 0 and occ_a is not None else None
        if ob == 0 or occ_b is None:
            R_b = None
        elif same_occ_frame:
            R_b = R_a
        else:
            R_b = _ext_half_fwd(occ_b)

        if oa == 0 and ob == 0:
            D_a = D_b = None
            K_a = np.zeros((nao, nao), dtype=np.float64)
            K_b = np.zeros((nao, nao), dtype=np.float64)
            rho_val = np.zeros(qjk, dtype=np.float64)
        else:
            D_a, D_b, K_a, K_b, rho_val = ext_fock_ops.fwd(
                jk_whitener,
                occ_a if oa > 0 else np.empty((nao, 0), dtype=np.float64),
                occ_b if ob > 0 else np.empty((nao, 0), dtype=np.float64),
                R_a,
                R_b,
                same_occ_frame=same_occ_frame,
            )
            if jk_contig_f64:
                ext_fock_ops.j_fwd(ao_coul_jk, rho_val, j_ao)
            else:
                for q0 in range(0, qjk, ext_tile_q):
                    q1 = min(qjk, q0 + ext_tile_q)
                    ext_fock_ops.j_fwd(ao_coul_jk[q0:q1], rho_val[q0:q1], j_ao)

        if jk_streaming:
            del ao_coul_jk
            release_host_memory()
        else:
            del ao_coul_jk

        g_ao = hcore_ao + j_ao
        primal_fock_a = np.ascontiguousarray(
            0.5 * ((g_ao - K_a) + (g_ao - K_a).T)
            if oa > 0
            else 0.5 * (g_ao + g_ao.T)
        )
        primal_exch_a = np.ascontiguousarray(
            0.5 * (K_a + K_a.T) if oa > 0 else np.zeros((nao, nao), dtype=np.float64)
        )
        if same_spin:
            primal_fock_b = primal_fock_a
            primal_exch_b = primal_exch_a
        else:
            primal_fock_b = np.ascontiguousarray(
                0.5 * ((g_ao - K_b) + (g_ao - K_b).T)
                if ob > 0
                else 0.5 * (g_ao + g_ao.T)
            )
            primal_exch_b = np.ascontiguousarray(
                0.5 * (K_b + K_b.T) if ob > 0 else np.zeros((nao, nao), dtype=np.float64)
            )
        fock["fock_alpha"] = run_ao_mo_bilin_fwd(aomo_ops, primal_fock_a, c_alpha, c_alpha)
        fock["exchange_alpha"] = run_ao_mo_bilin_fwd(aomo_ops, primal_exch_a, c_alpha, c_alpha)
        if same_spin:
            fock["fock_beta"] = fock["fock_alpha"]
            fock["exchange_beta"] = fock["exchange_alpha"]
        else:
            fock["fock_beta"] = run_ao_mo_bilin_fwd(aomo_ops, primal_fock_b, c_beta, c_beta)
            fock["exchange_beta"] = run_ao_mo_bilin_fwd(aomo_ops, primal_exch_b, c_beta, c_beta)
        del K_a, K_b, g_ao
    else:
        for spin, occ in occupied.items():
            if same_spin and spin == "beta":
                jk_occ_raw["beta"] = jk_occ_raw["alpha"]
                jk_fit["beta"] = jk_fit["alpha"]
                if occ:
                    charge *= 2.0
                continue
            if occ == 0:
                jk_occ_raw[spin] = np.empty((qjk, 0, ext), dtype=np.float64)
                jk_fit[spin] = np.empty((qjk, 0, ext), dtype=np.float64)
                continue
            occ_raw = project_ao_operator_tiled(
                nao, ext, occ, qjk, "raw_f12", "ao_f12", coeffs_by_spin[spin], ao_coul_jk, checked
            )
            jk_occ_raw[spin] = occ_raw
            if aomo_ops is not None:
                fit = run_coulomb_fit_fwd(
                    aomo_ops,
                    jk_whitener,
                    np.ascontiguousarray(occ_raw).reshape(qjk, occ * ext),
                ).reshape(qjk, occ, ext)
            else:
                solve = coulomb_fit_response(qjk, occ * ext)
                fit = checked(
                    Program({"fit": solve.outputs["fit"]}),
                    {
                        "metric_whitener": jk_whitener,
                        "three_center": np.ascontiguousarray(occ_raw).reshape(qjk, occ * ext),
                    },
                ).outputs["fit"].reshape(qjk, occ, ext)
            jk_fit[spin] = fit
            charge += np.einsum("aii->a", fit[:, :occ, :occ])

        for tile_len, slices in _tile_groups(qjk, ao_width):
            j_prog = _j_ao_program(tile_len, nao)
            with batch_execute(checked, j_prog) as exec_j:
                for rows in slices:
                    j_ao += exec_j(
                        {
                            "charge": np.ascontiguousarray(charge[rows]),
                            "ao": np.ascontiguousarray(ao_coul_jk[rows]),
                        }
                    ).outputs["j_ao"]

        if jk_streaming:
            del ao_coul_jk
            release_host_memory()
        else:
            del ao_coul_jk

        cong_fwd = _congruence_program(nao, ext) if aomo_ops is None else None
        for spin, occ in occupied.items():
            if same_spin and spin == "beta":
                fock["fock_beta"] = fock["fock_alpha"]
                fock["exchange_beta"] = fock["exchange_alpha"]
                continue
            c_s = coeffs_by_spin[spin]
            if aomo_ops is not None:
                h_s = run_ao_mo_bilin_fwd(aomo_ops, hcore_ao, c_s, c_s)
                j_s = run_ao_mo_bilin_fwd(aomo_ops, j_ao, c_s, c_s)
            else:
                assert cong_fwd is not None
                h_s = checked(cong_fwd, {"mat": hcore_ao, "coefficients": c_s}).outputs["out"]
                j_s = checked(cong_fwd, {"mat": j_ao, "coefficients": c_s}).outputs["out"]
            ex_s = np.zeros((ext, ext), dtype=np.float64)
            if occ:
                ex_width = default_auxiliary_batch_size(qjk, occ, ext)
                for tile_len, slices in _tile_groups(qjk, ex_width):
                    ex_prog = _exchange_forward_program(tile_len, occ, ext)
                    with batch_execute(checked, ex_prog) as exec_ex:
                        for rows in slices:
                            ex_s += exec_ex(
                                {
                                    "fit": np.ascontiguousarray(jk_fit[spin][rows]),
                                    "raw": np.ascontiguousarray(jk_occ_raw[spin][rows]),
                                }
                            ).outputs["exchange"]
            fock["exchange_" + spin] = ex_s
            fock["fock_" + spin] = h_s + j_s - ex_s
    record_ram_phase("after_jk_forward")

    # --- Phase 2: Streamed F12 pair projection ---
    data, _ = _orbital_sources(
        space, ao, checked, None, projected=None, role="pairs", occupied_coulomb=True
    )
    release_host_memory()
    record_ram_phase("after_f12_forward")

    # --- Phase 3: Pair energy + pullback without allocating [qjk, ext, ext] ---
    sources = {
        key.removeprefix("source/"): v
        for key, v in data.items()
        if key.startswith("source/")
    }
    spin_sym = is_spin_symmetric_pair_data(
        space, data, sources, fock, frozen, f12_weights_full
    )
    if pair_resp_fn is not None:
        use_rhf = _is_rhf_spatial_identity_pair_data(
            space, data, sources, fock, frozen, active_weights
        )
        f12_spin, _ord_spin, count, source_bars, raw_bars, fock_bars, _, _ = (
            run_mp2_f12_pair_response_native(
                pair_resp_fn,
                space=space,
                frozen_occupied=frozen,
                rhf_spatial=use_rhf,
                include_f12=True,
                include_ordinary=False,
                budget_bytes=budget << 20,
                data=data,
                sources=sources,
                fock=fock,
                occ_w_a=None if active_weights is None else active_weights[0],
                occ_w_b=None if active_weights is None else active_weights[1],
            )
        )
        f12_energy = math.fsum(f12_spin)
        if not math.isfinite(f12_energy):
            raise ValueError("Nonfinite unrestricted MP2-F12 energy")
        raw_bars.update({"source/" + k: v for k, v in source_bars.items()})
        del source_bars, sources
    else:
        disclose_host_assisted_stage(
            "streamed_f12_pair_residuals",
            requested_backend=request.backend,
            detail="NumPy host matrix-multiply pair residuals and pullback in streamed_f12_increment fallback",
        )
        source_bars = {k: np.zeros_like(v) for k, v in sources.items()}
        raw_bars = {
            k: np.zeros_like(v)
            for k, v in data.items()
            if k == "cross_overlap" or k.startswith(("raw/", "metric/"))
        }
        fock_bars = {k: np.zeros_like(v) for k, v in fock.items()}
        blocks = occupied_pair_blocks(
            space, frozen_occupied=frozen, prefactored_metric=True
        )
        residuals: dict[str, np.ndarray] = {}
        for op, _ in _F12_METRIC_OPS:
            metric = data["metric/" + op]
            qf_m = metric.shape[0]
            for spin in ("alpha", "beta"):
                raw_key = "raw/" + op + "/" + spin
                fit_key = "fit_" + spin
                if raw_key in data and fit_key in sources:
                    if spin_sym and spin == "beta" and ("res/" + op + "/alpha") in residuals:
                        residuals["res/" + op + "/beta"] = residuals["res/" + op + "/alpha"]
                        continue
                    raw = data[raw_key]
                    fit = sources[fit_key]
                    residuals["res/" + op + "/" + spin] = raw - (
                        metric @ fit.reshape(qf_m, -1)
                    ).reshape(fit.shape)
        res_bars = {k: np.zeros_like(v) for k, v in residuals.items()}

        f12_terms: list[float] = []
        count = 0
        selected = ("pair_energy",)
        for sl, sr, nl, nr, graph, pairs in blocks:
            if spin_sym and sl == "beta" and sr == "beta":
                continue
            primal_graph = Program({name: graph.outputs[name] for name in selected})
            rev = transpose_program(
                primal_graph, selected, max_elements=max(1_000_000, ext * ext)
            ).program
            kernel = Program({**rev.outputs, **primal_graph.outputs})
            shared = _shared_contiguous_operands(data, fock, sl, sr, include_metrics=False)
            mirror_bb = spin_sym and sl == "alpha" and sr == "alpha"
            mirror_ab_block = spin_sym and sl == "alpha" and sr == "beta"
            with batch_execute(checked, kernel) as exec_one:
                for i, j in pairs:
                    if mirror_ab_block and i > j:
                        continue
                    mirror_ab = mirror_ab_block and i < j
                    primal = pair_operands(
                        data, sources, fock, sl, sr, nl, nr, ext, i, j, residuals=residuals
                    )
                    feeds = dict(shared)
                    for k, v in primal.items():
                        if k not in shared:
                            feeds[k] = np.ascontiguousarray(v)
                    weight = (
                        1.0
                        if f12_weights_full is None
                        else (
                            f12_weights_full[sl == "beta"][i]
                            + f12_weights_full[sr == "beta"][j]
                        )
                        * 0.5
                    )
                    feeds["bar_pair_energy"] = np.array(weight)
                    result = exec_one(feeds).outputs
                    doubled = mirror_bb or mirror_ab
                    count += 2 if doubled else 1
                    val = float(result["pair_energy"]) * weight
                    f12_terms.append(val)
                    if doubled:
                        f12_terms.append(val)

                    buffers = pair_operands(
                        raw_bars, source_bars, fock_bars, sl, sr, nl, nr, ext, i, j,
                        residuals=res_bars,
                    )
                    for key in rev.outputs:
                        buffers[key.removeprefix("bar_")] += result[key]
                    if mirror_bb:
                        bb_buffers = pair_operands(
                            raw_bars, source_bars, fock_bars, "beta", "beta", nl, nr, ext, i, j,
                            residuals=res_bars,
                        )
                        for key in rev.outputs:
                            bb_buffers[key.removeprefix("bar_")] += result[key]
                    elif mirror_ab:
                        if "bar_fock_alpha" in result:
                            fock_bars["fock_beta"] += result["bar_fock_alpha"]
                        if "bar_fock_beta" in result:
                            fock_bars["fock_alpha"] += result["bar_fock_beta"]
                        if "bar_exchange_alpha" in result:
                            fock_bars["exchange_beta"] += result["bar_exchange_alpha"]
                        if "bar_exchange_beta" in result:
                            fock_bars["exchange_alpha"] += result["bar_exchange_beta"]
                        if "bar_cross_overlap" in result:
                            raw_bars["cross_overlap"] += result["bar_cross_overlap"].T
                        if "bar_fit_i_alpha" in result:
                            source_bars["fit_beta"][:, i] += result["bar_fit_i_alpha"]
                        if "bar_fit_j_beta" in result:
                            source_bars["fit_alpha"][:, j] += result["bar_fit_j_beta"]
                        if "bar_raw_coulomb_j_beta" in result:
                            source_bars["raw_coulomb_alpha"][:, j] += result["bar_raw_coulomb_j_beta"]
                        for op, _ in _F12_METRIC_OPS:
                            rk = "bar_raw_" + op + "_i_alpha"
                            if rk in result:
                                raw_bars["raw/" + op + "/beta"][:, i] += result[rk]
                            sk = "bar_res_" + op + "_j_beta"
                            if sk in result:
                                res_bars["res/" + op + "/alpha"][:, j] += result[sk]

        f12_energy = math.fsum(f12_terms)
        if not math.isfinite(f12_energy):
            raise ValueError("Nonfinite unrestricted MP2-F12 energy")
        pullback_operator_residuals(data, sources, res_bars, raw_bars, source_bars)
        del residuals, res_bars
        raw_bars.update({"source/" + k: v for k, v in source_bars.items()})
        del source_bars, sources

    # On exact spin-symmetric RHF, remove sub-1e-13 rounding asymmetry between alpha
    # and beta cotangents so _orbital_sources reverse shares alpha/beta cleanly.
    if spin_sym:
        for base in ("source/raw_coulomb_", "source/fit_"):
            ka, kb = base + "alpha", base + "beta"
            if ka in raw_bars and kb in raw_bars and np.allclose(raw_bars[ka], raw_bars[kb], rtol=1e-12, atol=1e-13):
                avg = 0.5 * (raw_bars[ka] + raw_bars[kb])
                raw_bars[ka] = avg
                raw_bars[kb] = avg
        for op, _ in _F12_METRIC_OPS:
            ka, kb = f"raw/{op}/alpha", f"raw/{op}/beta"
            if ka in raw_bars and kb in raw_bars and np.allclose(raw_bars[ka], raw_bars[kb], rtol=1e-12, atol=1e-13):
                avg = 0.5 * (raw_bars[ka] + raw_bars[kb])
                raw_bars[ka] = avg
                raw_bars[kb] = avg

    release_host_memory()
    record_ram_phase("after_pair_loop")

    # --- Phase 4: Streamed F12 reverse directly into ao_bar_sink ---
    # Run the operator loop explicitly so each operator's ao_mat is freed and trimmed
    # before allocating bar_ao_mat, and bar_ao_mat is freed with zero residual alias
    # before loading the next operator (eliminating 2x [Q, ext, ext] coexistence).
    from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_cross_frame_overlap

    from ._f12_spin_ao import _OPERATORS

    qf = len(f12.whitener)
    f12_whitener = np.ascontiguousarray(ao["metric_whitener"])
    f12_loader = ao.get("__ao_loader__")
    bars: dict[str, np.ndarray] = {
        "overlap_ao": np.zeros_like(ao["overlap_ao"]),
        "hcore_ao": np.zeros_like(ao["hcore_ao"]),
        "coefficients_alpha": np.zeros_like(c_alpha),
        "coefficients_beta": np.zeros_like(c_beta),
        "metric/coulomb": np.zeros_like(ao["metric/coulomb"]),
    }
    for op in ("geminal", "squared_geminal", "geminal_coulomb", "double_commutator"):
        mkey = "metric/" + op
        bars[mkey] = np.array(raw_bars[mkey], dtype=np.float64, copy=True)

    bar_cross_val = raw_bars.pop("cross_overlap")
    if aomo_ops is not None:
        run_ao_mo_bilin_rev(
            aomo_ops,
            ao["overlap_ao"],
            c_alpha,
            c_beta,
            bar_cross_val,
            bar_a=bars["overlap_ao"],
            bar_c_left=bars["coefficients_alpha"],
            bar_c_right=bars["coefficients_beta"],
        )
    else:
        cross_prog = unrestricted_cross_frame_overlap(nao, ext)
        cross_rev = transpose_program(cross_prog, ("cross_overlap",)).program
        cross_out = checked(
            cross_rev,
            {
                "overlap_ao": ao["overlap_ao"],
                "coefficients_alpha": c_alpha,
                "coefficients_beta": c_beta,
                "bar_cross_overlap": bar_cross_val,
            },
        ).outputs
        for name, value in cross_out.items():
            bars[name.removeprefix("bar_")] += value
        del cross_out
    del bar_cross_val

    for op in ("coulomb", "geminal", "squared_geminal", "geminal_coulomb", "double_commutator"):
        field = _OPERATORS[op]
        ao_name = "ao_" + field.removeprefix("raw_")
        use_occ_coul = op == "coulomb"
        graph_field = "raw_f12" if use_occ_coul else field
        graph_ao_name = "ao_f12" if use_occ_coul else ao_name

        def _f12_op_key(spin_name: str, _op: str = op, _field: str = field) -> str:
            return (
                "source/" + _field + "_" + spin_name
                if _op == "coulomb"
                else "raw/" + _op + "/" + spin_name
            )

        ka, kb = _f12_op_key("alpha"), _f12_op_key("beta")
        can_share_op = (
            same_spin
            and (raw_bars[ka] is raw_bars[kb] or np.array_equal(raw_bars[ka], raw_bars[kb]))
            and (
                op != "coulomb"
                or oa == 0
                or (
                    raw_bars["source/fit_alpha"] is raw_bars["source/fit_beta"]
                    or np.array_equal(raw_bars["source/fit_alpha"], raw_bars["source/fit_beta"])
                )
            )
        )
        op_rev_spins = (("alpha", oa),) if can_share_op else (("alpha", oa), ("beta", ob))
        op_seed_scale = 2.0 if can_share_op else 1.0

        tc_bars: dict[str, np.ndarray | None] = {}
        for spin, occ in op_rev_spins:
            if op == "coulomb" and occ:
                if aomo_ops is not None:
                    tc_bars[spin] = run_coulomb_fit_rev(
                        aomo_ops,
                        f12_whitener,
                        np.ascontiguousarray(data[_f12_op_key(spin)][:, :occ]).reshape(qf, -1),
                        np.ascontiguousarray(raw_bars["source/fit_" + spin]).reshape(qf, -1),
                        bars["metric/coulomb"],
                        metric_scale=op_seed_scale,
                    ).reshape(qf, occ, ext)
                else:
                    solve = coulomb_fit_response(qf, occ * ext)
                    solve_rev = Program(
                        {k: solve.outputs[k] for k in ("three_center_bar", "metric_bar")}
                    )
                    solved = checked(
                        solve_rev,
                        {
                            "metric_whitener": f12_whitener,
                            "three_center": np.ascontiguousarray(data[_f12_op_key(spin)][:, :occ]).reshape(qf, -1),
                            "fit_bar": np.ascontiguousarray(raw_bars["source/fit_" + spin]).reshape(qf, -1),
                        },
                    ).outputs
                    tc_bars[spin] = solved["three_center_bar"].reshape(qf, occ, ext)
                    bars["metric/coulomb"] += op_seed_scale * solved["metric_bar"]
                    del solved
            else:
                tc_bars[spin] = None

        stream_this = f12_loader is not None and ao_name not in ao
        ao_mat = f12_loader(ao_name) if stream_this else ao[ao_name]

        # Phase 4a: compute bar_coefficients using ao_mat, then free ao_mat before bar_ao_mat
        for spin, occ in op_rev_spins:
            if occ == 0:
                continue
            key = _f12_op_key(spin)
            bar_c_target = np.zeros_like(c_alpha) if can_share_op else bars["coefficients_" + spin]
            project_ao_operator_pullback_tiled(
                nao,
                ext,
                occ,
                qf,
                graph_field,
                graph_ao_name,
                coeffs_by_spin[spin],
                ao_mat,
                raw_bars[key],
                checked,
                bar_coefficients=bar_c_target,
                bar_ao=None,
                seed_scale=op_seed_scale,
                three_center_bar=tc_bars[spin],
                inputs_mode="coefficients",
            )
            if can_share_op:
                bars["coefficients_alpha"] += 0.5 * bar_c_target
                bars["coefficients_beta"] += 0.5 * bar_c_target
        del ao_mat
        if stream_this:
            release_host_memory()

        # Phase 4b: allocate bar_ao_mat cleanly (ao_mat already freed), contract via sink, then free
        bar_ao_mat = np.zeros((qf, nao, nao), dtype=np.float64)
        for spin, occ in op_rev_spins:
            if occ == 0:
                continue
            key = _f12_op_key(spin)
            project_ao_operator_pullback_tiled(
                nao,
                ext,
                occ,
                qf,
                graph_field,
                graph_ao_name,
                coeffs_by_spin[spin],
                None,
                raw_bars[key],
                checked,
                bar_coefficients=None,
                bar_ao=bar_ao_mat,
                seed_scale=op_seed_scale,
                three_center_bar=tc_bars[spin],
                inputs_mode="ao",
            )
        del tc_bars
        # Drop finished operator's MO tensors immediately to lower peak during remaining ops
        data.pop(ka, None)
        data.pop(kb, None)
        raw_bars.pop(ka, None)
        raw_bars.pop(kb, None)
        if op == "coulomb":
            data.pop("source/fit_alpha", None)
            data.pop("source/fit_beta", None)
            raw_bars.pop("source/fit_alpha", None)
            raw_bars.pop("source/fit_beta", None)

        if ao_bar_sink is not None:
            ao_bar_sink(ao_name, bar_ao_mat)
            del bar_ao_mat
            release_host_memory()
        else:
            bars[ao_name] = bar_ao_mat
            del bar_ao_mat

    del data, raw_bars
    release_host_memory()
    record_ram_phase("after_f12_reverse")

    # --- Phase 5a: Factorized JKFit reverse (coefficients + metric + occ seeds; defer [qjk, nao, nao]) ---
    bar_R_a = bar_R_b = None
    bar_g: np.ndarray | None = None
    m_j_c: np.ndarray | None = None
    rev_spins: tuple[tuple[str, int], ...] = ()
    spin_mult = 1.0
    total_bar_occ_raw: dict[str, np.ndarray] = {}

    if used_ext_fock_aot:
        assert ext_fock_ops is not None
        assert (
            primal_fock_a is not None
            and primal_fock_b is not None
            and primal_exch_a is not None
            and primal_exch_b is not None
            and rho_val is not None
        )
        jk_bar_c = {
            "alpha": np.zeros_like(c_alpha),
            "beta": np.zeros_like(c_beta),
        }
        ao_fock_a = ext_fock_ops.congruence_rev(
            c_alpha, primal_fock_a, fock_bars["fock_alpha"], jk_bar_c["alpha"]
        )
        ao_exch_a = ext_fock_ops.congruence_rev(
            c_alpha, primal_exch_a, fock_bars["exchange_alpha"], jk_bar_c["alpha"]
        )
        ao_fock_b = ext_fock_ops.congruence_rev(
            c_beta, primal_fock_b, fock_bars["fock_beta"], jk_bar_c["beta"]
        )
        ao_exch_b = ext_fock_ops.congruence_rev(
            c_beta, primal_exch_b, fock_bars["exchange_beta"], jk_bar_c["beta"]
        )
        bar_g = np.ascontiguousarray(ao_fock_a + ao_fock_b, dtype=np.float64)
        jk_bar_hcore = bar_g.copy()
        k_seed_a = (-ao_fock_a + ao_exch_a) if oa > 0 else None
        k_seed_b = (-ao_fock_b + ao_exch_b) if ob > 0 else None

        ao_coul_jk = (
            panel["__ao_loader__"]("ao_coulomb") if jk_streaming else panel["ao_coulomb"]
        )
        jk_pb_contig = ao_coul_jk.flags.c_contiguous and ao_coul_jk.dtype == np.float64
        if jk_pb_contig:
            bar_rho = ext_fock_ops.j_rev(ao_coul_jk, bar_g)
        else:
            bar_rho = np.empty(qjk, dtype=np.float64)
            for q0 in range(0, qjk, ext_tile_q):
                q1 = min(qjk, q0 + ext_tile_q)
                bar_rho[q0:q1] = ext_fock_ops.j_rev(ao_coul_jk[q0:q1], bar_g)

        if oa == 0 and ob == 0:
            bar_R_a = bar_R_b = None
            jk_bar_metric = np.zeros((qjk, qjk), dtype=np.float64)
        else:
            bar_occ_a = np.zeros((nao, oa), dtype=np.float64) if oa > 0 else None
            bar_occ_b = np.zeros((nao, ob), dtype=np.float64) if ob > 0 else None
            bar_R_a, bar_R_b, jk_bar_metric = ext_fock_ops.rev(
                jk_whitener,
                nao,
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
            tmp_bar_raw = np.empty((min(ext_tile_q, qjk), nao, nao), dtype=np.float64)
            for q0 in range(0, qjk, ext_tile_q):
                q1 = min(qjk, q0 + ext_tile_q)
                sub = q1 - q0
                ext_fock_ops.half_rev(
                    ao_coul_jk[q0:q1],
                    occ_a,
                    occ_b,
                    bar_R_a[q0:q1] if bar_R_a is not None else None,
                    bar_R_b[q0:q1] if bar_R_b is not None else None,
                    tmp_bar_raw[:sub],
                    bar_occ_a,
                    bar_occ_b,
                )
            del tmp_bar_raw
            if bar_occ_a is not None:
                jk_bar_c["alpha"][:, :oa] += bar_occ_a
            if bar_occ_b is not None:
                jk_bar_c["beta"][:, :ob] += bar_occ_b
            del bar_occ_a, bar_occ_b
        del ao_coul_jk, R_a, R_b, D_a, D_b, fock_bars, fock
    else:
        b_h = {"alpha": fock_bars["fock_alpha"], "beta": fock_bars["fock_beta"]}
        b_j = {"alpha": fock_bars["fock_alpha"], "beta": fock_bars["fock_beta"]}
        b_k = {
            "alpha": fock_bars["exchange_alpha"] - fock_bars["fock_alpha"],
            "beta": fock_bars["exchange_beta"] - fock_bars["fock_beta"],
        }
        same_spin_jk_rev = (
            same_spin
            and np.allclose(b_h["alpha"], b_h["beta"], rtol=1e-12, atol=1e-13)
            and np.allclose(b_k["alpha"], b_k["beta"], rtol=1e-12, atol=1e-13)
        )
        if same_spin_jk_rev:
            avg_h = 0.5 * (b_h["alpha"] + b_h["beta"])
            avg_k = 0.5 * (b_k["alpha"] + b_k["beta"])
            b_h["alpha"] = b_h["beta"] = avg_h
            b_j["alpha"] = b_j["beta"] = avg_h
            b_k["alpha"] = b_k["beta"] = avg_k

        cong_rev = (
            transpose_program(cong_fwd, ("out",)).program if aomo_ops is None else None
        )
        m_j = np.zeros((nao, nao), dtype=np.float64)
        jk_bar_hcore = np.zeros((nao, nao), dtype=np.float64)
        jk_bar_c = {
            "alpha": np.zeros_like(c_alpha),
            "beta": np.zeros_like(c_beta),
        }
        rev_spins = (("alpha", oa),) if same_spin_jk_rev else (("alpha", oa), ("beta", ob))
        spin_mult = 2.0 if same_spin_jk_rev else 1.0
        for spin, _ in rev_spins:
            c_s = coeffs_by_spin[spin]
            if aomo_ops is not None:
                bar_h_s = np.zeros((nao, nao), dtype=np.float64)
                bar_j_s = np.zeros((nao, nao), dtype=np.float64)
                c_delta = np.zeros_like(c_s)
                run_ao_mo_bilin_rev(
                    aomo_ops,
                    hcore_ao,
                    c_s,
                    c_s,
                    np.ascontiguousarray(b_h[spin]),
                    bar_a=bar_h_s,
                    bar_c_left=c_delta,
                    bar_c_right=c_delta,
                )
                run_ao_mo_bilin_rev(
                    aomo_ops,
                    j_ao,
                    c_s,
                    c_s,
                    np.ascontiguousarray(b_j[spin]),
                    bar_a=bar_j_s,
                    bar_c_left=c_delta,
                    bar_c_right=c_delta,
                )
                jk_bar_hcore += spin_mult * bar_h_s
                m_j += spin_mult * bar_j_s
            else:
                assert cong_rev is not None
                h_rev = checked(
                    cong_rev,
                    {
                        "mat": hcore_ao,
                        "coefficients": c_s,
                        "bar_out": np.ascontiguousarray(b_h[spin]),
                    },
                ).outputs
                jk_bar_hcore += spin_mult * h_rev["bar_mat"]
                j_rev = checked(
                    cong_rev,
                    {
                        "mat": j_ao,
                        "coefficients": c_s,
                        "bar_out": np.ascontiguousarray(b_j[spin]),
                    },
                ).outputs
                m_j += spin_mult * j_rev["bar_mat"]
                c_delta = h_rev["bar_coefficients"] + j_rev["bar_coefficients"]
            if same_spin_jk_rev:
                jk_bar_c["alpha"] += c_delta
                jk_bar_c["beta"] += c_delta
            else:
                jk_bar_c[spin] += c_delta

        ao_coul_jk = (
            panel["__ao_loader__"]("ao_coulomb") if jk_streaming else panel["ao_coulomb"]
        )
        bar_charge = np.zeros(qjk, dtype=np.float64)
        m_j_c = np.ascontiguousarray(m_j)
        for tile_len, slices in _tile_groups(qjk, ao_width):
            bc_prog = _bar_charge_program(tile_len, nao)
            with batch_execute(checked, bc_prog) as exec_bc:
                for rows in slices:
                    bar_charge[rows] = exec_bc(
                        {
                            "ao": np.ascontiguousarray(ao_coul_jk[rows]),
                            "m_j": m_j_c,
                        }
                    ).outputs["bar_charge"]

        jk_bar_metric = np.zeros((qjk, qjk), dtype=np.float64)
        for spin, occ in rev_spins:
            if occ == 0:
                continue
            bar_fit_s = np.zeros((qjk, occ, ext), dtype=np.float64)
            bar_occ_s = np.zeros((qjk, occ, ext), dtype=np.float64)
            ex_width = default_auxiliary_batch_size(qjk, occ, ext)
            bk_c = np.ascontiguousarray(b_k[spin])
            for tile_len, slices in _tile_groups(qjk, ex_width):
                exp_prog = _exchange_pullback_program(tile_len, occ, ext)
                with batch_execute(checked, exp_prog) as exec_exp:
                    for rows in slices:
                        part = exec_exp(
                            {
                                "fit": np.ascontiguousarray(jk_fit[spin][rows]),
                                "raw": np.ascontiguousarray(jk_occ_raw[spin][rows]),
                                "b_k": bk_c,
                            }
                        ).outputs
                        bar_fit_s[rows] += part["bar_fit"]
                        bar_occ_s[rows] += part["bar_raw"]
            idx = np.arange(occ)
            bar_fit_s[:, idx, idx] += bar_charge[:, None]

            if aomo_ops is not None:
                bar_occ_s += run_coulomb_fit_rev(
                    aomo_ops,
                    jk_whitener,
                    np.ascontiguousarray(jk_occ_raw[spin]).reshape(qjk, occ * ext),
                    np.ascontiguousarray(bar_fit_s).reshape(qjk, occ * ext),
                    jk_bar_metric,
                    metric_scale=spin_mult,
                ).reshape(qjk, occ, ext)
                del bar_fit_s
            else:
                solve = coulomb_fit_response(qjk, occ * ext)
                solve_rev = Program(
                    {k: solve.outputs[k] for k in ("three_center_bar", "metric_bar")}
                )
                solved = checked(
                    solve_rev,
                    {
                        "metric_whitener": jk_whitener,
                        "three_center": np.ascontiguousarray(jk_occ_raw[spin]).reshape(qjk, occ * ext),
                        "fit_bar": np.ascontiguousarray(bar_fit_s).reshape(qjk, occ * ext),
                    },
                ).outputs
                bar_occ_s += solved["three_center_bar"].reshape(qjk, occ, ext)
                jk_bar_metric += spin_mult * solved["metric_bar"]
                del bar_fit_s, solved
            total_bar_occ_raw[spin] = bar_occ_s

            bar_c_occ = np.zeros_like(coeffs_by_spin[spin])
            project_ao_operator_pullback_tiled(
                nao,
                ext,
                occ,
                qjk,
                "raw_f12",
                "ao_f12",
                coeffs_by_spin[spin],
                ao_coul_jk,
                bar_occ_s,
                checked,
                bar_coefficients=bar_c_occ,
                bar_ao=None,
                seed_scale=1.0,
                inputs_mode="coefficients",
            )
            if same_spin_jk_rev:
                jk_bar_c["alpha"] += bar_c_occ
                jk_bar_c["beta"] += bar_c_occ
            else:
                jk_bar_c[spin] += bar_c_occ

        del ao_coul_jk, jk_occ_raw, jk_fit, fock_bars, fock

    bars["hcore_ao"] += jk_bar_hcore
    bars["coefficients_alpha"] += jk_bar_c["alpha"]
    bars["coefficients_beta"] += jk_bar_c["beta"]
    # relaxed_f12_source_weights only updates the leading [qjk, obs, obs] OBS block;
    # deferring the full [qjk, nao, nao] JK AO allocation until after CPHF cuts peak by ~12 GiB.
    bars["jkfit/ao_coulomb"] = np.zeros((qjk, obs, obs), dtype=np.float64)
    bars["jkfit/metric/coulomb"] = jk_bar_metric
    release_host_memory()
    record_ram_phase("after_jk_reverse")

    # --- Phase 6: Relax OBS and CABS frame weights through canonical DF-HF ---
    if proj.rank == nao - obs:
        disclose_host_assisted_stage(
            "full_rank_cabs_overlap_bar",
            requested_backend=request.backend,
            detail="Ozaki-4 compensated host SciPy/NumPy full-rank CABS overlap cotangent",
        )
    one_e_w, jk_w, f12_w = relaxed_f12_source_weights(
        request,
        reference,
        one,
        jk,
        panel,
        proj,
        bars,
        checked,
        budget=budget,
        relative_cutoff=relative_cutoff,
        library=library,
        canonical_library=canonical_library,
        extra_obs_fock_seeds=extra_obs_fock_seeds,
        extra_obs_cbar=extra_obs_cbar,
    )
    del bars, panel
    release_host_memory()
    record_ram_phase("after_cphf_relax")

    # --- Phase 5b: Materialize full [qjk, nao, nao] JK AO weight after CPHF buffers are freed ---
    obs_jk_ao = jk_w.pop("ao_coulomb")
    if used_ext_fock_aot:
        assert ext_fock_ops is not None and rho_val is not None and bar_g is not None
        bar_ao_coul_jk = np.empty((qjk, nao, nao), dtype=np.float64)
        if oa == 0 and ob == 0:
            bar_ao_coul_jk.fill(0.0)
        else:
            zero_tile = np.zeros((min(ext_tile_q, qjk), nao, nao), dtype=np.float64)
            discard_occ_a = np.zeros((nao, oa), dtype=np.float64) if oa > 0 else None
            discard_occ_b = np.zeros((nao, ob), dtype=np.float64) if ob > 0 else None
            for q0 in range(0, qjk, ext_tile_q):
                q1 = min(qjk, q0 + ext_tile_q)
                sub = q1 - q0
                ext_fock_ops.half_rev(
                    zero_tile[:sub],
                    occ_a,
                    occ_b,
                    bar_R_a[q0:q1] if bar_R_a is not None else None,
                    bar_R_b[q0:q1] if bar_R_b is not None else None,
                    bar_ao_coul_jk[q0:q1],
                    discard_occ_a,
                    discard_occ_b,
                    rho_tile=rho_val[q0:q1],
                    bar_g=bar_g,
                    add_to_bar_raw=False,
                )
            del zero_tile, discard_occ_a, discard_occ_b
        del bar_R_a, bar_R_b
    else:
        assert m_j_c is not None
        bar_ao_coul_jk = np.empty((qjk, nao, nao), dtype=np.float64)
        for _, slices in _tile_groups(qjk, ao_width):
            for rows in slices:
                bar_ao_coul_jk[rows] = charge[rows, None, None] * m_j_c[None, :, :]
        for spin, occ in rev_spins:
            if occ == 0:
                continue
            project_ao_operator_pullback_tiled(
                nao,
                ext,
                occ,
                qjk,
                "raw_f12",
                "ao_f12",
                coeffs_by_spin[spin],
                None,
                total_bar_occ_raw[spin],
                checked,
                bar_coefficients=None,
                bar_ao=bar_ao_coul_jk,
                seed_scale=spin_mult,
                inputs_mode="ao",
            )
        del total_bar_occ_raw
    bar_ao_coul_jk[:, :obs, :obs] += obs_jk_ao
    del obs_jk_ao
    jk_w["ao_coulomb"] = bar_ao_coul_jk
    release_host_memory()
    return FocalF12Increment(f12_energy, count, proj.rank, one_e_w, jk_w, f12_w)
