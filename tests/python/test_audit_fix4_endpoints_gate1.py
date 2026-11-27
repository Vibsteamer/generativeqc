"""Gate 1 regression & counterexample tests for Audit Fix-4 (Python endpoints, budget, AOT, Delta-CV, hygiene)."""

from __future__ import annotations
import threading
from pathlib import Path
import pytest
from generativeqc._f12_ram_hygiene import (
    STRICT_GPU_ONLY_ENV,
    capture_hybrid_phases,
    capture_ram_trace,
    disclose_host_assisted_stage,
    record_ram_phase,
)


# ---------------------------------------------------------------------------
# M11: Structured Hybrid-Phase Telemetry Disclosure & STRICT_GPU_ONLY Guard
# ---------------------------------------------------------------------------


def test_m11_hybrid_phase_disclosure_and_strict_gpu_only_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify capture_hybrid_phases records host-assisted stages and STRICT_GPU_ONLY=1 rejects them on CUDA."""
    monkeypatch.delenv(STRICT_GPU_ONLY_ENV, raising=False)
    with capture_hybrid_phases() as events:
        disclose_host_assisted_stage(
            "full_rank_cabs_overlap_bar",
            requested_backend="cuda",
            detail="Ozaki-4 host full-rank CABS overlap bar",
        )
        disclose_host_assisted_stage(
            "streamed_f12_pair_residuals",
            requested_backend="cpu_reference",
            detail="host residual math",
        )
    assert len(events) == 2
    assert events[0]["stage"] == "full_rank_cabs_overlap_bar"
    assert events[0]["execution_mode"] == "hybrid_host_assisted"
    assert events[1]["execution_mode"] == "host"

    monkeypatch.setenv(STRICT_GPU_ONLY_ENV, "1")
    # Host backend stays permitted under STRICT_GPU_ONLY=1
    disclose_host_assisted_stage(
        "full_rank_cabs_overlap_bar",
        requested_backend="cpu_reference",
        detail="host mode allowed",
    )
    # CUDA backend with host-contracted scientific stage MUST raise NotImplementedError!
    with pytest.raises(NotImplementedError, match=STRICT_GPU_ONLY_ENV):
        disclose_host_assisted_stage(
            "full_rank_cabs_overlap_bar",
            requested_backend="cuda",
            detail="Ozaki-4 host full-rank CABS overlap bar",
        )


@pytest.mark.parametrize("capture", [capture_ram_trace, capture_hybrid_phases])
def test_overlapping_empty_traces_remove_only_the_closed_collector(capture) -> None:
    first, second, third = (capture() for _ in range(3))
    records_a = first.__enter__()
    records_b = second.__enter__()
    first.__exit__(None, None, None)
    with third as records_c:
        second.__exit__(None, None, None)
        record_ram_phase("active")
        disclose_host_assisted_stage("active", requested_backend="cpu_reference", detail="test")
    assert records_a == records_b == []
    assert len(records_c) == 1


def test_m8_capture_ram_trace_contextvar_interleaved_and_multithreaded() -> None:
    """Verify capture_ram_trace uses ContextVar and survives interleaved exits and concurrent threads."""
    # 1. Interleaved out-of-order manual __enter__ / __exit__ in the same thread
    cm_a = capture_ram_trace()
    cm_b = capture_ram_trace()
    rec_a = cm_a.__enter__()
    record_ram_phase("only_a")
    rec_b = cm_b.__enter__()
    record_ram_phase("both_ab")
    cm_a.__exit__(None, None, None)
    record_ram_phase("only_b_after_a_exit")
    cm_b.__exit__(None, None, None)
    record_ram_phase("none_active")

    assert [e["phase"] for e in rec_a] == ["only_a", "both_ab"]
    assert [e["phase"] for e in rec_b] == ["both_ab", "only_b_after_a_exit"]

    # 2. Concurrent threads never cross-contaminate each other's traces
    barrier = threading.Barrier(2)
    results: dict[str, list[str]] = {}

    def worker(tag: str) -> None:
        with capture_ram_trace() as trace:
            barrier.wait()
            record_ram_phase(f"phase_{tag}_1")
            barrier.wait()
            record_ram_phase(f"phase_{tag}_2")
            results[tag] = [str(e["phase"]) for e in trace]

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results["A"] == ["phase_A_1", "phase_A_2"]
    assert results["B"] == ["phase_B_1", "phase_B_2"]


def test_d4_host_libraries_reject_invalid_explicit_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An invalid explicit test library must fail before any alternative is selected."""
    from conftest import f12_host_libraries

    missing = tmp_path / "does_not_exist.so"
    monkeypatch.setenv("GENERATIVEQC_F12_HOST_SINK", str(missing))
    with pytest.raises(FileNotFoundError, match=missing.name):
        f12_host_libraries()

    shell = tmp_path / "shell.so"
    shell.touch()
    monkeypatch.setenv("GENERATIVEQC_F12_HOST_SINK", str(shell))
    monkeypatch.setenv("GENERATIVEQC_F12_ONE_ECP", str(missing))
    with pytest.raises(FileNotFoundError, match=missing.name):
        f12_host_libraries()
    monkeypatch.delenv("GENERATIVEQC_F12_ONE_ECP")
    assert f12_host_libraries() == (shell, shell)
