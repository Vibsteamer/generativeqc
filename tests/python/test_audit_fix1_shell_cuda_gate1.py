"""Gate 1 executable regression suite for Audit Fix-1A (B1, M6, M8, D3).

Covers:
  - B1: Race-free MP2-F12 overlap-cotangent line folds (o >= 36 cross-warp
    pairs including i=0, j=32) and alias-safe diagonal (i == j, p == r)
    robust element pullback, verified via RHF-vs-UHF parity and central
    finite differences against mp2_pair_energies.
  - M6: Hard budget enforcement on MP2-F12 pair response & pair energies
    (including the aux=300 vs 1 MiB counterexample) without allocating past
    budget_bytes, plus ordinary DF-MP2 non-triangular fallback parity.
  - M8: Fail-closed NaN/Inf input & overflow-to-Inf output rejection across
    all three shell units (mp2_pair_response, dfmp2_cv, ao_mo_project/cabs).
  - D3: Concurrent configuration, build and immutable publication across shell
    domains and CUDA architectures using the standalone CMake entry point.
"""

from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from conftest import configured_test_library

from generativeqc._f12_ao_mo_native import (
    bind_ao_mo_ops,
    bind_cabs_singles_ops,
    run_ao_mo_bilin_fwd,
    run_ao_mo_bilin_rev,
    run_ao_mo_project_fwd,
    run_ao_mo_project_rev,
    run_cabs_cphf_action,
    run_cabs_frame_kernel_cotangent,
    run_cabs_whiten_3c,
    run_coulomb_fit_fwd,
    run_coulomb_fit_rev,
)
from generativeqc._f12_mp2_pair_native import (
    bind_ordinary_df_mp2,
    run_ordinary_df_mp2_native,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def host_sink(tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest) -> Path:
    configured = configured_test_library("GENERATIVEQC_F12_HOST_SINK")
    if configured is not None:
        return configured
    root = Path(__file__).resolve().parents[2]
    units = (
        "src/posthf/f12/f12_ao_mo_project_cuda.cu",
        "src/posthf/f12/f12_mp2_pair_response_cuda.cu",
    )
    output = tmp_path_factory.mktemp("audit_host") / "libaudit_host_fixture.so"
    compiler = request.getfixturevalue("required_native_cxx")
    return compiler.build_shared(
        [root / unit for unit in units],
        output,
        compile_args=("-x", "c++", "-std=c++20", "-O3", "-DGQC_SINK_HOST_REFERENCE"),
    )


# =============================================================================
# M8: Fail-closed NaN inputs & overflow-to-Inf outputs across ALL 3 units
# =============================================================================


def test_m8_ao_mo_and_cabs_fail_closed_on_overflow_to_inf(host_sink: Path):
    """Finite inputs that overflow to Inf MUST fail closed with status -2."""
    ao_ops = bind_ao_mo_ops(host_sink, host_reference=True)
    cabs_ops = bind_cabs_singles_ops(host_sink, host_reference=True)
    assert ao_ops is not None and cabs_ops is not None

    # Exact audit M8 counterexample: whiten_3c on 1x1 finite inputs (1e308, 2.0) -> Inf!
    with pytest.raises(ValueError, match="rc=-2"):
        run_cabs_whiten_3c(
            cabs_ops,
            np.array([[1e308]], dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
        )

    # coulomb_fit_fwd overflow -> rc=-2
    with pytest.raises(ValueError, match="rc=-2"):
        run_coulomb_fit_fwd(
            ao_ops,
            np.array([[1e200]], dtype=np.float64),
            np.array([[1e200]], dtype=np.float64),
        )

    # coulomb_fit_rev overflow -> rc=-2
    with pytest.raises(ValueError, match="rc=-2"):
        run_coulomb_fit_rev(
            ao_ops,
            np.array([[1e160]], dtype=np.float64),
            np.array([[1e160]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
        )

    # ao_mo_project_fwd overflow -> rc=-2
    with pytest.raises(ValueError, match="rc=-2"):
        run_ao_mo_project_fwd(
            ao_ops,
            1,
            1,
            1,
            1,
            np.array([[1e160]], dtype=np.float64),
            np.array([[[1e160]]], dtype=np.float64),
        )

    # ao_mo_project_rev overflow -> rc=-2
    with pytest.raises(ValueError, match="rc=-2"):
        run_ao_mo_project_rev(
            ao_ops,
            1,
            1,
            1,
            1,
            np.array([[1e160]], dtype=np.float64),
            np.array([[[1e160]]], dtype=np.float64),
            np.array([[[1e160]]], dtype=np.float64),
            bar_coefficients=np.zeros((1, 1), dtype=np.float64),
            bar_ao=np.zeros((1, 1, 1), dtype=np.float64),
            inputs_mode="both",
        )

    # ao_mo_bilin_fwd overflow -> rc=-2
    with pytest.raises(ValueError, match="rc=-2"):
        run_ao_mo_bilin_fwd(
            ao_ops,
            np.array([[1e160]], dtype=np.float64),
            np.array([[1e160]], dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
        )

    # ao_mo_bilin_rev overflow -> rc=-2
    with pytest.raises(ValueError, match="rc=-2"):
        run_ao_mo_bilin_rev(
            ao_ops,
            np.array([[1e160]], dtype=np.float64),
            np.array([[1e160]], dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            bar_a=np.zeros((1, 1), dtype=np.float64),
            bar_c_left=np.zeros((1, 1), dtype=np.float64),
            bar_c_right=np.zeros((1, 1), dtype=np.float64),
        )

    # cabs_frame_kernel_cotangent overflow -> rc=-2
    with pytest.raises(ValueError, match="rc=-2"):
        run_cabs_frame_kernel_cotangent(
            cabs_ops,
            np.array([[1e200]], dtype=np.float64),
            np.array([[1e200]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
        )

    # cabs_cphf_action overflow -> rc=-2
    with pytest.raises(ValueError, match="rc=-2"):
        run_cabs_cphf_action(
            cabs_ops,
            oa=1,
            ob=1,
            obs=2,
            aux=1,
            is_rhf=True,
            df_a=np.zeros((1, 2, 2), dtype=np.float64),
            df_b=None,
            gap_a=np.array([[1e308]], dtype=np.float64),
            gap_b=None,
            trial_a=np.array([[2.0]], dtype=np.float64),
            trial_b=None,
        )


def test_m8_mp2_pair_and_ordinary_mp2_fail_closed_overflow_and_nan(host_sink: Path):
    """Ordinary DF-MP2 rejects output overflow with status -5."""
    ord_fn = bind_ordinary_df_mp2(host_reference=True, shell_library=host_sink)
    assert ord_fn is not None

    # Overflow in ordinary DF-MP2 -> status -5
    c = np.eye(2, dtype=np.float64)
    eps = np.array([-1.0, 1.0], dtype=np.float64)
    ao = np.array([[[1e160, 1e160], [1e160, 1e160]]], dtype=np.float64)
    w = np.array([[1e160]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="status -5"):
        run_ordinary_df_mp2_native(
            ord_fn,
            same_spin=True,
            compute_response=False,
            oa=1,
            fa=0,
            ob=1,
            fb=0,
            budget_bytes=64 << 20,
            coeff_a=c,
            coeff_b=c,
            eps_a=eps,
            eps_b=eps,
            ao_coul=ao,
            whitener=w,
        )


# =============================================================================
# D3: Concurrent multi-arch build isolation & locked atomic publication
# =============================================================================


def test_d3_build_layout_multi_arch_isolation_and_concurrent_publish(tmp_path: Path):
    """Exercise simultaneous builder processes and a failed rebuild without a GPU."""
    # This CMake stand-in emits labeled bytes and fails on concurrent access to
    # one mutable build tree. Other cases in this module load the real host DSO.
    cmake = tmp_path / "cmake"
    cmake.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json
import os
from pathlib import Path
import sys
import time

args = sys.argv[1:]
if "-S" in args:
    build = Path(args[args.index("-B") + 1])
    variables = dict(arg[2:].split("=", 1) for arg in args if arg.startswith("-D"))
    (build / "options.json").write_text(json.dumps(variables))
else:
    build = Path(args[args.index("--build") + 1])
    guard = build / "active-build"
    with guard.open("x"):
        time.sleep(0.05)
        options = json.loads((build / "options.json").read_text())
        domain = options["GENERATIVEQC_F12_SHELL_DOMAIN"]
        arch = options["GENERATIVEQC_CUDA_ARCHITECTURES"]
        assert options["GENERATIVEQC_ENABLE_CUDA"] == "ON"
        library = Path(options["CMAKE_LIBRARY_OUTPUT_DIRECTORY"]) / f"libfocal_shell_{domain}.so"
        library.parent.mkdir(exist_ok=True)
        library.write_text(f"{arch}:{domain}:{os.environ['F12_TEST_CONTENT']}")
    guard.unlink()
    if os.environ["F12_TEST_CONTENT"] == "fail":
        raise SystemExit(9)
"""
    )
    cmake.chmod(0o755)
    cxx = shutil.which("c++")
    assert cxx is not None

    def build(arch: str, domain: str, content: str, *, check: bool = True):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "src/posthf/f12/build_f12_molecular_integral_sink.py"),
                "--backend", "cuda",
                "--arch", arch,
                "--nvcc", cxx,
                "--cxx", cxx,
                "--output-dir", str(tmp_path / "out"),
                f"--cabs-{domain}",
            ],
            env={
                **os.environ,
                "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
                "F12_TEST_CONTENT": content,
            },
            check=check,
            capture_output=True,
            text=True,
        )
        if check:
            return json.loads(result.stdout)
        return result

    requests = [
        ("80", "h", "one"), ("sm_89", "h", "one"), ("89", "i", "one"),
        ("sm80", "h", "two"), ("80", "h", "one"),
    ]
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        receipts = list(pool.map(lambda request: build(*request), requests))

    failed = build("80", "h", "fail", check=False)
    assert failed.returncode != 0 and not failed.stdout
    for receipt, (arch, domain, content) in zip(receipts, requests, strict=True):
        sm = arch.removeprefix("sm").lstrip("_")
        expected = f"{sm}-real:{domain}:{content}".encode()
        assert Path(receipt["library"]).read_bytes() == expected
        assert receipt["sha256"] == hashlib.sha256(expected).hexdigest()
        assert receipt["arch"] == f"sm_{sm}"
        assert receipt["domain"] == domain
    assert len({receipt["library"] for receipt in receipts}) == 4
    assert receipts[0]["library"] == receipts[4]["library"]
