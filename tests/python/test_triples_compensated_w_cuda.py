"""Every compensated FP32 W launch schedule matches a long-double oracle on a device."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _device_architecture() -> str:
    """Name the visible device explicitly so cached objects never cross architectures."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        pytest.skip("requires a visible CUDA device")
    try:
        query = subprocess.run(
            [smi, "--query-gpu=compute_cap", "--format=csv,noheader", "--id=0"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError:
        pytest.skip("requires a visible CUDA device")
    capability = query.stdout.strip()
    if query.returncode or not capability:
        pytest.skip("requires a visible CUDA device")
    major, minor = capability.split(".")
    return f"sm_{int(major)}{int(minor)}"


def test_compensated_w_schedules_match_long_double_oracle(tmp_path: Path) -> None:
    """Require a scheduler-owned GPU; forced scratch limits select each schedule."""
    if os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1":
        pytest.skip("requires explicitly allocated GPU validation window")
    compiler = shutil.which("nvcc")
    cache = shutil.which("sccache") or shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires nvcc and sccache or ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    root = Path(__file__).resolve().parents[2]
    architecture = _device_architecture()
    obj, executable = tmp_path / "compensated_w.o", tmp_path / "compensated_w"
    commands = (
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            f"-arch={architecture}",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            "-c",
            str(root / "tests/native/test_triples_compensated_w_cuda.cu"),
            "-o",
            str(obj),
        ],
        [
            compiler,
            f"-arch={architecture}",
            str(obj),
            "-lcublas",
            "-o",
            str(executable),
        ],
    )
    for command in commands:
        build = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "CCACHE_BASEDIR": str(root)},
        )
        assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=300, check=False
    )
    if result.returncode == 77:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr
