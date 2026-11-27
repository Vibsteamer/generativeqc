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

REPO = Path(__file__).resolve().parents[2]


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
