"""Bundled native data must retain build identity and redistribution notices."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from generativeqc import autotune

ROOT = Path(__file__).resolve().parents[2]
LIBXC_LICENSE = "upstream/libxc/7.0.0/COPYING"
# Licenses and notices installed with every native library.
NATIVE_LICENSE_FILES = (
    "LICENSE",
    "LICENSES/bse-data-BSD-3-Clause.txt",
    "LICENSES/gpu4pyscf-Apache-2.0.txt",
    "LICENSES/pyscf-Apache-2.0.txt",
    "LICENSES/pyscf-NOTICE.txt",
    "LICENSES/psi4-COPYING.LESSER.txt",
    "LICENSES/psi4-COPYING.txt",
    "LICENSES/dftd4-COPYING.LESSER.txt",
    "LICENSES/dftd4-COPYING.txt",
    "LICENSES/xtbloom-CUDA_MKL_LINKING_EXCEPTION.txt",
    "LICENSES/mctc-lib-LICENSE.txt",
    "LICENSES/implib-MIT.txt",
    "LICENSES/simple-dftd3-COPYING.txt",
    "LICENSES/simple-dftd3-COPYING.LESSER.txt",
    LIBXC_LICENSE,
)


@pytest.mark.parametrize(
    "relative",
    [
        "tools/generate_cli_basis_pack.py",
        "python/generativeqc/data/basis_pack.json",
    ],
)
def test_bundled_catalog_inputs_change_native_source_identity(
    monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    before = autotune.source_identity(ROOT)
    original = autotune.file_hash
    changed = ROOT / relative
    monkeypatch.setattr(
        autotune,
        "file_hash",
        lambda path: "changed-input" if path == changed else original(path),
    )
    assert autotune.source_identity(ROOT) != before


def _installed_artifacts(build: Path) -> set[Path]:
    """Artifacts of every target with an install rule, from CMake's file API."""
    reply = build / ".cmake/api/v1/reply"
    (index,) = reply.glob("index-*.json")
    codemodel = json.loads(index.read_text())["reply"]["codemodel-v2"]["jsonFile"]
    artifacts = set()
    for configuration in json.loads((reply / codemodel).read_text())["configurations"]:
        for entry in configuration["targets"]:
            target = json.loads((reply / entry["jsonFile"]).read_text())
            if "install" in target:
                artifacts.update(build / item["path"] for item in target["artifacts"])
    return artifacts


@pytest.mark.skipif(sys.platform != "linux", reason="ELF installation fixture")
def test_cli_disabled_library_installs_bundled_data_notice(tmp_path: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the native install contract")
    build, prefix = tmp_path / "build", tmp_path / "install"
    query = build / ".cmake/api/v1/query/codemodel-v2"
    query.parent.mkdir(parents=True)
    query.touch()
    subprocess.run(
        [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(build),
            "-DGENERATIVEQC_ENABLE_CUDA=OFF",
            "-DGENERATIVEQC_BUILD_CLI=OFF",
            "-DGENERATIVEQC_BUILD_TESTS=OFF",
            "-DGENERATIVEQC_CPU_LINALG_PROVIDER=scalar",
            "-DGENERATIVEQC_PYTHON_WHEEL=ON",
            "-DGENERATIVEQC_BUNDLE_XTB_OPENBLAS=OFF",
            "-DCMAKE_INSTALL_DATADIR=share",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Exercise the real install graph without rebuilding scientific code. The
    # wheel layout installs unversioned ELF artifacts; /bin/true is only an
    # explicit packaging fixture, never executed or treated as a native library.
    for artifact in _installed_artifacts(build):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile("/bin/true", artifact)
    for component in (None, "Unspecified"):
        destination = prefix / (component or "default")
        command = [cmake, "--install", str(build), "--prefix", str(destination)]
        if component is not None:
            command.extend(["--component", component])
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        licenses = destination / "share/generativeqc/licenses"
        for path in NATIVE_LICENSE_FILES:
            source = ROOT / path
            name = "libxc-MPL-2.0.txt" if path == LIBXC_LICENSE else source.name
            assert (licenses / name).read_bytes() == source.read_bytes(), path
        assert list(destination.rglob("libgenerativeqc.so"))
        assert (destination / "include/generativeqc/generativeqc.h").is_file()


def test_native_license_texts_match_the_wheel_license_files() -> None:
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    shipped = {
        path for pattern in project["license-files"] for path in ROOT.glob(pattern)
    }
    assert {ROOT / path for path in NATIVE_LICENSE_FILES} <= shipped
