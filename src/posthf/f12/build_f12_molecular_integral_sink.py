"""Build and publish a content-addressed F12 shell DSO using its CMake target."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))
from generativeqc_compiler.common.compiler_cache import resolve_compiler_cache


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise argparse.ArgumentTypeError(f"executable not found: {name}")
    return os.path.abspath(executable)


def _sm_number(value: str) -> str:
    match = re.fullmatch(r"(?:sm_?)?([1-9][0-9]+)", value)
    if not match:
        raise argparse.ArgumentTypeError("--arch must be an SM number such as sm_89")
    return match[1]


def _publish_library(library: Path, directory: Path) -> tuple[Path, str]:
    # A hard link to CMake's output would expose it to subsequent incremental links.
    digest = hashlib.sha256()
    with library.open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            digest.update(block)
    sha256 = digest.hexdigest()
    destination = directory / sha256 / library.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        shutil.copy2(library, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc")
    parser.add_argument("--cxx", default="c++", type=_executable)
    parser.add_argument("--arch", default="sm_89", type=_sm_number)
    parser.add_argument("--backend", choices=("host", "cuda"), default="cuda")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cabs-h", action="store_true", help="Build through shell h.")
    group.add_argument("--cabs-i", action="store_true", help="Build through shell i.")
    args = parser.parse_args()

    cuda = args.backend == "cuda"
    try:
        cmake = _executable("cmake")
        nvcc = _executable(args.nvcc or "nvcc") if cuda else None
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    launcher = resolve_compiler_cache()
    domain = "i" if args.cabs_i else "h" if args.cabs_h else "g"
    variant = Path(f"sm_{args.arch}" if cuda else "host", domain)
    target = f"focal_shell_{domain}"
    configure = [
        cmake,
        "-S",
        str(REPO),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCMAKE_CXX_COMPILER={args.cxx}",
        f"-DGENERATIVEQC_COMPILER_CACHE={launcher.path}",
        f"-DCMAKE_CXX_COMPILER_LAUNCHER={launcher.path}",
        f"-DGENERATIVEQC_F12_SHELL_DOMAIN={domain}",
        f"-DGENERATIVEQC_ENABLE_CUDA={'ON' if cuda else 'OFF'}",
        "-DGENERATIVEQC_BUILD_TESTS=OFF",
        "-DGENERATIVEQC_BUILD_CLI=OFF",
        "-DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF",
        "-DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF",
    ]
    if cuda:
        configure.extend(
            [
                f"-DCMAKE_CUDA_COMPILER={nvcc}",
                f"-DCMAKE_CUDA_COMPILER_LAUNCHER={launcher.path}",
                f"-DGENERATIVEQC_CUDA_ARCHITECTURES={args.arch}-real",
            ]
        )
    # CMake reads these environment defaults at first configure. Keep independent
    # toolchains and command-line flags out of each other's incremental trees.
    cmake_environment = {
        key: os.environ.get(key)
        for key in (
            "CMAKE_GENERATOR", "CMAKE_GENERATOR_INSTANCE", "CMAKE_GENERATOR_PLATFORM",
            "CMAKE_GENERATOR_TOOLSET", "CMAKE_TOOLCHAIN_FILE", "CMAKE_PREFIX_PATH",
            "CXXFLAGS", "CUDAFLAGS", "LDFLAGS", "CUDAHOSTCXX", "CPATH",
            "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "PATH",
        )
    }
    compiler_versions = [
        subprocess.check_output([compiler, "--version"], text=True)
        for compiler in (args.cxx, nvcc)
        if compiler
    ]
    configuration = json.dumps([configure, cmake_environment, compiler_versions], sort_keys=True)
    config_key = hashlib.sha256(configuration.encode()).hexdigest()
    output_dir = args.output_dir.resolve()
    build = output_dir / "build" / variant / config_key
    build.mkdir(parents=True, exist_ok=True)
    products = build / "lib"
    configure.extend(
        [
            "-B",
            str(build),
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={products}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={products}",
        ]
    )
    compile_command = [cmake, "--build", str(build), "--config", "Release", "--target", target]
    if not os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL"):
        compile_command.extend(["--parallel", "1"])
    environment = dict(os.environ)
    environment.setdefault("CCACHE_BASEDIR", str(REPO))
    with (build / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stats = [str(launcher.path), "--show-stats"]
        (build / "cache-before.txt").write_bytes(subprocess.check_output(stats))
        # A surviving CMake child retains the lease if this CLI is interrupted.
        for command in (configure, compile_command):
            subprocess.run(
                command, check=True, env=environment, stdout=sys.stderr, pass_fds=(lock.fileno(),)
            )
        (build / "cache-after.txt").write_bytes(subprocess.check_output(stats))
        library = products / f"lib{target}{'' if cuda else '_host'}.so"
        published, digest = _publish_library(library, output_dir / "artifacts" / variant)
    receipt = {
        "library": str(published),
        "sha256": digest,
        "build_dir": str(build),
        "cache": launcher.version,
        "backend": args.backend,
        "domain": domain,
    }
    if cuda:
        receipt["arch"] = f"sm_{args.arch}"
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
