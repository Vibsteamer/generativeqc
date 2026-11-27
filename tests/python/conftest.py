"""Cached native probes and shared numerical-policy owners for Python tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Iterable, Sequence


def configured_test_library(variable: str) -> Path | None:
    """An explicit qualification artifact must exist, even when a bundled one is available."""
    value = os.environ.get(variable)
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{variable}={value!r} does not identify a library file")
    return path.resolve()


def f12_host_shell() -> Path:
    """Select the explicit or installed through-i host-reference shell."""
    from generativeqc import _native

    path = configured_test_library("GENERATIVEQC_F12_HOST_SINK")
    if path is None:
        path = _native.installed_f12_shell_library(host_reference=True)
    if path is None:
        pytest.skip("install the through-i host-reference shell or set GENERATIVEQC_F12_HOST_SINK")
    return path


def f12_host_libraries() -> tuple[Path, Path]:
    """The through-i shell also exports the host one-electron and projection ABI."""
    one_electron = configured_test_library("GENERATIVEQC_F12_ONE_ECP")
    shell = f12_host_shell()
    return shell, one_electron or shell


@dataclass(frozen=True)
class NativeCxx:
    """Compile standalone native test probes through a compiler cache, then link normally."""

    compiler: str
    cache: str
    base_dir: Path

    def _env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {**os.environ, "CCACHE_BASEDIR": str(self.base_dir)}
        if extra:
            env.update(extra)
        return env

    @staticmethod
    def _absolute_include_args(args: Sequence[str]) -> list[str]:
        """Keep include directories relative to the caller when changing cwd."""

        def absolute(directory: str) -> str:
            if directory == "-" or directory.startswith(("=", "$SYSROOT")):
                return directory
            return str(Path(directory).absolute())

        options = ("-I", "-isystem", "-iquote", "-idirafter")
        result = []
        directory_next = False
        for arg in args:
            if directory_next:
                result.append(absolute(arg))
                directory_next = False
            elif arg in options:
                result.append(arg)
                directory_next = True
            else:
                option = next((item for item in options if arg.startswith(item)), None)
                if option and len(arg) > len(option):
                    result.append(option + absolute(arg[len(option) :]))
                else:
                    result.append(arg)
        return result

    def compile_object(
        self,
        source: Path,
        output: Path,
        *,
        args: Sequence[str] = (),
        timeout: int = 120,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Compile one translation unit through the compiler cache."""
        return subprocess.run(
            [
                self.cache,
                self.compiler,
                *self._absolute_include_args(args),
                "-c",
                str(source.absolute()),
                "-o",
                str(output.absolute()),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self._env(env),
            # ccache rewrites paths relative to cwd, not to CCACHE_BASEDIR.
            # Both must be this worker's pytest temporary root for generated
            # probes to reuse objects across session and worker directories.
            cwd=self.base_dir,
        )

    def link(
        self,
        objects: Iterable[Path],
        output: Path,
        *,
        args: Sequence[str] = (),
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Link cached objects without routing the non-cacheable link through the cache."""
        return subprocess.run(
            [self.compiler, *map(str, objects), *args, "-o", str(output)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **(env or {})},
        )

    def build_executable(
        self,
        sources: Sequence[Path],
        output: Path,
        *,
        compile_args: Sequence[str] = (),
        link_args: Sequence[str] = (),
        compile_timeout: int = 120,
        link_timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> Path:
        """Compile each source independently through the cache and link an executable."""
        objects = []
        for index, source in enumerate(sources):
            obj = output.parent / f".{output.name}.{index}.o"
            self.compile_object(
                source,
                obj,
                args=compile_args,
                timeout=compile_timeout,
                env=env,
            )
            objects.append(obj)
        self.link(objects, output, args=link_args, timeout=link_timeout, env=env)
        return output

    def build_shared(
        self,
        sources: Sequence[Path],
        output: Path,
        *,
        compile_args: Sequence[str] = (),
        link_args: Sequence[str] = (),
        compile_timeout: int = 120,
        link_timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> Path:
        """Compile cached PIC objects and link a shared library."""
        objects = []
        for index, source in enumerate(sources):
            obj = output.parent / f".{output.name}.{index}.o"
            self.compile_object(
                source,
                obj,
                args=(*compile_args, "-fPIC"),
                timeout=compile_timeout,
                env=env,
            )
            objects.append(obj)
        self.link(
            objects,
            output,
            args=("-shared", *link_args),
            timeout=link_timeout,
            env=env,
        )
        return output


def _compiler_cache() -> tuple[str | None, str]:
    """Verify ``CCACHE`` if set, else select sccache then ccache as runtime compilation does."""
    from generativeqc_compiler.common.compiler_cache import (
        resolve_compiler_cache,
        verify_compiler_cache,
    )

    requested = os.environ.get("CCACHE")
    if requested is not None:
        executable = shutil.which(requested)
        if executable is None:
            return None, f"CCACHE={requested!r} is not an executable"
        # which() may retain a relative spelling; probes compile from another cwd.
        return str(verify_compiler_cache(Path(os.path.abspath(executable))).path), ""
    try:
        return str(resolve_compiler_cache().path), ""
    except RuntimeError as error:
        return None, str(error)


def _native_cxx(
    tmp_path_factory: pytest.TempPathFactory, *, required: bool
) -> NativeCxx:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    cache, reason = _compiler_cache()
    if compiler is None or cache is None:
        message = "native probe requires a host C++ compiler and sccache or ccache"
        if reason:
            message += f" ({reason})"
        if required:
            pytest.fail(message)
        pytest.skip(message)
    # which() may retain a relative spelling. Preserve the executable basename
    # (e.g. clang++), but make it independent of the probe compilation cwd.
    compiler = os.path.abspath(compiler)
    return NativeCxx(compiler, cache, tmp_path_factory.getbasetemp())


@pytest.fixture(scope="session")
def generated_classical_headers(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialize production scalar headers for probes built outside CMake."""
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("generated-classical-headers")
    for name in ("nuclear_repulsion", "dispersion_scalar", "gcp_scalar"):
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(root / "tools" / f"generate_{name}.py"),
                "--output",
                str(directory / f"generated_{name}.hpp"),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    return directory


@pytest.fixture(scope="session")
def native_cxx(tmp_path_factory: pytest.TempPathFactory) -> NativeCxx:
    """Provide a cached host compiler for probes with optional prerequisites."""
    return _native_cxx(tmp_path_factory, required=False)


@pytest.fixture(scope="session")
def required_native_cxx(tmp_path_factory: pytest.TempPathFactory) -> NativeCxx:
    """Fail mandatory native gates when compiler/cache prerequisites are absent."""
    return _native_cxx(tmp_path_factory, required=True)


@pytest.fixture(scope="module")
def policy_library() -> ctypes.CDLL:
    """Borrow actual host policy queries without executing molecular work.

    Molecular doubles are shared across test modules, so their native owner
    must remain visible when a consumer is collected or sharded independently.
    """
    from generativeqc import _native

    return _native.load_library(device="cpu", device_id=0)
