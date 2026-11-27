"""Resolve a backend-specific F12 ABI against a single native library."""

from __future__ import annotations

import ctypes as ct
import hashlib
import os
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from ._native import installed_f12_shell_library

_Binding = TypeVar("_Binding")
_LOADED: dict[str, ct.CDLL] = {}


def open_shell_library(path: Path | str, *, noload: bool = False) -> ct.CDLL | None:
    resolved = str(Path(path).resolve())
    if resolved in _LOADED:
        return _LOADED[resolved]
    mode = ct.RTLD_LOCAL
    if noload:
        if not hasattr(os, "RTLD_NOLOAD"):
            return None
        mode |= os.RTLD_NOLOAD
    try:
        library = ct.CDLL(resolved, mode=mode)
    except OSError:
        return None
    _LOADED[resolved] = library
    return library


def _loaded_sink_paths() -> list[str]:
    """List mapped focal shell libraries and other application DSOs with custom names."""
    paths = {}
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="ignore") as maps:
            for line in maps:
                fields = line.rstrip().split(None, 5)
                if len(fields) < 6:
                    continue
                path = fields[5]
                if not path.startswith("/") or ".so" not in path or path.endswith(" (deleted)"):
                    continue
                name = os.path.basename(path)
                if any(tag in name for tag in ("focal", "shell", "f12", "sink")) or not (
                    path.startswith(("/usr/", "/lib/", "/lib64/", "/opt/", "/etc/"))
                    or "site-packages" in path
                ):
                    paths[path] = None
    except OSError:
        return []
    return list(paths)


def _digest(path: Path) -> bytes:
    stat = path.stat()
    return _binary_digest(path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_ctime_ns))


@lru_cache(maxsize=16)
def _binary_digest(path: Path, identity: tuple[int, int, int, int]) -> bytes:
    # The identity participates in the cache key so replaced files get new digests.
    digest = hashlib.sha256()
    with path.open("rb") as binary:
        for block in iter(lambda: binary.read(1 << 20), b""):
            digest.update(block)
    return digest.digest()


def resolve_shell_binding(
    selected: Path | str | ct.CDLL | None,
    bind: Callable[[ct.CDLL], _Binding | None],
    *,
    host_reference: bool,
    env_keys: tuple[str, ...],
    mapped_paths: Callable[[], Iterable[str]],
    registered: Iterable[str] = (),
) -> _Binding | None:
    """An explicit owner, environment override or installed artifact owns all its bindings."""
    if selected is None:
        for key in env_keys:
            if value := os.environ.get(key, "").strip():
                selected = value
                break
    if selected is None:
        selected = installed_f12_shell_library(host_reference=host_reference)
    if selected is not None:
        library = selected if isinstance(selected, ct.CDLL) else open_shell_library(selected)
        return None if library is None else bind(library)

    candidates = {Path(path).resolve(): False for path in registered}
    for path in mapped_paths():
        candidates.setdefault(Path(path).resolve(), True)
    match = None
    first_path = None
    first_digest = None
    for path, noload in candidates.items():
        library = open_shell_library(path, noload=noload)
        if library is None:
            continue
        binding = bind(library)
        if binding is None:
            continue
        if first_path is None:
            first_path, match = path, binding
        elif not first_path.samefile(path):
            if first_digest is None:
                first_digest = _digest(first_path)
            if _digest(path) != first_digest:
                raise RuntimeError(
                    f"Ambiguous F12 shell libraries {first_path} and {path}. "
                    "Supply the shell library for this evaluation."
                )
    return match
