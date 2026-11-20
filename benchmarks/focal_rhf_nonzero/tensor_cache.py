"""Transport verified sm_89 TensorIR programs between stages from the same focal release."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_runtime import CudaArtifact
    from generativeqc_compiler.tensor.cuda_plan import TensorPlan

FILES = ("artifact.json", "program.so", "static.bin")
KEY = re.compile(r"[0-9a-f]{64}")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def entry(directory: Path) -> dict:
    metadata = json.loads((directory / "artifact.json").read_bytes())
    identity = metadata["identity"]
    key = metadata["key"]
    if directory.name != key or not KEY.fullmatch(key) or canonical(identity) != key:
        raise ValueError("Tensor artifact key differs from its full compiler identity")
    if "sm_89" not in identity["options"] or not all(
        k in identity
        for k in (
            "source",
            "generated",
            "plan",
            "toolchain",
            "compiler_environment",
            "compiler_executables",
            "host",
            "dependencies",
            "static_data",
        )
    ):
        raise ValueError("Incomplete sm_89 TensorIR compiler identity")
    if any(
        not (directory / name).is_file() or (directory / name).is_symlink()
        for name in FILES
    ):
        raise ValueError("Missing regular tensor cache member")
    contents = {
        name: {
            "bytes": (directory / name).stat().st_size,
            "sha256": sha(directory / name),
        }
        for name in FILES
    }
    so, data = contents["program.so"], contents["static.bin"]
    if (
        (so["sha256"], so["bytes"])
        != (metadata["binary_sha256"], metadata["binary_bytes"])
        or data != identity["static_data"]
        or (data["sha256"], data["bytes"])
        != (metadata["static_data_sha256"], metadata["static_data_bytes"])
    ):
        raise ValueError(
            "Tensor artifact binary or static payload differs from its metadata"
        )
    return contents


def export_cache(cache: Path, keys: set[str], archive: Path, context: dict) -> dict:
    if not keys or not all(KEY.fullmatch(k) for k in keys):
        raise ValueError("Expected executed TensorIR cache keys")
    entries = {k: entry(cache / k) for k in sorted(keys)}
    with (
        archive.open("wb") as target,
        gzip.GzipFile(
            fileobj=target, mode="wb", filename="", mtime=0, compresslevel=1
        ) as gz,
        tarfile.open(fileobj=gz, mode="w|") as out,
    ):
        for key, files in entries.items():
            for name, data in files.items():
                info = tarfile.TarInfo(f"{key}/{name}")
                info.size = data["bytes"]
                info.mode = 0o644
                with (cache / key / name).open("rb") as source:
                    out.addfile(info, source)
    return {
        "context": context,
        "entries": entries,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha(archive),
    }


def restore_cache(
    cache: Path, archive: Path, manifest: dict, context: dict
) -> set[str]:
    if (
        context != manifest["context"]
        or sha(archive) != manifest["archive_sha256"]
        or archive.stat().st_size != manifest["archive_bytes"]
    ):
        raise ValueError("Tensor cache bundle differs from its qualified pilot")
    entries = manifest["entries"]
    if not entries or not all(
        KEY.fullmatch(k) and set(v) == set(FILES) for k, v in entries.items()
    ):
        raise ValueError("Unexpected tensor cache manifest")
    expected = {
        f"{key}/{name}": record
        for key, files in entries.items()
        for name, record in files.items()
    }
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache.parent) as temporary:
        extract = Path(temporary)
        seen = set()
        with tarfile.open(archive, "r|gz") as source:
            for member in source:
                name = member.name
                if (
                    name not in expected
                    or name in seen
                    or not member.isfile()
                    or member.size != expected[name]["bytes"]
                ):
                    raise ValueError("Unexpected tensor cache archive member")
                path = extract / name
                path.parent.mkdir(exist_ok=True)
                with source.extractfile(member) as stream, path.open("xb") as target:
                    shutil.copyfileobj(stream, target)
                if sha(path) != expected[name]["sha256"]:
                    raise ValueError("Tensor cache member SHA mismatch")
                seen.add(name)
        if seen != set(expected):
            raise ValueError("Incomplete tensor cache bundle")
        for key, files in entries.items():
            if entry(extract / key) != files:
                raise ValueError("Restored tensor cache identity mismatch")
        for key, files in entries.items():
            target = cache / key
            if target.exists():
                if entry(target) != files:
                    raise ValueError("Conflicting existing TensorIR cache entry")
            else:
                os.rename(extract / key, target)
    return set(entries)


@contextmanager
def observe_compilation(
    cache: Path, seeded: set[str]
) -> Iterator[tuple[set[str], dict[str, int]]]:
    """Count resolved keys after the installed compiler validates the current plan and binary."""
    from generativeqc._ucc import native_tensor_cuda
    from generativeqc_compiler.tensor import cuda_execute

    original = cuda_execute.compile_cuda
    used = set()
    counts = {"calls": 0, "qualified_seed_hits": 0}

    def observed(
        plan: TensorPlan, compiler: CudaCompilerAdapter, location: Path
    ) -> CudaArtifact:
        if Path(location).resolve() != cache.resolve():
            raise ValueError("Unexpected focal cache directory")
        artifact = original(plan, compiler, location)
        key = artifact.metadata["key"]
        if artifact.library.parent != cache.resolve() / key:
            raise ValueError("Unexpected TensorIR artifact location")
        used.add(key)
        counts["calls"] += 1
        counts["qualified_seed_hits"] += int(key in seeded)
        return artifact

    with ExitStack() as stack:
        for owner in (cuda_execute, native_tensor_cuda):
            stack.enter_context(patch.object(owner, "compile_cuda", observed))
        yield used, counts
