"""Run a signed molecular driver while recording TensorIR cache provenance on an L20."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def run(
    data: Path, main: Path, shell: Path, response: Path, out: Path, backend: str
) -> None:
    sys.path.insert(0, str(data))
    from tensor_cache import export_cache, observe_compilation, restore_cache, sha

    task = json.loads((data / "task.json").read_bytes())
    context = json.loads((data / "jit_context.json").read_bytes())
    if (
        backend != "cuda"
        or context["architecture"] != "sm_89"
        or context["source_commit"] != task["source_commit"]
        or context["protocol"] != task["protocol"]
    ):
        raise ValueError("Forward cache context differs from the molecular release")
    source = data / "forward_driver.py"
    if (
        sha(source) != context["forward_driver_sha256"]
        or sha(data / "tensor_cache.py") != context["cache_module_sha256"]
    ):
        raise ValueError("Forward driver or cache module differs from its release")
    spec = importlib.util.spec_from_file_location("focal_forward_driver", source)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    cache = out.parent / "cache"
    seeded = set()
    receipt = {"mode": task["mode"], "context": context}
    if task["mode"] != "timing":
        parent = json.loads((data / "jit_parent.json").read_bytes())
        manifest_path = data / "jit_manifest.json"
        if sha(manifest_path) != parent["manifest_sha256"]:
            raise ValueError("Pilot manifest SHA mismatch")
        start = time.monotonic()
        seeded = restore_cache(
            cache,
            data / "jit_programs.tar.gz",
            json.loads(manifest_path.read_bytes()),
            context,
        )
        receipt.update(parent=parent, restore_seconds=time.monotonic() - start)
    with observe_compilation(cache, seeded) as (used, counts):
        driver.run(data, main, shell, response, out, backend)
    if not used:
        raise ValueError("No compiler-resolved TensorIR artifact was observed")
    if task["mode"] == "timing":
        start = time.monotonic()
        manifest = export_cache(cache, used, out / "jit_programs.tar.gz", context)
        (out / "jit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        receipt.update(
            export_seconds=time.monotonic() - start,
            archive_sha256=manifest["archive_sha256"],
            archive_bytes=manifest["archive_bytes"],
            manifest_sha256=sha(out / "jit_manifest.json"),
        )
    receipt.update(
        **counts,
        seeded_keys=sorted(seeded),
        executed_seed_keys=sorted(used & seeded),
        executed_new_keys=sorted(used - seeded),
    )
    (out / "jit_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("data", "main", "shell", "response", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--backend", required=True, choices=("cuda",))
    a = p.parse_args()
    run(a.data, a.main, a.shell, a.response, a.out, a.backend)
