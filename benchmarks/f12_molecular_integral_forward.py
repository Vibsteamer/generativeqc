"""Build molecular two-/three-center spherical AO tensors from native shell integrals."""

import argparse
import json
from pathlib import Path

import numpy as np
from generativeqc._f12_focal_integrals import bind_forward, forward_integrals

from benchmarks.f12_molecular_sink_test import sha256

__all__ = ["bind_forward", "forward_integrals"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("layout", "source", "library", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--host-reference", action="store_true")
    args = parser.parse_args()
    with np.load(args.layout, allow_pickle=False) as layout:
        metadata = json.loads(str(layout["manifest"]))
        sources, resources = forward_integrals(
            layout, args.library, host_reference=args.host_reference
        )
    expected_sha = metadata["inputs"]["source_fixture_sha256"]
    if sha256(args.source) != expected_sha:
        raise ValueError("independent source file does not match fixture provenance")
    case = metadata["case"]
    rows = {}
    with np.load(args.source, allow_pickle=False) as ref:
        for label, values in sources.items():
            expected = ref[case + "/" + label]
            if expected.shape != values.shape or not np.isfinite(values).all():
                raise ValueError(f"{label} incomplete spherical tensor")
            diff = values - expected
            err = float(np.max(np.abs(diff)))
            rows[label] = {
                **resources[label],
                "shape": values.shape,
                "max_abs_error": err,
                "rms_error": float(np.sqrt(np.mean(diff**2))),
                "max_abs_reference": float(np.max(np.abs(expected))),
                "passed": err <= 1.0e-9,
            }
    passed = len(rows) == 10 and all(r["passed"] for r in rows.values())
    report = {
        "case": case,
        "scope": "full spherical two- and three-center molecular forward tensors",
        "source_sha256": expected_sha,
        "layout_sha256": sha256(args.layout),
        "library_sha256": sha256(args.library),
        "tolerance": 1e-9,
        "passed": passed,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not passed:
        raise SystemExit("molecular forward failed full tensor oracle comparison")


if __name__ == "__main__":
    main()
