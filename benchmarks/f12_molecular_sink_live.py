"""Contract live molecular F12 adjoints in the orbital and fitting AO bases."""

import argparse
import json
from pathlib import Path

import numpy as np
from generativeqc._f12_focal_integral_sink import ShellProjection, contract_live_weights

__all__ = ["ShellProjection", "contract_live_weights"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("layout", "weights", "library", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--coulomb-library", type=Path)
    parser.add_argument("--host-reference", action="store_true")
    args = parser.parse_args()
    with (
        np.load(args.layout, allow_pickle=False) as topology,
        np.load(args.weights, allow_pickle=False) as weights,
    ):
        report = contract_live_weights(
            topology,
            weights,
            args.library,
            host_reference=args.host_reference,
            coulomb_library=args.coulomb_library,
        )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passes_contraction_checks"]:
        raise SystemExit(
            "molecular channel contraction exceeded its numerical or resource gate"
        )
    if report["passes_central_channel_references"] is False:
        raise SystemExit(
            "central molecular weights disagree with independent reference"
        )


if __name__ == "__main__":
    main()
