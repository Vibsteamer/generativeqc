"""Compare Q-batched conventional CC plus fixed-geminal terms with independent molecular energy and forces."""

import json
from functools import partial

from benchmarks.f12_cc_benchmark_center import check_center
from benchmarks.f12_closed_shell_cc_molecular import molecular_cc_correlation
from benchmarks.f12_h2_cc_molecular_force import validate
from benchmarks.f12_molecular_force_composition import (
    calculate_panel,
    endpoint_arguments,
    write_measurement,
)


def main() -> None:
    args = endpoint_arguments()
    panel = json.loads(args.panel.read_bytes())
    if "cc_reference" not in panel:
        raise ValueError("independent fixed-geminal CC reference data required")
    geometry = panel["geometry"]
    if geometry.get("multiplicity", 1) != 1:
        raise ValueError("closed-shell molecular reference required")
    control = panel["conventional_df_batch_size"]
    if type(control) is not int or control < 1:
        raise ValueError("positive explicit auxiliary batch size required")
    measured = calculate_panel(
        args,
        panel,
        full_rank_constraint=args.full_rank_cabs_response,
        molecular_correlation=partial(
            molecular_cc_correlation, auxiliary_batch_size=control
        ),
    )
    if panel.get("validation_scope") == "benchmark_center_and_native_direction":
        check_center(measured, panel)
    else:
        validate(measured, panel)
    measured["scope"] = (
        "Restricted DF-CC fixed geminal and analytic nuclear force. The conventional CC energy, amplitude residual, Lambda and stationary Hamiltonian weights operate on Q batches and at most two virtual indices. The same fully relaxed CABS singles and geminal sources are used throughout."
    )
    measured["conventional_auxiliary_batch_size"] = control
    write_measurement(args.output, measured)
    if not measured["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
