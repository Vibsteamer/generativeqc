"""Pulay, orbital and explicitly correlated force for a bent water with all occupied pairs active."""

import json

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
    if panel["molecule"] != "water" or panel["geometry"]["elements"] != ["O", "H", "H"]:
        raise ValueError("expected the bent all-electron water qualification")
    if int(panel["layout"]["space"][0]) != 5:
        raise ValueError("all five occupied water orbitals must participate")
    report = calculate_panel(
        args,
        panel,
        full_rank_constraint=args.full_rank_cabs_response,
        molecular_correlation=molecular_cc_correlation,
    )
    validate(report, panel)
    report["scope"] = (
        "All-active restricted fixed-geminal CC for bent water, robust DF, CABS singles and fully relaxed molecular force. Independent PySCF RCC plus explicit F12 spin-projected equations, Psi4 complementary-basis sources and twice-displaced Cartesian energies."
    )
    write_measurement(args.output, report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
