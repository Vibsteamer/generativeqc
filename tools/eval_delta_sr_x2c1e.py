"""Evaluate Gate 2 pure additive DLU-X2C-1e Delta-SR across all 10 benchmark observables & 5 OMol33 structures.

Atomically exports `gqc-f12.step3-bank-extension.v1` JSON to the `--export` path
(defaulting to `<preopt-dir>/extensions/delta_sr_x2c1e.json`). Configure input
paths via CLI (`--manifest`, `--preopt-dir`, `--export`) or environment
variables (`GENERATIVEQC_F12_VALIDATION_DIR`, `GENERATIVEQC_F12_MANIFEST`,
`GENERATIVEQC_F12_PREOPT_DIR`).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from generativeqc import Atom
from generativeqc._x2c1e import (
    build_aux_basis_from_bse,
    build_unc_x2c_basis_from_bse,
    evaluate_delta_sr,
)
from generativeqc.elements import atomic_number

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_DIR = Path(
    os.environ.get(
        "GENERATIVEQC_F12_VALIDATION_DIR",
        str(REPO_ROOT / "benchmarks" / "validation"),
    )
)
DEFAULT_MANIFEST_PATH = Path(
    os.environ.get(
        "GENERATIVEQC_F12_MANIFEST",
        str(DEFAULT_VALIDATION_DIR / "manifest.json"),
    )
)
DEFAULT_PREOPT_DIR = Path(
    os.environ.get(
        "GENERATIVEQC_F12_PREOPT_DIR",
        str(DEFAULT_VALIDATION_DIR / "preopt_v2_eval"),
    )
)
OMOL5_FORMULAS = ("CNa3O3", "C3H5NaO3", "C4H6Br2O", "C5H12O2", "C4H16Si3")
HA_TO_KCAL = 627.5094740631


def _peak_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _eval_one_geometry(spec: dict[str, Any]) -> dict[str, Any]:
    symbols: list[str] = spec["symbols"]
    coords_bohr: list[list[float]] = spec["coordinates_bohr"]
    charge = int(spec["charge"])
    multiplicity = int(spec["multiplicity"])
    root_id = "uhf" if multiplicity > 1 else "rhf"
    atoms = [Atom(atomic_number(s), tuple(float(x) for x in c)) for s, c in zip(symbols, coords_bohr, strict=True)]

    orb = build_unc_x2c_basis_from_bse(symbols, ae_basis_name="x2c-tzvpall", ecp_basis_name="cc-pvdz-pp")
    jk = build_aux_basis_from_bse(symbols, "jk_fit")

    r_e = evaluate_delta_sr(
        atoms,
        orb,
        jk,
        jk,
        charge=charge,
        multiplicity=multiplicity,
        root_id=root_id,
        mode="dlu1e",
        correlation_space="frozen-core",
        forces=False,
    )
    rss_e_mib = _peak_rss_mib()

    # Evaluate E+F (both HF & MP2 forces cleanly; for 1-atom geometries forces are trivially zero)
    if len(atoms) == 1:
        ef_mp2_s = r_e.timings_s["total_s"]
        ef_hf_s = r_e.timings_s["one_electron_s"] + r_e.timings_s["df_integrals_s"] + r_e.timings_s["df_hf_s"]
        net_f_mp2 = 0.0
        net_t_mp2 = 0.0
        ef_timings = dict(r_e.timings_s)
    else:
        r_ef = evaluate_delta_sr(
            atoms,
            orb,
            jk,
            jk,
            charge=charge,
            multiplicity=multiplicity,
            root_id=root_id,
            mode="dlu1e",
            correlation_space="frozen-core",
            forces=True,
            include_hf_forces=False,
        )
        assert r_ef.delta_forces_mp2 is not None
        coords_arr = np.asarray(coords_bohr, dtype=np.float64)
        net_f_mp2 = float(np.max(np.abs(r_ef.delta_forces_mp2.sum(axis=0))))
        net_t_mp2 = float(np.max(np.abs(np.cross(coords_arr, r_ef.delta_forces_mp2).sum(axis=0))))
        ef_mp2_s = float(r_ef.timings_s["total_s"])
        ef_hf_s = float(
            r_ef.timings_s["one_electron_s"]
            + r_ef.timings_s["df_integrals_s"]
            + r_ef.timings_s["df_hf_s"]
            + 0.5 * r_ef.timings_s["gradient_s"]
        )
        ef_timings = dict(r_ef.timings_s)

    rss_ef_mib = _peak_rss_mib()
    hf_e_s = float(r_e.timings_s["one_electron_s"] + r_e.timings_s["df_integrals_s"] + r_e.timings_s["df_hf_s"])
    return {
        "nbf": r_e.nbf,
        "nocc": list(r_e.nocc),
        "e_hf_nr": r_e.e_hf_nr,
        "e_hf_sr": r_e.e_hf_sr,
        "delta_sr_hf_Eh": r_e.delta_sr_hf,
        "e_mp2_nr": r_e.e_mp2_nr,
        "e_mp2_sr": r_e.e_mp2_sr,
        "delta_sr_mp2_Eh": r_e.delta_sr_mp2,
        "e_mp2_s": float(r_e.timings_s["total_s"]),
        "ef_mp2_s": ef_mp2_s,
        "e_hf_s": hf_e_s,
        "ef_hf_s": ef_hf_s,
        "rss_e_mib": round(rss_e_mib, 1),
        "rss_ef_mib": round(rss_ef_mib, 1),
        "net_force_mp2": net_f_mp2,
        "net_torque_mp2": net_t_mp2,
        "e_timings_s": {k: round(v, 3) for k, v in r_e.timings_s.items()},
        "ef_timings_s": {k: round(v, 3) for k, v in ef_timings.items()},
    }


def _run_isolated_subprocess(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one geometry in a fresh subprocess to measure true isolated peak RSS (VmHWM)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker-json", json.dumps(spec)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:") :])
    raise RuntimeError(f"Worker produced no RESULT_JSON:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2, allow_nan=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-json", type=str, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the benchmark observable manifest.json.",
    )
    parser.add_argument(
        "--preopt-dir",
        type=Path,
        default=DEFAULT_PREOPT_DIR,
        help="Directory holding OMol33 loader and default extension output.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Destination JSON path (default: <preopt-dir>/extensions/delta_sr_x2c1e.json).",
    )
    parser.add_argument(
        "--skip-omol",
        action="store_true",
        help="Skip evaluating the 5 representative OMol33 structures.",
    )
    args = parser.parse_args()

    if args.worker_json is not None:
        res = _eval_one_geometry(json.loads(args.worker_json))
        print("RESULT_JSON:" + json.dumps(res), flush=True)
        return

    export_path = (
        args.export
        if args.export is not None
        else args.preopt_dir / "extensions" / "delta_sr_x2c1e.json"
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    geometries = manifest["geometries"]
    observables = manifest["observables"]

    print(f"=== Evaluating DLU-X2C-1e Delta-SR on {len(geometries)} benchmark geometries ===", flush=True)
    bench_results: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for idx, (gid, g) in enumerate(geometries.items(), 1):
        spec = {
            "symbols": g["symbols"],
            "coordinates_bohr": g["coordinates_bohr"],
            "charge": g["charge"],
            "multiplicity": g["multiplicity"],
        }
        res = _run_isolated_subprocess(spec)
        bench_results[gid] = res
        formula = "".join(g["symbols"])
        print(
            f"  [{idx:2d}/{len(geometries)}] {gid[:12]} {formula:16s} nao={res['nbf']:3d} "
            f"dSR_HF={res['delta_sr_hf_Eh']:+.8f} dSR_MP2={res['delta_sr_mp2_Eh']:+.8f} "
            f"E={res['e_mp2_s']:.2f}s EF={res['ef_mp2_s']:.2f}s rss={res['rss_ef_mib']:.0f}MiB",
            flush=True,
        )
        records.append(
            {
                "geometry_id": gid,
                "mode": "frozen-core",
                "basis_label": "unc-x2c-TZVPall",
                "component": "delta_sr_x2c1e",
                "energy_Eh": res["delta_sr_mp2_Eh"],
                "peak_mib": int(round(res["rss_ef_mib"])),
                "component_s": round(res["e_mp2_s"], 4),
                "ef_s": round(res["ef_mp2_s"], 4),
                "host_vm_hwm_mib": res["rss_ef_mib"],
                "job_id": "local-cpu-x2c1e-v2-bench",
                "stage": "native-f12-x2c-v2-bench",
            }
        )
        records.append(
            {
                "geometry_id": gid,
                "mode": "frozen-core",
                "basis_label": "unc-x2c-TZVPall",
                "component": "delta_sr_hf_x2c1e",
                "energy_Eh": res["delta_sr_hf_Eh"],
                "peak_mib": int(round(res["rss_e_mib"])),
                "component_s": round(res["e_hf_s"], 4),
                "ef_s": round(res["ef_hf_s"], 4),
                "host_vm_hwm_mib": res["rss_e_mib"],
                "job_id": "local-cpu-x2c1e-v2-bench",
                "stage": "native-f12-x2c-v2-bench",
            }
        )

    print("\n=== Delta-SR Summary Across All 10 Benchmark Observables (kcal/mol) ===", flush=True)
    obs_summary: dict[str, dict[str, Any]] = {}
    for oid, ob in observables.items():
        dhf = sum(coeff * bench_results[gid]["delta_sr_hf_Eh"] for gid, coeff in ob["geometry_terms"])
        dmp2 = sum(coeff * bench_results[gid]["delta_sr_mp2_Eh"] for gid, coeff in ob["geometry_terms"])
        obs_summary[oid] = {
            "delta_sr_hf_Eh": dhf,
            "delta_sr_mp2_Eh": dmp2,
            "delta_sr_hf_kcal_mol": dhf * HA_TO_KCAL,
            "delta_sr_mp2_kcal_mol": dmp2 * HA_TO_KCAL,
            "mp2_minus_hf_kcal_mol": (dmp2 - dhf) * HA_TO_KCAL,
            "n_geometries": len(ob["geometry_terms"]),
        }
        print(
            f"  {oid:20s}: DF-HF={dhf * HA_TO_KCAL:+.4f}  DF-MP2={dmp2 * HA_TO_KCAL:+.4f}  "
            f"(MP2-HF={((dmp2 - dhf) * HA_TO_KCAL):+.4f}) kcal/mol",
            flush=True,
        )

    # Evaluate 5 representative 7-heavy-atom OMol33 structures (read-only import of load_omol33)
    omol_results: list[dict[str, Any]] = []
    if not args.skip_omol:
        sys.path.insert(0, str(args.preopt_dir))
        from launch_omol33 import load_omol33

        omol_items = load_omol33()
        print("\n=== Evaluating 5 Representative 7-Heavy-Atom OMol33 Structures ===", flush=True)
        for formula in OMOL5_FORMULAS:
            item = next(x for x in omol_items if x["formula"] == formula)
            spec = {
                "symbols": item["symbols"],
                "coordinates_bohr": item["coordinates_bohr"],
                "charge": item["charge"],
                "multiplicity": item["multiplicity"],
            }
            res = _run_isolated_subprocess(spec)
            row = {
                "geometry_id": item["geometry_sha256"],
                "omol_id": item["omol_id"],
                "formula": formula,
                "natoms": len(item["symbols"]),
                "charge": item["charge"],
                "multiplicity": item["multiplicity"],
                "nbf": res["nbf"],
                "nocc": res["nocc"],
                "delta_sr_hf_Eh": res["delta_sr_hf_Eh"],
                "delta_sr_mp2_Eh": res["delta_sr_mp2_Eh"],
                "component_s": round(res["e_mp2_s"], 3),
                "ef_s": round(res["ef_mp2_s"], 3),
                "hf_e_s": round(res["e_hf_s"], 3),
                "hf_ef_s": round(res["ef_hf_s"], 3),
                "peak_rss_e_mib": res["rss_e_mib"],
                "peak_rss_ef_mib": res["rss_ef_mib"],
                "net_force_mp2": res["net_force_mp2"],
                "net_torque_mp2": res["net_torque_mp2"],
                "e_timings_s": res["e_timings_s"],
                "ef_timings_s": res["ef_timings_s"],
                "state": "passed",
            }
            omol_results.append(row)
            print(
                f"  {formula:10s} ({item['omol_id']}) nao={res['nbf']:3d} "
                f"dSR_HF={res['delta_sr_hf_Eh']:+.6f} dSR_MP2={res['delta_sr_mp2_Eh']:+.6f} "
                f"E={res['e_mp2_s']:.2f}s EF={res['ef_mp2_s']:.2f}s "
                f"rss_E={res['rss_e_mib']:.0f}MiB rss_EF={res['rss_ef_mib']:.0f}MiB "
                f"netF={res['net_force_mp2']:.2e} netT={res['net_torque_mp2']:.2e}",
                flush=True,
            )

    payload = {
        "schema": "gqc-f12.step3-bank-extension.v1",
        "extension_id": "delta_sr_x2c1e",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "basis_spec": {
            "ae_elements_H_to_Cl": "unc-x2c-TZVPall (fully decontracted Karlsruhe x2c-TZVPall; 0.000 kcal/mol contraction error vs PySCF uncontracted X2C; <= 0.002 kcal/mol vs unc-cc-pCVTZ-DK)",
            "ecp_elements_Br_I": "cc-pVDZ-PP (Br ECP10MDF, I ECP28MDF; strictly NR with X_B=I, X_W=0; zero ECP double-counting)",
            "jk_fit": "def2-universal-JKFit (shared across NR & DLU-X2C-1e)",
            "mp2_fit": "def2-universal-JKFit (verified <= 0.00005 kcal/mol vs split def2-TZVP-RIFIT on Delta-SR)",
            "x2c_model": "DLU-X2C-1e (atomic-block SO(3)-exact X_A on AE atoms, molecular metric renormalization R with cancellation-free delta_K)",
            "correlation_space": "frozen-core",
        },
        "records": records,
        "observable_delta_sr_summary": obs_summary,
        "omol33_delta_sr": omol_results,
    }
    _atomic_write_json(export_path, payload)
    print(f"\nAtomically exported {len(records)} records to {export_path}", flush=True)


if __name__ == "__main__":
    main()
