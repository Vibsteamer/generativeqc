"""Physical molecular DF-UHF/MP2-F12 pair sum with separate alpha and beta orbitals."""

import argparse
import ctypes
import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_singles
from generativeqc_compiler.mp2.f12_opposite_spin import opposite_spin_projected_pair
from generativeqc_compiler.mp2.f12_unrestricted import (
    projected_same_spin_f12_pair,
    unrestricted_extended_df_fock,
)

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner


def evaluate(
    directory: Path,
    run: Callable,
    singles: Callable,
    checkpoint: Callable | None = None,
) -> list[dict]:
    start = time.monotonic()
    reference = json.loads((directory / "NH2_parent_DF_MP2_F12.json").read_text())
    with np.load(directory / "nh2_parent_D_f12.npz", allow_pickle=False) as stored:
        data = dict(stored)
    oa, ob, obs, n, aux = map(int, data["space"])
    names = ("alpha", "beta")
    occs = (oa, ob)
    src = {
        k.removeprefix("source/"): v for k, v in data.items() if k.startswith("source/")
    }
    fock = run(unrestricted_extended_df_fock(oa, ob, n, aux), src).outputs
    errors = {k: float(np.max(abs(a - data["expected_" + k]))) for k, a in fock.items()}
    if any(not np.isfinite(v) or v > 2e-8 for v in errors.values()):
        raise ValueError(f"UHF Fock component {errors}")
    single = [
        float(singles(fock["fock_" + s], o, spin_weight=1, pullback=False)[0])
        for s, o in zip(names, occs, strict=True)
    ]
    if not np.allclose(single, reference["CABS_singles_Eh"], rtol=0, atol=1e-10):
        raise ValueError("CABS singles")
    checkpoints = [
        {
            "case": "NH2_D_f12",
            "scope": "spin-dependent extended DF Fock and CABS",
            "errors_Eh": errors,
            "alpha_beta_CABS_Eh": single,
        }
    ]
    if checkpoint:
        checkpoint(checkpoints)
    corrections = []
    doubles = []
    expected = {(p["spin"], *p["pair"]): p for p in reference["pairs"]}
    for sa, sb in ((0, 0), (0, 1), (1, 1)):
        nl, nr = occs[sa], occs[sb]
        sl, sr = names[sa], names[sb]
        same = sa == sb
        graph = (
            projected_same_spin_f12_pair(nl, obs - nl, n - obs, auxiliaries=aux)
            if same
            else opposite_spin_projected_pair(nl, nr, obs, n, auxiliaries=aux)
        )
        for i in range(nl):
            for j in range(i + 1 if same else 0, nr):
                dl, dr = src["fit_" + sl][:, i], src["fit_" + sr][:, j]
                raw_c = src["raw_coulomb_" + sr][:, j]
                if same:
                    vals = {
                        "select_i": np.eye(nl)[i],
                        "select_j": np.eye(nr)[j],
                        "fock": fock["fock_" + sl],
                        "exchange": fock["exchange_" + sl],
                        "fit_i": dl,
                        "fit_j": dr,
                        "raw_coulomb_j": raw_c,
                    }
                else:
                    vals = {
                        "select_i": np.eye(n)[i],
                        "select_j": np.eye(n)[j],
                        "cross_overlap": data["cross_overlap"],
                        "fit_i_alpha": dl,
                        "fit_j_beta": dr,
                        "raw_coulomb_j_beta": raw_c,
                        **fock,
                    }
                for op, rel in (
                    ("geminal", "f12"),
                    ("squared_geminal", "f12_squared"),
                    ("geminal_coulomb", "f12_over_coulomb"),
                    ("double_commutator", "double_commutator"),
                ):
                    la, rb = (
                        data["raw/" + op + "/" + sl][:, i],
                        data["raw/" + op + "/" + sr][:, j],
                    )
                    if same:
                        vals.update(
                            {
                                "raw_" + rel + "_i": la,
                                "raw_" + rel + "_j": rb,
                                "metric_" + rel: data["metric/" + op],
                            }
                        )
                    else:
                        vals.update(
                            {
                                "raw_" + op + "_i_alpha": la,
                                "raw_" + op + "_j_beta": rb,
                                "metric_" + op: data["metric/" + op],
                            }
                        )
                # Prepared CUDA requires contiguous row-major source buffers.
                result = run(
                    graph, {k: np.ascontiguousarray(a) for k, a in vals.items()}
                ).outputs
                e, u = (
                    float(result["pair_energy"]),
                    float(result["ordinary_pair_energy"]),
                )
                ref = expected[(sl + "_" + sr, i, j)]
                diff = max(abs(e - ref["F12_Eh"]), abs(u - ref["ordinary_Eh"]))
                if not np.isfinite(diff) or diff > 2e-9:
                    raise ValueError(f"Spin pair {sl}/{sr} {i},{j} differs by {diff}")
                corrections.append(e)
                doubles.append(u)
                checkpoints.append(
                    {
                        "case": "NH2_D_f12",
                        "scope": "density fitted pair",
                        "spin": sl + "_" + sr,
                        "ij": [i, j],
                        "F12_Eh": e,
                        "ordinary_Eh": u,
                        "error_Eh": diff,
                    }
                )
                if checkpoint:
                    checkpoint(checkpoints)
    total = reference["UHF_Eh"] + sum(single) + sum(corrections) + sum(doubles)
    errors = {
        "F12_Eh": abs(sum(corrections) - reference["F12_increment_Eh"]),
        "ordinary_vs_Psi4_Eh": abs(
            reference["UHF_Eh"] + sum(doubles) - reference["Psi4_UMP2_total_Eh"]
        ),
        "total_Eh": abs(total - reference["total_Eh"]),
    }
    if len(corrections) != len(expected) or any(
        not np.isfinite(v) or v > 3e-8 for v in errors.values()
    ):
        raise ValueError(
            f"Incomplete or inaccurate all-spin sum: {len(corrections)}, {errors}"
        )
    checkpoints.append(
        {
            "case": "NH2_D_f12",
            "scope": "all-pair DF UHF F12 molecular energy on supplied three-center and orbital UHF sources",
            "pairs": len(corrections),
            "total_Eh": total,
            "errors_Eh": errors,
            "elapsed_seconds_including_graph_compiles": time.monotonic() - start,
        }
    )
    if checkpoint:
        checkpoint(checkpoints)
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--component-library", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_molecular_graph_budget(parser)
    args = parser.parse_args()
    run, metrics = cuda_runner(args.arch, args.cache, max_bytes=args.budget_mib << 20)

    def checkpoint(rows: list[dict]) -> None:
        tmp = args.output.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"rows": rows, "programs": metrics}, indent=2, allow_nan=False)
            + "\n"
        )
        tmp.replace(args.output)

    evaluate(
        args.fixtures,
        run,
        partial(cabs_singles, library=ctypes.CDLL(args.component_library)),
        checkpoint,
    )


if __name__ == "__main__":
    main()
