"""Four nonzero public closed-shell focal components and two-step Cartesian forces on LiH."""

import argparse
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

XYZ = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 3.0]])
STEPS = (0.0004, 0.0002)
ROLES = ("hf_cabs", "mp2_f12_correlation", "ccsd_f12b_minus_mp2_f12", "triples")


def geometries() -> Iterator[tuple[str, np.ndarray]]:
    yield "center", XYZ
    for atom, axis in np.ndindex(XYZ.shape):
        for h in STEPS:
            for sign in (-1, 1):
                xyz = XYZ.copy()
                xyz[atom, axis] += sign * h
                yield label(atom, axis, sign * h), xyz


def label(atom: int, axis: int, displacement: float) -> str:
    return f"{atom}-{axis}-{displacement:+.4f}"


def atoms(xyz: np.ndarray) -> list[tuple[str, list[float]]]:
    return list(zip(("Li", "H"), xyz.tolist(), strict=True))


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reference(path: Path) -> None:
    import pyscf
    from pyscf import cc, gto, scf

    rows = {}
    for key, xyz in geometries():
        molecule = gto.M(atom=atoms(xyz), unit="bohr", basis="sto-3g", verbose=0)
        hf = scf.RHF(molecule).density_fit(auxbasis="def2-svp")
        hf.conv_tol = 1e-14
        hf.conv_tol_grad = 1e-11
        hf.max_cycle = 200
        hf.kernel()
        coupled = cc.CCSD(hf, frozen=0).density_fit(auxbasis="def2-svp")
        coupled.conv_tol = 1e-13
        coupled.conv_tol_normt = 1e-11
        coupled.max_cycle = 200
        coupled.kernel()
        correction = float(coupled.ccsd_t())
        assert hf.converged and coupled.converged and abs(correction) > 1e-6
        occupied = hf.mo_coeff[:, hf.mo_occ > 0]
        assert occupied.shape == (6, 2)
        rows[key] = {
            "geometry_bohr": xyz.tolist(),
            "HF_Eh": hf.e_tot,
            "triples_Eh": correction,
            "occupied_coefficients": occupied.tolist(),
        }
    write(
        path,
        {
            "pyscf": pyscf.__version__,
            "basis": "sto-3g",
            "jkfit": "def2-svp",
            "correlation_fit": "def2-svp",
            "frozen": 0,
            "points": rows,
        },
    )


def grade(result: dict, independent: dict) -> dict:
    rows = result["points"]
    assert set(rows) == set(independent["points"]) == {key for key, _ in geometries()}
    expected_rank = result["cabs_rank"]
    triples_error = []
    for key, xyz in geometries():
        row, control = rows[key], independent["points"][key]
        assert row["geometry_bohr"] == control["geometry_bohr"] == xyz.tolist()
        assert set(row["components"]) == set(ROLES)
        assert abs(sum(row["components"].values()) - row["energy_Eh"]) < 1e-12
        assert abs(row["HF_Eh"] - control["HF_Eh"]) < 3e-8
        assert (
            len(row["occupied"]) == 1
            and row["occupied"] == [2]
            and row["rank"] == expected_rank
        )
        assert row["independent_occupied_overlap"] > 0.99999
        if key != "center":
            assert (
                row["transported_occupied_overlap"] > 0.999
                and row["reference_has_forces"] is False
            )
        triples_error.append(abs(row["components"]["triples"] - control["triples_Eh"]))
    assert max(triples_error) < 2e-10
    center = rows["center"]
    errors = {}
    for h in STEPS:
        errors[str(h)] = {}
        for role in (*ROLES, "total", "independent_triples"):
            analytic = np.array(
                result["forces_Eh_per_bohr"][
                    "triples" if role == "independent_triples" else role
                ]
            )
            fd = np.empty_like(analytic)
            for a, j in np.ndindex(XYZ.shape):
                minus, plus = (label(a, j, sign * h) for sign in (-1, 1))
                if role == "independent_triples":
                    em, ep = (
                        independent["points"][k]["triples_Eh"] for k in (minus, plus)
                    )
                elif role == "total":
                    em, ep = (rows[k]["energy_Eh"] for k in (minus, plus))
                else:
                    em, ep = (rows[k]["components"][role] for k in (minus, plus))
                fd[a, j] = (em - ep) / (2 * h)
            err = float(np.max(abs(fd - analytic)))
            errors[str(h)][role] = err
            assert err < (
                2e-8 if role in ("triples", "independent_triples") else 3e-6
            ), (h, role, err)
    assert abs(center["components"]["ccsd_f12b_minus_mp2_f12"]) > 1e-5
    assert abs(center["components"]["triples"]) > 1e-6
    forces = np.array(result["forces_Eh_per_bohr"]["total"])
    assert np.max(abs(forces.sum(axis=0))) < 2e-8
    assert np.max(abs(np.cross(XYZ, forces).sum(axis=0))) < 2e-8
    return {
        "nonzero_triples_Eh": center["components"]["triples"],
        "independent_triples_max_energy_error_Eh": max(triples_error),
        "max_cartesian_force_errors_Eh_per_bohr": errors,
    }


def run(
    data: Path, main: Path, shell: Path, response: Path, out: Path, backend: str
) -> None:
    os.environ["GENERATIVEQC_LIBRARY"] = str(main.resolve())
    from generativeqc.experimental.f12_focal import (
        FocalBasis,
        FocalSeries,
        MolecularFocalProvider,
        singlepoint,
        singlepoint_energy,
    )

    task = json.loads((data / "task.json").read_bytes())
    independent = json.loads((data / "independent.json").read_bytes())
    basis = FocalBasis(
        2,
        "LiH finite STO-3G",
        "sto-3g",
        jk_fit="def2-svp",
        mp2_fit="def2-svp",
        f12_fit="def2-svp",
        cabs="def2-svp",
        beta=1.0,
    )
    series = FocalSeries(basis)
    graph = {"all": 0, "cc": 0}

    def record(row: dict) -> None:
        assert (
            row["backend"] == "cuda-fp64-ordinary-stream"
            and row["metrics"]["precision"] == "fp64"
        )
        graph["all"] += 1
        graph["cc"] += row.get("stage") == "cc"

    if backend == "cuda":
        provider = MolecularFocalProvider.cuda(
            shell_library=shell,
            response_library=response,
            architecture="sm_89",
            cache_directory=out.parent / "cache",
            record_graph=record,
            budget_mib=2048,
        )
        assert Path(provider.canonical_library._name).samefile(main)
    else:
        provider = MolecularFocalProvider(shell_library=shell, budget_mib=2048)
    args = {
        "hf_cabs": series,
        "correlation": series,
        "post_mp2": None,
        "f12_doubles": series,
        "triples": series,
        "root_id": "rhf",
        "multiplicity": 1,
        "correlation_space_id": "all-active",
        "backend": backend,
        "protocol_id": "LiH/RHF/STO-3G/def2-SVP/all-active/four-role",
        "provider": provider,
    }
    begin = time.monotonic()
    rows = {}
    out.mkdir(exist_ok=True)
    for key, xyz in geometries():
        t = time.monotonic()
        if key == "center":
            center = singlepoint(atoms(xyz), **args)
            ans = center
            forces = {v.term.role: v.term.forces.tolist() for v in center.components}
            forces["total"] = center.forces.tolist()
            components = {v.term.role: v.term.energy for v in ans.components}
            assert all(v.term.basis_id == basis.basis_id for v in ans.components)
        else:
            ans = singlepoint_energy(atoms(xyz), center=center, **args)
            components = {v.request.role: v.energy for v in ans.components}
            assert all(
                v.request.basis.basis_id == basis.basis_id for v in ans.components
            )
        assert set(ans.references) == set(ans.cabs_ranks) == {basis.basis_id}
        ref = ans.references[basis.basis_id].reference
        assert ref.executed_backend == backend and ref._native_owner is not None
        coeff = np.array(independent["points"][key]["occupied_coefficients"])
        singular = np.linalg.svd(
            coeff.T @ ref.overlap @ ref.coefficients[0, :, :2], compute_uv=False
        )
        rows[key] = {
            "geometry_bohr": xyz.tolist(),
            "energy_Eh": ans.energy,
            "components": components,
            "HF_Eh": ref.energy,
            "occupied": list(ref.occupied),
            "rank": ans.cabs_ranks[basis.basis_id],
            "independent_occupied_overlap": float(min(singular)),
            "reference_has_forces": ref.forces is not None,
            "transported_occupied_overlap": None
            if key == "center"
            else min(ref.occupied_overlaps),
            "seconds": time.monotonic() - t,
        }
        print(key, ans.energy, rows[key]["seconds"], flush=True)
    result = {
        "backend": backend,
        "source_commit": task["source_commit"],
        "mode": task["mode"],
        "basis_id": basis.basis_id,
        "scope": "One finite STO-3G orbital basis with def2-SVP fitting and CABS.",
        "points": rows,
        "cabs_rank": center.cabs_ranks[basis.basis_id],
        "forces_Eh_per_bohr": forces,
        "graph_counts": graph,
        "analytic_centers": 1,
        "energy_only_points": len(rows) - 1,
        "wall_seconds": time.monotonic() - begin,
    }
    write(out / "observations.json", result)
    assert backend != "cuda" or graph["all"] > graph["cc"] > 0
    write(
        out / "measurements.json",
        {**result, **grade(result, independent), "passed": True},
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--oracle":
        reference(Path(sys.argv[2]))
    else:
        parser = argparse.ArgumentParser()
        for name in ("data", "main", "shell", "response", "out"):
            parser.add_argument(name, type=Path)
        parser.add_argument(
            "--backend", required=True, choices=("cpu_reference", "cuda")
        )
        arg = parser.parse_args()
        run(arg.data, arg.main, arg.shell, arg.response, arg.out, arg.backend)
