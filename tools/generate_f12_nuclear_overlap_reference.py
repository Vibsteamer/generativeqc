"""Independent physical-atom overlap gradients and all-Cartesian CABS probes."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyscf
from generate_f12_projector_reference import overlaps, qr_frame
from pyscf import gto


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError("Do not overwrite a registered scientific fixture")
    started = time.monotonic()
    payload = {"steps": np.array([4e-4, 2e-4])}
    metadata = {
        "scope": "CPU independent CABS projection norm and overlap gradient only; no SCF",
        "oracle": f"PySCF {pyscf.__version__}/libcint and independent Cholesky/QR complement",
        "cases": {},
    }
    cases = {
        "h2": ("HH", [[0, 0, 0], [0.13, -0.05, 1.39]]),
        "water": ("OHH", [[0, 0, 0], [1.45, 0.15, 1.09], [-1.42, 0.31, 1.08]]),
    }
    for name, (elements, positions) in cases.items():
        xyz = np.array(positions, dtype=float)
        atoms = list(zip(elements, xyz, strict=True))
        obs_mol = gto.M(atom=atoms, basis="cc-pvdz-f12", unit="Bohr", verbose=0)
        aux_mol = gto.M(atom=atoms, basis="cc-pvdz-f12-optri", unit="Bohr", verbose=0)
        probes = gto.M(
            atom=atoms,
            basis={s: [[0, [0.083 if s == "H" else 0.15, 1.0]]] for s in elements},
            unit="Bohr",
            verbose=0,
        )
        total = obs_mol + aux_mol + probes
        central = overlaps(elements, xyz)
        s, b, obs = central["s"], central["b"], central["obs"]
        n, natoms = len(s), len(atoms)
        weights = np.arange(1.0, probes.nao + 1)
        weights /= np.linalg.norm(weights)
        v = np.linalg.solve(s, b)
        vo = np.linalg.solve(s[:obs, :obs], b[:obs])
        ws = -np.outer(v, v)
        ws[:obs, :obs] += np.outer(vo, vo)
        wb = 2 * v
        wb[:obs] -= 2 * vo
        full_weight = np.zeros((total.nao, total.nao))
        full_weight[:n, :n] = ws
        full_weight[:n, n:] = wb[:, None] * weights[None, :]
        ip = total.intor("int1e_ipovlp", comp=3)
        gradient = np.zeros((natoms, 3))
        for atom, (_, _, lo, hi) in enumerate(total.aoslice_by_atom()):
            gradient[atom % natoms] -= np.einsum(
                "cij,ij->c", ip[:, lo:hi], (full_weight + full_weight.T)[lo:hi]
            )
        frame = qr_frame(s, obs)
        observable = float(np.linalg.norm(frame.T @ b) ** 2)
        displaced_values, displaced_s, displaced_b = [], [], []
        for step in payload["steps"]:
            values, metrics, vectors = [], [], []
            for coordinate in range(xyz.size):
                for sign in (1, -1):
                    moved = xyz.copy()
                    moved.flat[coordinate] += sign * step
                    d = overlaps(elements, moved)
                    c = qr_frame(d["s"], obs)
                    values.append(float(np.linalg.norm(c.T @ d["b"]) ** 2))
                    metrics.append(d["s"])
                    vectors.append(d["b_perp"])
            displaced_values.append(np.array(values).reshape(natoms, 3, 2))
            displaced_s.append(np.array(metrics))
            displaced_b.append(np.array(vectors))
        fd = [
            (a[:, :, 0] - a[:, :, 1]) / (2 * h)
            for a, h in zip(displaced_values, payload["steps"], strict=True)
        ]
        errors = [float(np.max(abs(a - gradient))) for a in fd]
        if max(errors) > 3e-8:
            raise ValueError(
                f"Independent all-Cartesian source failed for {name}: {errors}"
            )
        shells = []
        for ish in range(total.nbas):
            for column in total.bas_ctr_coeff(ish).T:
                shells.append(
                    {
                        "atom_index": int(total.bas_atom(ish)) % natoms,
                        "angular_momentum": int(total.bas_angular(ish)),
                        "primitives": [
                            [float(a), float(c)]
                            for a, c in zip(total.bas_exp(ish), column, strict=True)
                        ],
                    }
                )
        record = {
            "elements": list(elements),
            "coordinates_bohr": xyz.tolist(),
            "obs": obs,
            "aux": aux_mol.nao,
            "probe": probes.nao,
            "shells": shells,
            "max_angular_momentum": max(s["angular_momentum"] for s in shells),
            "source_order": "OBS all atoms, OptRI all atoms, probe all atoms; shell center IDs collapsed to physical atoms",
            "reference_fd_errors": errors,
        }
        metadata["cases"][name] = record
        arrays = {
            "s": s,
            "b": b,
            "b_perp": central["b_perp"],
            "weights": weights,
            "gradient": gradient,
            "probe_value": np.array(observable),
            "reference_overlap_weight": full_weight,
            "displaced_values": np.array(displaced_values),
            "displaced_s": np.array(displaced_s),
            "displaced_b": np.array(displaced_b),
        }
        payload.update({name + "/" + key: value for key, value in arrays.items()})
    metadata["elapsed_seconds"] = time.monotonic() - started
    payload["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(output, **payload)
    print(
        json.dumps(
            {
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "elapsed_seconds": metadata["elapsed_seconds"],
                "cases": {
                    k: v["reference_fd_errors"] for k, v in metadata["cases"].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
