"""Export independent spherical AO overlaps and a QR CABS oracle for H2 and water."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyscf
from pyscf import gto
from scipy.linalg import solve_triangular


def overlaps(
    elements: str, xyz: np.ndarray, direction: np.ndarray | None = None
) -> dict[str, Any]:
    atoms = list(zip(elements, xyz, strict=True))
    orbitals = gto.M(atom=atoms, basis="cc-pvdz-f12", unit="Bohr", verbose=0)
    auxiliary = gto.M(atom=atoms, basis="cc-pvdz-f12-optri", unit="Bohr", verbose=0)
    # A separate atom-centred probe samples the physical complementary space.
    probe_basis = {s: [[0, [0.083 if s == "H" else 0.15, 1.0]]] for s in elements}
    probes = gto.M(atom=atoms, basis=probe_basis, unit="Bohr", verbose=0)
    total = orbitals + auxiliary + probes
    obs = orbitals.nao
    n = obs + auxiliary.nao
    metric = total.intor_symmetric("int1e_ovlp")
    weights = np.arange(1.0, probes.nao + 1)
    weights /= np.linalg.norm(weights)
    b = metric[:n, n:] @ weights
    s = metric[:n, :n]
    obs_coeff = np.linalg.solve(s[:obs, :obs], b[:obs])
    result = {"s": s, "b": b, "b_perp": b - s[:, :obs] @ obs_coeff, "obs": obs}
    if direction is not None:
        center_velocity = np.tile(direction, (3, 1))
        ao_velocity = np.repeat(
            center_velocity,
            np.diff(total.aoslice_by_atom()[:, 2:], axis=1).ravel(),
            axis=0,
        )
        ip = total.intor("int1e_ipovlp", comp=3)
        bra = -np.einsum("ic,cij->ij", ao_velocity, ip)
        ds = bra + bra.T
        ds, db = ds[:n, :n], ds[:n, n:] @ weights
        v = np.linalg.solve(s, b)
        slope = -v @ ds @ v + obs_coeff @ ds[:obs, :obs] @ obs_coeff
        slope += 2.0 * (db @ v - db[:obs] @ obs_coeff)
        result.update(ds=ds, db=db, slope=slope)
    return result


def qr_frame(s: np.ndarray, obs: int) -> np.ndarray:
    # The AO feature vectors are columns of L.T. Complete QR removes the
    # original OBS in that Euclidean space without forming any Schur inverse.
    features = np.linalg.cholesky(s).T
    orthogonal, _ = np.linalg.qr(features[:, :obs], mode="complete")
    return solve_triangular(features, orthogonal[:, obs:], lower=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    output = parser.parse_args().output
    cases = {
        "h2_stretch": (
            "HH",
            [[0, 0, 0], [0.13, -0.05, 1.39]],
            [[0, 0, 0], [0.13, -0.05, 1.39]],
        ),
        "water_stretch": (
            "OHH",
            [[0, 0, 0], [1.45, 0.15, 1.09], [-1.42, 0.31, 1.08]],
            [[0, 0, 0], [1.45, 0.15, 1.09], [0, 0, 0]],
        ),
        "water_bend": (
            "OHH",
            [[0, 0, 0], [1.45, 0.15, 1.09], [-1.42, 0.31, 1.08]],
            [[0, 0, 0], [-1.09, 0, 1.45], [1.08, 0, 1.42]],
        ),
    }
    data = {"steps_bohr": np.array([4e-4, 2e-4])}
    meta = {
        "integrals": "PySCF/libcint spherical overlap and bra gradient",
        "pyscf": pyscf.__version__,
        "cases": {},
        "basis_sha256": {},
    }
    for basis in ("cc-pvdz-f12", "cc-pvdz-f12-optri"):
        records = {e: gto.basis.load(basis, e) for e in ("H", "O")}
        meta["basis_sha256"][basis] = hashlib.sha256(
            json.dumps(records).encode()
        ).hexdigest()
    for name, (elements, positions, displacement) in cases.items():
        xyz = np.array(positions, dtype=float)
        u = np.array(displacement, dtype=float)
        u -= u.mean(axis=0)
        u /= np.linalg.norm(u)
        central = overlaps(elements, xyz, u)
        obs = central["obs"]
        c = qr_frame(central["s"], obs)
        k = c @ c.T
        data.update({name + "_" + field: value for field, value in central.items()})
        data[name + "_k"] = k
        meta["cases"][name] = {
            "elements": list(elements),
            "coordinates_bohr": xyz.tolist(),
            "unit_direction": u.tolist(),
            "obs": obs,
            "aux": len(k) - obs,
        }
        for i, h in enumerate(data["steps_bohr"]):
            for sign, label in ((-1, "minus"), (1, "plus")):
                moved = overlaps(elements, xyz + sign * h * u)
                start = f"{name}_{i}_{label}_"
                data[start + "s"], data[start + "b_perp"] = moved["s"], moved["b_perp"]
                cm = qr_frame(moved["s"], obs)
                amplitudes = cm.T @ moved["b"]
                data[start + "observable"] = amplitudes @ amplitudes
    data["metadata"] = np.array(json.dumps(meta))
    np.savez_compressed(output, **data)


if __name__ == "__main__":
    main()
