"""Export libcint nuclear derivatives and independently re-solved molecular DF tiles."""

import argparse
from pathlib import Path

import numpy as np
from pyscf import df, gto
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def molecular_inputs(
    symbols: str, xyz: np.ndarray, direction: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    mol = gto.M(
        atom=list(zip(symbols, xyz, strict=True)),
        unit="Bohr",
        basis="cc-pvdz-f12",
        verbose=0,
    )
    aux = df.addons.make_auxmol(mol, "aug-cc-pvtz-ri")
    metric = aux.intor_symmetric("int2c2e")
    raw = df.incore.aux_e2(mol, aux, intor="int3c2e", aosym="s1")
    shells = mol.aoslice_by_atom()[:, 2:]
    # Include orbital products centred on the same and on different nuclei.
    left = np.array([shells[0, 0], shells[0, 1] - 1, shells[-1, 0], shells[1, 0] + 1])
    right = np.array(
        [shells[-1, 0], shells[0, 1] - 1, shells[-1, 0] + 1, shells[0, 0] + 1]
    )
    three = raw[left, right, :].T.copy()
    data = {"metric": metric, "three": three}
    if direction is not None:
        ao_vel = np.repeat(direction, np.diff(shells, axis=1).ravel(), axis=0)
        aux_vel = np.repeat(
            direction, np.diff(aux.aoslice_by_atom()[:, 2:], axis=1).ravel(), axis=0
        )
        dm = -np.einsum("xij,ix->ij", aux.intor("int2c2e_ip1", comp=3), aux_vel)
        dip1 = df.incore.aux_e2(mol, aux, intor="int3c2e_ip1", comp=3, aosym="s1")
        dip2 = df.incore.aux_e2(mol, aux, intor="int3c2e_ip2", comp=3, aosym="s1")
        dg = -np.einsum("xpQ,px->Qp", dip1[:, left, right], ao_vel[left])
        dg -= np.einsum("xpQ,px->Qp", dip1[:, right, left], ao_vel[right])
        dg -= np.einsum("xpQ,Qx->Qp", dip2[:, left, right], aux_vel)
        data.update(dmetric=dm + dm.T, dthree=dg)
    return data


def fixture(
    symbols: str, xyz: np.ndarray, direction: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    center = molecular_inputs(symbols, xyz, direction)
    j, g = center["metric"], center["three"]
    factor = cho_factor(j, lower=True)
    seed = rng.normal(size=g.shape)
    seed /= np.linalg.norm(seed)
    steps = np.array([4e-4, 2e-4])
    slopes = []
    for h in steps:
        vals = []
        for sign in (1, -1):
            displaced = molecular_inputs(symbols, xyz + sign * h * direction)
            vals.append(
                np.vdot(
                    seed, cho_solve(cho_factor(displaced["metric"]), displaced["three"])
                )
            )
        slopes.append((vals[0] - vals[1]) / (2 * h))
    return dict(
        **center,
        whitener=solve_triangular(factor[0], np.eye(len(j)), lower=True),
        seed=seed,
        expected=cho_solve(factor, g),
        steps=steps,
        slopes=np.array(slopes),
        xyz_bohr=xyz,
        direction=direction,
        eigenvalues=np.linalg.eigvalsh(j),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(17038)
    out = {}
    for name, symbols, xyz in [
        ("hydrogen", "HH", [[0.0, 0.0, 0.0], [0.09, 0.1, 1.40]]),
        ("water", "OHH", [[0.0, 0.0, 0.0], [1.45, 0.15, 1.09], [-1.42, 0.31, 1.08]]),
    ]:
        direction = np.zeros((len(symbols), 3))
        direction[-1] = [0.36, -0.48, 0.80]
        case = fixture(symbols, np.array(xyz), direction, rng)
        out.update({f"{name}/{k}": v for k, v in case.items()})
    np.savez_compressed(args.output, **out)


if __name__ == "__main__":
    main()
