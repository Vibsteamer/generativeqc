"""Build physical-atom S and hcore derivative oracles for a molecular F12 source."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import psi4
import pyscf
from psi4.driver import qcdb
from pyscf import gto, scf


def combined_shells(
    elements: list[str], xyz: np.ndarray, obs: str, opt: str
) -> tuple[Any, list[dict], np.ndarray]:
    """Keep one physical nucleus per site for both orbital and OptRI shells."""
    geometry = "0 1\nunits bohr\nsymmetry c1\nno_com\nno_reorient\n" + "\n".join(
        e + " " + " ".join(map(str, point))
        for e, point in zip(elements, xyz, strict=True)
    )
    molecule = psi4.geometry(geometry)
    molecule.update_geometry()
    spec = qcdb.libmintsbasisset.BasisSet.pyconstruct_combined(
        molecule.to_string(dtype="psi4"),
        ["BASIS", "CABS_BASIS"],
        [obs, opt],
        ["ORBITAL", "F12"],
        [obs, obs],
    )
    basis = psi4.core.BasisSet.construct_from_pydict(molecule, spec, spec["puream"])
    shells = []
    native_to_source = []
    source = sorted(
        (basis.shell(i) for i in range(basis.nshell())), key=lambda s: (s.ncenter, s.am)
    )
    for shell in source:
        angular = shell.am
        if shell.nfunction != 2 * angular + 1:
            raise ValueError("the F12 reference requires real spherical AOs")
        # Psi4 orders m as 0,+1,-1,... . Native/libcint use x,y,z for p
        # and increasing signed m for higher l. Both have the same phases.
        psi_order = [0] + [m for k in range(1, angular + 1) for m in (k, -k)]
        public_order = [1, -1, 0] if angular == 1 else range(-angular, angular + 1)
        native_to_source.extend(
            shell.function_index + psi_order.index(m) for m in public_order
        )
        shells.append(
            {
                "atom_index": shell.ncenter,
                "angular_momentum": angular,
                "primitives": [
                    [shell.exp(k), shell.original_coef(k)]
                    for k in range(shell.nprimitive)
                ],
            }
        )
    p = np.array(native_to_source, dtype=np.int64)
    np.testing.assert_array_equal(np.sort(p), np.arange(basis.nbf()))
    return basis, shells, p


def matrices(mol: gto.Mole) -> np.ndarray:
    return np.stack(
        (
            mol.intor_symmetric("int1e_ovlp"),
            mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc"),
        )
    )


def reference(
    case: dict, archived_s: np.ndarray, archived_h: np.ndarray
) -> tuple[dict, dict]:
    elements, xyz = case["elements"], np.asarray(case["geometry_bohr"])
    opts = case["options"]
    _, shells, p = combined_shells(
        elements, xyz, opts["basis"].upper(), opts["cabs_basis"].upper()
    )
    labels = [f"{element}{i}" for i, element in enumerate(elements)]
    basis = {label: [] for label in labels}
    for shell in shells:
        basis[labels[shell["atom_index"]]].append(
            [shell["angular_momentum"], *shell["primitives"]]
        )
    mol = gto.M(
        atom=list(zip(labels, xyz, strict=True)),
        unit="Bohr",
        basis=basis,
        cart=False,
        verbose=0,
    )
    if mol.natm != len(elements) or mol.has_ecp():
        raise ValueError(
            "the fixture assumes one all-electron nucleus per geometry site"
        )
    central = matrices(mol)
    errors = np.max(
        abs(central - np.stack([a[np.ix_(p, p)] for a in (archived_s, archived_h)])),
        axis=(1, 2),
    )
    if not np.isfinite(errors).all() or max(errors) > 2e-11:
        raise ValueError(
            f"source and native-order integral matrices disagree: {errors}"
        )
    ds = -mol.intor("int1e_ipovlp", comp=3)
    dh = scf.RHF(mol).nuc_grad_method().hcore_generator(mol)
    grads = np.zeros((mol.natm, 3, *central.shape))
    for atom, (_, _, lo, hi) in enumerate(mol.aoslice_by_atom()):
        grads[atom, :, 0, lo:hi, :] += ds[:, lo:hi, :]
        grads[atom, :, 0, :, lo:hi] += ds[:, lo:hi, :].transpose(0, 2, 1)
        grads[atom, :, 1] = dh(atom)
    steps = np.array([4e-4, 2e-4])
    fd = np.empty((len(steps), *grads.shape))
    fd_five = np.empty_like(fd)
    coordinate = (1, int(np.argmax(abs(xyz[1] - xyz[0]))))

    def displaced(atom: int, axis: int, step: float) -> np.ndarray:
        geom = xyz.copy()
        geom[atom, axis] += step
        return matrices(mol.set_geom_(geom, unit="Bohr", inplace=False))

    for atom in range(mol.natm):
        for axis in range(3):
            previous = None
            for k, step in enumerate(np.r_[2 * steps[0], steps]):
                slope = (displaced(atom, axis, step) - displaced(atom, axis, -step)) / (
                    2 * step
                )
                if previous is not None:
                    fd[k - 1, atom, axis] = slope
                    fd_five[k - 1, atom, axis] = (4 * slope - previous) / 3
                previous = slope
    # The nonlinear source scalar is much stiffer along the partial molecular
    # path with fixed OBS and two-electron sources. Store three nested pairs
    # (six points total) to form two distinct fourth-order estimates.
    nonlinear_steps = np.array([5e-5, 2.5e-5, 1.25e-5])
    moved = np.array(
        [
            [displaced(*coordinate, sign * h) for sign in (1, -1)]
            for h in nonlinear_steps
        ]
    )
    meta = {
        "elements": elements,
        "coordinates_bohr": xyz.tolist(),
        "shells": shells,
        "nonlinear_coordinate": coordinate,
        "basis": opts["basis"],
        "opt_ri": opts["cabs_basis"],
        "overlap_error": float(errors[0]),
        "hcore_error_Eh": float(errors[1]),
    }
    arrays = {
        "native_to_source": p,
        "derivatives": grads,
        "fd": fd,
        "fd_five": fd_five,
        "moved": moved,
        "steps": steps,
        "nonlinear_steps": nonlinear_steps,
    }
    return meta, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--oracle-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    psi4.core.be_quiet()
    metadata = {"psi4": psi4.__version__, "pyscf": pyscf.__version__, "cases": {}}
    packed = {}
    with np.load(args.source, allow_pickle=False) as source:
        for name in ("h2", "water"):
            case = json.loads((args.oracle_directory / name / "READY.json").read_text())
            meta, arrays = reference(
                case, source[name + "/overlap"], source[name + "/hcore_ao"]
            )
            metadata["cases"][name] = meta
            packed.update({name + "/" + k: x for k, x in arrays.items()})
    with args.output.open("xb") as out:
        np.savez_compressed(out, metadata=json.dumps(metadata), **packed)


if __name__ == "__main__":
    main()
