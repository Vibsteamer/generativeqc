"""Geometry and contracted shells shared by both live UHF spin frames."""

import ctypes
import json
import time
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_projector
from generativeqc.elements import atomic_number

from benchmarks.f12_molecular_integral_forward import forward_integrals
from benchmarks.f12_molecular_one_electron_forward import combined_matrices


def _check_geometry(science: dict, manifest: dict, snapshot_input: Path) -> None:
    """Bind both integral providers to the actual nuclei passed to the UHF solver."""
    case = science["combined_basis"]
    geom = science["geometry"]
    with snapshot_input.open() as stream:
        natoms, _, _, _, charge, multiplicity = map(int, stream.readline().split())
        atoms = [stream.readline().split() for _ in range(natoms)]
    numbers = [int(row[0]) for row in atoms]
    xyz = np.array([list(map(float, row[1:])) for row in atoms])
    if (
        xyz.shape != (natoms, 3)
        or not np.isfinite(xyz).all()
        or [atomic_number(e) for e in case["elements"]] != numbers
        or case["elements"] != geom["elements"]
        or case["elements"] != manifest["elements"]
        or not np.array_equal(case["coordinates_bohr"], xyz)
        or not np.array_equal(geom["geometry_bohr"], xyz)
        or not np.array_equal(manifest["geometry_bohr"], xyz)
        or case["charge"] != charge
        or geom["charge"] != charge
        or case["multiplicity"] != multiplicity
        or geom["multiplicity"] != multiplicity
        or manifest["case"] != science["molecule"]
        or manifest["options"] != science["options"]
        or manifest["geometry_sha256"] != geom["geometry_sha256"]
    ):
        raise ValueError("Gaussian and live UHF nuclei or electronic states disagree")


def _permutation(values: list[int], size: int) -> np.ndarray:
    order = np.asarray(values)
    if (
        order.dtype.kind not in "iu"
        or order.shape != (size,)
        or not np.array_equal(np.sort(order), np.arange(size))
    ):
        raise ValueError("Invalid spherical AO permutation in Gaussian layout")
    return order


def gaussian_sources(
    space: tuple[int, int, int, int, int],
    *,
    science_path: Path,
    layout_path: Path,
    snapshot_input: Path,
    sink_library: Path,
    one_electron_library: Path,
    component_library: ctypes.CDLL,
    budget_mib: int,
    frame_output: dict[str, np.ndarray] | None = None,
) -> tuple:
    """Build ten operators, S/H, and an OBS-orthogonal geometric CABS on CUDA."""
    start = time.monotonic()
    science = json.loads(science_path.read_bytes())
    basis = science["combined_basis"]
    mapping = science["layout"]
    _, _, obs, ext, aux = space
    if tuple(mapping["space"]) != tuple(space):
        raise ValueError("Gaussian integral and live-reference orbital spaces disagree")
    with np.load(layout_path, allow_pickle=False) as layout:
        _check_geometry(science, json.loads(str(layout["manifest"])), snapshot_input)
        size, nfit = map(int, layout["ao_sizes"])
        if nfit != aux or not obs < ext <= size:
            raise ValueError("Inconsistent combined or fitting basis dimensions")
        combined_order = _permutation(mapping["native_to_source"], size)
        order = _permutation(mapping["obs_native_to_source"], obs)
        permutation = _permutation(mapping["ao_permutation"], size)
        obs_rows = permutation[:obs]
        if not np.array_equal(obs_rows, mapping["obs_to_combined_source"]):
            raise ValueError("OBS embedding disagrees with the CABS projector order")
        operators, costs = forward_integrals(
            layout, sink_library, budget_mib=budget_mib
        )
    overlap, hcore, one = combined_matrices(
        basis, combined_order, one_electron_library, maximum_bytes=budget_mib << 20
    )
    if (
        len(costs) != 10
        or any(
            c["backend"] != "cuda_fp64" or c["device_bytes"] <= 0
            for c in costs.values()
        )
        or one["backend"] != "native_cuda_fp64"
        or one["device_bytes"] <= 0
    ):
        raise ValueError("Missing physical CUDA Gaussian integral execution")

    projector = cabs_projector(
        overlap[np.ix_(permutation, permutation)],
        obs,
        expected_rank=ext - obs,
        library=component_library,
    )
    inputs = spin_ao_inputs(
        operators, overlap, hcore, permutation, order, projector.frame
    )
    if frame_output is not None:
        _, embedding, complement, _, _, _ = inputs
        source_obs = np.ix_(obs_rows, obs_rows)
        frame_output.update(
            cabs=complement,
            embedding=embedding,
            overlap_ao=overlap,
            overlap_obs=overlap[source_obs],
            hcore_obs=hcore[source_obs],
            native_to_source=combined_order,
            obs_native_to_source=order,
            obs_to_combined_source=obs_rows,
            ao_permutation=permutation,
            space=np.asarray(space),
            geometry_sha256=np.asarray(science["geometry"]["geometry_sha256"]),
        )
    resources = {
        "integrals": costs,
        "one_electron": one,
        "cabs_projector": {
            "backend": "native_cuda_fp64",
            "rank": projector.rank,
            "cutoff_margin": projector.cutoff_margin,
        },
        "elapsed_seconds": time.monotonic() - start,
    }
    return (*inputs, resources)


def spin_ao_inputs(
    operators: dict[str, np.ndarray],
    overlap: np.ndarray,
    hcore: np.ndarray,
    permutation: np.ndarray,
    order: np.ndarray,
    cabs_frame: np.ndarray,
) -> tuple:
    """Put the OBS columns, geometric complement and native operators in one AO order."""
    size, obs = len(permutation), len(order)
    obs_rows = permutation[:obs]
    embedding = np.zeros((size, obs))
    embedding[obs_rows, np.arange(obs)] = 1.0
    complement = np.empty_like(cabs_frame)
    complement[permutation] = cabs_frame
    # The spin contraction uses chemical operator names for its five auxiliary metrics.
    ao = {
        "overlap_ao": overlap,
        "hcore_ao": hcore,
        "metric/coulomb": operators["metric"],
        "metric/geminal": operators["metric_f12"],
        "metric/squared_geminal": operators["metric_f12_squared"],
        "metric/geminal_coulomb": operators["metric_f12_over_coulomb"],
        "metric/double_commutator": operators["metric_double_commutator"],
        **{k: v for k, v in operators.items() if k.startswith("ao_")},
    }
    native_obs_rows = obs_rows[order]
    block = np.ix_(native_obs_rows, native_obs_rows)
    return ao, embedding, complement, order, overlap[block], hcore[block]
