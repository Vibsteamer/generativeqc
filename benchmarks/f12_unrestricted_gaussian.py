"""Geometry and contracted shells shared by both live UHF spin frames."""

import ctypes
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_projector
from generativeqc.elements import atomic_number
from generativeqc.f12_orbital_response import (
    unrestricted_canonical_source_seeds,
    unrestricted_frame_overlap_weights,
    unrestricted_nuclear_source_weights,
    unrestricted_source_orbital_adjoint,
)

from benchmarks.f12_cabs_ecp import CabsEcpOneElectron
from benchmarks.f12_cabs_h_batch import CabsHShells
from benchmarks.f12_molecular_integral_forward import forward_integrals
from benchmarks.f12_molecular_one_electron import native_contraction
from benchmarks.f12_molecular_one_electron_forward import combined_matrices
from benchmarks.f12_molecular_sink_live import contract_live_weights

_METRICS = {
    "metric/coulomb": "metric",
    "metric/geminal": "metric_f12",
    "metric/squared_geminal": "metric_f12_squared",
    "metric/geminal_coulomb": "metric_f12_over_coulomb",
    "metric/double_commutator": "metric_double_commutator",
}


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
    cabs_h_shells: CabsHShells | CabsEcpOneElectron | None = None,
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
    if cabs_h_shells is None:
        overlap, hcore, one = combined_matrices(
            basis, combined_order, one_electron_library, maximum_bytes=budget_mib << 20
        )
    else:
        overlap, hcore, one = cabs_h_shells.matrices(
            one_electron_library, derivatives=True, max_bytes=budget_mib << 20
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
        relative_cutoff=science["options"].get("cabs_relative_cutoff", 1e-9),
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
        **{source: operators[name] for source, name in _METRICS.items()},
        **{k: v for k, v in operators.items() if k.startswith("ao_")},
    }
    native_obs_rows = obs_rows[order]
    block = np.ix_(native_obs_rows, native_obs_rows)
    return ao, embedding, complement, order, overlap[block], hcore[block]


def gaussian_orbital_forces(
    space: tuple[int, int, int, int],
    central: Mapping[str, np.ndarray],
    orbital_adjoint: np.ndarray | None,
    direct: Mapping[str, np.ndarray],
    frame_overlap: Mapping[str, np.ndarray] | None,
    geometry: dict,
    layout: Mapping[str, np.ndarray],
    reference_forces: np.ndarray,
    sink_library: Path,
    run: Callable,
    *,
    budget_mib: int = 512,
    canonical_energies: Mapping[str, np.ndarray] | None = None,
    canonical_library: ctypes.CDLL | None = None,
    canonical_output: dict[str, np.ndarray] | None = None,
    orbital_response_library: ctypes.CDLL | None = None,
    orbital_response_output: dict[str, np.ndarray] | None = None,
    one_electron_forward_library: Path | None = None,
    projector_library: ctypes.CDLL | None = None,
    frame_output: dict[str, np.ndarray] | None = None,
    cabs_h_shells: CabsHShells | CabsEcpOneElectron | None = None,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Project central electronic weights through the native Gaussian nuclear gradients.

    The two providers contract every atomic direction directly from the shared
    shell geometry, including the nuclear attraction center. No displaced or
    analytic AO derivative arrays are needed. The reference force, direct F12
    source pullback and frame seeds must belong to this root. When the orbital
    adjoint is omitted, form its electronic driver and solve from this same center.
    Return total forces, provider diagnostics and the four corrected weights.
    """
    manifest = json.loads(str(layout["manifest"]))
    basis = geometry["combined_basis"]
    dims = geometry["layout"]["space"]
    if (
        tuple(dims[:3]) + tuple(dims[4:]) != space
        or manifest["case"] != geometry["molecule"]
        or manifest["geometry_sha256"] != geometry["geometry"]["geometry_sha256"]
        or manifest["elements"] != basis["elements"]
        or manifest["geometry_bohr"] != basis["coordinates_bohr"]
        or manifest["options"] != geometry["options"]
    ):
        raise ValueError(
            "Gaussian integral providers require the same spin space and shell geometry"
        )
    gauge = {}
    if canonical_energies is not None:
        seeds, gauge = unrestricted_canonical_source_seeds(
            space[:2],
            central,
            canonical_energies,
            direct,
            run,
            library=canonical_library,
            maximum_bytes=budget_mib << 20,
        )
        central = {**central, **seeds}
        if canonical_output is not None:
            canonical_output.update(seeds)
    if orbital_adjoint is None:
        if canonical_energies is None:
            raise ValueError("Computing the orbital adjoint requires canonical energies")
        driver, orbital_adjoint = unrestricted_source_orbital_adjoint(
            space, central, canonical_energies, run, maximum_bytes=budget_mib << 20,
            library=orbital_response_library,
        )
        if orbital_response_output is not None:
            orbital_response_output.update(driver=driver, rotations=orbital_adjoint)
    frame_resources = {}
    if frame_overlap is None:
        if one_electron_forward_library is None:
            raise ValueError("Regenerating the nuclear frame requires the combined-AO overlap provider")
        size = int(layout["ao_sizes"][0])
        order = _permutation(geometry["layout"]["native_to_source"], size)
        permutation = _permutation(geometry["layout"]["ao_permutation"], size)
        if not np.array_equal(permutation[:space[2]], geometry["layout"]["obs_to_combined_source"]):
            raise ValueError("OBS embedding disagrees with the orbital frame")
        if cabs_h_shells is None:
            overlap, _, one_forward = combined_matrices(
                basis, order, one_electron_forward_library, maximum_bytes=budget_mib << 20,
            )
        else:
            overlap, _, one_forward = cabs_h_shells.matrices(
                one_electron_forward_library, derivatives=True, max_bytes=budget_mib << 20
            )
        projector = cabs_projector(overlap[np.ix_(permutation, permutation)], space[2],
                                   expected_rank=dims[3] - space[2], library=projector_library,
                                   relative_cutoff=geometry["options"].get("cabs_relative_cutoff", 1e-9))
        complement = np.empty_like(projector.frame)
        complement[permutation] = projector.frame
        frame_overlap = unrestricted_frame_overlap_weights(central, complement, direct, run)
        if frame_output is not None:
            frame_output.update(frame_overlap)
        frame_resources = {"one_electron": one_forward, "rank": projector.rank,
                           "cutoff_margin": projector.cutoff_margin}
    corrected = unrestricted_nuclear_source_weights(
        space,
        central,
        orbital_adjoint,
        run,
        direct_weights=direct,
        frame_overlap=frame_overlap,
    )
    operators = {key: direct[key] for key in direct if key.startswith("ao_")}
    operators["ao_coulomb"] = corrected["ao_coulomb"]
    for key, name in _METRICS.items():
        operators[name] = corrected[key] if key in corrected else direct[key]
    gauss = contract_live_weights(
        layout, operators, sink_library, budget_mib=budget_mib
    )
    if cabs_h_shells is None:
        one = native_contraction(
            basis, corrected["overlap_ao"], corrected["hcore_ao"],
            np.asarray(geometry["layout"]["native_to_source"]),
        )
    else:
        one = cabs_h_shells.gradient(
            corrected["overlap_ao"], corrected["hcore_ao"], budget_mib << 20
        )
    ref = np.asarray(reference_forces)
    if (
        ref.shape != (len(geometry["combined_basis"]["elements"]), 3)
        or ref.dtype != np.float64
    ):
        raise ValueError("Expected a FP64 reference force for each nucleus")
    forces = ref - np.asarray(gauss["gradient_Eh_per_bohr"]) - one.gradient
    if not gauss["passes_contraction_checks"] or not np.isfinite(forces).all():
        raise ValueError("Gaussian response failed contraction checks")
    return (
        forces,
        {
            "Gaussian": gauss,
            "canonical_unresolved_torque": gauge,
            "regenerated_frame": frame_resources,
            "one_electron_gradient": one.gradient.tolist(),
            "one_electron_resources": one.resources,
        },
        corrected,
    )
