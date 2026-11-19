"""Evaluate one all-electron open-shell MP2-F12 center from Gaussian shells and a matched UHF state."""

import ctypes
import json
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_singles
from generativeqc._f12_spin_fock import spin_fock
from generativeqc._f12_unrestricted import unrestricted_mp2_f12_ao_response
from generativeqc.f12_orbital_response import unrestricted_frame_overlap_weights
from generativeqc_compiler.mp2.f12_density_fitting import molecular_orbital_sources
from generativeqc_compiler.tensor import Index, Program, TensorSpec, einsum, input_tensor

from benchmarks.f12_cabs_h_batch import CabsHShells
from benchmarks.f12_molecular_metric import metric_whitener
from benchmarks.f12_unrestricted_gaussian import (
    gaussian_orbital_forces,
    gaussian_sources,
)


def shell_reference_fit(
    space: tuple[int, int, int, int, int],
    ao: Mapping[str, np.ndarray],
    energies: Mapping[str, np.ndarray],
    run: Callable,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Transform and whiten the Gaussian OBS factors and check both physical Focks."""
    oa, ob, n, _, q = space
    atomic = ao["overlap_ao"].shape[0]
    transform = molecular_orbital_sources(atomic, n, 0, q).outputs
    raw = transform["raw_coulomb"]
    a = raw.spec.indices[0]
    b = Index("b", a.space)
    w = input_tensor("whitener", TensorSpec((a, b), role="input"))
    graph = Program({"df": einsum("ab,bpq->apq", w, raw), "hcore": transform["hcore"]})
    central = {"whitener": ao["metric_whitener"]}
    fock_inputs = {}
    for spin, occ in (("alpha", oa), ("beta", ob)):
        c = np.ascontiguousarray(ao["coefficients_" + spin][:, :n])
        inputs = {
            "coefficients": c,
            "ao_coulomb": ao["ao_coulomb"],
            "hcore_ao": ao["hcore_ao"],
            "whitener": central["whitener"],
        }
        values = run(graph, inputs).outputs
        df = values["df"]
        central["coefficients_" + spin] = c
        central["df_" + spin] = df
        central["occupied_energies_" + spin] = energies[spin][:occ]
        fock_inputs["hcore_" + spin] = values["hcore"]
        fock_inputs["raw_coulomb_" + spin] = df
        if occ:
            fock_inputs["fit_" + spin] = np.ascontiguousarray(df[:, :occ, :])
    focks = spin_fock((oa, ob, n, q), fock_inputs, run)
    errors = {
        spin: float(np.max(abs(focks["fock_" + spin] - np.diag(energies[spin]))))
        for spin in ("alpha", "beta")
    }
    if any(not np.isfinite(v) or v > 2e-8 for v in errors.values()):
        raise ValueError(f"Shell DF Fock differs from the UHF reference: {errors}")
    return central, errors


def center_energy_forces(
    geometry_path: Path,
    layout_path: Path,
    snapshot_input: Path,
    snapshot: dict,
    central: Mapping[str, np.ndarray] | None,
    run: Callable,
    *,
    sink_library: Path,
    forward_library: Path,
    cabs_h_one_electron_library: Path | None = None,
    metric_library: Path | None = None,
    projector_library: ctypes.CDLL,
    canonical_library: ctypes.CDLL,
    orbital_library: ctypes.CDLL,
    budget_mib: int = 1536,
    response_output: dict | None = None,
) -> tuple[dict, np.ndarray, dict]:
    """Evaluate a matched canonical UHF snapshot and its Gaussian F12 correlation response.

    With central=None the CUDA metric and the shell integrals determine the
    orbital-response factors of both spins. A supplied center must match the
    same fitted Hamiltonian. The integral forward, energy pullback and metric
    derivative share one CABS frame; its columns and cotangents never cross a
    process boundary. Reference forces are those of the supplied UHF root.
    """
    geometry = json.loads(geometry_path.read_bytes())
    space = tuple(geometry["layout"]["space"])
    oa, ob, n, ext, q = space
    if tuple(snapshot["occupations"]) != (oa, ob) or snapshot["nbf"] != n:
        raise ValueError("The snapshot and geometry have different spin spaces")
    high = None
    basis = geometry["combined_basis"]
    maximum_l = max(shell["angular_momentum"] for shell in basis["shells"])
    if maximum_l > 5:
        raise ValueError("The combined one-electron basis supports at most h shells")
    if maximum_l == 5:
        if cabs_h_one_electron_library is None:
            raise ValueError(
                "The combined h-shell basis requires its one-electron CUDA library"
            )
        with np.load(layout_path, allow_pickle=False) as layout:
            high = CabsHShells(
                basis,
                geometry["layout"]["native_to_source"],
                layout,
                cabs_h_one_electron_library,
            )
    ao, embedding, complement, order, so, hcore, forward = gaussian_sources(
        space,
        science_path=geometry_path,
        layout_path=layout_path,
        snapshot_input=snapshot_input,
        sink_library=sink_library,
        one_electron_library=forward_library,
        component_library=projector_library,
        budget_mib=budget_mib,
        cabs_h_shells=high,
    )
    overlap = np.asarray(snapshot["overlap"]).reshape(n, n)
    norms = np.sqrt(np.diag(overlap) / np.diag(so))
    scale = norms[:, None] * norms[None, :]
    errors = {
        "overlap": np.max(abs(scale * so - overlap)),
        "hcore": np.max(
            abs(scale * hcore - np.asarray(snapshot["hcore"]).reshape(n, n))
        ),
    }
    fit_diagnostics = {}
    if central is None:
        if metric_library is None:
            raise ValueError(
                "Building the shell DF factors requires the CUDA metric library"
            )
        whiten, fit_diagnostics["metric"] = metric_whitener(
            ao["metric/coulomb"], metric_library, budget_bytes=budget_mib << 20
        )
    else:
        whiten = np.asarray(central["whitener"])
    errors["whitening"] = np.max(
        abs(whiten @ ao["metric/coulomb"] @ whiten.T - np.eye(q))
    )
    ao["metric_whitener"] = whiten
    energies = {}
    for spin, occ in (("alpha", oa), ("beta", ob)):
        block = snapshot[spin]
        native_mo = np.asarray(block["mo"]).reshape(n, n)
        source = np.empty_like(native_mo)
        source[order] = norms[:, None] * native_mo
        c = embedding @ source
        frame = np.ascontiguousarray(np.column_stack((c, complement)))
        energies[spin] = np.asarray(block["eps"])
        errors["canonical_" + spin] = np.max(
            abs(
                native_mo.T @ np.asarray(block["fock"]).reshape(n, n) @ native_mo
                - np.diag(energies[spin])
            )
        )
        errors["gram_" + spin] = np.max(
            abs(frame.T @ ao["overlap_ao"] @ frame - np.eye(ext))
        )
        if central is not None:
            errors["mo_" + spin] = np.max(abs(c - central["coefficients_" + spin]))
            if occ:
                errors["occupation_" + spin] = np.max(
                    abs(energies[spin][:occ] - central["occupied_energies_" + spin])
                )
                transformed = (c.T @ ao["ao_coulomb"] @ c).reshape(q, n * n)
                errors["df_" + spin] = np.max(
                    abs((whiten @ transformed).reshape(q, n, n) - central["df_" + spin])
                )
        ao["coefficients_" + spin] = frame
    errors = {k: float(v) for k, v in errors.items()}
    if any(not np.isfinite(v) or v > 2e-7 for v in errors.values()):
        raise ValueError(
            f"UHF state or Coulomb fit differs from the shell geometry: {errors}"
        )
    if central is None:
        central, fit_diagnostics["canonical_Fock_errors_Eh"] = shell_reference_fit(
            space, ao, energies, run
        )
    terms, direct = unrestricted_mp2_f12_ao_response(
        space,
        ao,
        run,
        singles=partial(cabs_singles, library=projector_library),
    )
    frame_weights = unrestricted_frame_overlap_weights(central, complement, direct, run)
    response = {} if response_output is not None else None
    with np.load(layout_path, allow_pickle=False) as layout:
        forces, sinks, _ = gaussian_orbital_forces(
            (oa, ob, n, q),
            central,
            None,
            direct,
            frame_weights,
            geometry,
            layout,
            np.asarray(snapshot["forces"]).reshape(-1, 3),
            sink_library,
            run,
            canonical_energies=energies,
            canonical_library=canonical_library,
            orbital_response_library=orbital_library,
            budget_mib=budget_mib,
            orbital_response_output=response,
            cabs_h_shells=high,
        )
    energy = {
        "UHF_Eh": float(snapshot["energy"]),
        "UMP2_Eh": float(terms.ordinary_mp2),
        "F12_Eh": float(terms.f12_increment),
        "CABS_singles_Eh": list(terms.cabs_singles),
    }
    energy["total_Eh"] = (
        energy["UHF_Eh"]
        + energy["UMP2_Eh"]
        + energy["F12_Eh"]
        + sum(energy["CABS_singles_Eh"])
    )
    if not np.isfinite(energy["total_Eh"]) or not np.isfinite(forces).all():
        raise ValueError("Nonfinite molecular energy or force")
    if response_output is not None:
        response_output.update(
            frame_weights=frame_weights, orbital=response, direct=direct
        )
    return (
        energy,
        forces,
        {
            "active_pairs": terms.active_pairs,
            "shell_reference_fit": fit_diagnostics,
            "root_errors": errors,
            "forward": forward,
            "nuclear_response": sinks,
        },
    )
