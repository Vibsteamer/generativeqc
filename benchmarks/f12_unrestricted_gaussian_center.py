"""Evaluate one all-electron open-shell MP2-F12 center from Gaussian shells and a matched UHF state."""

from functools import partial
import json

import numpy as np

from generativeqc._f12_cabs import cabs_singles
from generativeqc._f12_unrestricted import unrestricted_mp2_f12_ao_response
from generativeqc.f12_orbital_response import unrestricted_frame_overlap_weights

from benchmarks.f12_unrestricted_gaussian import gaussian_orbital_forces, gaussian_sources


def center_energy_forces(
    geometry_path, layout_path, snapshot_input, snapshot, central, run, *,
    sink_library, forward_library, projector_library, canonical_library, orbital_library,
    budget_mib=1536, response_output=None,
):
    """Evaluate a matched canonical UHF snapshot and its Gaussian F12 correlation response.

    Central OBS coefficients, DF factors and whitener must refer to the same
    fitted Hamiltonian. The integral forward, energy pullback and metric
    derivative share one CABS frame; its columns and cotangents never cross a
    process boundary. Reference forces are those of the supplied UHF root.
    """
    geometry = json.loads(geometry_path.read_bytes())
    space = tuple(geometry["layout"]["space"])
    oa, ob, n, ext, q = space
    if tuple(snapshot["occupations"]) != (oa, ob) or snapshot["nbf"] != n:
        raise ValueError("The snapshot and geometry have different spin spaces")
    ao, embedding, complement, order, so, hcore, forward = gaussian_sources(
        space, science_path=geometry_path, layout_path=layout_path, snapshot_input=snapshot_input,
        sink_library=sink_library, one_electron_library=forward_library,
        component_library=projector_library, budget_mib=budget_mib,
    )
    overlap = np.asarray(snapshot["overlap"]).reshape(n, n)
    norms = np.sqrt(np.diag(overlap) / np.diag(so))
    scale = norms[:, None] * norms[None, :]
    errors = {
        "overlap": np.max(abs(scale * so - overlap)),
        "hcore": np.max(abs(scale * hcore - np.asarray(snapshot["hcore"]).reshape(n, n))),
    }
    whiten = np.asarray(central["whitener"])
    errors["whitening"] = np.max(abs(whiten @ ao["metric/coulomb"] @ whiten.T - np.eye(q)))
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
        errors["canonical_" + spin] = np.max(abs(native_mo.T @ np.asarray(block["fock"]).reshape(n,n)
                                                  @ native_mo - np.diag(energies[spin])))
        errors["gram_" + spin] = np.max(abs(frame.T @ ao["overlap_ao"] @ frame - np.eye(ext)))
        errors["mo_" + spin] = np.max(abs(c - central["coefficients_" + spin]))
        if occ:
            errors["occupation_" + spin] = np.max(abs(energies[spin][:occ] - central["occupied_energies_" + spin]))
            transformed = (c.T @ ao["ao_coulomb"] @ c).reshape(q, n*n)
            errors["df_" + spin] = np.max(abs((whiten @ transformed).reshape(q,n,n) - central["df_" + spin]))
        ao["coefficients_" + spin] = frame
    errors = {k:float(v) for k,v in errors.items()}
    if any(not np.isfinite(v) or v > 2e-7 for v in errors.values()):
        raise ValueError(f"UHF state or Coulomb fit differs from the shell geometry: {errors}")
    terms, direct = unrestricted_mp2_f12_ao_response(
        space, ao, run, singles=partial(cabs_singles, library=projector_library),
    )
    frame_weights = unrestricted_frame_overlap_weights(central, complement, direct, run)
    response = {} if response_output is not None else None
    with np.load(layout_path, allow_pickle=False) as layout:
        forces, sinks, _ = gaussian_orbital_forces(
            (oa,ob,n,q), central, None, direct, frame_weights, geometry, layout,
            np.asarray(snapshot["forces"]).reshape(-1,3), sink_library, run,
            canonical_energies=energies, canonical_library=canonical_library,
            orbital_response_library=orbital_library, budget_mib=budget_mib, orbital_response_output=response,
        )
    energy = dict(UHF_Eh=float(snapshot["energy"]), UMP2_Eh=float(terms.ordinary_mp2),
                  F12_Eh=float(terms.f12_increment), CABS_singles_Eh=[float(x) for x in terms.cabs_singles])
    energy["total_Eh"] = energy["UHF_Eh"] + energy["UMP2_Eh"] + energy["F12_Eh"] + sum(energy["CABS_singles_Eh"])
    if not np.isfinite(energy["total_Eh"]) or not np.isfinite(forces).all():
        raise ValueError("Nonfinite molecular energy or force")
    if response_output is not None:
        response_output.update(frame_weights=frame_weights, orbital=response, direct=direct)
    return energy, forces, {"active_pairs":terms.active_pairs, "root_errors":errors,
                            "forward":forward, "nuclear_response":sinks}
