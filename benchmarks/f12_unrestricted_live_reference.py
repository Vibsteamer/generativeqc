"""Run a native DF-UHF solve from geometry and use its two spin frames in MP2-F12."""

import argparse
import ctypes
import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_singles
from generativeqc._f12_spin_ao import unrestricted_orbital_sources
from generativeqc._f12_spin_pairs import molecular_pair_energies
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_extended_df_fock

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner
from benchmarks.f12_molecular_metric import metric_whitener


def evaluate(
    directory: Path,
    snap: dict,
    run: Callable,
    singles: Callable,
    whiten: Callable,
    *,
    geometry_sources: Callable | None = None,
) -> list[dict]:
    ref = json.loads((directory / "reference.json").read_bytes())
    space = tuple(ref["space"])
    na, nb, obs, ext, aux = space
    if tuple(snap["occupations"]) != (na, nb) or snap["nbf"] != obs:
        raise ValueError("The live UHF spin occupations or orbital space changed")
    provider = None
    if geometry_sources is None:
        with np.load(directory / "operator_input.npz", allow_pickle=False) as stored:
            ao = dict(stored)
        with np.load(directory / "orbital_frame.npz", allow_pickle=False) as f:
            embedding = f["embedding"]
            complement = f["cabs"]
            order = f["native_to_psi"]
            so = f["overlap_obs"][np.ix_(order, order)]
            hcore = f["hcore_obs"][np.ix_(order, order)]
    else:
        ao, embedding, complement, order, so, hcore, provider = geometry_sources(space)
    if any(k.startswith("coefficients_") or k == "metric_whitener" for k in ao):
        raise ValueError(
            "Unexpected precomputed spin orbitals or metric inverse in the live qualification"
        )
    if not np.array_equal(np.sort(order), np.arange(obs)):
        raise ValueError("Invalid AO ordering map")
    overlap = np.asarray(snap["overlap"]).reshape(obs, obs)
    norm = np.sqrt(np.diag(overlap) / np.diag(so))
    ao_error = float(np.max(abs(overlap - norm[:, None] * so * norm[None, :])))
    core_error = float(
        np.max(
            abs(
                np.asarray(snap["hcore"]).reshape(obs, obs)
                - norm[:, None] * hcore * norm[None, :]
            )
        )
    )
    if max(ao_error, core_error) > 1e-8:
        raise ValueError(
            f"Native and integral AO conventions disagree: {ao_error}, {core_error}"
        )
    gram_error = 0.0
    canonical = 0.0
    for spin in ("alpha", "beta"):
        orb = snap[spin]
        c = np.array(orb["mo"]).reshape(obs, obs)
        canonical = max(
            canonical,
            float(
                np.max(
                    abs(
                        c.T @ np.array(orb["fock"]).reshape(obs, obs) @ c
                        - np.diag(orb["eps"])
                    )
                )
            ),
        )
        csrc = np.empty_like(c)
        csrc[order] = norm[:, None] * c
        full = np.ascontiguousarray(np.hstack([embedding @ csrc, complement]))
        gram = full.T @ ao["overlap_ao"] @ full
        gram_error = max(gram_error, float(np.max(abs(gram - np.eye(ext)))))
        ao["coefficients_" + spin] = full
    if max(gram_error, canonical) > 2e-8:
        raise ValueError(
            f"Unqualified native UHF orbital frames {gram_error}, {canonical}"
        )
    w, info = whiten(ao["metric/coulomb"])
    if info["rank"] != aux:
        raise ValueError("Unexpected Coulomb fit rank")
    ao["metric_whitener"] = w
    data, _ = unrestricted_orbital_sources(space, ao, run)
    src = {
        k.removeprefix("source/"): a for k, a in data.items() if k.startswith("source/")
    }
    fock = run(unrestricted_extended_df_fock(na, nb, ext, aux), src).outputs
    sz = (na - nb) / 2
    ss = float(sz * (sz + 1) + nb - np.sum(data["cross_overlap"][:na, :nb] ** 2))
    if abs(ss - ref["reference_spin_square"]) > 2e-7:
        raise ValueError(f"Unrestricted spin root changed: <S^2>={ss}")
    discrepancies = {
        s: float(np.max(abs(fock["fock_" + s][:obs, :obs] - np.diag(snap[s]["eps"]))))
        for s in ("alpha", "beta")
    }
    if max(discrepancies.values()) > 2e-8:
        raise ValueError(
            f"Post-HF source and live UHF use different physical Fock operators {discrepancies}"
        )
    single = sum(
        singles(fock["fock_" + s], n, spin_weight=1, pullback=False)[0]
        for s, n in (("alpha", na), ("beta", nb))
    )
    pairs = list(molecular_pair_energies(space, data, src, fock, run))
    ump = sum(x.mp2 for x in pairs)
    f12 = sum(x.f12 for x in pairs)
    energy = snap["energy"] + ump + f12 + single
    diffs = {
        "DF_UHF": abs(snap["energy"] - ref["UHF_Eh"]),
        "UMP2_total": abs(snap["energy"] + ump - ref["Psi4_UMP2_total_Eh"]),
        "F12": abs(f12 - ref["F12_increment_Eh"]),
        "CABS": abs(single - sum(ref["CABS_singles_Eh"])),
        "total": abs(energy - ref["total_Eh"]),
    }
    forces = np.asarray(snap["forces"]).reshape(3, 3)
    err = float(np.max(np.abs(forces - np.array(ref["UHF_forces_Eh_per_bohr"]))))
    if (
        len(pairs) != 36
        or not np.isfinite([*diffs.values(), err]).all()
        or max(diffs.values()) > 2e-8
        or err > 3e-7
    ):
        raise ValueError(
            f"Live reference or molecular energy qualification failed {diffs}, force={err}"
        )
    scope = (
        "Native live DF-UHF and 36-pair UMP2-F12 energy from geometry and contracted "
        "Gaussian shells; native geometric CABS and UHF analytical forces."
        if provider is not None
        else "native live DF-UHF reference and 36-pair UMP2-F12 energy from supplied combined AO integrals, integral-only CABS subspace and metric. UHF reference analytical forces, correlation forces not taken in this test."
    )
    return [
        {
            "case": "NH2_D_f12",
            "scope": scope,
            **({"gaussian_provider": provider} if provider is not None else {}),
            "reference_backend": snap["backend"],
            "reference_iterations": snap["iterations"],
            "spin_square": ss,
            "native_reference_forces": forces.tolist(),
            "UHF_force_error": err,
            "UHF_Eh": snap["energy"],
            "UMP2_Eh": ump,
            "F12_Eh": f12,
            "CABS_Eh": single,
            "total_Eh": energy,
            "reference_errors": diffs,
            "obs_Fock_mismatch": discrepancies,
            "basis_alignment_error": max(ao_error, core_error),
            "orbital_gram_residual": gram_error,
            "pairs": len(pairs),
            "metric": info,
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--component-library", required=True)
    parser.add_argument("--snapshot-executable", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--save-reference", type=Path, help="Save the qualified live CUDA UHF snapshot"
    )
    parser.add_argument(
        "--save-gaussian-frame",
        type=Path,
        help="Save the qualified native geometric CABS, combined overlap and OBS mapping as NPZ",
    )
    parser.add_argument(
        "--gaussian-layout", type=Path, help="Contracted-shell integral layout NPZ"
    )
    parser.add_argument(
        "--gaussian-science", type=Path, help="Nuclear geometry and shell mapping JSON"
    )
    parser.add_argument(
        "--sink-library",
        type=Path,
        help="Native CUDA two-/three-center integral library",
    )
    parser.add_argument(
        "--one-electron-library", type=Path, help="Native combined-basis S/Hcore bridge"
    )
    add_molecular_graph_budget(parser)
    a = parser.parse_args()
    gaussian = (
        a.gaussian_layout,
        a.gaussian_science,
        a.sink_library,
        a.one_electron_library,
    )
    if any(gaussian) and not all(gaussian):
        parser.error(
            "Gaussian mode requires --gaussian-layout, --gaussian-science, --sink-library and --one-electron-library"
        )
    if a.save_gaussian_frame and not all(gaussian):
        parser.error("--save-gaussian-frame requires Gaussian mode")
    destinations = [
        path.resolve()
        for path in (a.output, a.save_reference, a.save_gaussian_frame)
        if path is not None
    ]
    if len(set(destinations)) != len(destinations):
        parser.error(
            "Measurement, reference and Gaussian frame output paths must be distinct"
        )
    with tempfile.TemporaryDirectory() as tmp:
        snap = Path(tmp) / "physical.json"
        subprocess.run(
            [
                str(a.snapshot_executable.resolve()),
                "cuda",
                str(a.fixtures / "nh2.in"),
                str(snap),
            ],
            check=True,
        )
        snapshot_bytes = snap.read_bytes()
        obs = json.loads(snapshot_bytes)
    if obs["backend"] != "cuda":
        raise RuntimeError("Reference was not on the CUDA DF-JK route")
    run, metrics = cuda_runner(a.arch, a.cache, max_bytes=a.budget_mib << 20)
    component = ctypes.CDLL(a.component_library)
    geometry_sources = None
    frame_output = {} if a.save_gaussian_frame else None
    if all(gaussian):
        from benchmarks.f12_unrestricted_gaussian import gaussian_sources

        geometry_sources = partial(
            gaussian_sources,
            science_path=a.gaussian_science,
            layout_path=a.gaussian_layout,
            snapshot_input=a.fixtures / "nh2.in",
            sink_library=a.sink_library,
            one_electron_library=a.one_electron_library,
            component_library=component,
            budget_mib=a.budget_mib,
            frame_output=frame_output,
        )
    rows = evaluate(
        a.fixtures,
        obs,
        run,
        partial(cabs_singles, library=component),
        partial(metric_whitener, library=a.component_library),
        geometry_sources=geometry_sources,
    )
    if a.save_reference:
        a.save_reference.write_bytes(snapshot_bytes)
    if frame_output is not None:
        with a.save_gaussian_frame.open("wb") as output:
            np.savez_compressed(
                output,
                snapshot_sha256=np.asarray(hashlib.sha256(snapshot_bytes).hexdigest()),
                **frame_output,
            )
    a.output.write_text(
        json.dumps({"rows": rows, "programs": metrics}, indent=2, allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
