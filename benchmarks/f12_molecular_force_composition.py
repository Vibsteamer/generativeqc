"""Compose the molecular MP2-F12/3C(FIX) Cartesian force on a native RI-JK reference."""

import argparse
import ctypes
import json
import time
from argparse import Namespace
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._canonical_frame import canonical_frame_seed
from generativeqc._f12_cabs import cabs_projector, cabs_singles
from generativeqc._f12_molecular_response import (
    F12ReferenceResponse,
    molecular_reference_response,
    orbital_source_rows,
    with_reference_orbitals,
)
from generativeqc._one_electron_gradient import IntegralGradient
from numpy.typing import ArrayLike

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner
from benchmarks.df_reference_response import DFReferenceSession
from benchmarks.f12_cabs_h_batch import CabsHShells
from benchmarks.f12_molecular_fock import compensated_fock
from benchmarks.f12_molecular_frame import chain
from benchmarks.f12_molecular_integral_forward import forward_integrals
from benchmarks.f12_molecular_metric import metric_whitener
from benchmarks.f12_molecular_one_electron import native_contraction
from benchmarks.f12_molecular_one_electron_forward import combined_matrices
from benchmarks.f12_molecular_sink_live import contract_live_weights


def full_energy(reference: float, increment: float, ordinary: dict) -> float:
    """Combine a live HF energy and F12 increment with the matched native MP2 correlation."""
    if ordinary["backend"] != "cuda" or ordinary["passed"] is not True:
        raise ValueError("an accepted native DF-MP2 result is required")
    if abs(reference - ordinary["reference_energy_Eh"]) > 5e-11:
        raise ValueError("the separately evaluated DF-MP2 has a different RHF energy")
    energy = reference + increment + ordinary["correlation_energy_Eh"]
    if not np.isfinite(energy):
        raise ValueError("nonfinite molecular energy")
    return float(energy)


def full_result(
    reference: F12ReferenceResponse,
    explicit_one: ArrayLike,
    explicit_two_three: ArrayLike,
    ordinary: dict,
) -> tuple[float, np.ndarray]:
    """Reuse conventional MP2 correlation only. The reference force includes nuclei."""
    nref = np.asarray(reference.reference_forces)
    correlation = np.asarray(ordinary["correlation_forces_Eh_per_bohr"])
    terms = (correlation, np.asarray(explicit_one), np.asarray(explicit_two_three))
    if (
        nref.ndim != 2
        or nref.shape[1] != 3
        or any(v.shape != nref.shape or not np.isfinite(v).all() for v in terms)
    ):
        raise ValueError("incompatible molecular Cartesian forces")
    energy = full_energy(
        reference.reference_energy, reference.increment_energy, ordinary
    )
    forces = nref + correlation - terms[1] - terms[2]
    if not np.isfinite(forces).all():
        raise ValueError("nonfinite molecular endpoint")
    return float(energy), forces


def component_forces(
    total: np.ndarray,
    hf_forces: np.ndarray,
    correlation_forces: ArrayLike,
    singles_reference: F12ReferenceResponse,
    singles_one_gradient: ArrayLike,
    singles_integral_gradient: ArrayLike,
) -> dict[str, list]:
    """Remove RHF and nuclei from the singles adjoint and resolve the remaining pair force."""
    hf, corr, ref, one, two = map(
        np.asarray,
        (
            hf_forces,
            correlation_forces,
            singles_reference.reference_forces,
            singles_one_gradient,
            singles_integral_gradient,
        ),
    )
    if (
        total.ndim != 2
        or total.shape[1] != 3
        or any(
            x.shape != total.shape or not np.isfinite(x).all()
            for x in (total, hf, corr, ref, one, two)
        )
    ):
        raise ValueError("invalid molecular component forces")
    singles = ref - hf - one - two
    return {
        "reference": hf.tolist(),
        "ordinary_corr": corr.tolist(),
        "f12_pair": (total - hf - corr - singles).tolist(),
        "cabs_singles": singles.tolist(),
    }


def _numpy_scalar(value: object) -> object:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    raise TypeError(f"unsupported molecular result field: {type(value).__name__}")


def write_measurement(path: Path, result: dict) -> None:
    """Serialize before replacing the previous numerical checkpoint."""
    payload = json.dumps(result, default=_numpy_scalar, indent=2, allow_nan=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload + "\n")
    tmp.replace(path)


def endpoint_arguments() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "scf-inputs",
        "component-library",
        "session-library",
        "cache",
        "output",
        "layout",
        "sink-library",
        "panel",
    ):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--cabs-h-one-electron-library", type=Path)
    parser.add_argument(
        "--full-rank-cabs-response",
        action="store_true",
        help="Use the constraint derivative for a complete CABS and the projector derivative for a truncated CABS.",
    )
    parser.add_argument(
        "--component-forces",
        action="store_true",
        help="Resolve analytic CABS singles and F12 pair forces for component extrapolation.",
    )
    parser.add_argument("--arch", required=True)
    add_molecular_graph_budget(parser)
    return parser.parse_args()


def combined_basis_for_geometry(target: dict) -> dict:
    """Require one electronic state and physical geometry for both integral paths."""
    case, spec = target["combined_basis"], target["geometry"]
    if (
        case["elements"] != spec["elements"]
        or not np.array_equal(case["coordinates_bohr"], spec["geometry_bohr"])
        or case.get("charge", 0) != spec.get("charge", 0)
        or case.get("multiplicity", 1) != spec.get("multiplicity", 1)
    ):
        raise ValueError("combined basis and ordinary MP2 geometry or state disagree")
    return case


def calculate_panel(
    args: Namespace,
    target: dict,
    *,
    central_component: Callable | None = None,
    full_rank_constraint: bool = False,
) -> dict:
    """Compute and checkpoint central Cartesian E/F and requested native energy displacements."""
    name = target["molecule"]
    ordinary = target["ordinary"]
    spec = target["geometry"]
    lib = ctypes.CDLL(str(args.component_library.resolve()))
    canonical = partial(canonical_frame_seed, library=lib)
    run, programs = cuda_runner(args.arch, args.cache, max_bytes=args.budget_mib << 20)
    fock_projections = []

    def physical_fock(*sources: np.ndarray) -> np.ndarray:
        value, receipt = compensated_fock(
            *sources, library=args.component_library, budget_bytes=args.budget_mib << 20
        )
        fock_projections.append(receipt)
        return value

    def metric_gram(overlap: np.ndarray, frame: np.ndarray) -> np.ndarray:
        # The same compensated C.T A C with zero Coulomb and exchange terms.
        columns = frame.shape[1]
        return physical_fock(
            overlap,
            np.zeros((1, *overlap.shape)),
            frame,
            np.zeros((1, 1, columns)),
            np.zeros((columns, columns)),
        )

    reverse = partial(
        chain,
        metric_gram=metric_gram,
        physical_fock=physical_fock,
        run=run,
        projector=partial(cabs_projector, library=lib),
        singles=partial(cabs_singles, library=lib, spin_weight=2),
        full_rank_constraint=full_rank_constraint,
    )
    begin = time.monotonic()
    case = combined_basis_for_geometry(target)
    indices = np.array(target["layout"]["native_to_source"])
    permutation = np.array(target["layout"]["ao_permutation"])
    space = np.array(target["layout"]["space"])
    rows = orbital_source_rows(indices, permutation, int(space[1]))
    base_xyz = np.array(spec["geometry_bohr"])
    with np.load(args.layout, allow_pickle=False) as file:
        template = {k: file[k] for k in file}
    meta = json.loads(str(template["manifest"]))
    if (
        meta["case"] != name
        or meta["elements"] != spec["elements"]
        or not np.array_equal(meta["geometry_bohr"], base_xyz)
    ):
        raise ValueError("two- and three-center layout and MP2 geometry disagree")

    has_h = any(s["angular_momentum"] == 5 for s in case["shells"])
    h_library = getattr(args, "cabs_h_one_electron_library", None)
    if has_h and h_library is None:
        raise ValueError(
            "combined h-shell basis requires the CABS h one-electron library"
        )
    central_one = None

    def geometry_sources(xyz: np.ndarray, *, derivative: bool = False) -> tuple:
        nonlocal central_one
        geometry = xyz.tolist()
        topology = {
            **template,
            "centers": np.ascontiguousarray(xyz[template["atom"]]),
            "manifest": np.asarray(json.dumps({**meta, "geometry_bohr": geometry})),
        }
        tensors, costs = forward_integrals(
            topology, args.sink_library, budget_mib=args.budget_mib
        )
        if len(costs) != 10 or any(rec["device_bytes"] <= 0 for rec in costs.values()):
            raise ValueError("missing GPU integral source")
        factor, whitening = metric_whitener(tensors["metric"], args.component_library)
        frame_case = {**case, "coordinates_bohr": geometry}
        if has_h:
            high = CabsHShells(frame_case, indices, topology, h_library)
            overlap, hcore, one_cost = high.matrices(
                args.session_library,
                derivatives=derivative,
                max_bytes=args.budget_mib << 20,
            )
            if derivative:
                central_one = high
        else:
            overlap, hcore, one_cost = combined_matrices(
                frame_case, indices, args.session_library
            )
        atomic = len(indices)
        if (
            one_cost["device_bytes"] <= 0
            or whitening["device_bytes"] <= 0
            or whitening["rank"] != len(factor)
        ):
            raise ValueError("missing full-rank native CUDA metric")
        sources = {
            **tensors,
            "space": space,
            "ao_permutation": permutation,
            "overlap": overlap,
            "hcore_ao": hcore,
            "metric_whitener": factor,
            "coefficients": np.zeros((atomic, space[2])),
        }
        return (
            sources,
            topology,
            {"integrals": costs, "metric": whitening, "one_electron": one_cost},
        )

    data, topology, forward_cost = geometry_sources(base_xyz, derivative=True)

    def contract_explicit(
        response: F12ReferenceResponse, *, singles_only: bool
    ) -> tuple[IntegralGradient, dict]:
        weights = response.explicit_weights
        ws, wh = weights["overlap"], weights["hcore_ao"]
        if central_one is None:
            one = native_contraction(case, ws, wh, indices)
        else:
            one = central_one.gradient(ws, wh, args.budget_mib << 20)
        sinks = contract_live_weights(
            topology,
            weights,
            args.sink_library,
            budget_mib=args.budget_mib,
            singles_only=singles_only,
        )
        return one, sinks

    scalar_checkpoint_seconds = None
    central_components = {}
    split_report = {}
    with DFReferenceSession(
        args.session_library, args.scf_inputs / (name + "_central.in")
    ) as session:
        if central_component is not None:
            state = session.snapshot()
            data["coefficients"] = with_reference_orbitals(
                data["coefficients"], rows, state["coefficients"]
            )
            return central_component(data, reverse, programs, forward_cost)

        def record_energy(increment: float, components: dict[str, float]) -> None:
            nonlocal scalar_checkpoint_seconds
            central_components.update(components)
            reference = float(session.snapshot()["energy"])
            scalar_checkpoint_seconds = time.monotonic() - begin
            write_measurement(
                args.output,
                {
                    "case": name,
                    "passed": False,
                    "phase": "central_energy_complete",
                    "total_energy_Eh": full_energy(reference, increment, ordinary),
                    "reference_energy_Eh": reference,
                    "increment_Eh": increment,
                    **components,
                    "ordinary_corr_Eh": ordinary["correlation_energy_Eh"],
                    "forward_resources": forward_cost,
                    "graph_gpu_programs": programs,
                    "physical_fock_projections": fock_projections,
                    "conventional_input": target["ordinary_receipt"],
                    "scalar_checkpoint_seconds": scalar_checkpoint_seconds,
                },
            )

        obs = molecular_reference_response(
            session,
            data,
            indices,
            partial(reverse, record_energy=record_energy),
            run,
            canonical=canonical,
        )
        explicit, sinks = contract_explicit(obs, singles_only=False)
        obs = obs._replace(explicit_weights={})
        energy, forces = full_result(
            obs, explicit.gradient, sinks["gradient_Eh_per_bohr"], ordinary
        )
        if getattr(args, "component_forces", False):
            tick = time.monotonic()
            single = molecular_reference_response(
                session,
                data,
                indices,
                partial(reverse, singles_only=True),
                run,
                canonical=canonical,
            )
            if (
                abs(
                    single.increment_energy
                    - central_components["cabs_singles_energy_Eh"]
                )
                > 5e-11
            ):
                raise ValueError("isolated and joint singles energies differ")
            one, two = contract_explicit(single, singles_only=True)
            null_source = np.zeros((session.n, session.n))
            hf_forces, _ = session.response(null_source, null_source)
            split_report = {
                "component_forces_Eh_per_bohr": component_forces(
                    forces,
                    hf_forces,
                    ordinary["correlation_forces_Eh_per_bohr"],
                    single,
                    one.gradient,
                    two["gradient_Eh_per_bohr"],
                ),
                "cabs_singles_response": single.diagnostics,
                "cabs_singles_integral_gradients": two,
                "component_resolution_seconds": time.monotonic() - tick,
            }
            del single
    del data
    central_seconds = time.monotonic() - begin
    df_checks = []
    checkpoint = {
        "case": name,
        "passed": False,
        "phase": "central_complete",
        "total_energy_Eh": energy,
        "reference_energy_Eh": obs.reference_energy,
        "increment_Eh": obs.increment_energy,
        **central_components,
        **split_report,
        "ordinary_corr_Eh": ordinary["correlation_energy_Eh"],
        "total_force_Eh_per_bohr": forces.tolist(),
        "central_seconds": central_seconds,
        "scalar_checkpoint_seconds": scalar_checkpoint_seconds,
        "forward_resources": forward_cost,
        "displacements": df_checks,
        "forward_input": "native GPU from supplied geometry and explicit contracted shells",
        "conventional_input": target["ordinary_receipt"],
        "force_terms": {
            "reference_plus_orbital": obs.reference_forces.tolist(),
            "ordinary_correlation": ordinary["correlation_forces_Eh_per_bohr"],
            "explicit_one_electron_gradient": explicit.gradient.tolist(),
            "explicit_two_three_gradient": sinks["gradient_Eh_per_bohr"],
        },
        "integral_gradients": sinks,
        "orbital_response": obs.diagnostics,
        "one_electron_resources": explicit.resources,
        "physical_fock_projections": fock_projections,
        "graph_gpu_programs": programs,
        "reused_ordinary_central_seconds": ordinary["central_seconds"],
    }
    write_measurement(args.output, checkpoint)
    for arm in target["native_displacements"]:
        tick = time.monotonic()
        vector = np.array(arm["direction"])
        h = arm["step"]
        endpoints = []
        signed_components = []
        for sign, filename, (emp2, ehf) in zip(
            (1, -1), arm["files"], arm["mp2_and_hf_Eh"], strict=True
        ):
            xyz = base_xyz + sign * h * vector
            atoms = (
                (args.scf_inputs / filename).read_text().splitlines()[1 : len(xyz) + 1]
            )
            if not np.array_equal(
                xyz, [[float(a) for a in line.split()[1:]] for line in atoms]
            ):
                raise ValueError(
                    "finite-difference native SCF and integral geometries disagree"
                )
            moved, _, _cost = geometry_sources(xyz)
            with DFReferenceSession(
                args.session_library, args.scf_inputs / filename
            ) as point:
                snap = point.snapshot()
                if abs(snap["energy"] - ehf) > 5e-11:
                    raise ValueError("displaced ordinary and F12 SCF energies disagree")
                ix = np.ix_(rows, rows)
                for key, scf_key in (("overlap", "overlap"), ("hcore_ao", "hcore")):
                    np.testing.assert_allclose(
                        moved[key][ix], snap[scf_key], atol=2e-8, rtol=0
                    )
                moved["coefficients"] = with_reference_orbitals(
                    moved["coefficients"], rows, snap["coefficients"]
                )
                increment, _, components = reverse(moved, pullback=False)
            del moved
            # Archived MP2 totals already include the HF energy at this displaced geometry.
            base_energy = (
                emp2
                if arm.get("ordinary_total_only")
                else snap["energy"] + (emp2 - ehf)
            )
            endpoints.append(float(base_energy + increment))
            signed_components.append(
                {
                    "sign": sign,
                    "reference_energy_Eh": float(snap["energy"]),
                    "ordinary_corr_Eh": float(base_energy - snap["energy"]),
                    "increment_Eh": increment,
                    **components,
                }
            )
        error = abs(-(endpoints[0] - endpoints[1]) / (2 * h) - np.vdot(forces, vector))
        component_errors = {}
        if split_report:
            resolved = split_report["component_forces_Eh_per_bohr"]
            for term in ("f12_pair", "cabs_singles", "reference", "ordinary_corr"):
                field = term + ("_energy_Eh" if term != "ordinary_corr" else "_Eh")
                plus, minus = (point[field] for point in signed_components)
                projected = np.vdot(resolved[term], vector)
                component_errors[term] = float(
                    abs(projected + (plus - minus) / (2 * h))
                )
        df_checks.append(
            {
                "direction": arm["name"],
                "step_bohr": h,
                "energies_Eh": endpoints,
                "energy_components": signed_components,
                "component_force_errors_Eh_per_bohr": component_errors,
                "seconds": time.monotonic() - tick,
                "max_error_Eh_per_bohr": error,
            }
        )
        write_measurement(args.output, checkpoint)
    checkpoint["native_directional_checks"] = checkpoint.pop("displacements")
    checkpoint["composition_seconds"] = time.monotonic() - begin
    return checkpoint


def common_checks(result: dict, target: dict) -> dict:
    programs = result["graph_gpu_programs"]
    sinks = result["integral_gradients"]
    forces = np.asarray(result["total_force_Eh_per_bohr"])
    translation = float(np.max(np.abs(forces.sum(axis=0))))
    torque = float(
        np.max(
            np.abs(np.cross(target["geometry"]["geometry_bohr"], forces).sum(axis=0))
        )
    )
    result.update(net_force_max_Eh_per_bohr=translation, torque_max_Eh=torque)
    checks = {}
    if "component_forces_Eh_per_bohr" in result:
        single = result["cabs_singles_integral_gradients"]
        checks["physical_singles_response"] = (
            single["backend"] == "cuda_fp64"
            and set(single["rows"]) == {"metric", "ao_coulomb"}
            and single["max_device_bytes"] > 0
            and single["passes_contraction_checks"]
            and result["cabs_singles_response"]["orbital_residual"] <= 1e-8
        )
        arms = result["native_directional_checks"]
        if arms:
            checks["component_native_directions"] = (
                max(
                    max(row["component_force_errors_Eh_per_bohr"].values())
                    for row in arms
                )
                <= 2e-6
            )
        else:
            result["component_native_directional_validation"] = "center_only"
    return {
        **checks,
        "physical_cuda": bool(programs)
        and all(p["backend"] == "cuda-fp64-ordinary-stream" for p in programs)
        and sinks["backend"] == "cuda_fp64"
        and sinks["max_device_bytes"] > 0
        and result["one_electron_resources"]["device_bytes"] > 0
        and result["orbital_response"]["canonical_bytes"] > 0,
        "orbital_response": result["orbital_response"]["orbital_residual"] <= 1e-8,
        "translation": translation <= 2e-7,
        "torque": torque <= 2e-6,
    }


def main() -> None:
    args = endpoint_arguments()
    target = json.loads(args.panel.read_bytes())
    report = calculate_panel(
        args, target, full_rank_constraint=args.full_rank_cabs_response
    )
    forces = np.asarray(report["total_force_Eh_per_bohr"])
    oracle = target["stock"]
    de = abs(report["total_energy_Eh"] - oracle["energy_Eh"])
    inc_error = abs(report["increment_Eh"] - oracle["increment_energy_Eh"])
    legacy = [
        float(np.max(np.abs(forces - values)))
        for values in oracle["cartesian_forces_Eh_per_bohr"]
    ]
    supplementary = [
        float(np.max(np.abs(forces - values)))
        for values in oracle.get("additional_forces_Eh_per_bohr", [])
    ]
    component_error = float(
        np.max(np.abs(forces - np.asarray(target["analytic_reference_force"])))
    )
    df_checks = report["native_directional_checks"]
    sinks = report["integral_gradients"]
    checks = {
        **common_checks(report, target),
        "native_two_step_derivatives": len(df_checks) == 4
        and max(r["max_error_Eh_per_bohr"] for r in df_checks) <= 2e-6,
        "physical_increment_energy": inc_error <= 5e-11,
        "total_energy": de <= 5e-11,
        "original_stock_cartesian_forces": len(legacy) == 2 and max(legacy) <= 2e-6,
        "analytic_components": component_error <= 2e-6,
        "integral_channels": len(sinks["rows"]) == 10
        and sinks["passes_central_channel_references"],
    }
    report.update(
        phase="complete",
        passed=all(checks.values()),
        checks=checks,
        total_energy_error_Eh=de,
        increment_error_Eh=inc_error,
        stock_cartesian_max_errors_Eh_per_bohr=legacy,
        additional_stencil_errors_Eh_per_bohr=supplementary,
        independent_analytic_force_max_error_Eh_per_bohr=component_error,
    )
    write_measurement(args.output, report)
    if not report["passed"]:
        raise SystemExit("molecular closure requires review, full results retained")


if __name__ == "__main__":
    main()
