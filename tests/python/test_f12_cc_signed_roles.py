"""A partial CC role cannot qualify a full molecular benchmark without its matched partner."""

import json
from pathlib import Path

import pytest

from benchmarks import f12_cc_signed_roles as roles


@pytest.fixture
def stages(tmp_path: Path) -> tuple:
    terms = {
        "reference_energy_Eh": -3.0,
        "ordinary_cc_correlation_Eh": -0.1,
        "cabs_singles_energy_Eh": -0.01,
        "common_geminal_Eh": -0.03,
        "extra_f12b_geminal_Eh": -0.001,
    }
    energy = sum(terms.values())
    geometry = {
        "elements": ["H", "H"],
        "geometry_bohr": [[0, 0, -1], [0, 0, 1]],
        "geometry_sha256": "geom",
    }
    arms = [
        {
            "name": "z",
            "step": step,
            "direction": [[0, 0, 1], [0, 0, 0]],
            "hf_energies_Eh": [-3.0, -3.0],
            "files": [f"plus-{step}.in", f"minus-{step}.in"],
        }
        for step in (0.0004, 0.0002)
    ]
    panel = {
        "geometry": geometry,
        "cc_reference": {"energy_Eh": energy, "components_Eh": terms},
        "molecule": "example",
        "native_displacements": arms,
        "scf_input_sha256": {
            name: f"digest-{name}" for arm in arms for name in arm["files"]
        },
    }
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel))
    code = [{"backend": "cuda-fp64-ordinary-stream"}]
    ident = {
        "geometry_sha256": "geom",
        "files": {"panel.json": roles.file_sha(panel_path)},
        "arch": "sm_89",
    }
    center = {
        **terms,
        "case": "example",
        "physical_gpu": {"arch": "sm_89", "uuid": "GPU-test", "name": "test L20"},
        "total_energy_Eh": energy,
        "central_seconds": 12.0,
        "total_force_Eh_per_bohr": [[0, 0, 0.015], [0, 0, -0.015]],
        "cc_residual": 1e-12,
        "lambda_residual": 0.0,
        "graph_gpu_programs": code,
        "integral_gradients": {
            "backend": "cuda_fp64",
            "max_device_bytes": 8,
            "rows": list(range(10)),
            "passes_contraction_checks": True,
        },
        "one_electron_resources": {"device_bytes": 8},
        "forward_resources": {"integrals": {str(i): {} for i in range(10)}},
        "orbital_response": {"canonical_bytes": 8, "orbital_residual": 0},
    }
    center = roles.qualify_role(center, "center", panel, ident)
    pairs = []
    for arm in arms:
        row = {
            "direction": "z",
            "step_bohr": arm["step"],
            "energies_Eh": [],
            "energy_components": [],
        }
        for sign in (1, -1):
            moved = terms.copy()
            moved["ordinary_cc_correlation_Eh"] -= sign * arm["step"] * 0.015
            row["energies_Eh"].append(sum(moved.values()))
            row["energy_components"].append(
                {
                    **moved,
                    "sign": sign,
                    "scf_filename": arm["files"][0 if sign == 1 else 1],
                    "scf_input_sha256": panel["scf_input_sha256"][
                        arm["files"][0 if sign == 1 else 1]
                    ],
                    "cc_residual": 0.0,
                    "integral_device_bytes": {str(i): 8 for i in range(10)},
                }
            )
        pairs.append(row)
    signed = roles.qualify_role(
        {
            "case": "example",
            "physical_gpu": {"arch": "sm_89", "uuid": "GPU-test", "name": "test L20"},
            "signed_pairs": pairs,
            "graph_gpu_programs": code,
            "composition_seconds": 4.0,
        },
        "signed",
        panel,
        ident,
    )
    assert (
        center["role_passed"]
        and signed["role_passed"]
        and not center["passed"]
        and not signed["passed"]
    )
    return panel_path, center, signed


def finish(panel: Path, center: dict, signed: dict) -> dict:
    a, b = panel.parent / "center.json", panel.parent / "signed.json"
    a.write_text(json.dumps(center))
    b.write_text(json.dumps(signed))
    return roles.join(panel, a, b)


def test_split_roles_join_only_under_the_same_scientific_identity(
    stages: tuple,
) -> None:
    panel, center, signed = stages
    result = finish(panel, center, signed)
    assert result["passed"] and result["checks"]["native_benchmark_direction_two_steps"]
    assert result["center_plus_signed_compute_seconds"] == 16.0
    assert result["total_force_Eh_per_bohr"] == center["total_force_Eh_per_bohr"]
    signed["role_identity"] = dict(signed["role_identity"], relative_cutoff=1e-4)
    with pytest.raises(ValueError, match="identities"):
        finish(panel, center, signed)


@pytest.mark.parametrize(
    "change",
    [
        "missing step",
        "changed step",
        "wrong sign",
        "missing integral",
        "different energy",
        "failed root",
    ],
)
def test_incomplete_signed_evidence_rejected(stages: tuple, change: str) -> None:
    panel, center, signed = stages
    pairs = signed["signed_pairs"]
    if change == "missing step":
        pairs.pop()
    if change == "changed step":
        pairs[1]["step_bohr"] = pairs[0]["step_bohr"]
    if change == "wrong sign":
        pairs[0]["energy_components"].reverse()
    if change == "missing integral":
        pairs[0]["energy_components"][0]["integral_device_bytes"].pop("9")
    if change == "different energy":
        pairs[0]["energies_Eh"][0] += 0.02
    if change == "failed root":
        pairs[0]["energy_components"][0]["cc_residual"] = 1e-7
    with pytest.raises(ValueError):
        finish(panel, center, signed)


def test_coupled_projection_still_rejects_incorrect_center_force(stages: tuple) -> None:
    panel, center, signed = stages
    center["total_force_Eh_per_bohr"] = [[0.0, 0.0, 0.025], [0.0, 0.0, -0.025]]
    assembled = finish(panel, center, signed)
    assert not assembled["checks"]["native_benchmark_direction_two_steps"]
    assert not assembled["passed"]


def test_two_individual_signed_steps_join_in_either_order(stages: tuple) -> None:
    panel_path, center, signed = stages
    panel = json.loads(panel_path.read_bytes())
    center_path = panel_path.parent / "center.json"
    center_path.write_text(json.dumps(center))
    paths = []
    for pair in reversed(signed["signed_pairs"]):
        step = pair["step_bohr"]
        partial = roles.qualify_role(
            dict(
                signed,
                selected_step_bohr=step,
                signed_pairs=[pair],
                composition_seconds=2.0,
            ),
            "signed",
            panel,
            signed["role_identity"],
        )
        assert partial["role_passed"] and not partial["passed"]
        path = panel_path.parent / f"signed-{step}.json"
        path.write_text(json.dumps(partial))
        paths.append(path)
    joined = roles.join(panel_path, center_path, *paths)
    assert joined["passed"] and joined["center_plus_signed_compute_seconds"] == 16.0
    with pytest.raises(ValueError, match="missing, duplicate"):
        roles.join(panel_path, center_path, paths[0], paths[0])


@pytest.mark.parametrize("selected", [None, "plus", "minus"])
def test_signed_worker_needs_only_displaced_scf_and_scalar_correlation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected: str | None
) -> None:
    """A worker with only scoped signed inputs can return its role without a center SCF."""
    from argparse import Namespace
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np

    from benchmarks import f12_molecular_force_composition as worker

    xyz = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    direction = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    geometry = {"elements": ["H", "H"], "geometry_bohr": xyz.tolist()}
    panel = {
        "molecule": "test",
        "geometry": geometry,
        "combined_basis": {
            "elements": ["H", "H"],
            "coordinates_bohr": xyz.tolist(),
            "shells": [],
        },
        "layout": {
            "native_to_source": [0, 1],
            "ao_permutation": [0, 1],
            "space": [1, 2, 2],
        },
        "validation_scope": "benchmark_center_and_native_direction",
        "native_displacements": [],
        "scf_input_sha256": {},
    }
    for h in (0.0002, 0.0004):
        names = [f"p-{h}.in", f"m-{h}.in"]
        if h == 0.0002:
            for sign, name in zip((1, -1), names, strict=True):
                if selected and sign != (1 if selected == "plus" else -1):
                    continue
                geom = xyz + sign * h * direction
                (tmp_path / name).write_text(
                    "atoms\n"
                    + "\n".join("H " + " ".join(str(x) for x in row) for row in geom)
                )
        panel["scf_input_sha256"].update({name: "digest-" + name for name in names})
        panel["native_displacements"].append(
            {
                "name": "z",
                "direction": direction.tolist(),
                "files": names,
                "step": h,
                "hf_energies_Eh": [-3.0, -3.0],
            }
        )
    layout = tmp_path / "layout.npz"
    np.savez(layout, atom=[0, 1], manifest=json.dumps({"case": "test", **geometry}))
    args = Namespace(
        component_library=tmp_path / "lib",
        session_library=tmp_path / "session",
        sink_library=tmp_path / "sink",
        scf_inputs=tmp_path,
        arch="sm_89",
        cache=tmp_path,
        budget_mib=32,
        cabs_relative_cutoff=None,
        layout=layout,
        cc_step=0.0002,
        cc_sign=selected,
    )
    monkeypatch.setattr(worker.ctypes, "CDLL", lambda _: None)
    monkeypatch.setattr(worker, "cuda_runner", lambda *args, **kwargs: (None, []))
    monkeypatch.setattr(
        worker,
        "forward_integrals",
        lambda *_args, **_kwargs: (
            {"metric": np.eye(2)},
            {str(i): {"device_bytes": 8} for i in range(10)},
        ),
    )
    monkeypatch.setattr(
        worker,
        "metric_whitener",
        lambda *_: (np.eye(2), {"device_bytes": 8, "rank": 2}),
    )
    monkeypatch.setattr(
        worker,
        "combined_matrices",
        lambda *_: (np.eye(2), np.eye(2), {"device_bytes": 8}),
    )

    def reference_session(lib: Path, path: Path) -> nullcontext:
        assert path.exists()
        snapshot = {
            "energy": -3.0,
            "hcore": np.eye(2),
            "overlap": np.eye(2),
            "coefficients": np.eye(2),
        }
        return nullcontext(SimpleNamespace(snapshot=lambda: snapshot))

    def energy(moved: dict, **kwargs: object) -> tuple:
        assert kwargs["pullback"] is False
        return -0.1, None, {"cc_residual": 0.0}

    monkeypatch.setattr(worker, "DFReferenceSession", reference_session)
    monkeypatch.setattr(worker, "chain", energy)
    result = worker.calculate_panel(
        args, panel, molecular_correlation=lambda: None, cc_role="signed"
    )
    assert (
        result["selected_step_bohr"] == 0.0002
        and result["phase"] == "signed_pairs_complete"
    )
    assert result["signed_pairs"][0]["energies_Eh"] == (
        [-3.1] if selected else [-3.1, -3.1]
    )
    assert not result["passed"] and "total_force_Eh_per_bohr" not in result


def test_four_single_point_shards_with_independent_receipt_identities(
    stages: tuple,
) -> None:
    panel_path, center, whole = stages
    panel = json.loads(panel_path.read_bytes())
    center_path = panel_path.parent / "center.json"
    center_path.write_text(json.dumps(center))
    single = []
    for pair in whole["signed_pairs"]:
        for idx, sgn in enumerate((1, -1)):
            point = dict(
                pair,
                energy_components=[pair["energy_components"][idx]],
                energies_Eh=[pair["energies_Eh"][idx]],
            )
            component = roles.qualify_role(
                dict(
                    whole,
                    selected_step_bohr=pair["step_bohr"],
                    selected_sign=sgn,
                    signed_pairs=[point],
                    composition_seconds=1,
                ),
                "signed",
                panel,
                center["role_identity"],
            )
            path = panel_path.parent / f"step-{pair['step_bohr']}-{sgn}.json"
            path.write_text(json.dumps(component))
            single.append(path)
    mixed = [single[2], single[1], single[3], single[0]]
    joined = roles.join(panel_path, center_path, *mixed)
    assert (
        joined["passed"]
        and joined["total_force_Eh_per_bohr"] == center["total_force_Eh_per_bohr"]
    )
    assert joined["center_plus_signed_compute_seconds"] == 16
    with pytest.raises(ValueError, match="missing, duplicate"):
        roles.join(panel_path, center_path, *mixed[:-1])
    with pytest.raises(ValueError, match="missing, duplicate"):
        roles.join(panel_path, center_path, *mixed[:-1], mixed[0])
    edited = json.loads(mixed[0].read_bytes())
    old = json.loads(mixed[0].read_bytes())
    edited["physical_gpu"] = {
        "name": "example GPU",
        "arch": "sm_80",
        "uuid": "GPU-test",
    }
    mixed[0].write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="physical GPU identity"):
        roles.join(panel_path, center_path, *mixed)
    edited = old
    edited["signed_pairs"][0]["energy_components"][0]["scf_input_sha256"] = (
        "another geometry"
    )
    mixed[0].write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="input identity"):
        roles.join(panel_path, center_path, *mixed)
