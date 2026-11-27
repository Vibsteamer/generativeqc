"""Gate 1 unit tests for Upstream-PR Hardening Track B (B1, H1, H5, M2, M7, M8)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from generativeqc._f12_focal_ecp_cuda import add_cuda_projector_residual
from generativeqc._x2c1e import build_aux_basis_from_bse, build_unc_x2c_basis_from_bse
from generativeqc.elements import atomic_number

ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# B1: Optional basis_set_exchange guard & packaging declaration
# ============================================================================


def test_b1_pyproject_declares_basis_set_exchange_optional_extra() -> None:
    """pyproject.toml declares basis-set-exchange in optional dependencies."""
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    opt = data["project"]["optional-dependencies"]
    assert any(dep.startswith("basis-set-exchange") for dep in opt["reference-test"])
    assert any(dep.startswith("basis-set-exchange") for dep in opt["bse"])


def test_b1_x2c1e_bse_helpers_raise_actionable_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime BSE helpers fail with actionable install guidance when BSE is absent."""
    monkeypatch.setitem(sys.modules, "basis_set_exchange", None)
    with pytest.raises(ImportError, match=r"generativeqc\[bse\]"):
        build_unc_x2c_basis_from_bse(["H"])
    with pytest.raises(ImportError, match=r"generativeqc\[bse\]"):
        build_aux_basis_from_bse(["H"], "jk_fit")


def test_b1_x2c1e_bse_bases_record_bse_release_and_license(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime BSE bases record the BSE release and license in their identity."""
    bse = pytest.importorskip("basis_set_exchange")
    orbital = build_unc_x2c_basis_from_bse(["H"])
    fitting = build_aux_basis_from_bse(["H"], "jk_fit")
    for basis in (orbital, fitting):
        assert basis.provenance.version == f"basis_set_exchange {bse.get_version()}"
        assert basis.provenance.license == "BSD-3-Clause"
    monkeypatch.setattr(bse, "get_version", lambda: "0.0+other")
    assert build_unc_x2c_basis_from_bse(["H"]).identity != orbital.identity
    assert build_aux_basis_from_bse(["H"], "jk_fit").identity != fitting.identity


def test_b1_x2c1e_production_bse_records_match_their_pins() -> None:
    """Every production ΔSR element builds from records equal to the pinned content."""
    pytest.importorskip("basis_set_exchange")
    elements = (
        "H",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Na",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
    )
    orbital = build_unc_x2c_basis_from_bse(elements)
    fitting = build_aux_basis_from_bse(elements, "jk_fit")
    zs = [atomic_number(symbol) for symbol in elements]
    assert [e.atomic_number for e in orbital.elements] == zs
    assert [e.atomic_number for e in fitting.elements] == zs
    assert [e.ecp_core_electrons for e in orbital.elements[-2:]] == [10, 28]


@pytest.mark.parametrize(
    ("symbol", "basis_name", "role"),
    [
        ("H", "x2c-tzvpall", None),
        ("Br", "cc-pvdz-pp", None),
        ("H", "def2-universal-jkfit", "jk_fit"),
        ("H", "def2-tzvp-rifit", "mp2_fit"),
    ],
)
def test_b1_x2c1e_bse_records_with_drifted_values_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
    basis_name: str,
    role: Literal["jk_fit", "mp2_fit"] | None,
) -> None:
    """Reject changed orbital, scalar ECP and fitting exponents."""
    bse = pytest.importorskip("basis_set_exchange")
    get_basis = bse.get_basis
    z = atomic_number(symbol)

    def drifted_basis(name: str, elements: list[int], fmt: str) -> str:
        data = json.loads(get_basis(name, elements=elements, fmt=fmt))
        record = data["elements"][str(z)]
        if symbol == "Br":
            exponents = record["ecp_potentials"][0]["gaussian_exponents"]
        else:
            exponents = record["electron_shells"][0]["exponents"]
        exponents[0] = repr(float(exponents[0]) * 1.01)
        return json.dumps(data)

    monkeypatch.setattr(bse, "get_basis", drifted_basis)
    with pytest.raises(ValueError, match=rf"{basis_name} Z={z} has content SHA-256"):
        if role is None:
            build_unc_x2c_basis_from_bse([symbol])
        else:
            build_aux_basis_from_bse([symbol], role)


def test_b1_x2c1e_bse_exports_must_hold_one_pinned_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extra elements in an export and records without a pin are refused."""
    bse = pytest.importorskip("basis_set_exchange")
    with pytest.raises(ValueError, match="x2c-tzvpall Z=19 has no pinned content"):
        build_unc_x2c_basis_from_bse(["K"])
    with pytest.raises(ValueError, match="x2c-qzvpall Z=1 has no pinned content"):
        build_unc_x2c_basis_from_bse(["H"], ae_basis_name="x2c-qzvpall")
    with pytest.raises(ValueError, match="def2-universal-jkfit Z=19 has no pinned"):
        build_aux_basis_from_bse(["K"], "jk_fit")
    get_basis = bse.get_basis
    monkeypatch.setattr(
        bse,
        "get_basis",
        lambda name, elements, fmt: get_basis(name, elements=[*elements, 2], fmt=fmt),
    )
    with pytest.raises(
        ValueError, match=r"x2c-tzvpall export holds elements Z=\[1, 2\]"
    ):
        build_unc_x2c_basis_from_bse(["H"])
    with pytest.raises(ValueError, match=r"jkfit export holds elements Z=\[1, 2\]"):
        build_aux_basis_from_bse(["H"], "jk_fit")


def test_b1_x2c1e_gate1_collects_cleanly_without_basis_set_exchange() -> None:
    """pytest --collect-only skips test_x2c1e_dlu_gate1.py cleanly when BSE is absent."""
    script = (
        "import sys, pytest\n"
        "sys.modules['basis_set_exchange'] = None\n"
        "rc = pytest.main(['--collect-only', '-q', 'tests/python/test_x2c1e_dlu_gate1.py'])\n"
        "raise SystemExit(rc)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # pytest exit code 5 = no tests collected (skipped at module level via importorskip),
    # 0 = collected without error; 2 = collection/usage error (MUST NOT happen).
    assert proc.returncode in (0, 5), f"Unexpected collection failure:\n{proc.stdout}\n{proc.stderr}"
    assert "ModuleNotFoundError" not in proc.stdout
    assert "ModuleNotFoundError" not in proc.stderr


# ============================================================================
# H1: Br/I ECP CUDA 3rd-level grid escalation (ArithmeticError vs ValueError)
# ============================================================================


def test_h1_ecp_cuda_escalates_to_640x112_on_arithmetic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When (384,80) vs (512,96) raises ArithmeticError, escalate to (640,112) and return refined."""
    from generativeqc import _f12_focal_ecp_cuda as ecp_cuda
    from generativeqc_compiler.integral.ecp_policy import (
        EXTENDED_POLAR_POINTS,
        EXTENDED_RADIAL_POINTS,
        MAX_CUDA_POLAR_POINTS,
        MAX_CUDA_RADIAL_POINTS,
        ORBITAL_G_POLAR_POINTS,
        ORBITAL_G_RADIAL_POINTS,
    )

    assert (ORBITAL_G_RADIAL_POINTS, ORBITAL_G_POLAR_POINTS) == (640, 112)

    coarse = np.full((1, 2, 2), 1.0, dtype=np.float64)
    # Differ by 1e-6 >> MATRIX_ABS_TOLERANCE (2e-9) so check_projector_grids raises ArithmeticError.
    fine = np.full((1, 2, 2), 1.0 + 1e-6, dtype=np.float64)
    # Converged within 1e-12 << 2e-9 against fine.
    refined = np.full((1, 2, 2), 1.0 + 1e-6 + 1e-12, dtype=np.float64)
    responses = {
        (EXTENDED_RADIAL_POINTS, EXTENDED_POLAR_POINTS): (coarse, 1024),
        (MAX_CUDA_RADIAL_POINTS, MAX_CUDA_POLAR_POINTS): (fine, 2048),
        (ORBITAL_G_RADIAL_POINTS, ORBITAL_G_POLAR_POINTS): (refined, 4096),
    }
    called_grids: list[tuple[int, int]] = []

    def fake_cuda_projectors(
        library, shells, transforms, xyz, terms, grid, derivatives, budget
    ) -> tuple[np.ndarray, int]:
        called_grids.append(grid)
        return responses[grid]

    monkeypatch.setattr(ecp_cuda, "cuda_projectors", fake_cuda_projectors)
    hcore = np.zeros((1, 2, 2), dtype=np.float64)
    max_w = add_cuda_projector_residual(
        hcore, Path("fake.so"), None, {}, np.zeros((1, 3)), {}, False, 1 << 20  # type: ignore[arg-type]
    )

    assert called_grids == [
        (EXTENDED_RADIAL_POINTS, EXTENDED_POLAR_POINTS),
        (MAX_CUDA_RADIAL_POINTS, MAX_CUDA_POLAR_POINTS),
        (ORBITAL_G_RADIAL_POINTS, ORBITAL_G_POLAR_POINTS),
    ]
    assert max_w == 4096
    np.testing.assert_array_equal(hcore, refined)


def test_h1_ecp_cuda_fails_closed_when_640x112_remains_unconverged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When (512,96) vs (640,112) still fails convergence, fail closed with ArithmeticError."""
    from generativeqc import _f12_focal_ecp_cuda as ecp_cuda
    from generativeqc_compiler.integral.ecp_policy import (
        EXTENDED_POLAR_POINTS,
        EXTENDED_RADIAL_POINTS,
        MAX_CUDA_POLAR_POINTS,
        MAX_CUDA_RADIAL_POINTS,
        ORBITAL_G_POLAR_POINTS,
        ORBITAL_G_RADIAL_POINTS,
    )

    coarse = np.full((4, 2, 2), 1.0, dtype=np.float64)
    fine = np.full((4, 2, 2), 1.0 + 1e-6, dtype=np.float64)
    # Still diverges in derivative channel (> 2e-8).
    refined = fine.copy()
    refined[1] += 1e-5
    responses = {
        (EXTENDED_RADIAL_POINTS, EXTENDED_POLAR_POINTS): (coarse, 1024),
        (MAX_CUDA_RADIAL_POINTS, MAX_CUDA_POLAR_POINTS): (fine, 2048),
        (ORBITAL_G_RADIAL_POINTS, ORBITAL_G_POLAR_POINTS): (refined, 4096),
    }
    called_grids: list[tuple[int, int]] = []

    def fake_cuda_projectors(
        library, shells, transforms, xyz, terms, grid, derivatives, budget
    ) -> tuple[np.ndarray, int]:
        called_grids.append(grid)
        return responses[grid]

    monkeypatch.setattr(ecp_cuda, "cuda_projectors", fake_cuda_projectors)
    hcore = np.full((4, 2, 2), 7.5, dtype=np.float64)
    with pytest.raises(ArithmeticError, match="did not converge"):
        add_cuda_projector_residual(
            hcore, Path("fake.so"), None, {}, np.zeros((1, 3)), {}, True, 1 << 20  # type: ignore[arg-type]
        )

    assert called_grids == [
        (EXTENDED_RADIAL_POINTS, EXTENDED_POLAR_POINTS),
        (MAX_CUDA_RADIAL_POINTS, MAX_CUDA_POLAR_POINTS),
        (ORBITAL_G_RADIAL_POINTS, ORBITAL_G_POLAR_POINTS),
    ]
    np.testing.assert_array_equal(hcore, np.full((4, 2, 2), 7.5, dtype=np.float64))


def test_h1_ecp_cuda_skips_third_grid_when_initial_pair_converges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When (384,80) vs (512,96) already converges, do not invoke (640,112)."""
    from generativeqc import _f12_focal_ecp_cuda as ecp_cuda
    from generativeqc_compiler.integral.ecp_policy import (
        EXTENDED_POLAR_POINTS,
        EXTENDED_RADIAL_POINTS,
        MAX_CUDA_POLAR_POINTS,
        MAX_CUDA_RADIAL_POINTS,
    )

    coarse = np.full((1, 2, 2), 2.0, dtype=np.float64)
    fine = np.full((1, 2, 2), 2.0 + 1e-12, dtype=np.float64)
    called_grids: list[tuple[int, int]] = []

    def fake_cuda_projectors(
        library, shells, transforms, xyz, terms, grid, derivatives, budget
    ) -> tuple[np.ndarray, int]:
        called_grids.append(grid)
        if grid == (EXTENDED_RADIAL_POINTS, EXTENDED_POLAR_POINTS):
            return coarse, 1024
        if grid == (MAX_CUDA_RADIAL_POINTS, MAX_CUDA_POLAR_POINTS):
            return fine, 2048
        raise AssertionError(f"Unexpected escalation to grid {grid}")

    monkeypatch.setattr(ecp_cuda, "cuda_projectors", fake_cuda_projectors)
    hcore = np.zeros((1, 2, 2), dtype=np.float64)
    max_w = add_cuda_projector_residual(
        hcore, Path("fake.so"), None, {}, np.zeros((1, 3)), {}, False, 1 << 20  # type: ignore[arg-type]
    )
    assert called_grids == [
        (EXTENDED_RADIAL_POINTS, EXTENDED_POLAR_POINTS),
        (MAX_CUDA_RADIAL_POINTS, MAX_CUDA_POLAR_POINTS),
    ]
    assert max_w == 2048
    np.testing.assert_array_equal(hcore, fine)


# ============================================================================
# H5: Locked Plan-A recipe enforcement, _x2c1e FC9 support, manifest TZ beta
# ============================================================================


def test_h5_validate_and_assemble_plan_a_recipe_and_reject_mismatches() -> None:
    """Locked Plan A (FC9 + Delta-CV9 + Delta-SR(FC9) with beta={2:0.9,3:1.0,4:1.0}) passes; mismatches reject."""
    from generativeqc._f12_focal_cv import (
        DEFAULT_PRODUCTION_CORRELATION_SPACE as CV_DEFAULT_SPACE,
        FocalDeltaCVResult,
    )
    from generativeqc._f12_focal_molecular import MolecularFocalProvider
    from generativeqc._f12_frozen_core import (
        DEFAULT_PRODUCTION_CORRELATION_SPACE,
        DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL,
        assemble_production_focal_energy_and_forces,
        validate_production_focal_recipe,
    )
    from generativeqc._x2c1e import DeltaSrResult
    from generativeqc.experimental.f12_focal import (
        FocalEvaluation,
        FocalRunResult,
        FocalTerm,
    )

    assert DEFAULT_PRODUCTION_CORRELATION_SPACE == "frozen-core-spd"
    assert CV_DEFAULT_SPACE == DEFAULT_PRODUCTION_CORRELATION_SPACE
    assert DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL == {2: 0.9, 3: 1.0, 4: 1.0}

    # Valid Plan A with consistent subsystems.
    recipe = validate_production_focal_recipe(
        valence_correlation_space="frozen-core-spd",
        delta_cv_valence_space="frozen-core-spd",
        delta_sr_correlation_space="frozen-core-spd",
        f12_beta_by_cardinal={2: 0.9, 3: 1.0, 4: 1.0},
        subsystem_f12_betas=({2: 0.9, 3: 1.0}, {3: 1.0, 4: 1.0}),
    )
    assert recipe.valence_correlation_space == "frozen-core-spd"
    assert recipe.f12_beta_by_cardinal == {2: 0.9, 3: 1.0, 4: 1.0}

    # Assemble with typed Delta-CV and Delta-SR result objects.
    f_val = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]], dtype=np.float64)
    f_cv = np.array([[0.01, 0.0, -0.02], [-0.01, 0.0, 0.02]], dtype=np.float64)
    f_sr = np.array([[-0.005, 0.002, 0.0], [0.005, -0.002, 0.0]], dtype=np.float64)
    valence = FocalRunResult(
        -100.0,
        f_val,
        (("hf_cabs", "basis", 2, 1.0),),
        (
            FocalEvaluation(
                FocalTerm("surface", "series", "basis", 2, "hf_cabs", -100.0, f_val),
                "cuda",
                True,
            ),
        ),
        valence_correlation_space="frozen-core-spd",
        frozen_core_definition_version=2,
    )
    fake_cv = FocalDeltaCVResult(
        reference=None,  # type: ignore[arg-type]
        cabs_rank=10,
        cv_frozen=(0, 0),
        fc_frozen=(9, 9),
        cv_ordinary_mp2_Eh=-1.0,
        fc_ordinary_mp2_Eh=-0.8,
        cv_f12_increment_Eh=0.0,
        fc_f12_increment_Eh=0.0,
        cv_mp2_f12_Eh=-1.0,
        fc_mp2_f12_Eh=-0.8,
        delta_ordinary_mp2_Eh=-0.2,
        delta_f12_increment_Eh=0.0,
        delta_mp2_f12_Eh=-0.2,
        delta_forces=f_cv,
        fc_mode="frozen-core-spd",
        frozen_core_definition_version=2,
    )
    fake_sr = DeltaSrResult(
        mode="dlu1e",
        correlation_space="frozen-core-spd",
        nbf=20,
        nocc=(13, 13),
        e_hf_nr=-10.0,
        e_hf_sr=-10.01,
        delta_sr_hf=-0.01,
        e_mp2_corr_nr=-0.5,
        e_mp2_corr_sr=-0.502,
        e_mp2_nr=-10.5,
        e_mp2_sr=-10.512,
        delta_sr_mp2=-0.012,
        forces_hf_nr=None,
        forces_hf_sr=None,
        delta_forces_hf=None,
        forces_mp2_nr=None,
        forces_mp2_sr=None,
        delta_forces_mp2=f_sr,
        timings_s={},
        frozen_core_definition_version=2,
    )
    assembled = MolecularFocalProvider.assemble_production_energy_and_forces(
        valence=valence,
        valence_correlation_space="frozen-core-spd",
        f12_beta_by_cardinal={2: 0.9, 3: 1.0, 4: 1.0},
        delta_cv=fake_cv,
        delta_sr=fake_sr,
    )
    assert assembled.total_energy_Eh == pytest.approx(-100.212, abs=1e-14)
    assert assembled.total_forces_Eh_per_bohr is not None
    np.testing.assert_allclose(assembled.total_forces_Eh_per_bohr, f_val + f_cv + f_sr, atol=1e-15)

    # Reject FC9 + Delta-CV4 (omits (n-1)d core-valence correlation!).
    with pytest.raises(ValueError, match="Mismatched valence.*Delta-CV"):
        validate_production_focal_recipe(
            valence_correlation_space="frozen-core-spd",
            delta_cv_valence_space="frozen-core",
            f12_beta_by_cardinal={2: 0.9, 3: 1.0},
        )

    # Reject FC4 + Delta-CV9 even when require_production_defaults=False.
    with pytest.raises(ValueError, match="Mismatched valence.*Delta-CV"):
        validate_production_focal_recipe(
            valence_correlation_space="frozen-core",
            delta_cv_valence_space="frozen-core-spd",
            f12_beta_by_cardinal={2: 0.9, 3: 1.0},
            require_production_defaults=False,
        )

    # Reject FC9 + Delta-CV9 + Delta-SR(FC4).
    with pytest.raises(ValueError, match="Mismatched valence.*Delta-SR"):
        validate_production_focal_recipe(
            valence_correlation_space="frozen-core-spd",
            delta_cv_valence_space="frozen-core-spd",
            delta_sr_correlation_space="frozen-core",
            f12_beta_by_cardinal={2: 0.9, 3: 1.0},
        )

    # Reject non-Plan-A FC4 under production defaults.
    with pytest.raises(ValueError, match="Plan A requires valence correlation space"):
        validate_production_focal_recipe(
            valence_correlation_space="frozen-core",
            delta_cv_valence_space="frozen-core",
            delta_sr_correlation_space="frozen-core",
            f12_beta_by_cardinal={2: 0.9, 3: 1.0},
        )

    # Reject TZ beta=0.9 under production defaults.
    with pytest.raises(ValueError, match="requires F12 beta=1.0 at cardinal 3"):
        validate_production_focal_recipe(
            valence_correlation_space="frozen-core-spd",
            f12_beta_by_cardinal={2: 0.9, 3: 0.9},
        )

    # Reject mixed TZ beta=0.9 vs 1.0 across subsystems even when diagnostic mode is enabled.
    with pytest.raises(ValueError, match="Inconsistent F12 beta at cardinal 3"):
        validate_production_focal_recipe(
            valence_correlation_space="frozen-core-spd",
            f12_beta_by_cardinal={2: 0.9, 3: 1.0},
            subsystem_f12_betas=({2: 0.9, 3: 0.9},),
            require_production_defaults=False,
        )

    # Reject assembling forces when a component supplies energy without forces.
    fake_cv_no_f = FocalDeltaCVResult(
        reference=None,  # type: ignore[arg-type]
        cabs_rank=10,
        cv_frozen=(0, 0),
        fc_frozen=(9, 9),
        cv_ordinary_mp2_Eh=-1.0,
        fc_ordinary_mp2_Eh=-0.8,
        cv_f12_increment_Eh=0.0,
        fc_f12_increment_Eh=0.0,
        cv_mp2_f12_Eh=-1.0,
        fc_mp2_f12_Eh=-0.8,
        delta_ordinary_mp2_Eh=-0.2,
        delta_f12_increment_Eh=0.0,
        delta_mp2_f12_Eh=-0.2,
        delta_forces=None,
        fc_mode="frozen-core-spd",
        frozen_core_definition_version=2,
    )
    with pytest.raises(ValueError, match="Delta-CV forces"):
        assemble_production_focal_energy_and_forces(
            valence=valence,
            valence_correlation_space="frozen-core-spd",
            f12_beta_by_cardinal={2: 0.9, 3: 1.0},
            delta_cv=fake_cv_no_f,
        )


def test_h5_x2c1e_delta_sr_fc9_parity_and_fd_forces_on_hbr() -> None:
    """evaluate_delta_sr supports frozen-core-spd (FC9) on ECP HBr matching PySCF and central FD."""
    pytest.importorskip("basis_set_exchange")
    gto = pytest.importorskip("pyscf.gto")
    mp = pytest.importorskip("pyscf.mp")
    scf = pytest.importorskip("pyscf.scf")

    from generativeqc import Atom
    from generativeqc._f12_frozen_core import resolve_frozen_occupied_for_atoms
    from generativeqc._x2c1e import evaluate_delta_sr
    from generativeqc.elements import SYMBOLS

    atoms = [Atom(35, (0.0, 0.0, 0.0)), Atom(1, (0.0, 0.0, 2.67))]
    orb = build_unc_x2c_basis_from_bse(["Br", "H"], ae_basis_name="x2c-svpall", ecp_basis_name="cc-pvdz-pp")
    jk = build_aux_basis_from_bse(["Br", "H"], "jk_fit")
    mp2_fit = build_aux_basis_from_bse(["Br", "H"], "mp2_fit")

    assert resolve_frozen_occupied_for_atoms(atoms, orb, (13, 13), "frozen-core") == (4, 4)
    assert resolve_frozen_occupied_for_atoms(atoms, orb, (13, 13), "frozen-core-spd") == (9, 9)

    res_fc9 = evaluate_delta_sr(
        atoms,
        orb,
        jk,
        mp2_fit,
        mode="dlu1e",
        correlation_space="frozen-core-spd",
        forces=True,
    )
    assert res_fc9.correlation_space == "frozen-core-spd"
    assert res_fc9.frozen_nr.counts("frozen-core-spd") == (9, 9)
    assert res_fc9.delta_forces_mp2 is not None and res_fc9.delta_forces_hf is not None

    # Verify independent PySCF parity for NR DF-MP2(frozen=9).
    def _to_pyscf(b):
        out = {}
        for atom in atoms:
            sym = SYMBOLS[atom.atomic_number - 1]
            if sym in out:
                continue
            eb = b.by_element[atom.atomic_number]
            out[sym] = [
                [sh.angular_momentum] + [[float(e), float(c)] for e, c in zip(sh.exponents, crow, strict=True)]
                for sh in eb.shells
                for crow in sh.coefficients
            ]
        return out

    mol = gto.M(
        atom=[("Br", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.67))],
        basis=_to_pyscf(orb),
        ecp={"Br": gto.basis.load_ecp("cc-pvdz-pp", "Br")},
        unit="Bohr",
        cart=False,
        verbose=0,
    )
    mf_nr = scf.RHF(mol).density_fit(auxbasis=_to_pyscf(jk))
    mf_nr.conv_tol = 1e-13
    mf_nr.conv_tol_grad = 1e-10
    e_hf_nr_py = mf_nr.kernel()
    assert mf_nr.converged
    e_corr_nr_py = mp.MP2(mf_nr, frozen=9).density_fit(auxbasis=_to_pyscf(mp2_fit)).kernel()[0]
    assert abs(res_fc9.e_hf_nr - e_hf_nr_py) < 1e-10
    assert abs(res_fc9.e_mp2_corr_nr - e_corr_nr_py) < 1e-10

    # Verify FC9 analytic Delta-SR force against two-step Richardson central FD (< 1e-8 Eh/bohr).
    def central(h: float) -> float:
        ap = [Atom(35, (0.0, 0.0, 0.0)), Atom(1, (0.0, 0.0, 2.67 + h))]
        am = [Atom(35, (0.0, 0.0, 0.0)), Atom(1, (0.0, 0.0, 2.67 - h))]
        ep = evaluate_delta_sr(ap, orb, jk, mp2_fit, mode="dlu1e", correlation_space="frozen-core-spd", forces=False).delta_sr_mp2
        em = evaluate_delta_sr(am, orb, jk, mp2_fit, mode="dlu1e", correlation_space="frozen-core-spd", forces=False).delta_sr_mp2
        return -(ep - em) / (2.0 * h)

    fd_rich = (4.0 * central(1e-3) - central(2e-3)) / 3.0
    assert abs(float(res_fc9.delta_forces_mp2[1, 2]) - fd_rich) < 1e-8
    assert float(np.max(np.abs(res_fc9.delta_forces_mp2.sum(axis=0)))) < 1e-12


# ============================================================================
# M2: Non-_ex integral normalization contract matches compiled kDefault=1 (1/beta)
# ============================================================================


def test_m2_non_ex_fallback_matches_compiled_1_over_beta_default(tmp_path: Path) -> None:
    """Non-_ex fallback bind and execute match compiled kDefaultSlaterFitNormalizationCode=1."""
    import ctypes as ct
    from unittest.mock import patch

    from generativeqc._f12_focal_integral_kernel import (
        DEFAULT_SLATER_FIT_NORMALIZATION,
        LEGACY_SLATER_FIT_NORMALIZATION,
        LEGACY_SLATER_NORM_CODE,
        contract_geometry,
        slater_normalization_code,
    )
    from generativeqc._f12_focal_integral_sink import _bind_fallback_weighted

    assert DEFAULT_SLATER_FIT_NORMALIZATION == "negative_exponential_over_beta"
    assert LEGACY_SLATER_FIT_NORMALIZATION == "negative_exponential_over_beta"
    assert LEGACY_SLATER_NORM_CODE == 1

    fake_lib = tmp_path / "lib_non_ex.so"
    fake_lib.write_bytes(b"")

    def fake_non_ex_kernel(
        ncenter, op, beta, nshell, nprim, nblocks, nweights,
        angular, prim_offsets, exponents, radial, centers,
        blocks, offsets, weights, output, budget, size_ptr,
    ) -> int:
        output[:] = 0.0
        output[:, 0] = 2.5
        ct.cast(size_ptr, ct.POINTER(ct.c_uint64))[0] = 64
        return 0

    def fake_bind(library, kernel, host_reference, *, extended: bool = False):
        if extended:
            raise AttributeError("non-_ex library")
        return fake_non_ex_kernel

    code_over_beta = slater_normalization_code("negative_exponential_over_beta")
    code_unscaled = slater_normalization_code("negative_exponential")
    assert code_over_beta == 1 and code_unscaled == 0

    with patch("generativeqc._f12_focal_integral_sink.bind", side_effect=fake_bind):
        fn, is_ext = _bind_fallback_weighted(fake_lib, "geminal", True, code_over_beta)
        assert fn is fake_non_ex_kernel and is_ext is False
        with pytest.raises(RuntimeError, match="SlaterFitNormalization"):
            _bind_fallback_weighted(fake_lib, "geminal", True, code_unscaled)
        # Coulomb is independent of Slater normalization.
        fn_c, is_ext_c = _bind_fallback_weighted(fake_lib, "coulomb", True, code_unscaled)
        assert fn_c is fake_non_ex_kernel and is_ext_c is False

    fixture = {
        "metric_blocks": np.array([[0, 0]], dtype=np.int32),
        "metric_offsets": np.array([0, 1], dtype=np.int32),
        "weights_metric_f12": np.array([1.0], dtype=np.float64),
        "weights_metric": np.array([1.0], dtype=np.float64),
        "angular": np.array([0], dtype=np.int32),
        "atom": np.array([0], dtype=np.int32),
        "prim_offsets": np.array([0, 1], dtype=np.int32),
        "exponents": np.array([1.0], dtype=np.float64),
        "radial": np.array([1.0], dtype=np.float64),
        "centers": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    }
    manifest = {
        "elements": ["H"],
        "geometry_bohr": [[0.0, 0.0, 0.0]],
        "options": {"f12_beta": 1.0},
    }
    _, row_ok = contract_geometry(
        fixture, manifest, {"geminal": fake_non_ex_kernel},
        "metric", "f12", 1 << 20, True,
        slater_normalization="negative_exponential_over_beta",
        extended_abi=False,
    )
    assert row_ok["weighted_value_Eh"] == pytest.approx(2.5)
    with pytest.raises(RuntimeError, match="SlaterFitNormalization"):
        contract_geometry(
            fixture, manifest, {"geminal": fake_non_ex_kernel},
            "metric", "f12", 1 << 20, True,
            slater_normalization="negative_exponential",
            extended_abi=False,
        )
    # Coulomb accepts non-default Slater normalization on non-_ex too.
    _, row_c = contract_geometry(
        fixture, manifest, {"coulomb": fake_non_ex_kernel},
        "metric", "coulomb", 1 << 20, True,
        slater_normalization="negative_exponential",
        extended_abi=False,
    )
    assert row_c["weighted_value_Eh"] == pytest.approx(2.5)


# ============================================================================
# M7: Frozen-core shell-splitting energy-gap guard (NaBr FC4 vs FC9)
# ============================================================================


def test_frozen_core_shell_gap_refuses_an_explicit_count_that_splits_a_shell() -> None:
    from generativeqc._f12_frozen_core import check_frozen_core_shell_gap

    # A count-based cut falls inside a fivefold near-degenerate d shell.
    eps = np.array([-40.5, -9.6, -7.2, -7.2, -7.2, -3.014, -3.012, -3.012,
                    -3.010, -3.008, -2.8, -1.55, -1.55, -1.55, -0.9, -0.35])
    with pytest.raises(ValueError, match=r"nearest frozen counts .* are 5 and 10\.$"):
        check_frozen_core_shell_gap(eps, 9, n_occupied=16, correlation_space_id="frozen:9")
    assert check_frozen_core_shell_gap(eps, 10, n_occupied=16) == pytest.approx(0.208)


def test_m7_real_df_hf_shell_gaps_pass_for_h2o_oh_hbr() -> None:
    """Real DF-HF canonical orbital gaps for H2O, OH, and ECP HBr (FC4 & FC9) are >> 0.05 Eh."""
    from generativeqc import Atom
    from generativeqc._f12_focal_reference import df_reference
    from generativeqc._f12_frozen_core import (
        DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH,
        select_frozen_occupied_frame,
    )
    from generativeqc.experimental.f12_focal import FocalBasis, FocalSeries, plan_singlepoint

    from _bse_basis_support import jk_fit_basis, pp_orbital_basis

    # 1. H2O (RHF) and OH (UHF) in sto-3g / def2-svp.
    light_basis = FocalBasis(
        2,
        "m7-light",
        "sto-3g",
        jk_fit="def2-svp",
        mp2_fit="def2-svp",
        f12_fit="def2-svp",
        cabs="def2-svp",
        beta=0.9,
    )
    for atoms, mult, root, cspace in (
        (
            (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.43, 1.1)), ("H", (0.0, -1.43, 1.1))),
            1,
            "rhf",
            "frozen-core",
        ),
        (
            (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.83))),
            2,
            "uhf",
            "frozen-core",
        ),
    ):
        plan = plan_singlepoint(
            atoms,
            FocalSeries(light_basis),
            FocalSeries(light_basis),
            None,
            charge=0,
            multiplicity=mult,
            root_id=root,
            correlation_space_id=cspace,
            backend="cpu_reference",
        )
        req = plan.steps[0][0]
        ref = df_reference(req, forces=False, budget_mib=256)
        frame = select_frozen_occupied_frame(
            req.atoms, req.basis.orbital, ref.overlap, ref.coefficients, ref.orbital_energies,
            ref.occupied, cspace,
        )
        assert frame.counts(cspace) == (1, 1)
        for channel in frame.channels(cspace):
            assert channel.minimum_gap_eh > 10.0 * DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH

    # 2. HBr (ECP10MDF, RHF) via native CPU DF-HF in BSE aug-cc-pVDZ(-PP) for FC4 & FC9.
    from generativeqc._x2c1e import (
        _build_sad_density,
        _factor_df,
        _solve_df_hf,
        df_integrals_cpu,
        one_electron_integrals_cpu,
    )

    orbital, fit = pp_orbital_basis((1, 35)), jk_fit_basis((1, 35))
    hbr_basis = FocalBasis(
        2, "m7-hbr", orbital, jk_fit=fit, mp2_fit=fit, f12_fit=fit, cabs=fit, beta=0.9
    )
    hbr_atoms = (Atom(35, (0.0, 0.0, 0.0)), Atom(1, (0.0, 0.0, 2.67)))
    s_mat, h_mat, e_nuc = one_electron_integrals_cpu(hbr_atoms, hbr_basis.orbital)
    jk_m, jk_3c = df_integrals_cpu(hbr_atoms, hbr_basis.orbital, hbr_basis.jk_fit)
    jk_fac = _factor_df(jk_m, jk_3c)
    sad = _build_sad_density(hbr_atoms, hbr_basis.orbital, s_mat, h_mat, jk_fac, (13, 13))
    hf_hbr = _solve_df_hf(
        s_mat, h_mat, e_nuc, jk_fac, (13, 13), unrestricted=False, initial_densities=sad
    )
    for cspace, expected_fc in (("frozen-core", 4), ("frozen-core-spd", 9)):
        plan = plan_singlepoint(
            hbr_atoms,
            FocalSeries(hbr_basis),
            FocalSeries(hbr_basis),
            None,
            charge=0,
            multiplicity=1,
            root_id="rhf",
            correlation_space_id=cspace,
            backend="cpu_reference",
        )
        req = plan.steps[0][0]
        frame = select_frozen_occupied_frame(
            req.atoms, req.basis.orbital, s_mat, hf_hbr.coefficients[:1],
            hf_hbr.orbital_energies[:1], (13,), cspace,
        )
        assert frame.counts(cspace) == (expected_fc, expected_fc)
        assert frame.channels(cspace)[0].minimum_gap_eh > 1.0


# ============================================================================
# M8: Guard single-pass fp32_single W schedule on forces
# ============================================================================


def test_m8_fp32_single_never_default_and_refused_on_non_l20_forces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fp32_single is never default; non-L20 forces refuse it before the native contraction."""
    import inspect
    import warnings
    from types import SimpleNamespace

    from generativeqc import _f12_focal_ucc_triples as ucc_triples
    from generativeqc._cuda_device import CudaDeviceIdentity
    from generativeqc._f12_focal_molecular import MolecularFocalProvider
    from generativeqc.experimental.f12_focal import TriplesWPrecisionError

    # 1. Verify defaults never select fp32_single.
    default_provider = MolecularFocalProvider(shell_library=Path("fake.so"))
    assert default_provider.triples_fp32_w is False
    assert default_provider.triples_w_schedule is None
    cuda_sig = inspect.signature(MolecularFocalProvider.cuda)
    assert cuda_sig.parameters["triples_fp32_w"].default is False
    assert cuda_sig.parameters["triples_w_schedule"].default is None

    # 2. Verify MolecularFocalProvider.triples_energy vs triples_energy_and_forces.
    native_calls: list[bool] = []

    def fake_triples(req: object, ref: object, **kw: object) -> tuple[float, np.ndarray | None]:
        native_calls.append(bool(kw.get("forces")))
        return -0.01, np.zeros((1, 3), dtype=np.float64) if kw.get("forces") else None

    monkeypatch.setattr(ucc_triples, "molecular_triples_on_reference", fake_triples)
    fake_req = SimpleNamespace(
        atoms=(),
        basis=SimpleNamespace(orbital=SimpleNamespace(by_element={})),
        backend="cuda",
        correlation_space_id="all-active",
        root_id="rhf",
    )
    fake_ref = SimpleNamespace(
        occupied=(2,),
        orbital_energies=None,
        _pair_spin_indices=lambda req: (0, 0),
    )

    def a100() -> CudaDeviceIdentity:
        return CudaDeviceIdentity("NVIDIA A100-SXM4-80GB", "sm_80")

    fp32_single_provider = MolecularFocalProvider(
        shell_library=Path("fake.so"), triples_fp32_w=True, _cuda_device=a100
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e = fp32_single_provider.triples_energy(fake_req, reference=fake_ref)  # type: ignore[arg-type]
    assert e == pytest.approx(-0.01)
    assert native_calls == [False]

    with pytest.raises(TriplesWPrecisionError, match="only on an NVIDIA L20"):
        fp32_single_provider.triples_energy_and_forces(
            fake_req, reference=fake_ref  # type: ignore[arg-type]
        )
    assert native_calls == [False]

    # Default FP64 and compensated fp32_split3 forces run without warnings.
    for prov in (
        MolecularFocalProvider(shell_library=Path("fake.so"), _cuda_device=a100),
        MolecularFocalProvider(
            shell_library=Path("fake.so"), triples_w_schedule="fp32_split3", _cuda_device=a100
        ),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            e_f, f_arr = prov.triples_energy_and_forces(fake_req, reference=fake_ref)  # type: ignore[arg-type]
        assert e_f == pytest.approx(-0.01)
        assert f_arr.shape == (1, 3)
    assert native_calls == [False, True, True]
