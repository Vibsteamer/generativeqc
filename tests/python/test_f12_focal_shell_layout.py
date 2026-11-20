"""Independent integral and nuclear-derivative controls for canonical focal shell inputs."""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Atom
from generativeqc.basis import BasisSet, BasisShell
from generativeqc.experimental.f12_focal import FocalBasis, FocalRequest

from benchmarks.f12_focal_shell_layout import focal_shell_layout
from benchmarks.f12_molecular_integral_forward import forward_integrals
from benchmarks.f12_molecular_sink_live import contract_live_weights


@pytest.fixture(scope="module")
def host_sink() -> Path:
    source = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not source:
        pytest.skip("set GENERATIVEQC_F12_HOST_SINK to the built molecular host sink")
    return Path(source)


def shell_inputs() -> tuple:
    refs = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "reference_data/f12/libint_df_primitives.json"
        ).read_text()
    )["rows"][:4]
    _, _, beta, ea, eb, _, *positions = refs[0]["input"]
    atoms = tuple(
        Atom.from_value((i + 1, positions[3 * i : 3 * i + 3])) for i in range(2)
    )
    source = FocalBasis(2, "source", "sto-3g", jk_fit="def2-svp").orbital

    def basis(name: str, shells: list[list[BasisShell]]) -> BasisSet:
        return replace(
            source,
            name=name,
            elements=tuple(
                replace(source.by_element[z], shells=tuple(rows))
                for z, rows in enumerate(shells, 1)
            ),
        )

    s = BasisShell(0, ("1.1", "0.3"), (("0.2", "1.0"), ("-0.3", "0.9")))
    p, d = (BasisShell(l, ("0.5",), (("1",),)) for l in (1, 2))
    orbital = basis("orbital", [[s], [s]])
    cabs = basis("OptRI", [[p, d], [p, d]])
    fit = basis("fit", [[BasisShell(0, (str(e),), (("1",),))] for e in (ea, eb)])
    b = FocalBasis(
        2,
        "test",
        orbital,
        jk_fit=orbital,
        mp2_fit=orbital,
        f12_fit=fit,
        cabs=cabs,
        beta=beta,
    )
    request = FocalRequest(
        atoms,
        0,
        2,
        "uhf",
        "all-active",
        "HeH",
        b,
        "mp2_f12_correlation",
        "cpu_reference",
    )
    return request, refs


def test_geometry_binding_rejects_wrong_backend_before_execution(
    tmp_path: Path,
) -> None:
    request, _ = shell_inputs()
    with pytest.raises(ValueError, match="backend"):
        forward_integrals(request, tmp_path / "absent.so")


def test_focal_geminal_metric_and_nuclear_gradient(host_sink: Path) -> None:
    req, refs = shell_inputs()
    values, resources = forward_integrals(req, host_sink, host_reference=True)
    assert len(resources) == 10 and values["ao_f12"].shape == (2, 20, 20)
    scale = np.prod((2 * np.asarray(refs[0]["input"][3:5]) / np.pi) ** 0.75)
    for name, row in zip(
        ("f12", "f12_over_coulomb", "f12_squared", "double_commutator"),
        refs,
        strict=True,
    ):
        field = "metric_" + name
        answer = scale * np.asarray(row["value_and_gradient"])
        assert abs(values[field][0, 1] - answer[0]) < 2e-12
        weights = {k: np.zeros_like(v) for k, v in values.items()}
        weights[field][0, 1] = 1.0
        out = contract_live_weights(req, weights, host_sink, host_reference=True)
        assert out["passes_contraction_checks"]
        np.testing.assert_allclose(
            out["gradient_Eh_per_bohr"], answer[1:7].reshape(2, 3), atol=3e-12, rtol=0
        )


def test_focal_coulomb_operators_against_libcint(host_sink: Path) -> None:
    pytest.importorskip("pyscf")
    from pyscf import df, gto

    req, refs = shell_inputs()
    atoms = [(a.atomic_number, a.position) for a in req.atoms]
    orb_shells = [[0, [1.1, 0.2, -0.3], [0.3, 1.0, 0.9]]]
    combined = orb_shells + [[1, [0.5, 1.0]], [2, [0.5, 1.0]]]
    mol = gto.M(
        atom=atoms,
        unit="Bohr",
        spin=1,
        basis={"H": combined, "He": combined},
        verbose=0,
    )
    # Libcint groups functions by atom. The focal source enumerates the OBS
    # for the full molecule before adding either atom's OptRI functions.
    order = np.r_[0:2, 10:12, 2:10, 12:20]
    for role in ("mp2_f12_correlation", "hf_cabs"):
        current = replace(req, role=role)
        if role == "hf_cabs":
            single = FocalBasis(
                2,
                "test",
                req.basis.orbital,
                jk_fit=req.basis.jk_fit,
                cabs=req.basis.cabs,
            )
            current = replace(current, basis=single)
            aux = {"H": orb_shells, "He": orb_shells}
        else:
            aux = {
                symbol: [[0, [e, 1.0]]]
                for symbol, e in zip(("H", "He"), refs[0]["input"][3:5], strict=True)
            }
        auxmol = df.addons.make_auxmol(mol, aux)
        values, _ = forward_integrals(current, host_sink, host_reference=True)
        if role == "hf_cabs":
            assert set(values) == {"metric", "ao_coulomb"}
        raw = df.incore.aux_e2(mol, auxmol, aosym="s1").transpose(2, 0, 1)
        np.testing.assert_allclose(
            values["metric"], auxmol.intor("int2c2e"), rtol=0, atol=3e-12
        )
        np.testing.assert_allclose(
            values["ao_coulomb"], raw[:, order][:, :, order], rtol=0, atol=3e-12
        )


def test_raw_cartesian_spherical_projection_through_i(host_sink: Path) -> None:
    gto = pytest.importorskip("pyscf.gto")
    req, _ = shell_inputs()
    shells = tuple(BasisShell(l, ("0.8",), (("1",),)) for l in range(7))
    cabs = replace(
        req.basis.orbital,
        elements=tuple(replace(e, shells=shells) for e in req.basis.orbital.elements),
    )
    basis = FocalBasis(2, "test", req.basis.orbital, jk_fit=req.basis.jk_fit, cabs=cabs)
    layout = focal_shell_layout(
        replace(req, role="hf_cabs", basis=basis),
        host_sink,
        host_reference=True,
        budget_mib=64,
    )
    assert layout["obs_ao_count"] == 4
    for l in range(7):
        # Libcint normalizes the solid angular harmonic on the unit sphere.
        angular_scale = 1.0 if l < 2 else np.sqrt(4 * np.pi / (2 * l + 1))
        expected = angular_scale * gto.cart2sph(l, normalized="sp").T
        np.testing.assert_allclose(
            layout[f"transform_{l}"], expected, atol=1e-14, rtol=0
        )
