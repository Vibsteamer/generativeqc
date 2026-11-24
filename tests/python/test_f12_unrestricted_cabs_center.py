"""Check the CABS singles increment at a stationary open-shell Gaussian center."""

import json
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.tensor import execute

from benchmarks import f12_unrestricted_gaussian_center as center
from benchmarks.f12_cabs_ecp import ResidualEcpPanel
from benchmarks.f12_cabs_singles_nuclear import restricted_singles
from benchmarks.f12_unrestricted_gaussian import spin_ao_inputs


@pytest.mark.parametrize(
    ("beta_occupied", "singles_only", "ecp"),
    [(0, True, False), (1, True, False), (0, False, False), (1, True, True)],
)
def test_center_singles_increment_and_reference_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    beta_occupied: int,
    singles_only: bool,
    ecp: bool,
) -> None:
    """An occupied-to-CABS coupling moves while the entire OBS Hamiltonian is fixed."""
    obs, ext = 3, 4
    coupling = 0.12
    hcore = np.diag([-2.0, -1.0, 1.0, 3.0])
    hcore[0, -1] = hcore[-1, 0] = coupling
    gap = hcore[-1, -1] - hcore[0, 0]
    xyz = np.array([[0., 0., 0.], [0., 0., 2.]])
    source_order = np.array([3, 2, 1, 0])
    geometry = tmp_path / "geometry.json"
    geometry.write_text(
        json.dumps(
            {
                "layout": {"space": [2, beta_occupied, obs, ext, 1], "native_to_source": source_order.tolist()},
                "combined_basis": {
                    "elements": ["Br" if ecp else "H", "H"],
                    "charge": 26 - (2 + beta_occupied) if ecp else 0,
                    "multiplicity": 3 - beta_occupied,
                    "shells": [{"angular_momentum": 6 if ecp else 0}],
                },
            }
        )
    )
    layout = tmp_path / "layout.npz"
    np.savez(layout)
    ops = {}
    for operator in (
        "coulomb",
        "f12",
        "f12_squared",
        "f12_over_coulomb",
        "double_commutator",
    ):
        ops["metric" if operator == "coulomb" else "metric_" + operator] = np.eye(1)
        ops["ao_" + operator] = np.zeros((1, ext, ext))
    def sources(*args, cabs_h_shells, **kwargs):
        s, h = np.eye(ext), hcore
        if ecp:
            s, h, _ = cabs_h_shells.matrices(derivatives=True, max_bytes=1 << 20)
        return (*spin_ao_inputs(ops, s, h, np.arange(ext), np.arange(obs), np.eye(ext)[:, obs:]), {})

    monkeypatch.setattr(center, "gaussian_sources", sources)
    ecp_options = {}
    if ecp:
        derivative = np.zeros((2, 3, ext, ext))
        derivative[:, 2, 0, -1] = derivative[:, 2, -1, 0] = [-1., 1.]
        ix = np.ix_(source_order, source_order)
        panel = ResidualEcpPanel(
            (hcore - np.diag(np.diag(hcore)))[ix], derivative[:, :, source_order[:, None], source_order],
            coordinates_bohr=xyz, ao_to_source=source_order, effective_charges=np.array([25, 1]),
        )
        class FixedBareCore:
            """The residual supplies the only position-dependent Hamiltonian term."""

            def matrices(self, *, derivatives, max_bytes):
                h = np.diag(np.diag(hcore))
                h[ix] += panel.matrix
                return np.eye(ext), h, {}

            def gradient(self, ws, wh, max_bytes):
                from generativeqc._one_electron_gradient import IntegralGradient

                return IntegralGradient(np.einsum("axij,ij->ax", panel.derivative, wh[ix]), {})

        def selected(basis, mapping, layout_path, occupations, h_library, i_library, residual):
            assert residual is panel
            return FixedBareCore()

        monkeypatch.setattr(center, "_center_one_electron", selected)
        ecp_options = {"cabs_i_one_electron_library": tmp_path / "i", "residual_ecp": panel}

    def singles(
        fock: np.ndarray,
        occupied: int,
        *,
        spin_weight: int,
        pullback: bool,
        library: object,
    ) -> tuple[float, np.ndarray]:
        if occupied == 0:
            return 0.0, np.zeros_like(fock)
        value, weight = restricted_singles(fock, occupied, pullback=pullback)
        assert weight is not None
        return 0.5 * spin_weight * value, 0.5 * spin_weight * weight

    def contract(
        space: tuple,
        central: dict,
        orbital: object,
        direct: dict,
        frame: dict,
        geom: dict,
        layout: object,
        reference: np.ndarray,
        *args: object,
        **kw: object,
    ) -> tuple[np.ndarray, dict, dict]:
        force = reference.copy()
        if ecp:
            tangent = kw["cabs_h_shells"].gradient(np.zeros_like(hcore), direct["hcore_ao"], 1 << 20)
            return force - tangent.gradient, {}, {}
        slope = direct["hcore_ao"][0, -1] + direct["hcore_ao"][-1, 0]
        force[:, 2] += np.array([slope, -slope])
        return force, {}, {}

    monkeypatch.setattr(center, "cabs_singles", singles)
    monkeypatch.setattr(center, "gaussian_orbital_forces", contract)
    root = np.diag(hcore)[:obs]
    central = {"whitener": np.eye(1)}
    snapshot = {
        "occupations": [2, beta_occupied],
        "nbf": obs,
        "overlap": np.eye(obs),
        "hcore": np.diag(root),
        "energy": -3.0,
        "forces": [[0.0, 0.0, 0.4], [0.0, 0.0, -0.4]],
    }
    for spin, occupied in (("alpha", 2), ("beta", beta_occupied)):
        snapshot[spin] = {"mo": np.eye(obs), "fock": np.diag(root), "eps": root}
        central["coefficients_" + spin] = np.eye(ext)[:, :obs]
        central["df_" + spin] = np.zeros((1, obs, obs))
        central["occupied_energies_" + spin] = root[:occupied]
    energy, forces, info = center.center_energy_forces(
        geometry,
        layout,
        tmp_path / "geometry.in",
        snapshot,
        central,
        execute,
        sink_library=None,
        forward_library=None,
        projector_library=None,
        canonical_library=None,
        orbital_library=None,
        **({"singles_only": True} if singles_only else {}),
        **ecp_options,
    )
    per_spin = -(coupling**2) / gap
    expected = per_spin * (1 + beta_occupied)
    np.testing.assert_allclose(
        energy["CABS_singles_Eh"],
        [per_spin, beta_occupied * per_spin],
        atol=1e-14,
        rtol=0,
    )
    if singles_only:
        assert set(energy) == {"CABS_energy_Eh", "CABS_singles_Eh"}
        assert energy["CABS_energy_Eh"] == pytest.approx(expected, abs=1e-14)
        assert info["active_pairs"] == 0
    else:
        assert energy["total_Eh"] == pytest.approx(
            snapshot["energy"] + expected, abs=1e-14
        )
        assert info["active_pairs"] == 1
    for h in (2e-4, 1e-4):
        slope = (
            (-((coupling + h) ** 2) + (coupling - h) ** 2)
            * (1 + beta_occupied)
            / (2 * h * gap)
        )
        expected_force = np.array([[0, 0, slope], [0, 0, -slope]])
        if not singles_only:
            expected_force += snapshot["forces"]
        np.testing.assert_allclose(forces, expected_force, atol=1e-12, rtol=0)
