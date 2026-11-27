"""Check the spin-frame mapping when combined Gaussian shells interleave OBS and CABS."""
import numpy as np
import pytest

from benchmarks.f12_unrestricted_gaussian import spin_ao_inputs


@pytest.mark.parametrize(
    ("angular_momentum", "message"),
    [(5, "h-shell basis requires"), (6, "i-shell ECP center requires"), (7, "at most i shells")],
)
def test_unrestricted_center_requires_a_matching_angular_provider(
    tmp_path, angular_momentum, message
):
    import json

    from benchmarks.f12_unrestricted_gaussian_center import center_energy_forces

    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({
        "layout": {"space": [2, 0, 2, 20, 8]},
        "combined_basis": {"shells": [{"angular_momentum": angular_momentum}]},
    }))
    with pytest.raises(ValueError, match=message):
        center_energy_forces(
            geometry, tmp_path / "layout.npz", tmp_path / "snapshot.in",
            {"occupations": (2, 0), "nbf": 2}, {}, None,
            sink_library=tmp_path / "sink", forward_library=tmp_path / "forward",
            projector_library=None, canonical_library=None, orbital_library=None,
        )


def test_spin_frames_preserve_physical_ao_matrix_elements():
    # Three orbital functions and two complementary functions interleaved in source order.
    perm = np.array([3, 0, 2, 4, 1])
    obs = np.array([2, 0, 1])
    rng = np.random.default_rng(617)
    raw = rng.normal(size=(5, 5))
    overlap = raw.T @ raw + np.eye(5)
    hcore = raw.T + raw
    ordered_overlap = overlap[np.ix_(perm, perm)]
    chol = np.linalg.cholesky(ordered_overlap)
    frame = np.linalg.inv(chol.T)[:, 3:]
    values = {key: rng.standard_normal((2, 2)) for key in (
        "metric", "metric_f12", "metric_f12_squared", "metric_f12_over_coulomb",
        "metric_double_commutator",
    )}
    for tag in ("coulomb", "f12", "f12_squared", "f12_over_coulomb", "double_commutator"):
        values["ao_" + tag] = rng.standard_normal((2, 5, 5))
    ao, embedding, cabs, order, so, ho = spin_ao_inputs(values, overlap, hcore, perm, obs, frame)
    np.testing.assert_array_equal(order, obs)
    expected_native_columns = np.eye(5)[:, [2, 3, 0]]
    np.testing.assert_allclose(so, expected_native_columns.T @ overlap @ expected_native_columns)
    np.testing.assert_allclose(ho, expected_native_columns.T @ hcore @ expected_native_columns)
    np.testing.assert_allclose(embedding.T @ overlap @ cabs, 0, atol=1e-14)
    np.testing.assert_allclose(cabs.T @ overlap @ cabs, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(ao["metric/geminal_coulomb"], values["metric_f12_over_coulomb"])
    np.testing.assert_allclose(ao["ao_double_commutator"], values["ao_double_commutator"])


def test_spin_frame_metric_weights_match_independent_moving_frames():
    from generativeqc.f12_orbital_response import unrestricted_frame_overlap_weights
    from generativeqc_compiler.tensor import execute

    rng = np.random.default_rng(78)
    q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    obs, complement = q[:, :3], q[:, 3:6]
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    central = {"coefficients_alpha": obs, "coefficients_beta": obs @ rot}
    bars = {key: rng.normal(size=(7, 6)) for key in central}
    ds = rng.normal(size=(7, 7))
    ds = (ds + ds.T) / 2
    weights = unrestricted_frame_overlap_weights(central, complement, bars, execute)

    def normalize(columns, metric):
        vals, vecs = np.linalg.eigh(columns.T @ metric @ columns)
        return columns @ (vecs / np.sqrt(vals)) @ vecs.T

    def energy(step, spin):
        metric = np.eye(7) + step * ds
        c = normalize(central["coefficients_" + spin], metric)
        ca = normalize(complement - c @ (c.T @ metric @ complement), metric)
        return np.vdot(bars["coefficients_" + spin], np.column_stack((c, ca)))

    for spin in weights:
        h = 1e-5
        finite = (energy(h, spin) - energy(-h, spin)) / (2 * h)
        np.testing.assert_allclose(np.vdot(weights[spin], ds), finite, atol=2e-8)


@pytest.mark.parametrize("beta_occupation", [0, 1])
def test_shell_reference_fit_recovers_both_canonical_focks(beta_occupation):
    from benchmarks.f12_unrestricted_gaussian_center import shell_reference_fit
    from generativeqc_compiler.tensor import execute

    rng = np.random.default_rng(813)
    frame, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    # AO operators are diagonal in an underlying rotated basis. The spin MOs
    # label its first three orbitals in two different orders.
    spectrum = np.array([-2.0, -0.8, 0.4, 0.9])
    coulomb_diagonals = rng.normal(size=(2, 4)) * 0.1
    whitener = np.array([[1.1, 0.2], [0.0, 0.7]])
    ao = {
        "overlap_ao": np.eye(4),
        "hcore_ao": (frame * spectrum) @ frame.T,
        "ao_coulomb": np.stack([(frame * v) @ frame.T for v in coulomb_diagonals]),
        "metric_whitener": whitener,
        "coefficients_alpha": frame,
        "coefficients_beta": frame[:, [1, 0, 2, 3]],
    }
    fitted = whitener @ coulomb_diagonals
    charge = fitted[:, :2].sum(axis=1) + beta_occupation * fitted[:, 1]
    energies, expected = {}, {}
    for spin, occ, order in (("alpha", 2, [0, 1, 2]), ("beta", beta_occupation, [1, 0, 2])):
        factors = fitted[:, order]
        expected[spin] = np.stack([np.diag(v) for v in factors])
        eps = spectrum[order] + charge @ factors
        eps[:occ] -= np.sum(factors[:, :occ] ** 2, axis=0)
        energies[spin] = eps
    space = (2, beta_occupation, 3, 4, 2)
    central, errors = shell_reference_fit(space, ao, energies, execute)
    assert max(errors.values()) < 1e-13
    for spin in energies:
        np.testing.assert_allclose(central["df_" + spin], expected[spin], atol=1e-13)
    energies["beta"][2] += 1e-4
    with pytest.raises(ValueError, match="Shell DF Fock differs"):
        shell_reference_fit(space, ao, energies, execute)
