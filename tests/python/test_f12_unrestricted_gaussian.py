"""Check the spin-frame mapping when combined Gaussian shells interleave OBS and CABS."""
import numpy as np

from benchmarks.f12_unrestricted_gaussian import spin_ao_inputs


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
