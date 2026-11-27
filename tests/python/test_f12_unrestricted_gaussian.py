"""Check the spin-frame mapping when combined Gaussian shells interleave OBS and CABS."""
import numpy as np
import pytest







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
