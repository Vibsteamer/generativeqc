"""Independent invariant-space and response checks for the CUDA CABS kernel."""

import ctypes
import os
import unittest
from typing import Any

import numpy as np
from generativeqc._f12_cabs import CABSProjector, cabs_projector


def svd_projector(s: np.ndarray, obs: int, cutoff: float) -> np.ndarray:
    features = np.linalg.cholesky(s).T
    qo, _ = np.linalg.qr(features[:, :obs])
    aux = features[:, obs:]
    left, singular, _ = np.linalg.svd(aux - qo @ (qo.T @ aux), full_matrices=False)
    frame = np.linalg.solve(features, left[:, singular**2 > cutoff * singular[0] ** 2])
    return frame @ frame.T


class BindingTest(unittest.TestCase):
    def test_invalid_before_backend(self) -> None:
        with self.assertRaises(ValueError):
            cabs_projector(np.eye(2), 2)
        with self.assertRaises(ValueError):
            cabs_projector(np.eye(2), 1, relative_cutoff=1e-14)
        with self.assertRaises(ValueError):
            cabs_projector(np.eye(2), 1, cotangent=np.ones((1, 2)))


@unittest.skipUnless(
    os.environ.get("GENERATIVEQC_TEST_CUDA") == "1", "requires allocated CUDA"
)
class CABSProjectorDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = os.environ.get("GENERATIVEQC_TEST_CABS_LIBRARY")
        cls.lib = ctypes.CDLL(path) if path else None

    def run_kernel(self, s: np.ndarray, o: int, **kwargs: Any) -> CABSProjector:
        return cabs_projector(s, o, library=self.lib, **kwargs)

    def test_partial_rank_and_cross_subspace_pullback(self) -> None:
        rng = np.random.default_rng(3209)
        a = np.array([[1.2, 0.12], [0.12, 1.7]])
        d = rng.normal(size=(2, 4)) / 3
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        t = (q * [0.015, 0.55, 0.55, 1.2]) @ q.T
        s = np.block([[a, d], [d.T, t + d.T @ np.linalg.solve(a, d)]])
        seed = rng.normal(size=s.shape)
        got = self.run_kernel(
            s, 2, relative_cutoff=0.05, expected_rank=3, cotangent=seed
        )
        ref = svd_projector(s, 2, 0.05)
        np.testing.assert_allclose(got.kernel, ref, atol=2e-12)
        np.testing.assert_allclose((s @ got.kernel)[:2], 0, atol=2e-13)
        np.testing.assert_allclose(got.kernel @ s @ got.kernel, got.kernel, atol=2e-13)
        np.testing.assert_allclose(got.overlap_bar, got.overlap_bar.T, atol=1e-14)
        self.assertEqual(got.rank, 3)
        self.assertGreater(got.cutoff_margin, 0.03)
        # This direction only rotates across the kept/discarded boundary.
        cross = np.zeros_like(s)
        cross[2:, 2:] = np.outer(q[:, 0], q[:, 1]) + np.outer(q[:, 1], q[:, 0])
        random = rng.normal(size=s.shape)
        for direction in (cross, (random + random.T) / 2):
            v = direction / np.linalg.norm(direction)
            slope = np.vdot(got.overlap_bar, v)
            self.assertGreater(abs(slope), 1e-4)
            for h in (1e-4, 5e-5):
                fd = np.vdot(
                    seed,
                    svd_projector(s + h * v, 2, 0.05)
                    - svd_projector(s - h * v, 2, 0.05),
                ) / (2 * h)
                self.assertAlmostEqual(slope, fd, delta=3e-7)

    def test_linear_dependence_and_scale(self) -> None:
        # First auxiliary duplicates the OBS, second lies outside it.
        s = np.array([[1.0, 1.0, 0.3], [1.0, 1.0, 0.3], [0.3, 0.3, 1.0]])
        r = self.run_kernel(s, 1, expected_rank=1)
        self.assertIsNone(r.overlap_bar)
        np.testing.assert_allclose((s @ r.kernel)[:1], 0.0, atol=1e-14)
        np.testing.assert_allclose(r.kernel @ s @ r.kernel, r.kernel, atol=2e-14)
        np.testing.assert_allclose(
            self.run_kernel(5 * s, 1).kernel, r.kernel / 5, atol=2e-14
        )

    def test_rank_and_input_guards(self) -> None:
        s = np.diag([1.0, 0.1, 1.0])
        with self.assertRaisesRegex(RuntimeError, "cutoff guard"):
            self.run_kernel(s, 1, relative_cutoff=0.1)
        with self.assertRaisesRegex(RuntimeError, "rank changed"):
            self.run_kernel(s, 1, expected_rank=1)
        with self.assertRaisesRegex(RuntimeError, "indefinite"):
            self.run_kernel(np.diag([1.0, -0.2, 1.0]), 1)
        with self.assertRaisesRegex(RuntimeError, "OBS"):
            self.run_kernel(np.diag([0.0, 1.0, 1.0]), 2)
        s[1, 0] = 0.3
        with self.assertRaisesRegex(RuntimeError, "asymmetric"):
            self.run_kernel(s, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
