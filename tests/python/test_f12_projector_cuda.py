"""Independent invariant-space and response checks for the CUDA CABS kernel."""

import ctypes
import os
import unittest
from types import SimpleNamespace
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
    def test_native_rectangular_frame_contract(self) -> None:
        pointer = ctypes.POINTER(ctypes.c_double)
        signature = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_size_t,
            pointer,
            ctypes.POINTER(ctypes.c_int),
            pointer,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        )
        calls = []

        def native(
            device: int,
            n: int,
            obs: int,
            cutoff: float,
            guard: float,
            expected: int,
            s: Any,
            seed: Any,
            k: Any,
            c: Any,
            capacity: int,
            bar: Any,
            rank: Any,
            margin: Any,
            error: Any,
            error_size: int,
        ) -> int:
            calls.append((device, n, obs, capacity, bool(seed), bool(bar)))
            np.ctypeslib.as_array(k, shape=(n * n,))[:] = np.diag(
                [0.0, 1.0, 0.0]
            ).ravel()
            np.ctypeslib.as_array(c, shape=(capacity,))[:] = np.nan
            np.ctypeslib.as_array(c, shape=(capacity,))[:n] = [0.0, 1.0, 0.0]
            rank[0], margin[0] = 1, 0.1
            return 0

        library = SimpleNamespace(
            generativeqc_f12_cabs_projector_cuda_v2=signature(native)
        )
        result = cabs_projector(np.eye(3), 1, library=library)
        self.assertEqual(calls, [(0, 3, 1, 6, False, False)])
        self.assertEqual(result.frame.shape, (3, 1))
        self.assertTrue(result.frame.flags.c_contiguous)
        np.testing.assert_array_equal(result.frame @ result.frame.T, result.kernel)
        self.assertEqual(result.rank, 1)

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

    def check_frame(self, result: CABSProjector, s: np.ndarray, obs: int) -> None:
        self.assertEqual(result.frame.shape, (len(s), result.rank))
        np.testing.assert_allclose(
            result.frame.T @ s @ result.frame, np.eye(result.rank), atol=2e-12
        )
        np.testing.assert_allclose((s @ result.frame)[:obs], 0.0, atol=2e-13)
        np.testing.assert_allclose(
            result.frame @ result.frame.T, result.kernel, rtol=2e-12, atol=2e-12
        )

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
        self.check_frame(got, s, 2)
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
        self.check_frame(r, s, 1)
        self.assertIsNone(r.overlap_bar)
        np.testing.assert_allclose((s @ r.kernel)[:1], 0.0, atol=1e-14)
        np.testing.assert_allclose(r.kernel @ s @ r.kernel, r.kernel, atol=2e-14)
        np.testing.assert_allclose(
            self.run_kernel(5 * s, 1).kernel, r.kernel / 5, atol=2e-14
        )

    def test_resolved_observable_complement_with_conditioned_obs(self) -> None:
        # Distorted methane with an F12-D basis has a small projected overlap
        # eigenvalue despite full retained rank. The inverse-times-block path
        # amplified its forward error when constructing the normalized columns.
        rng = np.random.default_rng(441)
        obs, aux = 20, 32
        q = np.linalg.qr(rng.normal(size=(obs, obs)))[0]
        u = np.linalg.qr(rng.normal(size=(aux, aux)))[0]
        lift = 0.5 * rng.normal(size=(obs, aux))
        a = (q * np.geomspace(3e-5, 2, obs)) @ q.T
        a = (a + a.T) / 2
        t = (u * np.geomspace(1e-5, 1, aux)) @ u.T
        cross = a @ lift
        s = np.block([[a, cross], [cross.T, t + cross.T @ lift]])
        result = self.run_kernel(s, obs, expected_rank=aux, cotangent=s)
        frame = result.frame
        np.testing.assert_allclose(
            frame.T @ (s @ frame), np.eye(aux), atol=2e-9, rtol=0
        )
        np.testing.assert_allclose(s[:obs] @ frame, 0, atol=2e-9)
        assert result.overlap_bar is not None
        # The seed is fixed to the central S: K((1+h)S)=K(S)/(1+h).
        self.assertAlmostEqual(float(np.vdot(result.overlap_bar, s)), -aux, delta=2e-6)

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

    def test_insufficient_frame_capacity_publishes_nothing(self) -> None:
        from generativeqc import _native

        library = self.lib or _native.load_library(device="cuda", device_id=0)
        cabs_projector(np.eye(3), 1, library=library)
        routine = library.generativeqc_f12_cabs_projector_cuda_v2
        pointer = ctypes.POINTER(ctypes.c_double)
        overlap = np.eye(3)
        kernel = np.full((3, 3), -123.0)
        frame = np.full(6, -234.0)
        rank, margin = ctypes.c_int(-7), ctypes.c_double(-8.0)
        error = ctypes.create_string_buffer(512)
        status = routine(
            0,
            3,
            1,
            1e-9,
            1e-12,
            -1,
            overlap.ctypes.data_as(pointer),
            None,
            kernel.ctypes.data_as(pointer),
            frame.ctypes.data_as(pointer),
            5,
            None,
            ctypes.byref(rank),
            ctypes.byref(margin),
            error,
            len(error),
        )
        self.assertNotEqual(status, 0)
        self.assertIn(b"frame capacity", error.value)
        np.testing.assert_array_equal(kernel, -123.0)
        np.testing.assert_array_equal(frame, -234.0)
        self.assertEqual((rank.value, margin.value), (-7, -8.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
