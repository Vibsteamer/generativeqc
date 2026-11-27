"""Allocated-device qualification of the CABS energy and fixed-frame Fock response."""

import ctypes
import os
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc._f12_cabs import cabs_singles

FIXTURES = Path(__file__).parents[1] / "reference_data/f12/psi4_cabs_singles.npz"


def direct_energy(fock: np.ndarray, occupied: int, weight: int) -> float:
    """Independent finite-dimensional solve in the supplied noncanonical frame."""
    a = fock[:occupied, :occupied]
    b = fock[occupied:, occupied:]
    c = fock[:occupied, occupied:]
    operator = np.kron(np.eye(len(b)), a) - np.kron(b.T, np.eye(len(a)))
    rhs = c.ravel(order="F")
    return weight * (rhs @ np.linalg.solve(operator, rhs))


@unittest.skipUnless(
    os.environ.get("GENERATIVEQC_TEST_CUDA") == "1", "requires allocated CUDA"
)
class CabsSinglesCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = os.environ.get("GENERATIVEQC_TEST_CABS_LIBRARY")
        cls.library = ctypes.CDLL(path) if path else None

    def solve(
        self, fock: np.ndarray, occupied: int, weight: int = 2, **kwargs: Any
    ) -> tuple[float, np.ndarray | None]:
        return cabs_singles(
            fock, occupied, spin_weight=weight, library=self.library, **kwargs
        )

    def test_psi4_component_energies(self) -> None:
        with np.load(FIXTURES, allow_pickle=False) as data:
            for name in ("he", "h2", "h2_df", "water_df"):
                with self.subTest(name=name):
                    f = data[name + "_fock"]
                    o = int(data[name + "_occupied"])
                    energy, bar = self.solve(f, o)
                    self.assertAlmostEqual(
                        energy, float(data[name + "_energy"]), delta=2e-11
                    )
                    np.testing.assert_allclose(bar, bar.T, atol=1e-13, rtol=0)
                    scalar, absent = self.solve(f, o, pullback=False)
                    self.assertIsNone(absent)
                    self.assertAlmostEqual(scalar, energy, delta=1e-14)

    def test_full_pullback_matches_independent_block_solve(self) -> None:
        rng = np.random.default_rng(24)
        # Repeated eigenvalues within each subspace leave the Sylvester solve regular.
        oo = np.diag([-1.1, -1.1, -0.6])
        vv = np.diag([0.2, 0.2, 0.8, 1.4])
        c = 0.13 * rng.normal(size=(3, 4))
        qo = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        qv = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        f = np.block(
            [[qo.T @ oo @ qo, qo.T @ c @ qv], [qv.T @ c.T @ qo, qv.T @ vv @ qv]]
        )
        for weight in (1, 2):
            energy, bar = self.solve(f, 3, weight)
            self.assertAlmostEqual(energy, direct_energy(f, 3, weight), delta=5e-14)
            for i, j in zip(*np.tril_indices(len(f)), strict=True):
                direction = np.zeros_like(f)
                direction[i, j] = direction[j, i] = 1.0
                prediction = np.sum(bar * direction)
                for h in (3e-4, 1e-4):
                    diff = (
                        direct_energy(f + h * direction, 3, weight)
                        - direct_energy(f - h * direction, 3, weight)
                    ) / (2 * h)
                    self.assertAlmostEqual(diff, prediction, delta=4e-8)

    def test_spin_channels_and_rotation(self) -> None:
        f = np.array([[-1.2, 0.04, 0.12], [0.04, -0.5, 0.13], [0.12, 0.13, 0.31]])
        restricted, bar = self.solve(f, 1, 2)
        alpha, spin_bar = self.solve(f, 1, 1)
        self.assertAlmostEqual(restricted, 2 * alpha, delta=1e-14)
        np.testing.assert_allclose(bar, 2 * spin_bar, atol=1e-14)
        beta, beta_bar = self.solve(f, 2, 1)
        self.assertAlmostEqual(
            alpha + beta, direct_energy(f, 1, 1) + direct_energy(f, 2, 1), delta=1e-14
        )
        u = np.eye(3)
        u[:2, :2] = [[0.6, -0.8], [0.8, 0.6]]
        rotated, rotated_bar = self.solve(u.T @ f @ u, 2, 1)
        self.assertAlmostEqual(rotated, beta, delta=1e-14)
        np.testing.assert_allclose(rotated_bar, u.T @ beta_bar @ u, rtol=0, atol=2e-14)

    def test_empty_channel_and_invalid_input(self) -> None:
        f = np.diag([-1.0, 0.0, 1.0])
        for o in (0, 3):
            energy, bar = self.solve(f, o, 1)
            self.assertEqual(energy, 0.0)
            np.testing.assert_array_equal(bar, np.zeros_like(f))
        with self.assertRaisesRegex(RuntimeError, "denominator"):
            self.solve(np.diag([-1.0, -1.0]), 1)
        f[0, 1] += 0.2
        with self.assertRaisesRegex(RuntimeError, "asymmetric"):
            self.solve(f, 1)
        f[0, 1] = np.nan
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            self.solve(f, 1)
        self.assertEqual(self.solve(np.diag([-1.0, 1.0]), 1)[0], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
