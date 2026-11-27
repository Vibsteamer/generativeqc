"""Independent radial quadrature and Cartesian-center tests of F12 DF primitives."""

from __future__ import annotations

import ctypes
import itertools
import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
# The shortest Gaussian in the beta=2.2 fit needs the denser real-space
# quadrature. Increasing from 320 to 480 nodes changes relative values by <1e-13.
NODES, WEIGHTS = np.polynomial.legendre.leggauss(320)
U, W = (NODES + 1) / 2, WEIGHTS / 2
MAX_ORDER = 20


def moment_reference(op: int, t: float, rho: float, z: float) -> np.ndarray:
    orders = np.arange(MAX_ORDER + 1)
    if op == 1:
        # Direct quadrature of the transformed integral, with no Boys recurrence.
        v = (z + rho * U**2) / (z + rho)
        return (rho / (z + rho)) * (
            (v[None, :] ** orders[:, None]) @ (W * np.exp(-v * t))
        )

    def pure(exponent: complex) -> np.ndarray:
        p = rho + exponent
        fraction = exponent / p
        return (
            (0.5 * math.sqrt(math.pi) * rho / p**1.5)
            * np.exp(-t * fraction)
            * fraction**orders
        )

    if op == 0:
        return pure(z)
    # r² e^(-z r²) is a derivative with respect to z; complex step avoids subtraction.
    step = 1e-20 * (rho + z)
    return -np.imag(pure(z + 1j * step)) / step


def gaussian_kernel(op: int, z: float, r: np.ndarray) -> np.ndarray:
    return r ** (0, -1, 2)[op] * np.exp(-z * r**2)


def fitted_kernel(
    op: int, first: np.ndarray, second: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """Evaluate the archived two radial fits pointwise before quadrature."""

    def values(terms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        exponents, coefficients = terms.T
        gaussian = np.exp(-exponents[:, None] * r[None, :] ** 2)
        return coefficients @ gaussian, -2 * r * ((coefficients * exponents) @ gaussian)

    a, da = values(first)
    if op == 3:
        return a
    if op == 4:
        return a / r
    b, db = values(second)
    return a * b if op == 5 else da * db


def independent_s_integral(
    exponents: np.ndarray,
    centers: np.ndarray,
    kernel: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Integrate the kernel against the three-dimensional relative Gaussian."""
    a, b, c, d = exponents
    p, q = a + b, c + d
    rho = p * q / (p + q)
    length = np.linalg.norm(
        (a * centers[0] + b * centers[1]) / p - (c * centers[2] + d * centers[3]) / q
    )
    upper = length + 12 / np.sqrt(rho)
    r = U * upper
    w = 4 * rho * r * length
    angular = np.ones_like(r) if length == 0 else -np.expm1(-w) / w
    radial = np.exp(-rho * (r - length) ** 2) * angular
    norm = (math.pi**2 / (p * q)) ** 1.5 * (rho / math.pi) ** 1.5 * 4 * math.pi
    norm *= math.exp(
        -a * b / p * np.sum((centers[0] - centers[1]) ** 2)
        - c * d / q * np.sum((centers[2] - centers[3]) ** 2)
    )
    return float(norm * upper * (W @ (r**2 * radial * kernel(r))))


class GeminalIntegralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        gpu = os.environ.get("GENERATIVEQC_TEST_CUDA") == "1"
        compiler = shutil.which("nvcc" if gpu else "c++")
        cache = shutil.which("sccache") or shutil.which("ccache")
        if not compiler or not cache:
            if gpu:
                raise RuntimeError(
                    "allocated CUDA test needs nvcc and the project compiler cache"
                )
            raise unittest.SkipTest("missing host C++ compiler or compiler cache")
        subprocess.run([cache, "--version"], check=True, capture_output=True)
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        flags = ["-x", "c++", "-fPIC"]
        if gpu:
            cap = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                text=True,
            ).splitlines()[0]
            flags = ["-arch=sm_" + cap.strip().replace(".", ""), "-Xcompiler=-fPIC"]
        obj = str(folder / "probe.o")
        subprocess.run(
            [
                cache,
                compiler,
                "-std=c++17",
                "-O2",
                *flags,
                "-I" + str(ROOT / "src"),
                "-c",
                str(ROOT / "tests/native/f12_radial_probe.cu"),
                "-o",
                obj,
            ],
            check=True,
        )
        lib = folder / "probe.so"
        subprocess.run([compiler, "-shared", obj, "-o", str(lib)], check=True)
        cls.probe = ctypes.CDLL(str(lib)).evaluate_f12_rows
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
        cls.probe.argtypes = (ctypes.c_int, ptr, ptr)
        cls.probe.restype = ctypes.c_int

    def evaluate(self, rows: list[list[float]]) -> np.ndarray:
        inputs = np.asarray(rows, dtype=np.float64)
        outputs = np.zeros((len(inputs), 35), dtype=np.float64)
        self.assertEqual(self.probe(len(inputs), inputs, outputs), 0)
        return outputs

    def test_radials_through_order_twenty(self) -> None:
        entries = list(
            itertools.product(
                range(3),
                (0.0, 1e-5, 0.2, 8.0, 40.0, 300.0),
                (0.03, 1.0, 35.0),
                (0.0, 0.01, 0.9, 4.0, 130.0),
            )
        )
        rows = [
            [op, -1, MAX_ORDER, z, t, rho, *([0.0] * 16)] for op, t, rho, z in entries
        ]
        for (op, t, rho, z), row in zip(entries, self.evaluate(rows), strict=True):
            self.assertEqual(row[0], 1, (op, t, rho, z))
            expected = moment_reference(op, t, rho, z)
            # The r² moment can cross zero. Scale its absolute gate by the row.
            np.testing.assert_allclose(
                row[1:22],
                expected,
                rtol=3e-11,
                atol=3e-14 * max(np.max(abs(expected)), 1e-280),
                err_msg=str((op, t, rho, z)),
            )

    def test_raw_metric_three_and_four_center_values_and_derivatives(self) -> None:
        centers = np.array(
            [
                [0.1, -0.3, 0.24],
                [-0.5, 0.17, 1.13],
                [0.4, 1.1, -0.72],
                [0.33, 0.11, 0.68],
            ]
        )
        for mask, op in itertools.product((10, 8, 0), range(3)):
            e = np.array([0.8, 0.61, 0.34, 1.04])
            for index in range(4):
                if mask & (1 << index):
                    e[index] = 0.0
            z = 0.73
            (row,) = self.evaluate([[op, mask, 1, z, 0.0, 0.0, *e, *centers.ravel()]])
            self.assertEqual(row[0], 1)
            radial = partial(gaussian_kernel, op, z)
            target = independent_s_integral(e, centers, radial)
            self.assertAlmostEqual(row[22], target, delta=1e-12 * max(1.0, abs(target)))
            grad = row[23:].reshape(4, 3)
            np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=3e-13)
            for i, a in np.ndindex(4, 3):
                for step in (3e-4, 1e-4):
                    dx = np.zeros_like(centers)
                    dx[i, a] = step
                    diff = (
                        independent_s_integral(e, centers + dx, radial)
                        - independent_s_integral(e, centers - dx, radial)
                    ) / (2 * step)
                    self.assertAlmostEqual(
                        grad[i, a], diff, delta=4e-8 * max(abs(target), 1.0)
                    )

    def test_mixed_fitted_slater_df_integrals(self) -> None:
        fixture = ROOT / "tests/reference_data/f12/psi4_geminal_coefficients.json"
        fits = json.loads(fixture.read_text())["terms_exponent_coefficient"]
        centers = np.array(
            [[0.3, 0.0, 0.2], [-0.1, 0.7, 0.8], [0.3, -0.5, 0.1], [0.5, -0.4, 0.0]]
        )
        direction = np.array(
            [[1.0, 2.0, 0.0], [-0.5, 0.0, 0.5], [1.0, 0.0, -1.0], [0.0, -1.0, 0.0]]
        )
        direction /= np.linalg.norm(direction)
        for mask, op, (left, right), normalized in itertools.product(
            (8, 10), range(3, 7), ((0.9, 2.2), (2.2, 0.9), (1.4, 1.4)), (False, True)
        ):
            exponents = np.array([0.28, 0.0 if mask == 10 else 1.2, 0.43, 0.0])
            a, b = (np.array(fits[str(beta)]) for beta in (left, right))
            if normalized:
                a[:, 1] /= left
                b[:, 1] /= right
            radial = partial(fitted_kernel, op, a, b)
            ref = independent_s_integral(exponents, centers, radial)
            (row,) = self.evaluate(
                [
                    [
                        op,
                        mask,
                        1,
                        left,
                        right,
                        int(normalized),
                        *exponents,
                        *centers.ravel(),
                    ]
                ]
            )
            self.assertEqual(row[0], 1)
            self.assertAlmostEqual(row[22], ref, delta=3e-11 * max(1.0, abs(ref)))
            response = row[23:] @ direction.ravel()
            for step in (3e-4, 1e-4):
                plus = independent_s_integral(
                    exponents, centers + step * direction, radial
                )
                minus = independent_s_integral(
                    exponents, centers - step * direction, radial
                )
                self.assertAlmostEqual(
                    response,
                    (plus - minus) / (2 * step),
                    delta=4e-8 * max(1.0, abs(ref)),
                )

    def test_invalid_exponent_order_and_unit_slot(self) -> None:
        rows = [
            [0, -1, 21, 0.9, 0.0, 1.0, *([0.0] * 16)],
            [0, -1, 1, -0.1, 0.0, 1.0, *([0.0] * 16)],
            [0, 8, 1, 0.4, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, *([0.0] * 12)],
            [5, 10, 1, 0.0, 1.4, 1.0, 1.0, 0.0, 1.0, 0.0, *([0.0] * 12)],
            [5, 10, 1, 0.9, 1.4, 3.0, 1.0, 0.0, 1.0, 0.0, *([0.0] * 12)],
        ]
        np.testing.assert_array_equal(self.evaluate(rows)[:, 0], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
