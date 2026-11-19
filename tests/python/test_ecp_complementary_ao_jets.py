"""Execute generated complementary g/h/i Gaussian jets against independent derivatives."""

from __future__ import annotations

import ctypes
import math
import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.integral.ecp import emit_ecp_ao_cuda

if TYPE_CHECKING:
    from pathlib import Path


def test_compiled_high_angular_jets(tmp_path: Path) -> None:
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip(
            "C++ compiler is required to exercise the emitted scalar instructions"
        )
    cpp = tmp_path / "probe.cpp"
    cpp.write_text(
        emit_ecp_ao_cuda(6).replace("__device__ inline", "inline")
        + 'extern "C" void jet(unsigned x,unsigned y,unsigned z,double a,double b,double c,double p,double*out){generativeqc::generated::ecp_ao(x,y,z,a,b,c,p,out);}\n'
    )
    so = tmp_path / "jets.so"
    subprocess.run(
        [cxx, "-O1", "-shared", "-fPIC", str(cpp), "-o", str(so)], check=True
    )
    jet = ctypes.CDLL(str(so)).jet
    jet.argtypes = (
        [ctypes.c_uint] * 3 + [ctypes.c_double] * 4 + [ctypes.POINTER(ctypes.c_double)]
    )
    jet.restype = None
    xyz = np.array([0.34, -0.74, 0.48])
    alpha = 0.86
    for l in range(7):
        for x in range(l + 1):
            for y in range(l - x + 1):
                p = np.array([x, y, l - x - y])

                def value(pos: np.ndarray, powers: np.ndarray = p) -> float:
                    return float(
                        math.exp(-alpha * float(pos @ pos)) * np.prod(pos**powers)
                    )

                out = np.empty(4)
                jet(
                    *p, *xyz, alpha, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                )
                assert out[0] == pytest.approx(value(xyz), abs=1e-14)
                for h in (0.0004, 0.0002):
                    shifts = np.eye(3) * h
                    fd = np.array(
                        [
                            (
                                value(xyz - 2 * d)
                                - 8 * value(xyz - d)
                                + 8 * value(xyz + d)
                                - value(xyz + 2 * d)
                            )
                            / (-12 * h)
                            for d in shifts
                        ]
                    )
                    np.testing.assert_allclose(out[1:], fd, rtol=1e-9, atol=1e-11)
