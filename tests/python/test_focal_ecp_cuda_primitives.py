"""Validate contracted scalar arithmetic shared with the through-i CUDA ECP kernel."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import test_focal_ecp_one_electron_sources as focal_tests
from generativeqc._f12_focal_shells import pack_shells, spherical_projections

if TYPE_CHECKING:
    from conftest import NativeCxx

ecp_library = focal_tests.ecp_library


def test_geometry_jets_and_projector_radial_input(
    tmp_path: Path, required_native_cxx: NativeCxx, ecp_library: Path
) -> None:
    xyz = np.array([[0.13, -0.16, 0.27], [1.23, 0.52, -0.39], [-0.68, 0.81, 1.43]])
    req, oracle = focal_tests._request("Br", xyz)
    mol, _ = oracle(xyz)
    shells = pack_shells(
        req.atoms,
        (("orbital", req.basis.orbital), ("CABS", req.basis.cabs)),
        64 * 2**20,
    )
    n = sum(shells.ao_counts)
    coef = np.zeros((n, 28))
    ao = np.empty((n, 4), dtype=np.int32)
    transforms = spherical_projections(ecp_library, set(shells.angular.tolist()))
    start = 0
    for shell, (l, a) in enumerate(zip(shells.angular, shells.atom, strict=True)):
        rows = slice(start, start + 2 * l + 1)
        ao[rows] = [l, a, *shells.prim_offsets[shell : shell + 2]]
        coef[rows, : transforms[l].shape[1]] = transforms[l]
        start += 2 * l + 1
    src = tmp_path / "probe.cpp"
    src.write_text("""
#include "f12_focal_ecp_kernels.cuh"
using namespace generativeqc::focal_ecp;
extern "C" void sample(int n,int nq,int center,const int* ao,const double* xyz,const double* alpha,
const double* radial,const double* transform,const double* r,const double* u,double* out) {
  Basis b{ao,xyz,alpha,radial,transform};
  for(int i=0;i<nq;++i) for(int a=0;a<n;++a)
    point_jet(b,a,center,4,r[i],u+3*i,out+4*(i*n+a));
}
extern "C" double weight(int center,int channel,int count,const int* t,const double* p,double r,double dr) {
  return channel_weight(center,channel,count,t,p,r,dr);
}
""")
    path = required_native_cxx.build_shared(
        [src],
        tmp_path / "probe.so",
        compile_args=[
            "-std=c++17",
            "-O2",
            "-I" + str(Path(__file__).resolve().parents[2] / "benchmarks"),
        ],
    )
    lib = ctypes.CDLL(str(path))
    d = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    i = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    lib.sample.argtypes = [ctypes.c_int] * 3 + [i] + [d] * 7
    lib.sample.restype = None
    rng = np.random.default_rng(37)
    u = rng.normal(size=(11, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    radii = np.linspace(0.12, 2.35, len(u))
    for center in range(len(xyz)):
        result = np.empty((len(u), n, 4))
        lib.sample(
            n,
            len(u),
            center,
            ao,
            xyz,
            shells.exponents,
            shells.radial,
            coef,
            radii,
            u,
            result,
        )
        ref = mol.eval_gto(
            "GTOval_sph_deriv1", xyz[center] + radii[:, None] * u
        ).transpose(1, 2, 0)
        np.testing.assert_allclose(result, ref, atol=7e-14, rtol=0)
    lib.weight.argtypes = [ctypes.c_int] * 3 + [i, d, ctypes.c_double, ctypes.c_double]
    lib.weight.restype = ctypes.c_double
    terms = np.array(
        [(at, l, p) for at in (0, 2) for l in range(4) for p in range(5)],
        dtype=np.int32,
    )
    par = rng.uniform(0.02, 2, (len(terms), 2))
    par[::2, 1] *= -1
    for at in range(3):
        for l in range(4):
            expected = sum(
                0.16 * c * 0.67**p * np.exp(-a * 0.67**2)
                for (j, h, p), (a, c) in zip(terms, par, strict=True)
                if (j, h) == (at, l)
            )
            assert lib.weight(
                at, l, len(terms), terms, par, 0.67, 0.16
            ) == pytest.approx(expected, abs=2e-15)
