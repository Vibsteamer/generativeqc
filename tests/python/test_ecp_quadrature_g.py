"""Check all normalized real and Cartesian g functions in the generated ECP grid."""

import ctypes
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.ecp_projector import emit_ecp_quadrature_cpp


def test_g_grid_spherical_contraction_and_nuclear_jets(tmp_path: Path) -> None:
    scipy_special = pytest.importorskip("scipy.special")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("The compiled host quadrature needs a C++ compiler")
    repo = Path(__file__).resolve().parents[2]
    (tmp_path / "ecp.hpp").write_text(emit_ecp_quadrature_cpp())
    source = tmp_path / "probe.cpp"
    source.write_text(
        """
#include "ecp.hpp"
#include "molecule/basis.hpp"
#include <algorithm>
#include <numbers>
using namespace generativeqc;
extern "C" int points(bool spherical, double x, double y, double z, double* result) {
  using namespace generated;
  const auto kind = spherical ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
  const auto expansions = molecule::ao_expansions(4, kind);
  struct Component { unsigned x, y, z; double coefficient; };
  struct AO { double x{},y{},z{}; int term_count{}, primitive_count{1}, primitive_offset{};
              Component components[molecule::kMaximumAuxiliaryAoExpansionTerms]; };
  struct Jet { double v[4]; };
  struct Point { double x,y,z; };
  const double exponent=0.7;
  const core::Primitive primitive{exponent, std::pow(2*exponent/std::numbers::pi,0.75)
                                           *std::pow(4*exponent,2)};
  int largest=0;
  for(const auto& expansion:expansions){
    AO ao{};
    if(expansion.size()>std::size(ao.components))return -1;
    for(const auto& entry:expansion){
      const auto [a,b,c]=entry.component;
      ao.components[ao.term_count++]={a,b,c,ecp_component_coefficient(a,b,c,entry.coefficient)};
    }
    largest=std::max(largest,ao.term_count);
    const auto values=ecp_evaluate_ao<Jet>(ao,&primitive,Point{x,y,z},1,0,0,0,true);
    result=std::copy(values.v,values.v+4,result);
  }
  return largest;
}
"""
    )
    so = tmp_path / "probe.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            f"-I{repo / 'src'}",
            f"-I{repo / 'include'}",
            str(source),
            str(repo / "src/molecule/basis.cpp"),
            "-o",
            str(so),
        ],
        check=True,
    )
    points = ctypes.CDLL(str(so)).points
    points.argtypes = (
        [ctypes.c_bool]
        + [ctypes.c_double] * 3
        + [np.ctypeslib.ndpointer(dtype=np.float64)]
    )
    points.restype = ctypes.c_int
    alpha = 0.7
    powers = [(x, 4 - x - z, z) for x in range(4, -1, -1) for z in range(5 - x)]

    def reference(r: np.ndarray, spherical: bool) -> np.ndarray:
        radius = float(np.linalg.norm(r))
        radial = math.exp(-alpha * radius**2)
        if not spherical:
            norm = (2 * alpha / math.pi) ** 0.75 * (4 * alpha) ** 2
            return (
                radial
                * norm
                * np.array(
                    [
                        np.prod(r**p)
                        / math.sqrt(math.prod(math.prod(range(1, 2 * v, 2)) for v in p))
                        for p in powers
                    ]
                )
            )
        theta, phi = math.acos(r[2] / radius), math.atan2(r[1], r[0])
        norm = math.sqrt(2 * (2 * alpha) ** 5.5 / math.gamma(5.5))
        angular = []
        for m in range(-4, 5):
            if hasattr(scipy_special, "sph_harm_y"):
                y = scipy_special.sph_harm_y(4, abs(m), theta, phi)
            else:
                y = scipy_special.sph_harm(abs(m), 4, phi, theta)
            component = y.imag if m < 0 else y.real
            angular.append(component * (math.sqrt(2) * (-1) ** abs(m) if m else 1))
        return radial * norm * radius**4 * np.array(angular)

    for spherical, n, maximum in ((False, 15, 1), (True, 9, 6)):
        for xyz in (np.array([0.27, -0.48, 0.72]), np.array([-0.31, 0.83, 0.41])):
            measured = np.empty((n, 4))
            assert points(spherical, *xyz, measured) == maximum
            np.testing.assert_allclose(
                measured[:, 0], reference(xyz, spherical), rtol=2e-13, atol=3e-15
            )
            for h in (2e-4, 1e-4):
                for axis, shift in enumerate(h * np.eye(3)):
                    fd = (
                        -reference(xyz - 2 * shift, spherical)
                        + 8 * reference(xyz - shift, spherical)
                        - 8 * reference(xyz + shift, spherical)
                        + reference(xyz + 2 * shift, spherical)
                    ) / (12 * h)
                    np.testing.assert_allclose(
                        measured[:, axis + 1], fd, rtol=2e-9, atol=1e-11
                    )
