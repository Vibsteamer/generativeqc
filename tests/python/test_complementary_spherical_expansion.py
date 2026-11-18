"""Check the native complementary-shell ordering and normalization against libcint."""
from pathlib import Path
import ctypes
import math
import shutil
import subprocess

import numpy as np
import pytest


def test_complete_spherical_i_shell(tmp_path: Path) -> None:
    gto = pytest.importorskip("pyscf.gto")
    cxx = shutil.which("c++")
    cache = shutil.which("sccache") or shutil.which("ccache")
    if not cxx or not cache:
        pytest.skip("requires a cached C++ compiler")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    repo = Path(__file__).resolve().parents[2]
    source = tmp_path / "expansion.cpp"
    source.write_text(r'''
#include "molecule/basis.hpp"
extern "C" int expansion(unsigned l, double* out) {
  using namespace generativeqc::molecule;
  if (l < 2 || l > 6) return -1;
  auto rows = complementary_spherical_expansions(l);
  const unsigned ncart = cartesian_count(l);
  for (unsigned m = 0; m < rows.size(); ++m)
    for (const auto& t : rows[m]) {
      unsigned offset = l-t.component[0];
      unsigned slot = offset*(offset+1)/2 + t.component[2];
      out[slot*rows.size()+m] = t.coefficient * cartesian_component_normalization(t.component);
    }
  return ncart;
}
''')
    libpath = tmp_path / "expansion.so"
    subprocess.run([cache, cxx, "-std=c++20", "-shared", "-fPIC", "-O2",
                    "-I"+str(repo/"src"), "-I"+str(repo/"include"), str(source),
                    str(repo/"src/molecule/basis.cpp"), "-o", str(libpath)], check=True)
    fn = ctypes.CDLL(str(libpath)).expansion
    fn.argtypes = [ctypes.c_uint, np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")]
    fn.restype = ctypes.c_int
    for l in range(2, 7):
        expected = gto.cart2sph(l)
        coeff = np.zeros(expected.shape)
        assert fn(l, coeff) == coeff.shape[0]
        alpha = 0.73
        native_radial = (2*alpha/math.pi)**0.75 * (4*alpha)**(l/2)
        np.testing.assert_allclose(coeff*native_radial/gto.gto_norm(l,alpha), expected,
                                   rtol=2e-13, atol=2e-13)
