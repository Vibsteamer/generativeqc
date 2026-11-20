"""Exercise the opt-in i-shell CABS one-electron ABI and independent S/T/V."""

import ctypes as ct
from math import pi, prod
from pathlib import Path

import numpy as np
import pytest


_DOUBLE = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
_INT = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
_SIZE = ct.c_size_t
_SIZE_REF = ct.POINTER(_SIZE)
_XYZ = np.array([[0.32, -0.57, 0.21], [-0.48, 0.42, 0.19], [0.2, 0.61, -0.35]])
# Effective charges for Br/ECP10, I/ECP28 and a hydrogen nucleus.
_CHARGES = np.array([25, 25, 1], dtype=np.int32)


@pytest.fixture(scope="module")
def libraries(tmp_path_factory, required_native_cxx):
    root = Path(__file__).resolve().parents[2]
    source = root / "benchmarks/f12_cabs_h_one_electron.cu"
    build = tmp_path_factory.mktemp("cabs-one-electron-cohorts")
    result = {}
    for cohort in ("h", "i"):
        flags = [
            "-x", "c++", "-std=c++20", "-O2",
            "-I" + str(root / "src"), "-I" + str(root / "include"),
        ]
        if cohort == "i":
            flags.append("-DGQC_F12_CABS_I_ONE_ELECTRON")
        binary = required_native_cxx.build_shared(
            [source], build / f"cabs-{cohort}.so", compile_args=flags
        )
        result[cohort] = ct.CDLL(str(binary))
    return result


def _bind(lib, cohort, suffix, argtypes):
    name = f"gqc_f12_cabs_{cohort}_one_electron{suffix}_host_reference"
    function = getattr(lib, name)
    function.argtypes = argtypes
    function.restype = ct.c_int
    return function


def _cartesian_norms(l: int, exponent: float) -> np.ndarray:
    """Exact raw primitive moments in x-descending, then y-descending order."""
    ratio = 4 * exponent
    return (pi / (2 * exponent)) ** 1.5 * np.array(
        [
            prod(prod(range(1, 2 * n, 2)) / ratio**n for n in (x, y, l - x - y))
            for x in range(l, -1, -1)
            for y in range(l - x, -1, -1)
        ]
    )


def _libcint_values(gto, angular, atoms, exps, radial, xyz):
    mol = gto.M(
        atom=[("ghost-H", xyz[atoms[0]]), ("ghost-He", xyz[atoms[1]])],
        basis={"H": [[angular[0], [exps[0], 1.0]]], "He": [[angular[1], [exps[1], 1.0]]]},
        unit="Bohr", cart=True, verbose=0,
    )
    s = mol.intor("int1e_ovlp_cart")
    t = mol.intor("int1e_kin_cart")
    v = np.zeros_like(s)
    for q, position in zip(_CHARGES, xyz, strict=True):
        with mol.with_rinv_origin(position):
            v -= q * mol.intor("int1e_rinv_cart")
    first, second = (
        _cartesian_norms(l, a) for l, a in zip(angular, exps, strict=True)
    )
    cut = len(first)
    scales = np.sqrt(s.diagonal() / np.concatenate((first, second)))
    # Undo Libcint's normalized primitive factors and use the explicit radial
    # coefficients supplied to the cohort ABI.
    scale = np.outer(radial[0] / scales[:cut], radial[1] / scales[cut:])
    return np.array([item[:cut, cut:] * scale for item in (s, t, v)])


def _pair_binding(lib, cohort):
    return _bind(
        lib, cohort, "",
        [ct.c_int] * 7 + [_DOUBLE, _DOUBLE, _DOUBLE, _INT, _DOUBLE, _SIZE, _SIZE, _SIZE_REF],
    )


def test_i_cohort_and_legacy_h_domain(libraries):
    for cohort, other, angular in (("h", "i", 5), ("i", "h", 6)):
        lib = libraries[cohort]
        with pytest.raises(AttributeError):
            _pair_binding(lib, other)
        function = _pair_binding(lib, cohort)
        inputs = (0, 1, 1, 1, 3, np.ones(2), np.ones(2), _XYZ, _CHARGES)
        out = np.full(3 * 10 * (angular + 1) * (angular + 2) // 2, np.nan)
        used = _SIZE()
        args = (*inputs, out, out.size, 1 << 20, ct.byref(used))
        assert function(angular - 1, 0, *args) == 1
        assert function(angular + 1, 0, *args) == 1
        assert np.isnan(out).all()
        assert function(angular, 0, *args) == 0
        assert np.isfinite(out).all()


@pytest.mark.parametrize("angular", [(6, 6), (6, 5), (5, 6), (0, 6)])
def test_i_pair_cartesian_values_and_nuclear_response(libraries, angular):
    gto = pytest.importorskip("pyscf.gto")
    function = _pair_binding(libraries["i"], "i")
    atoms, exps, radial = (0, 1), np.array([0.73, 0.91]), np.array([0.52, 0.47])

    def values(xyz):
        return _libcint_values(gto, angular, atoms, exps, radial, xyz)

    expected = values(_XYZ)
    native = np.full((3, 10, *expected.shape[1:]), np.nan)
    used = _SIZE()
    args = (*angular, *atoms, 1, 1, 3, exps, radial, _XYZ, _CHARGES, native, native.size)
    assert function(*args, 1, ct.byref(used)) == 2 and np.isnan(native).all()
    assert function(*args, 1 << 20, ct.byref(used)) == 0
    assert np.isfinite(native).all()
    np.testing.assert_allclose(native[:, 0], expected, rtol=3e-12, atol=2e-10)
    for step in (2e-4, 1e-4):
        def derivatives(nuclear):
            def shifted(scale):
                xyz = _XYZ.copy()
                xyz.ravel()[nuclear] += scale * step
                return values(xyz)
            return (8 * (shifted(1) - shifted(-1)) - (shifted(2) - shifted(-2))) / (12 * step)

        reference = np.stack([derivatives(i) for i in range(9)], axis=1)
        np.testing.assert_allclose(native[:, 1:], reference, rtol=1e-8, atol=3e-8)
    np.testing.assert_allclose(
        native[:, 1:].reshape(3, 3, 3, *expected.shape[1:]).sum(axis=1), 0,
        atol=1e-10, rtol=0,
    )


def test_i_batch_compact_projection_and_nonsymmetric_weights(libraries):
    gto = pytest.importorskip("pyscf.gto")
    lib = libraries["i"]
    batch = _bind(
        lib, "i", "_batch",
        [ct.c_int] * 4 + [_INT, _DOUBLE, _DOUBLE, _DOUBLE, _INT, _DOUBLE, _DOUBLE,
                          _SIZE, _SIZE, _SIZE_REF],
    )
    contract = _bind(
        lib, "i", "_weights", [ct.c_int, _SIZE, _DOUBLE, _DOUBLE, _DOUBLE, _SIZE, _SIZE_REF]
    )
    ang, atoms = (0, 5, 6), (0, 1, 0)
    exps, radial = np.array([0.8, 0.9, 1.0]), np.array([0.51, 0.6, 0.63])
    transform = [gto.cart2sph(l, normalized="sp").T for l in range(7)]
    transform_table = np.concatenate([table.ravel() for table in transform])
    shells = np.array([(l, atom, k, k + 1) for k, (l, atom) in enumerate(zip(ang, atoms, strict=True))],
                      dtype=np.int32)
    output = np.empty((10, 2, sum(13 * (2 * m + 1) for m in ang)))
    used = _SIZE()
    args = (3, 3, 3, shells, exps, radial, _XYZ, _CHARGES, transform_table)
    assert batch(1, *args, output, output.size, 1 << 20, ct.byref(used)) == 0
    def reference(xyz):
        blocks = []
        for k in range(3):
            pair = [2, k]
            s, t, v = _libcint_values(
                gto, (ang[2], ang[k]), (atoms[2], atoms[k]),
                exps[pair], radial[pair], xyz,
            )
            projected = transform[6] @ np.stack((s, t + v)) @ transform[ang[k]].T
            blocks.append(projected.reshape(2, -1))
        return np.concatenate(blocks, axis=1)

    np.testing.assert_allclose(output[0], reference(_XYZ), rtol=3e-12, atol=2e-10)
    # Values-only mode must return the same compact source independently of its Jacobian.
    value = np.empty_like(output[:1])
    assert batch(0, *args, value, value.size, 1 << 20, ct.byref(used)) == 0
    np.testing.assert_allclose(value[0], output[0], rtol=1e-13, atol=1e-12)
    weights = np.random.default_rng(71).normal(size=output[0].shape)
    gradient = np.empty(9)
    assert contract(
        9, weights.size, output[1:], weights, gradient, 1 << 20, ct.byref(used)
    ) == 0
    np.testing.assert_allclose(
        gradient, np.einsum("qk,xqk->x", weights, output[1:]),
        rtol=1e-12, atol=1e-10,
    )
    # Both the s and i functions move with the first nucleus.
    step = 1e-4
    direction = np.zeros_like(_XYZ)
    direction[0, 0] = step
    derivative = (
        8 * (reference(_XYZ + direction) - reference(_XYZ - direction))
        - (reference(_XYZ + 2 * direction) - reference(_XYZ - 2 * direction))
    ) / (12 * step)
    np.testing.assert_allclose(gradient[0], np.vdot(weights, derivative), rtol=0, atol=3e-8)
