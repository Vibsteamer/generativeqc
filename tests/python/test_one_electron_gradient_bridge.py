"""Check fixed-weight native contraction ownership and bounded input admission."""

import ctypes as ct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator, _native
from generativeqc._api_types import Primitive, Shell
from generativeqc._one_electron_gradient import cuda_one_electron_gradient


class Function:
    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation

    def __call__(self, *args: Any) -> Any:
        return self.implementation(*args)


def calculator(*, failure: Any = False, creation_failure: Any = False) -> Any:
    live = set()

    def create(descriptor: Any, output: Any) -> int:
        ct.cast(output, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(51)
        live.add(51)
        return 0

    def destroy(handle: Any) -> None:
        live.remove(handle.value)

    def system(
        context: Any, atoms: Any, charge: Any, multiplicity: Any, **domain: Any
    ) -> Any:
        assert context.value in live
        if creation_failure:
            raise RuntimeError("injected system creation failure")
        live.add(61)
        return ct.c_void_p(61)

    def gradient(
        context: Any,
        system: Any,
        overlap: Any,
        kinetic: Any,
        attraction: Any,
        size: Any,
        schedule: Any,
        budget: Any,
        output: Any,
        count: Any,
        resources: Any,
    ) -> Any:
        assert context.value in live and system.value in live
        assert size == 4 and count == 3
        if failure:
            return _native.STATUS_OUT_OF_MEMORY
        assert not attraction
        np.testing.assert_array_equal(
            np.ctypeslib.as_array(overlap, (4,)), [0, 1, 2, 3]
        )
        np.testing.assert_array_equal(
            np.ctypeslib.as_array(kinetic, (4,)), [0, 2, 1, 3]
        )
        np.ctypeslib.as_array(output, (count,))[:] = [1, 2, 3]
        return 0

    lib = SimpleNamespace(
        generativeqc_context_create=Function(create),
        generativeqc_context_destroy=Function(destroy),
        generativeqc_system_destroy=Function(destroy),
        generativeqc_system_one_electron_gradient_cuda=Function(gradient),
        generativeqc_status_message=Function(lambda status: b"native capacity limit"),
        generativeqc_context_get_last_detail=Function(
            lambda context: b"native capacity limit"
        ),
    )
    shells = (Shell(0, 0, (Primitive(1.0, 1.0),)),) * 2
    calc = SimpleNamespace(
        _library=lib,
        _backend=_native.BACKEND_CUDA,
        _basis_representation=_native.BASIS_SPHERICAL,
        _shells_for_atoms=lambda *args, **kwargs: shells,
        _create_native_system=system,
        _context_descriptor=lambda: _native.ContextDescriptor(
            ct.sizeof(_native.ContextDescriptor),
            _native.ABI_VERSION,
            0,
            _native.BACKEND_CUDA,
        ),
    )
    return (calc, live)


@pytest.mark.parametrize("representation,nao", [("spherical", 9), ("cartesian", 15)])
def test_standalone_g_shell_reaches_the_contraction(
    representation: str, nao: int
) -> None:
    # Exercise the real Python preflight, shell expansion and descriptor path.
    # Only the CUDA ABI boundary is replaced on this host-only test.
    stub, live = calculator()
    calc = Calculator.__new__(Calculator)
    calc._method = _native.METHOD_RHF
    calc._device_name = "cuda"
    calc._device_id = 0
    calc._backend = _native.BACKEND_CUDA
    calc._density_fitting_mode = _native.DENSITY_FITTING_NONE
    calc._auxiliary_basis = None
    calc._dispersion_method_ir = None
    calc._representation_name = representation
    calc._basis_representation = (
        _native.BASIS_SPHERICAL
        if representation == "spherical"
        else _native.BASIS_CARTESIAN
    )
    calc._basis = (Shell(0, 4, (Primitive(0.9, 1.0),)),)
    calc._library = stub._library
    atoms = (Atom.from_value(("He", (0.0, 0.0, 0.0))),)

    def create(context: Any, descriptor: Any, out: Any) -> int:
        d = ct.cast(descriptor, ct.POINTER(_native.SystemDescriptor)).contents
        assert d.shell_count == 1 and d.shells[0].angular_momentum == 4
        assert d.basis_representation == calc._basis_representation
        ct.cast(out, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(61)
        live.add(61)
        return 0

    def contract(*args: Any) -> int:
        assert args[1].value in live and args[5] == nao * nao
        np.testing.assert_array_equal(
            np.ctypeslib.as_array(args[2], (nao * nao,)), np.eye(nao).ravel()
        )
        np.ctypeslib.as_array(args[8], (3,))[:] = 0.0
        return 0

    calc._library.generativeqc_system_create = Function(create)
    calc._library.generativeqc_system_one_electron_gradient_cuda = Function(contract)
    np.testing.assert_array_equal(
        cuda_one_electron_gradient(calc, atoms, overlap=np.eye(nao)).gradient,
        np.zeros((1, 3)),
    )
    with pytest.raises(NotImplementedError, match="cuda/eri"):
        calc._create_native_system(ct.c_void_p(), atoms, 0, 1)
    assert not live


@pytest.mark.parametrize(
    "failure,creation_failure", [(False, False), (True, False), (False, True)]
)
def test_transaction_success_and_native_failure_cleanup(
    failure: Any, creation_failure: Any
) -> None:
    (calc, live) = calculator(failure=failure, creation_failure=creation_failure)
    w = np.arange(4.0).reshape(2, 2)
    before = w.copy()
    args = {"overlap": w, "kinetic": w.T, "maximum_bytes": 1024}
    if failure or creation_failure:
        with pytest.raises(RuntimeError):
            cuda_one_electron_gradient(calc, [("H", (0, 0, 0))], **args)
    else:
        result = cuda_one_electron_gradient(calc, [("H", (0, 0, 0))], **args)
        np.testing.assert_array_equal(result.gradient, [[1, 2, 3]])
    assert not live
    np.testing.assert_array_equal(w, before)


@pytest.mark.parametrize(
    "matrix", [np.eye(2, dtype=np.float32), np.eye(3), np.full((2, 2), np.nan)]
)
def test_reject_invalid_weights_before_native_ownership(matrix: Any) -> None:
    (calc, live) = calculator()
    with pytest.raises(ValueError):
        cuda_one_electron_gradient(calc, [("H", (0, 0, 0))], overlap=matrix)
    assert not live


def test_requires_cuda_and_bounded_staging() -> None:
    (calc, live) = calculator()
    with pytest.raises(MemoryError):
        cuda_one_electron_gradient(
            calc, [("H", (0, 0, 0))], overlap=np.eye(2), maximum_bytes=16
        )
    calc._backend = _native.BACKEND_CPU_REFERENCE
    with pytest.raises(ValueError, match="CUDA"):
        cuda_one_electron_gradient(calc, [("H", (0, 0, 0))], overlap=np.eye(2))
    assert not live


def test_size_t_budget_is_not_restricted_to_int32() -> None:
    calc, live = calculator()
    w = np.arange(4.0).reshape(2, 2)
    result = cuda_one_electron_gradient(
        calc, [("H", (0, 0, 0))], overlap=w, kinetic=w.T, maximum_bytes=43 << 30
    )
    np.testing.assert_array_equal(result.gradient, [[1, 2, 3]])
    assert not live


@pytest.mark.parametrize("indices", [None, np.array([1, 0])])
def test_hcore_uses_one_contiguous_matrix_for_two_native_channels(indices: Any) -> None:
    calc, live = calculator()
    # A transpose requires a host copy. Both native channels must see that same copy.
    hcore = np.arange(4.0).reshape(2, 2).T

    def inspect(*args: Any) -> int:
        _, _, overlap, kinetic, attraction, *_ = args
        assert not overlap
        assert ct.addressof(kinetic.contents) == ct.addressof(attraction.contents)
        expected = hcore if indices is None else hcore[np.ix_(indices, indices)]
        np.testing.assert_array_equal(
            np.ctypeslib.as_array(kinetic, (4,)), expected.ravel()
        )
        np.ctypeslib.as_array(args[8], (3,))[:] = [0, 1, 2]
        return 0

    calc._library.generativeqc_system_one_electron_gradient_cuda = Function(inspect)
    result = cuda_one_electron_gradient(
        calc,
        [("H", (0, 0, 0))],
        hcore=hcore,
        native_to_source=indices,
        maximum_bytes=hcore.nbytes,
    )
    np.testing.assert_array_equal(result.gradient, [[0, 1, 2]])
    assert not live


@pytest.mark.parametrize("indices", [[0, 0], [-1, 1], [1, 2], [0], [0.0, 1.0]])
def test_reject_invalid_source_map_before_native_ownership(indices: list) -> None:
    calc, live = calculator()
    with pytest.raises(ValueError, match="bijective"):
        cuda_one_electron_gradient(
            calc,
            [("H", (0, 0, 0))],
            hcore=np.eye(2),
            native_to_source=np.asarray(indices),
        )
    assert not live


@pytest.mark.parametrize("separate", ["kinetic", "attraction"])
def test_hcore_rejects_duplicate_operator_weights(separate: str) -> None:
    calc, live = calculator()
    with pytest.raises(ValueError, match="hcore"):
        cuda_one_electron_gradient(
            calc, [("H", (0, 0, 0))], hcore=np.eye(2), **{separate: np.eye(2)}
        )
    assert not live
