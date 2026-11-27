"""The public ECP ABI admits orbital g shells exactly on the Python fitted-state domain."""

import ctypes
import hashlib
import itertools
import json
import typing

import pytest
from generativeqc import (
    Atom,
    BasisProvenance,
    BasisSet,
    BasisShell,
    Calculator,
    ElementBasis,
    _native,
)
from generativeqc.ecp import _fitted_ecp_orbital_angular_limit

LOCAL_POTENTIAL = json.dumps(
    [
        {
            "ecp_type": "scalar_ecp",
            "angular_momentum": [0],
            "r_exponents": [2],
            "gaussian_exponents": ["0.8"],
            "coefficients": [["-2.0"]],
        }
    ]
)

Topology = tuple[tuple[int, int], ...]
Native = tuple[typing.Any, ctypes.c_void_p]

# Atoms as (atomic number, ECP core electrons).
TOPOLOGIES: dict[str, Topology] = {
    "HBr": ((1, 0), (35, 10)),
    "Br": ((35, 10),),
    "BrH2": ((35, 10), (1, 0), (1, 0)),
    "BrH3": ((35, 10),) + ((1, 0),) * 3,
    "BrH4": ((35, 10),) + ((1, 0),) * 4,
    "HI": ((1, 0), (53, 28)),
    "CH3I": ((6, 0), (1, 0), (1, 0), (1, 0), (53, 28)),
    "H2O-Br": ((8, 0), (1, 0), (1, 0), (35, 10)),
    "C4H6Br2O": ((6, 0),) * 4 + ((1, 0),) * 6 + ((35, 10),) * 2 + ((8, 0),),
    "Br2": ((35, 10), (35, 10)),
    "IBr": ((53, 28), (35, 10)),
    "CH2BrCl": ((6, 0), (1, 0), (1, 0), (35, 10), (17, 0)),
    "NaI": ((11, 0), (53, 28)),
    "NaI-Na-ECP": ((11, 10), (53, 28)),
    "KI": ((19, 0), (53, 28)),
    "all-electron-Br-I": ((35, 0), (53, 28)),
    "large-core-I": ((1, 0), (53, 46)),
    "Na-ECP-hydride": ((11, 10), (1, 0)),
}

# (topology, charge, multiplicity) -> orbital g admitted, independent of either implementation.
ANCHORS = {
    ("HBr", 0, 1): True,
    ("HBr", 1, 1): False,
    ("CH3I", 2, 1): False,
    ("CH3I", 1, 2): True,
    ("C4H6Br2O", -1, 2): True,
    ("C4H6Br2O", -2, 1): False,
    ("NaI", 0, 1): True,
    ("NaI", 0, 3): False,
    ("NaI-Na-ECP", 0, 1): False,
    ("BrH4", 0, 2): False,
}


def python_basis(atoms: Topology, representation: str) -> BasisSet:
    elements = []
    for z, core in sorted(set(atoms)):
        shells = [BasisShell(0, ("0.9",), (("1.0",),))]
        if core:
            shells.append(BasisShell(4, ("0.6",), (("1.0",),)))
        elements.append(
            ElementBasis(
                z,
                tuple(shells),
                ecp_core_electrons=core,
                ecp_data=LOCAL_POTENTIAL if core else None,
            )
        )
    checksum = hashlib.sha256(repr(sorted(set(atoms))).encode()).hexdigest()
    provenance = BasisProvenance(
        "synthetic admission probe", "1", "test-only", checksum
    )
    return BasisSet(
        "fitted-g admission probe", tuple(elements), provenance, representation
    )


def native_status(
    library: typing.Any,
    context: ctypes.c_void_p,
    atoms: Topology,
    charge: int,
    multiplicity: int,
    representation: str,
    orbital: int,
) -> tuple[int, str]:
    n = len(atoms)
    atom_array = (_native.AtomDescriptor * n)(
        *(
            _native.AtomDescriptor(z, 0.1 * (a % 3), -0.2 * (a % 2), 3.1 * a)
            for a, (z, _) in enumerate(atoms)
        )
    )
    shells, primitives = [], []
    for a, (_, core) in enumerate(atoms):
        for angular, exponent in ((0, 0.9), (orbital, 0.6)) if core else ((0, 0.9),):
            shells.append(_native.ShellDescriptor(a, angular, len(primitives), 1))
            primitives.append(_native.PrimitiveDescriptor(exponent, 1.0))
    ecp_atoms = [a for a, (_, core) in enumerate(atoms) if core]
    shell_array = (_native.ShellDescriptor * len(shells))(*shells)
    primitive_array = (_native.PrimitiveDescriptor * len(primitives))(*primitives)
    descriptor = _native.SystemDescriptor(
        ctypes.sizeof(_native.SystemDescriptor),
        _native.ABI_VERSION,
        atom_array,
        n,
        shell_array,
        len(shells),
        primitive_array,
        len(primitives),
        charge,
        multiplicity,
        _native.BASIS_SPHERICAL
        if representation == "spherical"
        else _native.BASIS_CARTESIAN,
    )
    cores = (ctypes.c_int32 * n)(*(core for _, core in atoms))
    terms = (_native.EcpTermDescriptor * len(ecp_atoms))(
        *(_native.EcpTermDescriptor(a, -1, 2, 0.8, -2.0) for a in ecp_atoms)
    )
    system = ctypes.c_void_p()
    status = library.generativeqc_system_create_ecp(
        context,
        ctypes.byref(descriptor),
        cores,
        terms,
        len(ecp_atoms),
        ctypes.byref(system),
    )
    if system:
        library.generativeqc_system_destroy(system)
    if status == _native.STATUS_SUCCESS:
        return status, ""
    return status, (
        library.generativeqc_context_get_last_detail(context) or b""
    ).decode()


@pytest.fixture(scope="module")
def native() -> typing.Iterator[Native]:
    calculator = Calculator(device="cpu")
    library = calculator._library
    context = ctypes.c_void_p()
    _native.check(
        library,
        library.generativeqc_context_create(
            ctypes.byref(calculator._context_descriptor()), ctypes.byref(context)
        ),
    )
    yield library, context
    library.generativeqc_context_destroy(context)


def test_native_orbital_g_admission_matches_python_fitted_domain(
    native: Native,
) -> None:
    library, context = native
    decisions = {}
    states = itertools.product(TOPOLOGIES.items(), range(-2, 3), range(1, 5))
    for (label, atoms), charge, multiplicity in states:
        for representation in ("spherical", "cartesian"):
            if representation == "cartesian" and (charge, multiplicity) != (0, 1):
                continue
            base, _ = native_status(
                library, context, atoms, charge, multiplicity, representation, 3
            )
            status, detail = native_status(
                library, context, atoms, charge, multiplicity, representation, 4
            )
            if base != _native.STATUS_SUCCESS:
                assert status == base, (label, charge, multiplicity, detail)
                continue
            molecule = [Atom(z, (0.0, 0.0, 3.1 * a)) for a, (z, _) in enumerate(atoms)]
            limit = _fitted_ecp_orbital_angular_limit(
                python_basis(atoms, representation), molecule, charge, multiplicity
            )
            expected = representation == "spherical" and limit >= 4
            assert (status == _native.STATUS_SUCCESS) == expected, (
                label,
                charge,
                multiplicity,
                representation,
                limit,
                status,
                detail,
            )
            if not expected:
                assert status == _native.STATUS_INVALID_ARGUMENT
                assert "supports orbital s/p/d/f shells" in detail
            if representation == "spherical":
                decisions[label, charge, multiplicity] = expected
    for state, admitted in ANCHORS.items():
        assert decisions[state] is admitted, state
    assert {label for label, _, _ in decisions} == set(TOPOLOGIES)
    assert 0 < sum(decisions.values()) < len(decisions)
