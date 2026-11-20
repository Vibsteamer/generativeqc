"""Scalar ECP preflight, independent of native libraries and reference packages."""

import json
import typing
from dataclasses import replace

import pytest
from generativeqc import Atom, BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.basis_capabilities import basis_capability
from generativeqc.ecp import resolve_ecp


def potential(**changes: typing.Any) -> typing.Any:
    return {
        "ecp_type": "scalar_ecp",
        "angular_momentum": [0],
        "r_exponents": [2],
        "gaussian_exponents": ["0.8"],
        "coefficients": [["-2"]],
    } | changes


def element(records: typing.Any) -> typing.Any:
    return ElementBasis(
        11,
        (BasisShell(0, ("0.7",), (("1",),)),),
        ecp_core_electrons=10,
        ecp_data=json.dumps(records),
    )


def resolve(record: typing.Any) -> typing.Any:
    basis = BasisSet(
        "synthetic scalar ECP",
        (record,),
        BasisProvenance("synthetic test", "1", "CC0", "0" * 64),
    )
    return resolve_ecp(basis, (Atom(11, (0.0, 0.0, 0.0)),))


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
@pytest.mark.parametrize("order", (0, 1, 2, 3))
def test_ecp_ao_spatial_jets_have_backend_specific_capabilities(
    backend: typing.Any, order: typing.Any
) -> None:
    basis = BasisSet(
        "synthetic scalar ECP",
        (element([potential()]),),
        BasisProvenance("synthetic test", "1", "CC0", "0" * 64),
    )
    report = basis_capability(
        basis,
        [("Na", (0, 0, 0))],
        backend=backend,
        operator="ao",
        derivative_order=order,
    )
    assert report["eligible"] == (backend == "cpu" or order <= 1)


def test_ecp_ao_preflight_does_not_strip_unsupported_potential_metadata() -> None:
    # Local label g is valid with f projectors; h remains outside the contract.
    basis = BasisSet(
        "unsupported scalar ECP",
        (element([potential(angular_momentum=[5])]),),
        BasisProvenance("synthetic test", "1", "CC0", "0" * 64),
    )
    assert not basis_capability(basis, [("Na", (0, 0, 0))], operator="ao")["eligible"]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"ecp_type": "spin_orbit_ecp"}, "scalar_ecp"),
        ({"ecp_type": "unknown"}, "scalar_ecp"),
        ({"ecp_type": None}, "scalar_ecp"),
        ({"angular_momentum": [0, 1]}, "one angular channel"),
        ({"angular_momentum": []}, "one angular channel"),
        ({"angular_momentum": 0}, "one angular channel"),
    ],
)
def test_scalar_contract_at_construction_and_resolution(
    changes: typing.Any, message: typing.Any
) -> None:
    malformed = [potential(**changes)]
    with pytest.raises(ValueError, match=message):
        element(malformed)

    # Simulate a stale/corrupted owned record to exercise the resolver's own
    # boundary independently of the normal immutable constructor's validation.
    record = element([potential()])
    object.__setattr__(record, "ecp_data", json.dumps(malformed))
    with pytest.raises(ValueError, match=message):
        resolve(record)


def test_highest_singleton_channel_is_local() -> None:
    cores, terms = resolve(element([potential(angular_momentum=[1]), potential()]))
    assert cores == (10,)
    assert terms == ((0, -1, 2, 0.8, -2.0), (0, 0, 2, 0.8, -2.0))


def test_g_local_label_enables_f_projector_without_g_orbitals() -> None:
    cores, terms = resolve(
        element([potential(angular_momentum=[4]), potential(angular_momentum=[3])])
    )
    assert cores == (10,)
    assert terms == ((0, -1, 2, 0.8, -2.0), (0, 3, 2, 0.8, -2.0))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"angular_momentum": [5]}, "local channel"),
        ({"spin_orbit": True}, "unknown ECP parameter fields"),
    ],
)
def test_unsupported_execution_conventions_remain_rejected(
    changes: typing.Any, message: typing.Any
) -> None:
    with pytest.raises(NotImplementedError, match=message):
        resolve(element([potential(**changes)]))


def test_f_orbitals_are_resolved_but_g_remains_unsupported() -> None:
    record = element([potential()])
    for angular in (3, 4):
        changed = replace(record, shells=(BasisShell(angular, ("0.7",), (("1",),)),))
        if angular == 3:
            assert resolve(changed)[0] == (10,)
        else:
            with pytest.raises(NotImplementedError, match="orbital s/p/d/f"):
                resolve(changed)


def test_mixed_all_electron_atom_obeys_the_same_orbital_boundary() -> None:
    for angular in (3, 4):
        hydrogen = ElementBasis(1, (BasisShell(angular, ("0.7",), (("1",),)),))
        basis = BasisSet(
            "mixed scalar ECP",
            (element([potential()]), hydrogen),
            BasisProvenance("synthetic test", "1", "CC0", "0" * 64),
        )
        atoms = (Atom(11, (0.0, 0.0, 0.0)), Atom(1, (0.3, 0.1, 3.0)))
        if angular == 3:
            assert resolve_ecp(basis, atoms)[0] == (10, 0)
        else:
            with pytest.raises(NotImplementedError, match="orbital s/p/d/f"):
                resolve_ecp(basis, atoms)


@pytest.mark.parametrize(("z", "core_electrons"), ((35, 10), (53, 28)))
def test_g_shell_df_uhf_halogen_dihydrides(z: int, core_electrons: int) -> None:
    from generativeqc.ecp import fitted_uhf_hydride_domain

    h = ElementBasis(1, (BasisShell(0, ("1",), (("1",),)),))
    x = ElementBasis(
        z,
        (BasisShell(4, ("0.7",), (("1",),)),),
        ecp_core_electrons=core_electrons,
        ecp_data=json.dumps([potential()]),
    )
    basis = BasisSet(
        "spherical g-shell ECP admission test",
        (h, x),
        BasisProvenance("synthetic", "1", "CC0", "0" * 64),
        representation="spherical",
    )
    atoms = (
        Atom(1, (0.0, 0.2, -1.0)),
        Atom(z, (0.0, 0.0, 0.0)),
        Atom(1, (0.0, 0.3, 1.0)),
    )
    assert fitted_uhf_hydride_domain(basis, atoms, 0, 2)
    assert resolve_ecp(basis, atoms)[0] == (0, core_electrons, 0)
    assert not fitted_uhf_hydride_domain(basis, atoms, 1, 2)
    assert not fitted_uhf_hydride_domain(basis, atoms, 0, 1)
    assert not fitted_uhf_hydride_domain(basis, atoms[:2], 0, 2)
    assert not fitted_uhf_hydride_domain(
        replace(basis, representation="cartesian"), atoms, 0, 2
    )
    angular_h = replace(x, shells=(BasisShell(5, ("0.7",), (("1",),)),))
    assert not fitted_uhf_hydride_domain(
        replace(basis, elements=(h, angular_h)), atoms, 0, 2
    )


@pytest.mark.parametrize(("z", "core_electrons"), ((35, 10), (53, 28)))
def test_restricted_fitted_scalar_hydride_domain(z, core_electrons):
    from generativeqc.ecp import fitted_hf_hydride_domain

    hydrogen = ElementBasis(1, (BasisShell(0, ("1",), (("1",),)),))
    halogen = ElementBasis(
        z,
        (BasisShell(4, ("0.7",), (("1",),)),),
        ecp_core_electrons=core_electrons,
        ecp_data=json.dumps([potential()]),
    )
    orbital = BasisSet(
        "spherical hydrogen halide",
        (hydrogen, halogen),
        BasisProvenance("synthetic", "1", "CC0", "0" * 64),
        representation="spherical",
    )
    atoms = (Atom(1, (0.0, 0.0, 1.0)), Atom(z, (0.0, 0.0, 0.0)))
    assert fitted_hf_hydride_domain("rhf", orbital, atoms, 0, 1)
    assert fitted_hf_hydride_domain("rhf", orbital, atoms[::-1], 0, 1)
    assert resolve_ecp(orbital, atoms)[0] == (0, core_electrons)
    for method, charge, spin in (("uhf", 0, 1), ("rhf", 1, 1), ("rhf", 0, 3)):
        assert not fitted_hf_hydride_domain(method, orbital, atoms, charge, spin)
    assert not fitted_hf_hydride_domain("rhf", orbital, atoms + atoms[:1], 0, 1)
    assert not fitted_hf_hydride_domain(
        "rhf", replace(orbital, representation="cartesian"), atoms, 0, 1
    )
    for invalid in (
        replace(halogen, ecp_core_electrons=core_electrons - 2),
        replace(halogen, shells=(BasisShell(5, ("0.7",), (("1",),)),)),
    ):
        assert not fitted_hf_hydride_domain(
            "rhf", replace(orbital, elements=(hydrogen, invalid)), atoms, 0, 1
        )
