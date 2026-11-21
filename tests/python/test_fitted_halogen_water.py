"""Host-side qualification for the scalar small-core Br/I and water boundary."""

import ctypes
from dataclasses import replace

import numpy as np
import pytest
from generativeqc import Atom, BasisShell, Calculator, _native
from generativeqc.ecp import ecp_integrals, fitted_hf_ecp_domain, resolve_ecp
from generativeqc.experimental.f12_focal import FocalBasis
from generativeqc.experimental.f12_methyl_halide import methyl_halide_basis
from test_ecp_heavy import reference_components


def water_halide(symbol: str) -> tuple[Atom, ...]:
    # Fixed TA13_3 geometry (Br and water), converted from Å without recentering.
    xyz = (
        np.array(
            [
                [0, 0, 0],
                [0, 0, 2.7049230935],
                [0.9514568132, 0, 2.8405557142],
                [-0.2649203354, 0.9138311022, 2.8405557142],
            ]
        )
        / 0.529177210544
    )
    return tuple(
        Atom.from_value((element, position))
        for element, position in zip((symbol, "O", "H", "H"), xyz, strict=True)
    )


@pytest.mark.parametrize("symbol", ["Br", "I"])
def test_water_halide_fitted_hf_preflight_and_native_g_shell(symbol: str) -> None:
    atoms = water_halide(symbol)
    parent = methyl_halide_basis(symbol, 2)
    orbitals = {symbol: parent.orbital, "O": "sto-3g", "H": "sto-3g"}
    fits = {symbol: parent.jk_fit, "O": "sto-3g", "H": "sto-3g"}
    basis = FocalBasis(2, "water-halide", orbitals, jk_fit=fits)
    # One extra orbital g shell exercises the native expanded-basis admission.
    z = atoms[0].atomic_number
    heavy = basis.orbital.by_element[z]
    expanded = replace(
        heavy, shells=(*heavy.shells, BasisShell(4, ("0.2",), (("1",),)))
    )
    orbital = replace(
        basis.orbital,
        elements=tuple(
            expanded if e.atomic_number == z else e for e in basis.orbital.elements
        ),
    )
    cores, terms = resolve_ecp(orbital, atoms)
    assert cores == ((10 if symbol == "Br" else 28), 0, 0, 0)
    assert {term[0] for term in terms} == {0}
    assert fitted_hf_ecp_domain("uhf", orbital, atoms, 0, 2)
    assert fitted_hf_ecp_domain("rhf", orbital, atoms, 1, 1)
    assert not fitted_hf_ecp_domain("rhf", orbital, atoms, 0, 1)
    assert not fitted_hf_ecp_domain("uhf", orbital, atoms, 0, 4)
    assert not fitted_hf_ecp_domain("uhf", orbital, atoms, -2, 2)
    assert not fitted_hf_ecp_domain("uhf", orbital, atoms[:-1], 0, 2)

    calc = Calculator(
        method="uhf",
        basis=orbital,
        auxiliary_basis=basis.jk_fit,
        density_fitting="cuda",
        device="cuda",
        initial_guess=None,
    )
    assert calc.basis_capability_report(atoms, compute_forces=True)["eligible"]
    cpu = Calculator(basis=orbital, device="cpu")
    lib = cpu._library
    context = ctypes.c_void_p()
    _native.check(
        lib,
        lib.generativeqc_context_create(
            ctypes.byref(cpu._context_descriptor()), ctypes.byref(context)
        ),
    )
    try:
        for record in (None, basis.jk_fit):
            system = calc._create_native_system(context, atoms, 0, 2, record)
            lib.generativeqc_system_destroy(system)
        with pytest.raises(NotImplementedError, match="ECP density-fitting"):
            calc._create_native_system(context, atoms, 0, 4)
    finally:
        lib.generativeqc_context_destroy(context)


def test_ta13_bromine_and_off_center_oxygen_ecp_ao_derivatives() -> None:
    gto = pytest.importorskip("pyscf.gto")
    atoms = water_halide("Br")
    methyl = methyl_halide_basis("Br", 2)
    basis = FocalBasis(
        2,
        "water-halide",
        {"Br": methyl.orbital, "O": "sto-3g", "H": "sto-3g"},
        jk_fit={"Br": methyl.jk_fit, "O": "sto-3g", "H": "sto-3g"},
    ).orbital
    labels = ("Br", "O", "H", "H")
    orb = {
        label: [
            [
                shell.angular_momentum,
                *[
                    list(map(float, row))
                    for row in zip(shell.exponents, *shell.coefficients, strict=True)
                ],
            ]
            for shell in basis.by_element[z].shells
        ]
        for label, z in (("Br", 35), ("O", 8), ("H", 1))
    }
    xyz = np.asarray([a.position for a in atoms])
    mol = gto.M(
        atom=list(zip(labels, xyz, strict=True)),
        basis=orb,
        ecp={"Br": gto.basis.load_ecp("cc-pvdz-pp", "Br")},
        unit="Bohr",
        spin=1,
        verbose=0,
    )
    assert mol.nelec == (18, 17)
    raw = ecp_integrals(
        atoms,
        basis,
        multiplicity=2,
        radial_points=224,
        polar_points=44,
    )
    np.testing.assert_allclose(
        [raw.local, raw.nonlocal_], reference_components(mol), atol=2e-8, rtol=0
    )
    derivatives = np.stack([raw.local_derivative, raw.nonlocal_derivative])
    np.testing.assert_allclose(derivatives.sum(axis=1), 0, atol=2e-10, rtol=0)
    # Move the oxygen and the Br projector independently using the external oracle.
    for atom, axis in ((0, 2), (1, 2), (2, 0)):
        for step in (0.0004, 0.0002):
            shift = np.zeros_like(xyz)
            shift[atom, axis] = step
            minus, plus = (
                reference_components(
                    mol.copy().set_geom_(xyz + direction * shift, unit="Bohr")
                )
                for direction in (-1, 1)
            )
            np.testing.assert_allclose(
                derivatives[:, atom, axis],
                (plus - minus) / (2 * step),
                atol=2e-6,
                rtol=0,
            )
