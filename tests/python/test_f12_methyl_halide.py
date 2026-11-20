import pytest
from generativeqc.experimental.f12_methyl_halide import (
    methyl_halide_basis,
    methyl_halide_plan,
)


@pytest.mark.parametrize(
    "element,cardinal,nobs,ncabs,nri,njk,core",
    [
        ("Br", 2, 115, 352, 422, 395, 10),
        ("Br", 3, 202, 437, 572, 395, 10),
        ("I", 2, 115, 327, 420, 347, 28),
        ("I", 3, 205, 446, 559, 347, 28),
    ],
)
def test_actual_methyl_halide_fits(
    element: str, cardinal: int, nobs: int, ncabs: int, nri: int, njk: int, core: int
) -> None:
    z = 35 if element == "Br" else 53
    basis = methyl_halide_basis(element, cardinal)
    assert basis.orbital.by_element[z].ecp_core_electrons == core
    assert basis.mp2_fit.identity == basis.f12_fit.identity
    assert basis.jk_fit.identity != basis.f12_fit.identity
    for record, expected in zip(
        (basis.orbital, basis.cabs, basis.mp2_fit, basis.jk_fit),
        (nobs, ncabs, nri, njk),
        strict=True,
    ):
        assert set(record.by_element) == {1, 6, z}
        count = sum(
            (2 * s.angular_momentum + 1) * len(s.coefficients)
            for atom in (6, z, 1, 1, 1)
            for s in record.by_element[atom].shells
        )
        assert count == expected
    for root, charge, mult in [("rhf", 0, 1), ("uhf", 1, 2)]:
        plan = methyl_halide_plan(
            element, cardinal, root=root, f12_doubles=True, conventional_triples=True
        )
        assert [q.role for q, _ in plan.steps] == [
            "hf_cabs",
            "mp2_f12_correlation",
            "ccsd_f12b_minus_mp2_f12",
            "triples",
        ]
        assert all(
            (q.basis.basis_id, q.root_id, q.charge, q.multiplicity)
            == (basis.basis_id, root, charge, mult)
            for q, _ in plan.steps
        )


def test_displaced_organic_geometry() -> None:
    fixed = methyl_halide_plan("Br", 2).steps[0][0]
    coords = [list(atom.position) for atom in fixed.atoms]
    coords[1][2] += 0.0004
    shifted = methyl_halide_plan("Br", 2, coordinates_bohr=coords).steps[0][0]
    assert shifted.atoms[1].position[2] == coords[1][2]
    assert fixed.surface_id != shifted.surface_id
    assert fixed.basis.basis_id == shifted.basis.basis_id


@pytest.mark.parametrize("element", ("Br", "I"))
@pytest.mark.parametrize("cardinal", (2, 3))
@pytest.mark.parametrize("root,charge,multiplicity", (("rhf", 0, 1), ("uhf", 1, 2)))
def test_methyl_halide_fitted_ecp_template(
    element: str, cardinal: int, root: str, charge: int, multiplicity: int
) -> None:
    import ctypes

    from generativeqc import Calculator, _native
    from generativeqc.ecp import fitted_hf_ecp_domain, resolve_ecp

    q = methyl_halide_plan(element, cardinal, root=root).steps[0][0]
    basis, atoms = q.basis.orbital, q.atoms
    assert fitted_hf_ecp_domain(root, basis, atoms, charge, multiplicity)
    assert not fitted_hf_ecp_domain(root, basis, atoms, charge + 2, multiplicity)
    assert not fitted_hf_ecp_domain(root, basis, atoms[:-1], charge, multiplicity)
    cores, terms = resolve_ecp(basis, atoms)
    assert cores == (0, 10 if element == "Br" else 28, 0, 0, 0)
    assert all(term[0] == 1 for term in terms)
    calc = Calculator(
        method=root,
        basis=basis,
        auxiliary_basis=q.basis.jk_fit,
        density_fitting="cuda",
        device="cuda",
        precision="fp64",
    )
    assert calc._fitted_ecp_hf(basis, atoms, charge, multiplicity)
    assert calc._full_df_basis_consumer("df_three_center", 1, atoms=atoms) == "full_df"
    assert calc.basis_capability_report(atoms, compute_forces=True)["eligible"]
    # Template expansion owns no electronic solve and is identical on host and device.
    ctx = ctypes.c_void_p()
    cpu = Calculator(basis=basis, device="cpu")
    lib = calc._library
    _native.check(
        lib,
        lib.generativeqc_context_create(
            ctypes.byref(cpu._context_descriptor()), ctypes.byref(ctx)
        ),
    )
    try:
        handle = calc._create_native_system(ctx, atoms, charge, multiplicity)
        lib.generativeqc_system_destroy(handle)
    finally:
        lib.generativeqc_context_destroy(ctx)
