import ctypes
from dataclasses import replace

import pytest
from _bse_basis_support import jk_fit_basis, pp_orbital_basis
from generativeqc import Atom, Calculator, _native
from generativeqc.ecp import fitted_hf_ecp_domain, resolve_ecp
from generativeqc_compiler.common.resources import ResourceBudget


def bromoacetylene() -> tuple[Atom, ...]:
    # Linear H-C#C-Br with round bond lengths (angstrom): C#C 1.20, C-Br 1.79, C-H 1.06.
    return tuple(
        Atom.from_value((element, (0.0, 0.0, z / 0.529177210544)))
        for element, z in (("C", 0.0), ("C", 1.20), ("Br", 2.99), ("H", -1.06))
    )


@pytest.mark.parametrize("cardinal", [2, 3])
@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
def test_bromoacetylene_fitted_hf_host_admission(
    method: str, charge: int, multiplicity: int, cardinal: int
) -> None:
    atoms = bromoacetylene()
    orbital = pp_orbital_basis((1, 6, 35), cardinal)
    jk_fit = jk_fit_basis((1, 6, 35))
    cores, terms = resolve_ecp(orbital, atoms)
    assert cores == (0, 0, 10, 0)
    assert {term[0] for term in terms} == {2}
    calc = Calculator(
        method=method,
        basis=orbital,
        auxiliary_basis=jk_fit,
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=1 << 30,
        device="cuda",
        initial_guess=None,
    )
    assert calc.basis_capability_report(atoms, compute_forces=True)["eligible"]
    cpu = Calculator(basis=orbital, device="cpu")
    ctx = ctypes.c_void_p()
    lib = cpu._library
    _native.check(
        lib,
        lib.generativeqc_context_create(
            ctypes.byref(cpu._context_descriptor()), ctypes.byref(ctx)
        ),
    )
    try:
        for record in (None, jk_fit):
            handle = calc._create_native_system(
                ctx, atoms, charge, multiplicity, record
            )
            lib.generativeqc_system_destroy(handle)
        with pytest.raises(NotImplementedError, match="ECP density-fitting"):
            calc._create_native_system(ctx, atoms, charge + 2, multiplicity)
    finally:
        lib.generativeqc_context_destroy(ctx)

    states = {"charges": [charge], "multiplicities": [multiplicity]}
    plan = calc.estimate_resources([atoms], **states)
    if method == "rhf":
        assert plan.status == "feasible"
        assert dict(plan.selections) == {"hf": "cuda-df-source"}
        ecp = [
            estimate
            for estimate in plan.requests[0].candidates[0].estimates
            if estimate.name.startswith("serialized scalar ECP ")
        ]
        assert len(ecp) == 2 and all(estimate.bytes > 0 for estimate in ecp)
        constrained = calc.estimate_resources(
            [atoms], budget=ResourceBudget(host_bytes=1 << 20), **states
        )
        assert constrained.status == "infeasible"
    else:
        assert plan.status == "unsupported"
        assert "budgeted RHF" in plan.requests[0].unsupported_reason


@pytest.mark.parametrize("element", ["Br", "I"])
def test_carbon_halide_electron_parity_and_topology(element: str) -> None:
    basis = pp_orbital_basis((1, 6, 35 if element == "Br" else 53))
    atoms = (
        Atom.from_value(("C", (0.0, 0.0, 0.0))),
        Atom.from_value((element, (0.0, 0.0, 3.5))),
    )
    assert fitted_hf_ecp_domain("uhf", basis, atoms, 0, 2)
    assert fitted_hf_ecp_domain("rhf", basis, atoms, 1, 1)
    assert not fitted_hf_ecp_domain("uhf", basis, atoms, 1, 2)
    assert not fitted_hf_ecp_domain("rhf", basis, atoms, 0, 1)
    assert not fitted_hf_ecp_domain("uhf", basis, atoms, 0, 4)
    assert not fitted_hf_ecp_domain("uhf", basis, atoms, -2, 2)
    assert not fitted_hf_ecp_domain("uhf", basis, atoms + atoms[1:], 0, 2)
    assert not fitted_hf_ecp_domain(
        "uhf", replace(basis, representation="cartesian"), atoms, 0, 2
    )
    wrong_core = replace(basis.elements[-1], ecp_core_electrons=2)
    assert not fitted_hf_ecp_domain(
        "uhf", replace(basis, elements=(*basis.elements[:-1], wrong_core)), atoms, 0, 2
    )
