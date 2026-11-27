"""Scalar semilocal Gaussian ECPs in the BSE/NWChem convention.

The highest angular channel is the local residual potential; lower channels
are projector differences, not absolute potentials. Radial coefficients
multiply r**(n-2) exp(-alpha*r*r). The separate Coulomb tail uses Z-core.
No runtime reference-package dependency or implicit parameter download.
"""

from __future__ import annotations

import ctypes
import json
import typing
from dataclasses import dataclass

import numpy as np
from generativeqc_compiler.integral.ecp_policy import (
    MAX_CPU_POLAR_POINTS,
    MAX_CPU_RADIAL_POINTS,
)

from .basis import BasisSet, validate_ecp_data
from .elements import checked_integer

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from .calculator import Atom


def g_shell_halogen_hydride(basis, atoms):
    """Return whether orbital g quadrature was qualified for this atom/core topology."""
    if not isinstance(basis, BasisSet) or basis.representation != "spherical":
        return False
    elements = sorted(a.atomic_number for a in atoms)
    if elements not in ([1, 35], [1, 53], [1, 1, 35], [1, 1, 53]):
        return False
    hydrogen = basis.by_element.get(1)
    heavy = basis.by_element.get(elements[-1])
    return (
        hydrogen is not None
        and hydrogen.ecp_core_electrons == 0
        and heavy is not None
        and heavy.ecp_core_electrons == {35: 10, 53: 28}[elements[-1]]
    )


def g_shell_carbon_halide(basis: object, atoms: Sequence[Atom]) -> bool:
    """One small-core Br or I, at least one all-electron C and optional H."""
    if not isinstance(basis, BasisSet) or basis.representation != "spherical":
        return False
    elements = [a.atomic_number for a in atoms]
    if 6 not in elements or sum(z in (35, 53) for z in elements) != 1:
        return False
    cores = {1: 0, 6: 0, 35: 10, 53: 28}
    return all(
        z in cores
        and z in basis.by_element
        and basis.by_element[z].nuclear_charge == z
        and basis.by_element[z].ecp_core_electrons == cores[z]
        for z in set(elements)
    )


def g_shell_halogen_water(basis: object, atoms: Sequence[Atom]) -> bool:
    """One small-core Br or I with all-electron oxygen and two hydrogens."""
    if not isinstance(basis, BasisSet) or basis.representation != "spherical":
        return False
    elements = sorted(a.atomic_number for a in atoms)
    if elements not in ([1, 1, 8, 35], [1, 1, 8, 53]):
        return False
    return all(
        z in basis.by_element
        and basis.by_element[z].nuclear_charge == z
        and basis.by_element[z].ecp_core_electrons == core
        for z, core in ((1, 0), (8, 0), (elements[-1], {35: 10, 53: 28}[elements[-1]]))
    )


def g_shell_small_core_halide(basis: object, atoms: Sequence[Atom]) -> bool:
    """One or more small-core Br(ECP10MDF) / I(ECP28MDF) atoms with no other ECP elements."""
    if not isinstance(basis, BasisSet) or basis.representation != "spherical":
        return False
    has_halide = False
    for atom in atoms:
        eb = basis.by_element.get(atom.atomic_number)
        if eb is None or eb.nuclear_charge != atom.atomic_number:
            return False
        if atom.atomic_number == 35 and eb.ecp_core_electrons == 10:
            has_halide = True
        elif atom.atomic_number == 53 and eb.ecp_core_electrons == 28:
            has_halide = True
        elif eb.ecp_core_electrons != 0:
            return False
    return has_halide


def g_shell_halogen_dihydride(basis, atoms):
    return len(atoms) == 3 and g_shell_halogen_hydride(basis, atoms)


def fitted_uhf_hydride_domain(basis, atoms, charge, multiplicity):
    """Scalar small-core monohalogen hydrides covered by native CUDA DF-UHF."""
    if (
        charge != 0
        or multiplicity != 2
        or not isinstance(basis, BasisSet)
        or basis.representation != "spherical"
    ):
        return False
    shell_limit = 4 if g_shell_halogen_dihydride(basis, atoms) else 3
    heavy = []
    for atom in atoms:
        entry = basis.by_element.get(atom.atomic_number)
        if entry is None or any(s.angular_momentum > shell_limit for s in entry.shells):
            return False
        if atom.atomic_number != 1 or entry.ecp_core_electrons:
            heavy.append((atom.atomic_number, entry.ecp_core_electrons))
    return len(heavy) == 1 and heavy[0] in ((35, 10), (53, 28))


def fitted_hf_hydride_domain(method, basis, atoms, charge, multiplicity):
    """Qualified CUDA fitted spin and orbital domains for small-core hydrides."""
    if method == "uhf":
        return fitted_uhf_hydride_domain(basis, atoms, charge, multiplicity)
    return (
        method == "rhf"
        and charge == 0
        and multiplicity == 1
        and len(atoms) == 2
        and g_shell_halogen_hydride(basis, atoms)
        and all(
            shell.angular_momentum <= 4
            for atom in atoms
            for shell in basis.by_element[atom.atomic_number].shells
        )
    )


def _fitted_ecp_orbital_angular_limit(
    basis: BasisSet, atoms: Sequence[Atom], charge: int, multiplicity: int
) -> int:
    """Mirror C++ SmallCoreHalideComposition::fitted_orbital_angular_limit exactly.

    Returns 0 when the electronic state/topology is outside the fitted scalar ECP domain,
    3 for non-monohydride/dihydride doublet hydrides (e.g. BrH4 doublet -> s..f only),
    and 4 when orbital g shells are admitted.
    """
    hydrogen = carbon = oxygen = bromine = iodine = other = other_ecp = 0
    unsupported_other = 0
    neutral_valence_electrons = 0
    electrons = -charge
    for atom in atoms:
        eb = basis.by_element.get(atom.atomic_number)
        if eb is None or eb.nuclear_charge != atom.atomic_number:
            return 0
        core = eb.ecp_core_electrons
        if atom.atomic_number > core:
            neutral_valence_electrons += atom.atomic_number - core
        electrons += atom.atomic_number - core
        if atom.atomic_number == 1 and core == 0:
            hydrogen += 1
        elif atom.atomic_number == 6 and core == 0:
            carbon += 1
        elif atom.atomic_number == 8 and core == 0:
            oxygen += 1
        elif atom.atomic_number == 35 and core == 10:
            bromine += 1
        elif atom.atomic_number == 53 and core == 28:
            iodine += 1
        else:
            other += 1
            if core != 0:
                other_ecp += 1
            if core != 0 or atom.atomic_number < 1 or atom.atomic_number > 18:
                unsupported_other += 1

    ligand = (carbon > 0 and oxygen == 0) or (
        carbon == 0 and oxygen == 1 and hydrogen == 2
    )
    molecular_halide = ligand and (bromine + iodine == 1) and other == 0
    if (
        molecular_halide
        and charge in (0, 1)
        and multiplicity in (1, 2)
        and electrons == 6 * carbon + 8 * oxygen + hydrogen + 25 - charge
        and electrons % 2 == multiplicity - 1
    ):
        return 4

    hydride = carbon == 0 and oxygen == 0 and (bromine + iodine == 1) and other == 0
    if hydride:
        if charge == 0:
            if multiplicity == 1 and hydrogen == 1 and electrons == 26:
                return 4
            if multiplicity == 2 and hydrogen in (0, 2) and electrons == 25 + hydrogen:
                return 4
            if multiplicity == 2 and electrons == 25 + hydrogen and electrons % 2 == 1:
                return 3
        elif charge == 1:
            if multiplicity == 2 and hydrogen == 1 and electrons == 25:
                return 4
            if multiplicity == 1 and hydrogen in (0, 2) and electrons == 24 + hydrogen:
                return 4
        elif charge == -1:
            if multiplicity == 1 and hydrogen == 0 and electrons == 26:
                return 4
        return 0

    if (
        bromine + iodine >= 1
        and other_ecp == 0
        and unsupported_other == 0
        and not molecular_halide
        and -1 <= charge <= 1
        and multiplicity in (1, 2)
        and electrons > 0
        and electrons == neutral_valence_electrons - charge
        and electrons % 2 == multiplicity - 1
    ):
        return 4
    return 0


def fitted_hf_ecp_domain(
    method: str, basis: BasisSet, atoms: Sequence[Atom], charge: int, multiplicity: int
) -> bool:
    """Admit the supported small-core halide states and orbital angular momenta."""
    if not isinstance(basis, BasisSet) or basis.representation != "spherical":
        return False
    if (method, multiplicity) not in (("rhf", 1), ("uhf", 2)):
        return False
    shell_limit = _fitted_ecp_orbital_angular_limit(basis, atoms, charge, multiplicity)
    if shell_limit <= 0:
        return False
    return all(
        shell.angular_momentum <= shell_limit
        for z in {atom.atomic_number for atom in atoms}
        for shell in basis.by_element[z].shells
    )


def resolve_ecp(basis: typing.Any, atoms: typing.Any) -> typing.Any:
    """Resolve immutable source records to atom-mapped scalar residual terms."""
    orbital_g = (
        g_shell_halogen_hydride(basis, atoms)
        or g_shell_carbon_halide(basis, atoms)
        or g_shell_halogen_water(basis, atoms)
        or g_shell_small_core_halide(basis, atoms)
    )
    shell_limit = 4 if orbital_g else 3
    cores, terms = [], []
    metadata = basis.by_element if isinstance(basis, BasisSet) else {}
    for atom_index, atom in enumerate(atoms):
        element = metadata.get(atom.atomic_number)
        core = element.ecp_core_electrons if element else 0
        cores.append(core)
        if not core:
            continue
        if core >= atom.atomic_number:
            raise ValueError("ECP core count must leave a positive ionic charge")
        potentials = json.loads(element.ecp_data)
        # Requires ecp_type == "scalar_ecp" and one angular channel per record,
        # as well as valid radial arrays, before interpreting the local channel.
        validate_ecp_data(potentials)
        channels = [p["angular_momentum"][0] for p in potentials]
        if len(set(channels)) != len(channels):
            raise ValueError("duplicate ECP angular channel")
        local = max(channels)
        if local > 4:
            raise NotImplementedError(
                "ECP local channel must be s/p/d/f/g; projectors at most f"
            )
        if any(s.angular_momentum > shell_limit for s in element.shells):
            raise NotImplementedError(
                "ECP execution supports orbital s/p/d/f shells"
                if shell_limit == 3
                else "scalar-ECP orbital shells in this domain extend through g"
            )
        for potential in potentials:
            if set(potential) - {
                "ecp_type",
                "angular_momentum",
                "r_exponents",
                "gaussian_exponents",
                "coefficients",
            }:
                raise NotImplementedError(
                    "unknown ECP parameter fields require an explicit format conversion"
                )
            if len(potential["coefficients"]) != 1:
                raise NotImplementedError(
                    "ECP requires exactly one scalar coefficient row"
                )
            channel = potential["angular_momentum"][0]
            for power, exponent, coefficient in zip(
                potential["r_exponents"],
                potential["gaussian_exponents"],
                potential["coefficients"][0],
                strict=True,
            ):
                if power > 4:
                    raise NotImplementedError(
                        "ECP radial powers are supported from 0 through 4"
                    )
                terms.append(
                    (
                        atom_index,
                        -1 if channel == local else channel,
                        power,
                        float(exponent),
                        float(coefficient),
                    )
                )
    if terms:
        # Applies to mixed all-electron/ECP atoms too.
        for atom in atoms:
            if any(
                s.angular_momentum > shell_limit
                for s in metadata[atom.atomic_number].shells
            ):
                raise NotImplementedError(
                    "ECP execution supports orbital s/p/d/f shells"
                    if shell_limit == 3
                    else "scalar-ECP orbital shells in this domain extend through g"
                )
    return tuple(cores), tuple(terms)


@dataclass(frozen=True)
class ECPIntegrals:
    """Separate residual matrices and atom/xyz energy derivatives (Hartree/bohr)."""

    local: np.ndarray
    nonlocal_: np.ndarray
    local_derivative: np.ndarray | None
    nonlocal_derivative: np.ndarray | None
    radial_points: int
    polar_points: int
    backend: str
    model_identity: str

    def quadrature_difference(self, refined: typing.Any) -> typing.Any:
        """Empirical discretization evidence; not a rigorous error bound."""
        if (
            not isinstance(refined, ECPIntegrals)
            or refined.local.shape != self.local.shape
            or refined.model_identity != self.model_identity
        ):
            raise ValueError(
                "quadrature comparisons require matching ECP models, geometry and matrix layouts"
            )
        result = {
            "local_matrix_max_abs": float(np.max(np.abs(self.local - refined.local))),
            "nonlocal_matrix_max_abs": float(
                np.max(np.abs(self.nonlocal_ - refined.nonlocal_))
            ),
        }
        if self.local_derivative is not None and refined.local_derivative is not None:
            result.update(
                local_derivative_max_abs=float(
                    np.max(np.abs(self.local_derivative - refined.local_derivative))
                ),
                nonlocal_derivative_max_abs=float(
                    np.max(
                        np.abs(self.nonlocal_derivative - refined.nonlocal_derivative)
                    )
                ),
            )
        return result

    @property
    def matrix(self) -> typing.Any:
        return self.local + self.nonlocal_

    def contract(self, weights: typing.Any) -> typing.Any:
        """Contract fixed arbitrary real AO weights, including nonsymmetric ones."""
        if self.local_derivative is None:
            raise ValueError("ECP derivatives were not requested")
        if np.iscomplexobj(weights):
            raise TypeError("ECP weights must be real")
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != self.local.shape or not np.isfinite(weights).all():
            raise ValueError("ECP weights must be finite and match the full AO matrix")
        return np.einsum(
            "axij,ij->ax", self.local_derivative + self.nonlocal_derivative, weights
        )


def ecp_integrals(
    atoms: typing.Any,
    basis: typing.Any,
    *,
    charge: typing.Any = 0,
    multiplicity: typing.Any = 1,
    device: typing.Any = "cpu",
    derivatives: typing.Any = True,
    radial_points: typing.Any = 160,
    polar_points: typing.Any = 32,
) -> typing.Any:
    """Evaluate native ECP residuals, without SCF or evaluating electron repulsion.

    Quadrature settings are explicit. Compare a refined grid to assess
    discretization error independently from floating-point/backend differences.
    CPU raw-integral diagnostics admit up to 1536 radial and 192 polar points;
    CUDA retains its 512/96 limits. These controls do not change the checked
    SCF/force quadrature or promote any electronic/second-order capability.
    """
    from . import _native
    from .calculator import Atom, Calculator
    from .profiles import canonical_hash

    if not isinstance(basis, BasisSet):
        raise TypeError("ECP evaluation requires an owned canonical BasisSet")
    atoms = tuple(Atom.from_value(a) for a in atoms)
    radial_points = checked_integer(
        radial_points,
        "radial points",
        low=16,
        high=MAX_CPU_RADIAL_POINTS if device == "cpu" else 512,
    )
    polar_points = checked_integer(
        polar_points,
        "polar points",
        low=8,
        high=MAX_CPU_POLAR_POINTS if device == "cpu" else 96,
    )
    if type(derivatives) is not bool:
        raise TypeError("derivatives must be boolean")
    resolve_ecp(basis, atoms)
    calculator = Calculator(basis=basis, device=device)
    context = ctypes.c_void_p()
    _native.check(
        calculator._library,
        calculator._library.generativeqc_context_create(
            ctypes.byref(calculator._context_descriptor()), ctypes.byref(context)
        ),
    )
    system = None
    try:
        system = calculator._create_native_system(context, atoms, charge, multiplicity)
        n = sum(
            (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2
            if basis.representation == "cartesian"
            else 2 * s.angular_momentum + 1
            for s in basis.shells_for(atoms)
        )
        ncoord = 3 * len(atoms) if derivatives else 0
        output = np.empty((2, 1 + ncoord, n, n), dtype=np.float64)
        _native.check(
            calculator._library,
            calculator._library.generativeqc_system_ecp_integrals(
                context,
                system,
                radial_points,
                polar_points,
                derivatives,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                output.size,
            ),
        )
        if not np.isfinite(output).all():
            raise ArithmeticError("nonfinite ECP integral output")
        output.setflags(write=False)
        return ECPIntegrals(
            output[0, 0],
            output[1, 0],
            output[0, 1:].reshape(len(atoms), 3, n, n) if derivatives else None,
            output[1, 1:].reshape(len(atoms), 3, n, n) if derivatives else None,
            radial_points,
            polar_points,
            device,
            canonical_hash(
                {
                    "basis": calculator.basis_metadata(
                        atoms, charge=charge, multiplicity=multiplicity
                    )["orbital"]["mathematical_identity"],
                    "geometry": [(a.atomic_number, a.position) for a in atoms],
                }
            ),
        )
    finally:
        if system is not None:
            calculator._library.generativeqc_system_destroy(system)
        calculator._library.generativeqc_context_destroy(context)
