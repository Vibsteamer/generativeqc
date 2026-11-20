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


def resolve_ecp(basis: typing.Any, atoms: typing.Any) -> typing.Any:
    """Resolve immutable source records to atom-mapped scalar residual terms."""
    shell_limit = 4 if g_shell_halogen_hydride(basis, atoms) else 3
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
                else "scalar-ECP hydride orbital shells extend through g"
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
                    else "scalar-ECP hydride orbital shells extend through g"
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

    def contract(
        self,
        weights: typing.Any,
        *,
        orbital_coefficients: np.ndarray | None = None,
        coefficient_derivatives: np.ndarray | None = None,
        device: str = "cpu",
    ) -> np.ndarray:
        """Differentiate a fixed real weight contracted with V or C.T @ V @ C.

        C uses the same AO row order as the matrices and can be rectangular.
        Supply dC[atom, xyz] for a moving orbital frame. Alpha and beta frames
        can be contracted independently and their gradients added. The result
        uses the positive energy-derivative sign. CUDA evaluates and returns
        the whole Cartesian panel in one call.
        """
        if self.local_derivative is None or self.nonlocal_derivative is None:
            raise ValueError("ECP derivatives were not requested")
        if device not in {"cpu", "cuda"}:
            raise ValueError("ECP gradient device must be cpu or cuda")
        if np.iscomplexobj(weights):
            raise TypeError("ECP weights must be real")
        weights = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(weights).all():
            raise ValueError("ECP weights must be finite")
        if orbital_coefficients is None:
            if coefficient_derivatives is not None:
                raise ValueError("orbital coefficient derivatives require their frame")
            if weights.shape != self.local.shape:
                raise ValueError("ECP weights must match the full AO matrix")
            frame = None
        else:
            frame = np.asarray(orbital_coefficients)
            if (
                frame.ndim != 2
                or frame.dtype != np.float64
                or frame.shape[0] != self.local.shape[0]
                or not frame.shape[1]
                or weights.shape != (frame.shape[1], frame.shape[1])
                or not np.isfinite(frame).all()
            ):
                raise ValueError(
                    "ECP orbital frame and weights require matching finite dimensions"
                )
        motion = None
        if coefficient_derivatives is not None:
            motion = np.asarray(coefficient_derivatives)
            if (
                motion.dtype != np.float64
                or motion.shape != (*self.local_derivative.shape[:2], *frame.shape)
                or not np.isfinite(motion).all()
            ):
                raise ValueError(
                    "ECP frame derivatives must match atom, Cartesian and orbital dimensions"
                )
        if device == "cuda":
            from ._ecp_cuda import prepare, projected_inputs

            program, feeds = projected_inputs(self, weights, frame, motion)
            with prepare(program) as owner:
                return owner.execute(feeds).outputs["gradient"]
        ao_weights = weights if frame is None else frame @ weights @ frame.T
        gradient = np.einsum("axij,ij->ax", self.local_derivative, ao_weights)
        gradient += np.einsum("axij,ij->ax", self.nonlocal_derivative, ao_weights)
        if motion is not None:
            # The scalar residual ECP is real and symmetric; W need not be.
            frame_weights = self.matrix @ frame @ (weights + weights.T)
            gradient += np.einsum("axij,ij->ax", motion, frame_weights)
        return gradient

    def occupied_expectation_gradient(
        self,
        occupied: np.ndarray | tuple[np.ndarray, np.ndarray],
        virtual: np.ndarray | tuple[np.ndarray, np.ndarray],
        overlap_derivatives: np.ndarray,
        virtual_occupied_response: np.ndarray | tuple[np.ndarray, np.ndarray],
        *,
        occupation: int = 1,
        device: str = "cpu",
    ) -> np.ndarray:
        """Nuclear derivative of the scalar ECP expectation in occupied orbitals.

        The two real coefficient matrices have AO rows and S-orthonormal
        columns at the reference geometry. Supply dS in the same AO basis and
        the solved virtual-occupied coefficient response Uvo, both ordered by
        atom and Cartesian component. Use occupation 2 for a restricted
        determinant. For UHF, pass pairs of alpha/beta occupied, virtual and
        response arrays with occupation 1 to receive (2, atom, xyz). On CUDA
        both spins share the AO inputs and all Cartesian components are returned
        in one host transfer.
        """
        occupation = checked_integer(occupation, "orbital occupation", low=1, high=2)
        if device not in {"cpu", "cuda"}:
            raise ValueError("ECP expectation device must be cpu or cuda")
        pair = (
            isinstance(occupied, tuple)
            and len(occupied) == 2
            and all(np.ndim(c) == 2 for c in occupied)
        )
        if pair:
            if occupation != 1 or not all(
                isinstance(x, tuple) and len(x) == 2
                for x in (virtual, virtual_occupied_response)
            ):
                raise ValueError(
                    "a spin pair requires paired frames and responses with occupation 1"
                )
            frames = tuple(zip(occupied, virtual, virtual_occupied_response, strict=True))
        else:
            frames = ((occupied, virtual, virtual_occupied_response),)
        ds = np.asarray(overlap_derivatives)
        nao = self.local.shape[0]
        if self.local_derivative is None or self.nonlocal_derivative is None:
            raise ValueError("ECP derivatives were not requested")
        directions = self.local_derivative.shape[:2]
        if (
            ds.shape != (*directions, nao, nao)
            or ds.dtype != np.float64
            or not np.isfinite(ds).all()
        ):
            raise ValueError("ECP overlap derivatives must be finite with matching AO dimensions")
        blocks = []
        for frame in frames:
            co, cv, uvo = map(np.asarray, frame)
            if (
                co.ndim != 2
                or cv.ndim != 2
                or co.shape[0] != nao
                or cv.shape[0] != nao
                or uvo.shape != (*directions, cv.shape[1], co.shape[1])
            ):
                raise ValueError(
                    "ECP orbitals and response require matching AO and orbital dimensions"
                )
            if any(
                a.dtype != np.float64 or not np.isfinite(a).all() for a in (co, cv, uvo)
            ):
                raise ValueError("ECP orbitals and response must be finite real double precision")
            blocks.append((co, cv, uvo))
        if device == "cuda":
            from ._ecp_occupied_cuda import inputs, prepare

            program, feeds = inputs(self, ds, blocks, occupation)
            with prepare(program) as owner:
                gradients = owner.execute(feeds).outputs["gradient"]
            return gradients if pair else gradients[0]
        results = [
            self._occupied_spin_gradient(co, cv, ds, uvo, occupation)
            for co, cv, uvo in blocks
        ]
        return np.stack(results) if pair else results[0]

    def _occupied_spin_gradient(
        self, co: np.ndarray, cv: np.ndarray, ds: np.ndarray, uvo: np.ndarray, occupation: int
    ) -> np.ndarray:
        vco = self.matrix @ co
        gradient = self.contract(co @ co.T)
        # The symmetric occupied rotation follows from d(Co.T @ S @ Co) = 0.
        # Its antisymmetric gauge drops out of this trace with equal occupations.
        normalization_weight = co @ (co.T @ vco) @ co.T
        gradient -= np.einsum("axij,ij->ax", ds, normalization_weight)
        gradient += 2 * np.einsum("axvi,vi->ax", uvo, cv.T @ vco)
        return occupation * gradient


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
