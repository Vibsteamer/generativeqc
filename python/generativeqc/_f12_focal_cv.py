"""Build conventional DF-MP2 requests for a focal reference."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .experimental.f12_focal import FocalRequest


from ._f12_frozen_core import (
    FROZEN_CORE_DEFINITION_VERSION,
)


def conventional_mp2_request(
    atoms,
    basis,
    *,
    root_id: str,
    correlation_space_id: str,
    charge: int = 0,
    multiplicity: int = 1,
    backend: str = "cuda",
    protocol_id: str | None = None,
) -> FocalRequest:
    """Build a pure conventional DF-MP2 FocalRequest (zero CABS, zero F12, beta=None)."""
    import json

    from ._api_types import Atom
    from .elements import checked_integer, electron_state
    from .experimental.f12_focal import FocalBasis, FocalRequest
    from .profiles import canonical_hash

    geometry = tuple(Atom.from_value(atom) for atom in atoms)
    if not geometry:
        raise ValueError("at least one atom is required")
    charge = checked_integer(charge, "ionic charge", low=-(2**31))
    multiplicity = checked_integer(multiplicity, "multiplicity", low=1)
    if root_id not in ("rhf", "uhf") or (root_id == "rhf" and multiplicity != 1):
        raise ValueError("conventional DF-MP2 requires singlet rhf or uhf root_id")
    if not isinstance(correlation_space_id, str) or not correlation_space_id.strip():
        raise ValueError("conventional DF-MP2 requires a nonempty correlation_space_id")
    if backend not in ("cuda", "cpu_reference"):
        raise ValueError("focal backend must be cuda or cpu_reference")
    if not isinstance(basis, FocalBasis):
        raise TypeError("conventional_mp2_request requires a FocalBasis")
    nuclei = {atom.atomic_number for atom in geometry}
    for name, record in (
        ("orbital", basis.orbital),
        ("JKFit", basis.jk_fit),
        ("MP2Fit", basis.mp2_fit),
    ):
        if record is None:
            raise ValueError(f"conventional DF-MP2 requires an explicit {name} basis")
        if nuclei - record.by_element.keys():
            raise ValueError(f"{name} basis {record.name!r} lacks a nucleus in the geometry")

    orbital = basis.orbital.by_element
    state = electron_state(
        geometry, charge=charge, multiplicity=multiplicity, element_metadata=orbital
    )
    hamiltonian = tuple(
        (z, core, canonical_hash(json.loads(data)) if data is not None else None)
        for z, core, data in zip(
            state.nuclear_charges,
            state.ecp_core_electrons,
            (orbital[atom.atomic_number].ecp_data for atom in geometry),
            strict=True,
        )
    )
    surface_id = canonical_hash(
        {
            "geometry_bohr": [
                (
                    atom.atomic_number,
                    tuple((v if v else 0.0).hex() for v in atom.position),
                )
                for atom in geometry
            ],
            "charge": charge,
            "multiplicity": multiplicity,
            "hamiltonian": hamiltonian,
            "electronic_root": root_id,
            "correlation_space": correlation_space_id,
            "frozen_core_definition_version": FROZEN_CORE_DEFINITION_VERSION,
            **({"protocol_id": protocol_id} if protocol_id is not None else {}),
        }
    )
    return FocalRequest(
        geometry,
        charge,
        multiplicity,
        root_id,
        correlation_space_id,
        surface_id,
        basis,
        "mp2_f12_correlation",
        backend,  # type: ignore[arg-type]
    )
