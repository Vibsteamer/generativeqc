"""Explicit offline import of BSE JSON 0.1 (complete or minimal) and NWChem text."""

import hashlib
import json
import typing
from pathlib import Path

from .basis import (
    BasisProvenance,
    BasisSet,
    BasisShell,
    ElementBasis,
    read_local_bytes,
    read_local_json,
)
from .elements import checked_integer

_BSE_SCHEMAS = {
    "complete": {"schema_type": "complete", "schema_version": "0.1"},
    "minimal": {"schema_type": "minimal", "schema_version": "0.1"},
}


def import_bse(
    path: typing.Any,
    *,
    source: typing.Any,
    source_version: typing.Any,
    license: typing.Any,
    representation: typing.Any = None,
    name: typing.Any = None,
) -> typing.Any:
    """Import a supplied file; declare its provenance/license, never query a service.

    Complete JSON carries its own name and basis-data version. Minimal JSON,
    as written by BSE format converters, has neither: its name defaults to
    the file stem and ``source_version`` must identify the data. BSE
    single-l general contractions remain matrices. Combined shells require
    one coefficient row per angular momentum and retain their source group.
    Ambiguous combined/general layouts are rejected instead of guessed.
    """
    _require_source_version(source_version)
    data, checksum = read_local_json(path)
    return _basis_from_bse(
        data,
        checksum,
        source=source,
        source_version=source_version,
        license=license,
        representation=representation,
        name=name,
        default_name=Path(path).stem,
    )


def import_nwchem(
    path: typing.Any,
    *,
    source: typing.Any,
    source_version: typing.Any,
    license: typing.Any,
    representation: typing.Any = None,
    name: typing.Any = None,
) -> typing.Any:
    """Import a supplied NWChem basis/ECP file with the BSE NWChem reader.

    The reader comes from the optional ``basis-set-exchange`` package
    (``generativeqc[bse]``); nothing is downloaded. The provenance checksum is
    the SHA-256 of the file bytes and the name defaults to the file stem. An
    s/p-only file cannot reveal its AO representation, so pass
    ``representation`` for one.
    """
    _require_source_version(source_version)
    path = Path(path)
    raw = read_local_bytes(path)
    try:
        from basis_set_exchange import readers
    except ImportError as error:
        raise ImportError(
            "NWChem import uses the basis_set_exchange reader; install "
            "`pip install 'generativeqc[bse]'` (or `pip install basis-set-exchange`)"
        ) from error
    data = readers.read_formatted_basis_str(raw.decode("utf-8"), "nwchem")
    return _basis_from_bse(
        data,
        hashlib.sha256(raw).hexdigest(),
        source=source,
        source_version=source_version,
        license=license,
        representation=representation,
        name=name,
        default_name=path.stem,
    )


def _require_source_version(source_version: typing.Any) -> None:
    if not isinstance(source_version, str) or not source_version.strip():
        raise ValueError("source version must be explicitly supplied")


def bse_element_records(data: typing.Any) -> dict[int, ElementBasis]:
    """Convert the element records of an in-memory BSE 0.1 dictionary.

    Shell source groups are the BSE shell indices, as in :func:`import_bse`.
    """
    _, kinds = _bse_schema_kinds(data)
    return {element.atomic_number: element for element in _bse_elements(data, kinds)}


def _basis_from_bse(
    data: typing.Any,
    checksum: str,
    *,
    source: typing.Any,
    source_version: str,
    license: typing.Any,
    representation: typing.Any,
    name: typing.Any,
    default_name: str,
) -> BasisSet:
    complete, kinds = _bse_schema_kinds(data)
    if representation is None:
        if {"gto_cartesian", "gto_spherical"} <= kinds:
            raise ValueError(
                "mixed Cartesian/spherical BSE input requires an explicit representation"
            )
        representation = "spherical" if "gto_spherical" in kinds else "cartesian"
    elements = _bse_elements(data, kinds)
    version = data.get("version")
    if complete or version is not None:
        if not isinstance(version, str) or not version:
            raise ValueError("BSE source requires an explicit basis-data version")
    else:
        version = "unversioned"
    if name is None:
        name = data.get("name", "") if complete else default_name
    return BasisSet(
        name,
        elements,
        BasisProvenance(
            source, f"{source_version}; basis-data={version}", license, checksum
        ),
        representation,
    )


def _bse_schema_kinds(data: typing.Any) -> tuple[bool, set[str]]:
    schema = data.get("molssi_bse_schema")
    if schema not in _BSE_SCHEMAS.values():
        raise ValueError("import requires BSE complete or minimal JSON schema 0.1")
    complete = schema == _BSE_SCHEMAS["complete"]
    if not isinstance(data.get("elements"), dict) or not data["elements"]:
        raise ValueError("BSE input requires element records")
    declared_kinds = data.get("function_types")
    if complete or declared_kinds is not None:
        if not isinstance(declared_kinds, list) or any(
            not isinstance(kind, str) for kind in declared_kinds
        ):
            raise TypeError("BSE function_types must be an array of strings")
    else:
        declared_kinds = sorted(
            {
                shell.get("function_type")
                for element in data["elements"].values()
                if isinstance(element, dict)
                for shell in element.get("electron_shells", [])
                if isinstance(shell, dict)
            }
            - {None}
        )
    kinds = set(declared_kinds)
    if kinds - {"gto", "gto_cartesian", "gto_spherical", "scalar_ecp"}:
        raise NotImplementedError(
            "only Gaussian shells and scalar ECP metadata can be imported"
        )
    return complete, kinds


def _bse_elements(data: typing.Any, kinds: set[str]) -> tuple[ElementBasis, ...]:
    elements = []
    for key, element in data["elements"].items():
        if not key.isdecimal() or str(int(key)) != key:
            raise ValueError("BSE element keys must be canonical atomic numbers")
        z = checked_integer(int(key), "BSE atomic number", low=1, high=118)
        if not isinstance(element, dict) or not isinstance(
            element.get("electron_shells"), list
        ):
            raise TypeError("BSE element records require electron_shells arrays")
        shells = []
        for group, shell in enumerate(element.get("electron_shells", [])):
            if not isinstance(shell, dict):
                raise TypeError("BSE shell records must be objects")
            if shell.get("function_type") not in (
                "gto",
                "gto_cartesian",
                "gto_spherical",
            ):
                raise NotImplementedError(
                    f"element Z={z} shell {group}: unsupported function type"
                )
            if shell["function_type"] not in kinds:
                raise ValueError(
                    "BSE shell function type is missing from function_types"
                )
            angular, rows, exponents = (
                shell[k] for k in ("angular_momentum", "coefficients", "exponents")
            )
            if (
                not isinstance(angular, list)
                or not angular
                or not isinstance(rows, list)
            ):
                raise ValueError("malformed BSE angular/contraction arrays")
            if len(angular) == 1:
                shells.append(BasisShell(angular[0], exponents, rows, group))
            else:
                if len(rows) != len(angular) or len(set(angular)) != len(angular):
                    raise ValueError("ambiguous combined/general BSE shell")
                shells.extend(
                    BasisShell(l, exponents, (row,), group)
                    for l, row in zip(angular, rows, strict=True)
                )
        core = element.get("ecp_electrons", 0)
        potentials = element.get("ecp_potentials")
        elements.append(
            ElementBasis(
                z,
                tuple(shells),
                ecp_core_electrons=core,
                ecp_data=json.dumps(potentials, sort_keys=True, separators=(",", ":"))
                if potentials
                else None,
            )
        )
    return tuple(elements)
