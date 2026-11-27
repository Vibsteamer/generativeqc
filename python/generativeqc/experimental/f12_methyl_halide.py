"""Scalar-ECP F12 basis selections and experimental geometries for CH₃Br and CH₃I.

The heavy-augmented D and T selections augment carbon and the halogen and use
the parent hydrogen orbital basis. The JK fit is diagnostic def2-universal-JKFit.

The package holds only SHA-256 records (``data/f12_methyl_halide_external.json``).
Records redistributed by the MolSSI Basis Set Exchange come from the optional
``basis_set_exchange`` package. The Peterson/Hill ccRepo records, the carbon
diffuse exponents and the NIST CCCBDB geometries are files the user obtains and
places in ``<data_dir>/f12_methyl_halide``, where ``data_dir`` is the explicit
argument or the ``GENERATIVEQC_EXTERNAL_DATA_DIR`` environment variable; see
``docs/user/restricted_reference_data.md``. Files above 64 MiB are refused
before parsing, and a basis file must hold its recorded element and no other.
Every element record and geometry must match its recorded content hash and
every assembled selection its recorded basis identity. Nothing is substituted
when a check fails.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..basis import (
    BasisProvenance,
    BasisSet,
    ElementBasis,
    element_content_sha256,
    read_local_bytes,
)
from ..basis_import import bse_element_records
from ..elements import checked_integer
from ..profiles import canonical_hash
from .f12_focal import FocalBasis, FocalSeries, plan_singlepoint

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .f12_focal import FocalPlan

DATA_DIR_ENV = "GENERATIVEQC_EXTERNAL_DATA_DIR"
SUBDIRECTORY = "f12_methyl_halide"
GUIDE = "docs/user/restricted_reference_data.md"
GEOMETRY_CONTENT_SCHEMA = "generativeqc.geometry-content/v1"
# CODATA 2010 bohr radius in angstrom; the recorded plan geometries use it.
BOHR_ANGSTROM = 0.52917721092
_MANIFEST = (
    Path(__file__).resolve().parents[1] / "data" / "f12_methyl_halide_external.json"
)
_ROLES = ("orbital", "jkfit", "ri", "cabs")


def _manifest() -> dict[str, Any]:
    return json.loads(_MANIFEST.read_text())


def _bse() -> Any:
    try:
        import basis_set_exchange
    except ImportError as error:
        raise ImportError(
            "CH3Br/CH3I F12 bases take their redistributable records from the "
            "basis_set_exchange package; install it with "
            "`pip install 'generativeqc[bse]'`"
        ) from error
    return basis_set_exchange


def _key(halogen: str, cardinal: int) -> tuple[str, int]:
    if halogen not in ("Br", "I"):
        raise ValueError("Choose Br or I")
    cardinal = checked_integer(cardinal, "methyl-halide cardinal", low=2, high=3)
    return f"{halogen.lower()}-{'d' if cardinal == 2 else 't'}", cardinal


def required_files(halogen: str, cardinal: int | None = None) -> tuple[str, ...]:
    """Name the user-supplied files for one halogen and optionally one cardinal.

    Without a cardinal the list covers both selections and the geometry.
    """
    manifest = _manifest()
    keys = [_key(halogen, n)[0] for n in ((2, 3) if cardinal is None else (cardinal,))]
    names = {
        record["file"]
        for key in keys
        for role in manifest["series"][key]["roles"].values()
        for element in role["elements"]
        for record in element["records"]
        if "file" in record
    }
    if cardinal is None:
        names.add(manifest["geometries"][halogen])
    return tuple(sorted(names))


def _obtain(name: str, manifest: dict[str, Any]) -> str:
    entry = manifest["files"][name]
    source = manifest["sources"][entry["source"]]
    url = entry.get("url", source["url"])
    request = source.get("request")
    detail = f"{request} {url}" if request else url
    if "query" in entry:
        detail += f" with {json.dumps(entry['query'], sort_keys=True)}"
    return (
        f"{name}: {entry['description']}; obtain it from {detail} "
        f"({source['procedure']}). Expected content SHA-256 "
        f"{entry['content_sha256']}. See {GUIDE}."
    )


def external_data_directory(
    data_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve ``<data_dir>/f12_methyl_halide`` from the argument or environment."""
    root = data_dir if data_dir is not None else os.environ.get(DATA_DIR_ENV)
    if root is None or not str(root):
        raise FileNotFoundError(
            "CH3Br/CH3I F12 basis sets and geometries are not distributed with "
            f"GenerativeQC. Pass data_dir= or set {DATA_DIR_ENV} to a directory "
            f"whose {SUBDIRECTORY}/ subdirectory holds the files listed in {GUIDE}."
        )
    folder = Path(root) / SUBDIRECTORY
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} does not exist. Create it and place the files listed in "
            f"{GUIDE} there."
        )
    return folder


def _checked_file(folder: Path, name: str, manifest: dict[str, Any]) -> Path:
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}. {_obtain(name, manifest)}")
    return path


def _read_input(path: Path, name: str, manifest: dict[str, Any]) -> str:
    try:
        raw = read_local_bytes(path)
    except ValueError as error:
        raise ValueError(f"{error}. {_obtain(name, manifest)}") from error
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{path} is not UTF-8 text. {_obtain(name, manifest)}"
        ) from error


def _file_record(folder: Path, name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    entry = manifest["files"][name]
    path = _checked_file(folder, name, manifest)
    text = _read_input(path, name, manifest)
    try:
        data = _bse().readers.read_formatted_basis_str(text, "nwchem")
        records = bse_element_records(data)
    except (KeyError, ValueError, TypeError, RuntimeError) as error:
        raise ValueError(
            f"{path} is not the recorded record. {_obtain(name, manifest)}"
        ) from error
    z = entry["atomic_number"]
    if set(records) != {z}:
        raise ValueError(
            f"{path} holds elements Z={sorted(records)}; the recorded record is "
            f"Z={z} alone, so other elements are refused. {_obtain(name, manifest)}"
        )
    found = element_content_sha256(records[z], "spherical")
    if found != entry["content_sha256"]:
        raise ValueError(
            f"{path} has content SHA-256 {found}, not the recorded record; "
            f"refusing to substitute another basis set. {_obtain(name, manifest)}"
        )
    return data


def _standard_order(data: dict[str, Any]) -> dict[str, Any]:
    bse = _bse()
    data = bse.manip.uncontract_general(data, True)
    data = bse.manip.uncontract_spdf(data, 0, True)
    return bse.sort.sort_basis(data, True)


def _assemble(
    spec: dict[str, Any], folder: Path, manifest: dict[str, Any]
) -> ElementBasis:
    bse = _bse()
    z = spec["atomic_number"]
    data = None
    for record in spec["records"]:
        if "file" in record:
            part = _file_record(folder, record["file"], manifest)
        else:
            part = bse.get_basis(record["bse"], elements=[z], header=False)
        if data is None:
            data = part
        else:
            element = data["elements"][str(z)]
            shells = (
                element["electron_shells"] + part["elements"][str(z)]["electron_shells"]
            )
            data = {
                **data,
                "elements": {
                    **data["elements"],
                    str(z): {**element, "electron_shells": shells},
                },
            }
    if spec["layout"] == "bse-standard":
        data = _standard_order(data)
    element = bse_element_records(data)[z]
    shells = element.shells
    if spec["layout"] == "sorted-by-l":
        shells = tuple(sorted(shells, key=lambda shell: shell.angular_momentum))
    if spec["groups"] == "zero":
        shells = tuple(replace(shell, source_group=0) for shell in shells)
    elif spec["groups"] == "position":
        shells = tuple(
            replace(shell, source_group=index) for index, shell in enumerate(shells)
        )
    element = replace(element, shells=shells)
    if "ecp" in spec:
        ecp = bse.get_basis(spec["ecp"]["bse"], elements=[z], header=False)
        ecp = ecp["elements"][str(z)]
        potentials = ecp["ecp_potentials"]
        if spec["ecp"]["text"] == "float-repr":
            text = json.dumps(
                [
                    {
                        "ecp_type": term["ecp_type"],
                        "angular_momentum": term["angular_momentum"],
                        "r_exponents": term["r_exponents"],
                        "gaussian_exponents": [
                            repr(float(x)) for x in term["gaussian_exponents"]
                        ],
                        "coefficients": [
                            [repr(float(c)) for c in row]
                            for row in term["coefficients"]
                        ],
                    }
                    for term in potentials
                ]
            )
        else:
            text = json.dumps(potentials, sort_keys=True, separators=(",", ":"))
        element = replace(
            element, ecp_core_electrons=ecp["ecp_electrons"], ecp_data=text
        )
    found = element_content_sha256(element, "spherical")
    if found != spec["content_sha256"]:
        sources = ", ".join(
            record.get("file") or f"BSE {record['bse']}" for record in spec["records"]
        )
        if "ecp" in spec:
            sources += f", BSE {spec['ecp']['bse']} ECP"
        raise ValueError(
            f"assembled Z={z} record from {sources} has content SHA-256 {found}, "
            f"not {spec['content_sha256']}. The user files passed their own checks, "
            f"so basis_set_exchange {bse.get_version()} differs from the verified "
            f"version(s) {', '.join(manifest['bse']['verified_versions'])}; "
            "refusing to substitute another basis set."
        )
    return element


def methyl_halide_basis(
    halogen: str,
    cardinal: int,
    *,
    data_dir: str | os.PathLike[str] | None = None,
) -> FocalBasis:
    """Select the D (2) or T (3) C/H/Br or C/H/I orbital and all fitting roles.

    ``data_dir`` (or ``GENERATIVEQC_EXTERNAL_DATA_DIR``) names the directory
    whose ``f12_methyl_halide`` subdirectory holds the user-supplied files.
    """
    key, cardinal = _key(halogen, cardinal)
    manifest = _manifest()
    entry = manifest["series"][key]
    folder = external_data_directory(data_dir)
    missing = [
        name
        for name in required_files(halogen, cardinal)
        if not (folder / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{folder} lacks {len(missing)} file(s) for the {key} selection:\n"
            + "\n".join(_obtain(name, manifest) for name in missing)
        )
    roles = {}
    for role in _ROLES:
        spec = entry["roles"][role]
        basis = BasisSet(
            role,
            tuple(_assemble(item, folder, manifest) for item in spec["elements"]),
            BasisProvenance(**spec["provenance"]),
            "spherical",
        )
        if basis.identity != spec["identity"]:
            raise ValueError(
                f"the {key} {role} records match their content hashes but the "
                "assembled basis differs from the recorded identity"
            )
        roles[role] = basis
    basis = FocalBasis(
        cardinal,
        manifest["series_id"],
        roles["orbital"],
        jk_fit=roles["jkfit"],
        mp2_fit=roles["ri"],
        f12_fit=roles["ri"],
        cabs=roles["cabs"],
        beta=manifest["beta_bohr_inverse_by_cardinal"][str(cardinal)],
    )
    if basis.basis_id != entry["basis_id"]:
        raise ValueError(
            f"the assembled {key} basis differs from its recorded identity"
        )
    return basis


def geometry_content_sha256(
    symbols: Iterable[str], coordinates_angstrom: Iterable[Iterable[float]]
) -> str:
    """Hash element symbols and binary64 angstrom coordinates, not their spelling."""
    return canonical_hash(
        {
            "schema": GEOMETRY_CONTENT_SCHEMA,
            "symbols": list(symbols),
            "coordinates_angstrom": [
                [(float(x) + 0.0).hex() for x in row] for row in coordinates_angstrom
            ],
        }
    )


def methyl_halide_geometry(
    halogen: str, *, data_dir: str | os.PathLike[str] | None = None
) -> tuple[tuple[str, ...], tuple[tuple[float, float, float], ...]]:
    """Return the checked experimental CH₃X geometry as symbols and bohr positions."""
    if halogen not in ("Br", "I"):
        raise ValueError("Choose Br or I")
    manifest = _manifest()
    name = manifest["geometries"][halogen]
    path = _checked_file(external_data_directory(data_dir), name, manifest)
    lines = _read_input(path, name, manifest).splitlines()
    try:
        count = int(lines[0])
        rows = [line.split() for line in lines[2 : 2 + count]]
        symbols = tuple(row[0] for row in rows)
        angstrom = tuple(tuple(float(x) + 0.0 for x in row[1:]) for row in rows)
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"{path} is not an XYZ file. {_obtain(name, manifest)}"
        ) from error
    if (
        len(rows) != count
        or any(len(row) != 4 for row in rows)
        or any(line.strip() for line in lines[2 + count :])
    ):
        raise ValueError(f"{path} is not an XYZ file. {_obtain(name, manifest)}")
    found = geometry_content_sha256(symbols, angstrom)
    if found != manifest["files"][name]["content_sha256"]:
        raise ValueError(
            f"{path} has geometry content SHA-256 {found}, not the recorded "
            f"geometry. {_obtain(name, manifest)}"
        )
    return symbols, tuple(
        (x / BOHR_ANGSTROM, y / BOHR_ANGSTROM, z / BOHR_ANGSTROM)
        for x, y, z in angstrom
    )


def methyl_halide_plan(
    halogen: str,
    cardinal: int,
    *,
    root: str = "rhf",
    coordinates_bohr: Iterable[Iterable[float]] | None = None,
    backend: str = "cuda",
    f12_doubles: bool = False,
    conventional_triples: bool = False,
    data_dir: str | os.PathLike[str] | None = None,
) -> FocalPlan:
    """Plan CH₃X neutral RHF or vertical cation UHF at the recorded or caller geometry.

    Positions are in bohr, ordered C, halogen, H, H, H. The optional high-order
    terms request all-active CCSD-F12b minus MP2-F12 and conventional (T) in
    this exact cardinal basis. Execution uses the supplied public provider.
    """
    if root not in ("rhf", "uhf"):
        raise ValueError("Choose rhf (neutral singlet) or uhf (cation doublet)")
    basis = methyl_halide_basis(halogen, cardinal, data_dir=data_dir)
    if coordinates_bohr is None:
        coords = methyl_halide_geometry(halogen, data_dir=data_dir)[1]
    else:
        coords = coordinates_bohr
    atoms = tuple(zip(("C", halogen, "H", "H", "H"), coords, strict=True))
    series = FocalSeries(basis)
    charge = int(root == "uhf")
    return plan_singlepoint(
        atoms,
        series,
        series,
        None,
        root_id=root,
        charge=charge,
        multiplicity=charge + 1,
        correlation_space_id="all-active",
        backend=backend,
        protocol_id=f"CH3{halogen}/ha-PP-F12/split-fit/all-active",
        f12_doubles=basis if f12_doubles else None,
        triples=basis if conventional_triples else None,
    )
