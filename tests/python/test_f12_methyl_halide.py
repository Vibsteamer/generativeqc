"""CH3Br/CH3I F12 selections assembled from hash-checked, user-supplied inputs.

Required CI runs the manifest, refusal and synthetic-data tests. The tier that
reproduces the recorded selections needs the restricted inputs described in
docs/user/restricted_reference_data.md and skips when they are absent.
"""

import ctypes
import hashlib
import json
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest
from _bse_basis_support import br_ecp10mdf, jk_fit_basis, pp_orbital_basis
from generativeqc import Atom, Calculator, _native, import_bse, import_nwchem
from generativeqc._f12_frozen_core import DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL
from generativeqc.basis import (
    MAX_FILE_BYTES,
    BasisProvenance,
    BasisSet,
    element_content_sha256,
)
from generativeqc.ecp import fitted_hf_ecp_domain, resolve_ecp
from generativeqc.experimental import f12_methyl_halide as mh
from generativeqc.experimental.f12_focal import FocalBasis

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "python/generativeqc/data/f12_methyl_halide_external.json"
SHA256 = re.compile("[0-9a-f]{64}")
ROLES = ("hf_cabs", "mp2_f12_correlation", "ccsd_f12b_minus_mp2_f12", "triples")


def _bse_reader() -> None:
    pytest.importorskip(
        "basis_set_exchange",
        reason="NWChem and BSE records need the reference-test extra",
    )


def _walk(value: object) -> list[tuple[str, object]]:
    if isinstance(value, dict):
        return [(k, v) for k, v in value.items()] + [
            item for v in value.values() for item in _walk(v)
        ]
    if isinstance(value, list):
        return [item for v in value for item in _walk(v)]
    return []


def test_manifest_is_hash_only_and_names_every_source() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["schema"] == "generativeqc.f12-methyl-halide-external/v1"
    keys = {key for key, _ in _walk(manifest)}
    assert not keys & {
        "exponents",
        "coefficients",
        "coordinates_bohr",
        "coordinates_angstrom",
    }
    assert not any(
        isinstance(v, list) and any(isinstance(x, float) for x in v)
        for _, v in _walk(manifest)
    )
    for source in manifest["sources"].values():
        assert source["url"] and source["procedure"] and source["version"]
    for name, entry in manifest["files"].items():
        assert SHA256.fullmatch(entry["file_sha256"]), name
        assert SHA256.fullmatch(entry["content_sha256"]), name
        assert entry["source"] in manifest["sources"], name
        assert entry["citation"] and entry["retrieved"] and entry["description"]
        if entry["source"].startswith("ccrepo-"):
            assert entry["query"]["program"] == "NWChem"
    assert set(manifest["series"]) == {"br-d", "br-t", "i-d", "i-t"}
    for key, entry in manifest["series"].items():
        z = 35 if entry["halogen"] == "Br" else 53
        assert SHA256.fullmatch(entry["basis_id"])
        assert set(entry["roles"]) == {"orbital", "jkfit", "ri", "cabs"}
        for role in entry["roles"].values():
            assert SHA256.fullmatch(role["identity"])
            assert [e["atomic_number"] for e in role["elements"]] == [1, 6, z]
            for spec in role["elements"]:
                assert SHA256.fullmatch(spec["content_sha256"])
                for record in spec["records"]:
                    assert record.get("file") in manifest["files"] or "bse" in record
        halogen = entry["roles"]["orbital"]["elements"][2]
        assert halogen["ecp"]["bse"] == f"cc-pV{key[-1].upper()}Z-PP"
    for halogen in ("Br", "I"):
        names = mh.required_files(halogen)
        assert len(names) == 11 and manifest["geometries"][halogen] in names
        assert len(mh.required_files(halogen, 2)) == 5
        assert set(names) <= set(manifest["files"])


def test_bse_records_reproduce_recorded_content(tmp_path: Path) -> None:
    bse = pytest.importorskip(
        "basis_set_exchange", reason="BSE records need the reference-test extra"
    )
    manifest = json.loads(MANIFEST.read_text())
    checked = 0
    for key, entry in manifest["series"].items():
        for role in entry["roles"].values():
            for spec in role["elements"]:
                if "ecp" in spec or any("file" in r for r in spec["records"]):
                    continue
                ((_, name),) = (item for r in spec["records"] for item in r.items())
                z = spec["atomic_number"]
                path = tmp_path / f"{name}-{z}.json"
                path.write_text(
                    json.dumps(bse.get_basis(name, elements=[z], header=False))
                )
                element = import_bse(
                    path,
                    source="MolSSI Basis Set Exchange",
                    source_version=bse.get_version(),
                    license="BSD-3-Clause",
                    representation="spherical",
                ).by_element[z]
                if spec["layout"] == "sorted-by-l":
                    shells = sorted(element.shells, key=lambda s: s.angular_momentum)
                    element = replace(element, shells=tuple(shells))
                else:
                    assert spec["layout"] == "native"
                found = element_content_sha256(element, "spherical")
                assert found == spec["content_sha256"], (key, name, z)
                checked += 1
    assert checked == 28


def test_missing_inputs_are_refused_with_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(mh.DATA_DIR_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match=mh.DATA_DIR_ENV):
        mh.methyl_halide_basis("Br", 2)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mh.methyl_halide_basis("Br", 2, data_dir=tmp_path)
    (tmp_path / mh.SUBDIRECTORY).mkdir()
    with pytest.raises(FileNotFoundError, match="lacks 5 file") as missing:
        mh.methyl_halide_plan("I", 3, data_dir=tmp_path)
    manifest = json.loads(MANIFEST.read_text())
    entry = manifest["files"]["I_aug-cc-pVTZ-PP-F12_MP2FIT.nwchem"]
    message = str(missing.value)
    assert entry["content_sha256"] in message
    assert (
        "HTTP POST form http://www.grant-hill.group.shef.ac.uk/ccrepo/iodine/ibasis.php"
        in message
    )
    assert '"basis": "aug-cc-pVTZ-PP-F12_MP2FIT"' in message
    assert mh.GUIDE in message
    with pytest.raises(FileNotFoundError, match=r"CH3I\.xyz.*cccbdb"):
        mh.methyl_halide_geometry("I", data_dir=tmp_path)


def test_wrong_inputs_are_refused_not_substituted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bse_reader()
    folder = tmp_path / mh.SUBDIRECTORY
    folder.mkdir()
    for name in mh.required_files("Br", 2):
        symbol = name.split("_", 1)[0]
        (folder / name).write_text(
            f'BASIS "ao basis" SPHERICAL PRINT\n{symbol} S\n  0.5 1.0\nEND\n'
        )
    monkeypatch.setenv(mh.DATA_DIR_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="refusing to substitute") as wrong:
        mh.methyl_halide_basis("Br", 2)
    assert "C_aug-cc-pVDZ-F12_diffuse.nwchem" in str(wrong.value)
    (folder / "C_aug-cc-pVDZ-F12_diffuse.nwchem").write_text("not a basis\n")
    with pytest.raises(ValueError, match="not the recorded record"):
        mh.methyl_halide_basis("Br", 2)
    (folder / "CH3Br.xyz").write_text("2\n\nC 0 0 0\n")
    with pytest.raises(ValueError, match="not an XYZ file"):
        mh.methyl_halide_geometry("Br")
    (folder / "CH3Br.xyz").write_text("1\nwrong molecule\nC 0 0 0\n")
    with pytest.raises(ValueError, match="geometry content SHA-256"):
        mh.methyl_halide_geometry("Br")


def test_geometry_hash_ignores_spelling_only() -> None:
    symbols = ("C", "Br")
    a = mh.geometry_content_sha256(symbols, [[0, 0, -0.0], [0, 0, 1.9]])
    assert a == mh.geometry_content_sha256(
        symbols, [["0.000", "0", "0"], ["0", 0, "1.90"]]
    )
    assert a != mh.geometry_content_sha256(symbols, [[0, 0, 0], [0, 0, 1.91]])
    assert a != mh.geometry_content_sha256(("C", "I"), [[0, 0, 0], [0, 0, 1.9]])


def test_user_guide_lists_every_external_file_and_hash() -> None:
    guide = (ROOT / mh.GUIDE).read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        row = f"| `{name}` | `{entry['file_sha256']}` | `{entry['content_sha256']}` |"
        assert row in guide


def _nwchem(symbol: str, shells: dict[str, tuple[float, ...]]) -> str:
    lines = ['BASIS "ao basis" SPHERICAL PRINT']
    for label, exponents in shells.items():
        for exponent in exponents:
            lines += [f"{symbol} {label}", f"  {exponent!r} 1.0"]
    return "\n".join([*lines, "END", ""])


# Small even-tempered shells per role; jkfit and ri must differ.
SYNTHETIC = {
    "orbital": {"S": (3.1, 0.62, 0.15), "P": (1.4, 0.33), "D": (0.7,)},
    "jkfit": {"S": (6.0, 1.5, 0.4), "P": (2.2, 0.6), "D": (1.2,), "F": (0.9,)},
    "ri": {"S": (5.0, 1.2, 0.3), "P": (1.8, 0.5), "D": (1.1,), "F": (0.8,)},
    "cabs": {"S": (8.0, 2.0), "P": (3.0, 0.9), "D": (1.6,)},
}
# Project test geometry (angstrom): C-Br 1.95, C-H 1.09, tetrahedral angles.
SYNTHETIC_XYZ = """5
synthetic methyl bromide
C 0.0 0.0 0.0
Br 0.0 0.0 1.95
H 1.0277 0.0 -0.3633
H -0.51385 0.89002 -0.3633
H -0.51385 -0.89002 -0.3633
"""


def _synthetic_inputs(tmp_path: Path) -> tuple[dict, FocalBasis]:
    folder = tmp_path / mh.SUBDIRECTORY
    folder.mkdir()
    files, roles, expected = {}, {}, {}
    ecp = br_ecp10mdf()
    for role, shells in SYNTHETIC.items():
        records, specs = [], []
        for z, symbol in ((1, "H"), (6, "C"), (35, "Br")):
            name = f"synthetic_{role}_{symbol}.nwchem"
            (folder / name).write_text(_nwchem(symbol, shells))
            element = import_nwchem(
                folder / name,
                source="synthetic",
                source_version="test",
                license="test",
                representation="spherical",
            ).by_element[z]
            files[name] = {
                "description": f"synthetic {role} {symbol}",
                "kind": "basis",
                "format": "nwchem",
                "atomic_number": z,
                "source": "synthetic",
                "retrieved": "test",
                "file_sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest(),
                "content_sha256": element_content_sha256(element, "spherical"),
                "citation": "synthetic test data",
            }
            spec = {"records": [{"file": name}], "layout": "native", "groups": "source"}
            if role == "orbital" and z == 35:
                spec["ecp"] = {"bse": "cc-pVDZ-PP", "text": "bse"}
                element = replace(
                    element,
                    ecp_core_electrons=ecp.ecp_core_electrons,
                    ecp_data=ecp.ecp_data,
                )
            records.append(element)
            spec["content_sha256"] = element_content_sha256(element, "spherical")
            specs.append({"atomic_number": z, **spec})
        provenance = {
            "source": "synthetic",
            "version": "D",
            "license": "test",
            "checksum": "0" * 64,
        }
        expected[role] = BasisSet(
            role, tuple(records), BasisProvenance(**provenance), "spherical"
        )
        roles[role] = {
            "identity": expected[role].identity,
            "provenance": provenance,
            "elements": specs,
        }
    (folder / "CH3Br.xyz").write_text(SYNTHETIC_XYZ)
    rows = [line.split() for line in SYNTHETIC_XYZ.splitlines()[2:]]
    files["CH3Br.xyz"] = {
        "description": "synthetic CH3Br geometry",
        "kind": "geometry",
        "format": "xyz-angstrom",
        "source": "synthetic",
        "retrieved": "test",
        "file_sha256": hashlib.sha256(SYNTHETIC_XYZ.encode()).hexdigest(),
        "content_sha256": mh.geometry_content_sha256(
            [r[0] for r in rows], [r[1:] for r in rows]
        ),
        "citation": "synthetic test data",
    }
    basis = FocalBasis(
        2,
        "synthetic-methyl-bromide",
        expected["orbital"],
        jk_fit=expected["jkfit"],
        mp2_fit=expected["ri"],
        f12_fit=expected["ri"],
        cabs=expected["cabs"],
        beta=0.9,
    )
    manifest = {
        "schema": "generativeqc.f12-methyl-halide-external/v1",
        "series_id": "synthetic-methyl-bromide",
        "beta_bohr_inverse_by_cardinal": {"2": 0.9, "3": 1.0},
        "bse": {"verified_versions": ["test"]},
        "sources": {
            "synthetic": {
                "url": "file:synthetic",
                "procedure": "written by the test",
                "version": "test",
            }
        },
        "files": files,
        "geometries": {"Br": "CH3Br.xyz"},
        "series": {
            "br-d": {
                "halogen": "Br",
                "cardinal": 2,
                "basis_id": basis.basis_id,
                "roles": roles,
            }
        },
    }
    return manifest, basis


def test_synthetic_inputs_assemble_verify_and_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bse_reader()
    manifest, expected = _synthetic_inputs(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(mh, "_MANIFEST", path)

    basis = mh.methyl_halide_basis("Br", 2, data_dir=tmp_path)
    assert basis.basis_id == expected.basis_id
    assert basis.beta == pytest.approx(DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL[2])
    assert basis.orbital == expected.orbital
    assert basis.orbital.by_element[35].ecp_core_electrons == 10
    assert basis.mp2_fit.identity == basis.f12_fit.identity
    assert basis.jk_fit.identity != basis.f12_fit.identity

    symbols, bohr = mh.methyl_halide_geometry("Br", data_dir=tmp_path)
    assert symbols == ("C", "Br", "H", "H", "H")
    assert bohr[1] == (0.0, 0.0, 1.95 / mh.BOHR_ANGSTROM)
    for root, charge, mult in [("rhf", 0, 1), ("uhf", 1, 2)]:
        plan = mh.methyl_halide_plan(
            "Br",
            2,
            root=root,
            f12_doubles=True,
            conventional_triples=True,
            data_dir=tmp_path,
        )
        assert [q.role for q, _ in plan.steps] == list(ROLES)
        assert all(
            (q.basis.basis_id, q.root_id, q.charge, q.multiplicity)
            == (basis.basis_id, root, charge, mult)
            for q, _ in plan.steps
        )
    fixed = mh.methyl_halide_plan("Br", 2, data_dir=tmp_path).steps[0][0]
    coords = [list(atom.position) for atom in fixed.atoms]
    coords[1][2] += 0.0004
    shifted = mh.methyl_halide_plan(
        "Br", 2, coordinates_bohr=coords, data_dir=tmp_path
    ).steps[0][0]
    assert shifted.atoms[1].position[2] == coords[1][2]
    assert fixed.surface_id != shifted.surface_id
    assert fixed.basis.basis_id == shifted.basis.basis_id

    # A changed exponent, role identity or basis identity is refused.
    changed = tmp_path / mh.SUBDIRECTORY / "synthetic_ri_Br.nwchem"
    original = changed.read_text()
    changed.write_text(original.replace("0.3 1.0", "0.31 1.0"))
    with pytest.raises(ValueError, match="synthetic_ri_Br.nwchem has content"):
        mh.methyl_halide_basis("Br", 2, data_dir=tmp_path)
    changed.write_text(original)
    for field, value, message in (
        ("identity", "0" * 64, "assembled basis differs"),
        ("basis_id", "0" * 64, "differs from its recorded identity"),
    ):
        broken = json.loads(json.dumps(manifest))
        target = broken["series"]["br-d"]
        (target["roles"]["cabs"] if field == "identity" else target)[field] = value
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match=message):
            mh.methyl_halide_basis("Br", 2, data_dir=tmp_path)


def _use_synthetic_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FocalBasis:
    _bse_reader()
    manifest, expected = _synthetic_inputs(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(mh, "_MANIFEST", path)
    return expected


def test_basis_files_with_unrecorded_elements_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _use_synthetic_inputs(tmp_path, monkeypatch)
    changed = tmp_path / mh.SUBDIRECTORY / "synthetic_ri_Br.nwchem"
    original = changed.read_text()
    # The added hydrogen shell leaves the bromine record and its hash unchanged.
    for text, found in (
        (original.replace("END\n", "H S\n  0.5 1.0\nEND\n"), r"\[1, 35\]"),
        (_nwchem("H", SYNTHETIC["ri"]), r"\[1\]"),
    ):
        changed.write_text(text)
        with pytest.raises(
            ValueError, match=rf"synthetic_ri_Br\.nwchem holds elements Z={found}"
        ):
            mh.methyl_halide_basis("Br", 2, data_dir=tmp_path)
    changed.write_text(original)
    assert (
        mh.methyl_halide_basis("Br", 2, data_dir=tmp_path).basis_id == expected.basis_id
    )


def test_inputs_above_the_size_limit_are_refused_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_synthetic_inputs(tmp_path, monkeypatch)
    folder = tmp_path / mh.SUBDIRECTORY
    loaders = {
        "synthetic_ri_Br.nwchem": lambda: mh.methyl_halide_basis(
            "Br", 2, data_dir=tmp_path
        ),
        "CH3Br.xyz": lambda: mh.methyl_halide_geometry("Br", data_dir=tmp_path),
    }
    for name, load in loaders.items():
        path = folder / name
        original = path.read_bytes()
        # Trailing blanks parse to the same records, so only the size refuses them.
        path.write_bytes(original.ljust(MAX_FILE_BYTES + 1))
        with pytest.raises(
            ValueError, match=rf"{re.escape(name)} exceeds the 64 MiB input limit"
        ):
            load()
        path.write_bytes(original)
        load()


def _project_methyl_halide(element: str) -> tuple[Atom, ...]:
    halogen = 1.95 if element == "Br" else 2.15
    rows = [line.split() for line in SYNTHETIC_XYZ.splitlines()[2:]]
    rows[1] = [element, "0.0", "0.0", str(halogen)]
    return tuple(
        Atom.from_value((s, tuple(float(x) / mh.BOHR_ANGSTROM for x in xyz)))
        for s, *xyz in rows
    )


def _check_fitted_ecp_template(
    basis: BasisSet,
    jk_fit: BasisSet,
    atoms: tuple[Atom, ...],
    element: str,
    root: str,
    charge: int,
    multiplicity: int,
) -> None:
    assert fitted_hf_ecp_domain(root, basis, atoms, charge, multiplicity)
    assert not fitted_hf_ecp_domain(root, basis, atoms, charge + 2, multiplicity)
    assert not fitted_hf_ecp_domain(root, basis, atoms[:-1], charge, multiplicity)
    cores, terms = resolve_ecp(basis, atoms)
    assert cores == (0, 10 if element == "Br" else 28, 0, 0, 0)
    assert all(term[0] == 1 for term in terms)
    calc = Calculator(
        method=root,
        basis=basis,
        auxiliary_basis=jk_fit,
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


@pytest.mark.parametrize("element", ("Br", "I"))
@pytest.mark.parametrize("cardinal", (2, 3))
@pytest.mark.parametrize("root,charge,multiplicity", (("rhf", 0, 1), ("uhf", 1, 2)))
def test_methyl_halide_fitted_ecp_template_with_redistributable_bases(
    element: str, cardinal: int, root: str, charge: int, multiplicity: int
) -> None:
    z = 35 if element == "Br" else 53
    _check_fitted_ecp_template(
        pp_orbital_basis((1, 6, z), cardinal),
        jk_fit_basis((1, 6, z)),
        _project_methyl_halide(element),
        element,
        root,
        charge,
        multiplicity,
    )


def _external_data(halogen: str) -> Path:
    _bse_reader()
    root = os.environ.get(mh.DATA_DIR_ENV)
    if not root:
        pytest.skip(
            "restricted CH3Br/CH3I inputs are user-supplied; set "
            f"{mh.DATA_DIR_ENV} as described in {mh.GUIDE} to run this tier"
        )
    folder = Path(root) / mh.SUBDIRECTORY
    missing = [n for n in mh.required_files(halogen) if not (folder / n).is_file()]
    if missing:
        pytest.skip(f"{folder} lacks {', '.join(missing)} (see {mh.GUIDE})")
    return Path(root)


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
    data = _external_data(element)
    z = 35 if element == "Br" else 53
    basis = mh.methyl_halide_basis(element, cardinal, data_dir=data)
    assert basis.beta == pytest.approx(
        DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL[cardinal]
    )
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
        plan = mh.methyl_halide_plan(
            element,
            cardinal,
            root=root,
            f12_doubles=True,
            conventional_triples=True,
            data_dir=data,
        )
        assert [q.role for q, _ in plan.steps] == list(ROLES)
        assert all(
            (q.basis.basis_id, q.root_id, q.charge, q.multiplicity)
            == (basis.basis_id, root, charge, mult)
            for q, _ in plan.steps
        )


def test_displaced_organic_geometry() -> None:
    data = _external_data("Br")
    fixed = mh.methyl_halide_plan("Br", 2, data_dir=data).steps[0][0]
    coords = [list(atom.position) for atom in fixed.atoms]
    coords[1][2] += 0.0004
    shifted = mh.methyl_halide_plan(
        "Br", 2, coordinates_bohr=coords, data_dir=data
    ).steps[0][0]
    assert shifted.atoms[1].position[2] == coords[1][2]
    assert fixed.surface_id != shifted.surface_id
    assert fixed.basis.basis_id == shifted.basis.basis_id


def test_actual_files_with_an_added_element_or_padding_are_refused(
    tmp_path: Path,
) -> None:
    data = _external_data("Br") / mh.SUBDIRECTORY
    folder = tmp_path / mh.SUBDIRECTORY
    folder.mkdir()
    for name in mh.required_files("Br"):
        (folder / name).write_bytes((data / name).read_bytes())
    name = "Br_aug-cc-pVDZ-PP-F12.nwchem"
    original = (folder / name).read_bytes()
    head, tail = original.decode().rsplit("end", 1)
    (folder / name).write_text(f"{head}H s\n0.5 1.0\nend{tail}")
    with pytest.raises(
        ValueError, match=rf"{re.escape(name)} holds elements Z=\[1, 35\]"
    ):
        mh.methyl_halide_basis("Br", 2, data_dir=tmp_path)
    (folder / name).write_bytes(original.ljust(MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds the 64 MiB input limit"):
        mh.methyl_halide_basis("Br", 2, data_dir=tmp_path)
    (folder / name).write_bytes(original)
    copied = mh.methyl_halide_basis("Br", 2, data_dir=tmp_path)
    assert (
        copied.basis_id
        == mh.methyl_halide_basis("Br", 2, data_dir=data.parent).basis_id
    )


@pytest.mark.parametrize("element", ("Br", "I"))
@pytest.mark.parametrize("cardinal", (2, 3))
@pytest.mark.parametrize("root,charge,multiplicity", (("rhf", 0, 1), ("uhf", 1, 2)))
def test_methyl_halide_fitted_ecp_template(
    element: str, cardinal: int, root: str, charge: int, multiplicity: int
) -> None:
    data = _external_data(element)
    q = mh.methyl_halide_plan(element, cardinal, root=root, data_dir=data).steps[0][0]
    _check_fitted_ecp_template(
        q.basis.orbital, q.basis.jk_fit, q.atoms, element, root, charge, multiplicity
    )
