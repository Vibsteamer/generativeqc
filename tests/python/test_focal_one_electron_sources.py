"""Geometry-first OBS/OptRI S, Hcore and nuclear derivatives of the OH radical."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalRequest,
    FocalSeries,
    one_electron_sources,
    plan_singlepoint,
)
from generativeqc.profiles import canonical_hash

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.fixture(scope="module")
def molecular_library(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    if path := os.environ.get("GENERATIVEQC_F12_ONE_ELECTRON_HOST_LIBRARY"):
        return Path(path)
    root = Path(__file__).resolve().parents[2]
    sources = [
        root / rel
        for rel in (
            "src/molecule/basis.cpp",
            "benchmarks/f12_shell_projection.cpp",
            "benchmarks/f12_cabs_h_one_electron.cu",
        )
    ]
    return required_native_cxx.build_shared(
        sources,
        tmp_path_factory.mktemp("focal-one-electron") / "libone.so",
        compile_args=[
            "-x",
            "c++",
            "-std=c++20",
            "-O2",
            "-DGQC_F12_CABS_I_ONE_ELECTRON",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
        ],
    )


@pytest.fixture(scope="module")
def hydroxyl() -> tuple[FocalRequest, dict, dict]:
    gto = pytest.importorskip("pyscf.gto")

    def basis(name: str) -> tuple[BasisSet, dict]:
        raw = {label: gto.basis.load(name, label) for label in ("O", "H")}
        elements = []
        for z, label in ((8, "O"), (1, "H")):
            shells = []
            for l, *rows in raw[label]:
                exponents, *coefficients = zip(*rows, strict=True)
                shells.append(BasisShell(l, exponents, tuple(coefficients)))
            elements.append(ElementBasis(z, tuple(shells)))
        return BasisSet(
            name,
            tuple(elements),
            BasisProvenance(
                "PySCF locally resolved basis",
                name,
                "external data for testing",
                canonical_hash(raw),
            ),
            representation="spherical",
        ), raw

    orbital, obs = basis("cc-pvdz-f12")
    opt_ri, cabs = basis("cc-pvdz-f12-optri")
    b = FocalBasis(
        2,
        "D-F12",
        orbital,
        jk_fit="sto-3g",
        mp2_fit="sto-3g",
        f12_fit="sto-3g",
        cabs=opt_ri,
        beta=0.9,
    )
    atoms = [("O", (0.12, -0.28, 0.21)), ("H", (1.75, 0.26, -0.36))]
    plan = plan_singlepoint(
        atoms,
        FocalSeries(b),
        FocalSeries(b),
        b,
        root_id="uhf",
        correlation_space_id="all-active",
        multiplicity=2,
        backend="cpu_reference",
    )
    return next(req for req, _ in plan.steps if req.role == "hf_cabs"), obs, cabs


def test_oh_physical_d_f12_obs_optri(molecular_library: Path, hydroxyl: tuple) -> None:
    from pyscf import gto, scf

    req, obs, opt = hydroxyl
    xyz = np.array([a.position for a in req.atoms])

    def independent(coords: np.ndarray) -> tuple:
        # Zero-charge centers give independent libcint shell ownership and explicit
        # OBS-first order. Each OptRI center follows its physical atom at every displacement.
        labels = ("O", "H", "ghost-O", "ghost-H")
        mol = gto.M(
            atom=list(zip(labels, np.vstack([coords, coords]), strict=True)),
            basis={**obs, "ghost-O": opt["O"], "ghost-H": opt["H"]},
            unit="Bohr",
            spin=1,
            charge=0,
            verbose=0,
            cart=False,
        )
        return mol, np.stack(
            (mol.intor("int1e_ovlp"), mol.intor("int1e_kin") + mol.intor("int1e_nuc"))
        )

    out = one_electron_sources(req, library=molecular_library)
    assert out.executed_backend == "cpu_reference" and out.surface_id == req.surface_id
    assert out.basis_id == req.basis.basis_id and out.native_workspace_bytes > 0
    mol, values = independent(xyz)
    aos = mol.aoslice_by_atom()[:, 2:]
    assert out.obs_ao_count == aos[1, 1]
    np.testing.assert_allclose((out.overlap, out.hcore), values, atol=3e-10, rtol=0)
    only = one_electron_sources(req, library=molecular_library, derivatives=False)
    assert only.d_hcore is None and only.d_overlap is None
    np.testing.assert_allclose((only.overlap, only.hcore), values, atol=3e-10, rtol=0)
    ip = -mol.intor("int1e_ipovlp", comp=3)
    gen_h = scf.RHF(mol).nuc_grad_method().hcore_generator(mol)
    assert out.d_overlap is not None and out.d_hcore is not None
    for atom in range(2):
        ds = np.zeros_like(ip)
        for first, last in (aos[atom], aos[atom + 2]):
            ds[:, first:last, :] += ip[:, first:last, :]
            ds[:, :, first:last] += ip[:, first:last, :].swapaxes(1, 2)
        np.testing.assert_allclose(out.d_overlap[atom], ds, atol=3e-10, rtol=0)
        np.testing.assert_allclose(
            out.d_hcore[atom], gen_h(atom) + gen_h(atom + 2), atol=3e-9, rtol=0
        )
        for axis in range(3):
            for h in (4e-4, 2e-4):
                shift = np.zeros_like(xyz)
                shift[atom, axis] = h
                fd = (
                    8 * (independent(xyz + shift)[1] - independent(xyz - shift)[1])
                    - (
                        independent(xyz + 2 * shift)[1]
                        - independent(xyz - 2 * shift)[1]
                    )
                ) / (12 * h)
                np.testing.assert_allclose(
                    (out.d_overlap[atom, axis], out.d_hcore[atom, axis]),
                    fd,
                    atol=2e-8,
                    rtol=0,
                )


def test_backend_and_budget_must_be_honored(
    molecular_library: Path, hydroxyl: tuple
) -> None:
    request = hydroxyl[0]
    with pytest.raises(RuntimeError, match="cuda one-electron kernel"):
        one_electron_sources(
            replace(request, backend="cuda"), library=molecular_library
        )
    with pytest.raises(MemoryError, match="budget"):
        one_electron_sources(request, library=molecular_library, budget_mib=1)
