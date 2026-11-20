"""Geometry-first scalar Br and I ECP on three moving centers including complementary i AOs."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import test_focal_one_electron_sources as one_electron_tests
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalRequest,
    one_electron_sources,
)
from generativeqc.profiles import canonical_hash

ecp_library = one_electron_tests.molecular_library

if TYPE_CHECKING:
    from pathlib import Path


def _request(symbol: str, xyz: np.ndarray) -> tuple:
    gto = pytest.importorskip("pyscf.gto")
    core, raw = gto.basis.load_ecp("cc-pvtz-pp", symbol)
    ecp = []
    for l, groups in raw:
        rows = [(p, a, c) for p, entries in enumerate(groups) for a, c in entries] or [
            (2, 1.0, 0.0)
        ]
        ecp.append(
            {
                "ecp_type": "scalar_ecp",
                "angular_momentum": [4 if l == -1 else l],
                "r_exponents": [p for p, _, _ in rows],
                "gaussian_exponents": [str(a) for _, a, _ in rows],
                "coefficients": [[str(c) for _, _, c in rows]],
            }
        )
    z = {"Br": 35, "I": 53}[symbol]
    ao = {
        symbol: [[0, [1.51, 0.6], [0.81, 0.5]], [1, [0.53, 1.0]]],
        "H": [[0, [0.92, 1.0]]],
    }
    opt = {symbol: [[6, [0.9, 1.0]]], "H": [[2, [0.45, 1.0]], [6, [0.78, 1.0]]]}

    def basis(name: str, records: dict, with_ecp: bool = False) -> BasisSet:
        elements = []
        for label, atomic in ((symbol, z), ("H", 1)):
            shells = [
                BasisShell(
                    l, tuple(row[0] for row in rows), (tuple(row[1] for row in rows),)
                )
                for l, *rows in records[label]
            ]
            fitted = with_ecp and atomic == z
            elements.append(
                ElementBasis(
                    atomic,
                    tuple(shells),
                    ecp_core_electrons=core if fitted else 0,
                    ecp_data=json.dumps(ecp) if fitted else None,
                )
            )
        return BasisSet(
            name,
            tuple(elements),
            BasisProvenance(
                "PySCF ECP and test shells",
                "1",
                "test inputs",
                canonical_hash([records, ecp]),
            ),
            representation="spherical",
        )

    orb, comp = basis("orbital", ao, True), basis("complement", opt)
    b = FocalBasis(
        2, "test", orb, jk_fit=orb, mp2_fit=orb, f12_fit=orb, cabs=comp, beta=0.9
    )
    from generativeqc import Atom

    atoms = tuple(
        Atom.from_value((label, row))
        for label, row in zip((symbol, "H", "H"), xyz.tolist(), strict=True)
    )
    req = FocalRequest(
        atoms, 0, 2, "uhf", "all", "physical", b, "hf_cabs", "cpu_reference"
    )

    def oracle(coords: np.ndarray) -> tuple:
        labels = (symbol, "H", "H", "ghost-" + symbol, "ghost-H", "ghost-H")
        mol = gto.M(
            atom=list(zip(labels, np.vstack([coords, coords]), strict=True)),
            basis={**ao, **{"ghost-" + k: v for k, v in opt.items()}},
            ecp={symbol: [core, raw]},
            unit="Bohr",
            spin=1,
            verbose=0,
        )
        # Only the orbital center carries the ECP; its complementary ghost follows the same atom.
        return mol, mol.intor_symmetric("int1e_kin") + mol.intor_symmetric(
            "int1e_nuc"
        ) + mol.intor_symmetric("ECPscalar")

    return req, oracle


@pytest.mark.parametrize("symbol", ["Br", "I"])
def test_direct_semilocal_center_derivatives(symbol: str, ecp_library: Path) -> None:
    from pyscf import scf

    xyz = np.array([[0.13, -0.16, 0.27], [1.23, 0.52, -0.39], [-0.68, 0.81, 1.43]])
    req, oracle = _request(symbol, xyz)
    out = one_electron_sources(req, library=ecp_library, budget_mib=96)
    mol, h = oracle(xyz)
    np.testing.assert_allclose(out.hcore, h, rtol=0, atol=2e-9)
    np.testing.assert_allclose(
        out.overlap, mol.intor_symmetric("int1e_ovlp"), rtol=0, atol=2e-10
    )
    assert out.obs_ao_count == 6 and out.d_hcore is not None
    gen = scf.RHF(mol).nuc_grad_method().hcore_generator(mol)
    for atom in range(3):
        np.testing.assert_allclose(
            out.d_hcore[atom], gen(atom) + gen(atom + 3), rtol=0, atol=6e-9
        )
    np.testing.assert_allclose(out.d_hcore.sum(axis=0), 0, rtol=0, atol=2e-10)
    direction = np.array([[0.14, -0.23, 0.36], [-0.64, 0.77, 0.16], [0.81, 0.09, 0.31]])
    analytic = np.einsum("ax,axij->ij", direction, out.d_hcore)
    for step in (2e-4, 1e-4):
        delta = step * direction
        numerical = (
            8 * (oracle(xyz + delta)[1] - oracle(xyz - delta)[1])
            - (oracle(xyz + 2 * delta)[1] - oracle(xyz - 2 * delta)[1])
        ) / (12 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=0, atol=3e-8)
    value = one_electron_sources(
        req, library=ecp_library, derivatives=False, budget_mib=96
    )
    np.testing.assert_allclose(value.hcore, out.hcore, atol=2e-10, rtol=0)
    assert value.d_hcore is None
    with pytest.raises(RuntimeError, match="cuda one-electron kernel"):
        one_electron_sources(replace(req, backend="cuda"), library=ecp_library)


@pytest.mark.parametrize("symbol", ["Br", "I"])
def test_cuda_geometry_projectors_against_three_center_oracle(symbol: str) -> None:
    """Run the built molecular CUDA artifact against fresh independent AO matrices."""
    import os
    from pathlib import Path

    from pyscf import scf

    if not (library := os.environ.get("GENERATIVEQC_F12_FOCAL_CUDA_LIBRARY")):
        pytest.skip("requires the compiled focal CUDA library and a CUDA device")
    xyz = np.array([[0.13, -0.16, 0.27], [1.23, 0.52, -0.39], [-0.68, 0.81, 1.43]])
    req, oracle = _request(symbol, xyz)
    req = replace(req, backend="cuda")
    out = one_electron_sources(req, library=Path(library), budget_mib=160)
    mol, h = oracle(xyz)
    np.testing.assert_allclose(out.hcore, h, atol=2e-9, rtol=0)
    gen = scf.RHF(mol).nuc_grad_method().hcore_generator(mol)
    assert out.d_hcore is not None and out.executed_backend == "cuda"
    for atom in range(3):
        np.testing.assert_allclose(
            out.d_hcore[atom], gen(atom) + gen(atom + 3), atol=8e-9, rtol=0
        )
    only = one_electron_sources(
        req, library=Path(library), derivatives=False, budget_mib=160
    )
    np.testing.assert_allclose(only.hcore, h, atol=2e-9, rtol=0)
