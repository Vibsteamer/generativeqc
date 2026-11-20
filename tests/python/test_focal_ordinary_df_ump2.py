"""Physical metric and spin-pair force transposes for the focal UMP2 provider."""

import os
from pathlib import Path

import numpy as np
import pytest
import test_focal_df_reference as reference_tests
from generativeqc_compiler.mp2.ordinary_df_factors import ordinary_df_factors
from generativeqc_compiler.mp2.unrestricted_df_response import df_ump2_pair_response
from generativeqc_compiler.tensor import Program, execute, transpose_program

hydrogen_bases = reference_tests.hydrogen_bases


def test_whitened_ov_integral_and_metric_pullback() -> None:
    rng = np.random.default_rng(691)
    n, o, q = 7, 2, 9
    c, dc = rng.normal(size=(2, n, n))
    raw, dr = rng.normal(size=(2, q, n, n))
    raw += raw.transpose(0, 2, 1).copy()
    dr += dr.transpose(0, 2, 1).copy()
    z = rng.normal(size=(q, q))
    metric = z @ z.T + np.eye(q)
    dm = rng.normal(size=(q, q))
    dm += dm.T.copy()
    weight = rng.normal(size=(o * (n - o), o * (n - o)))
    weight += weight.T.copy()

    def state(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coef = c + t * dc
        b = np.einsum("qui,uj,iv->qjv", raw + t * dr, coef[:, :o], coef[:, o:]).reshape(
            q, -1
        )
        w = np.linalg.inv(np.linalg.cholesky(metric + t * dm))
        b = w @ b
        return b, w, np.einsum("Qi,ij,Qj->", b, weight, b)

    b, w, _ = state(0.0)
    graph = ordinary_df_factors(n, o, q)
    inputs = {"coefficients": c, "ao_coulomb": raw, "whitener": w}
    fwd = Program({"B": graph.outputs["B"]})
    np.testing.assert_allclose(execute(fwd, inputs).outputs["B"].reshape(q, -1), b)
    seed = (2 * b @ weight).reshape(q, o, n - o)
    bars = execute(
        transpose_program(fwd, ("B",)).program, {**inputs, "bar_B": seed}
    ).outputs
    mbar = execute(
        Program({"metric": graph.outputs["metric_bar"]}), {**inputs, "factor_bar": seed}
    ).outputs["metric"]
    deriv = (
        np.vdot(bars["bar_coefficients"], dc)
        + np.vdot(bars["bar_ao_coulomb"], dr)
        + np.vdot(mbar, dm)
    )
    for step in (2e-5, 1e-5):
        assert deriv == pytest.approx(
            (state(step)[2] - state(-step)[2]) / (2 * step), rel=2e-7
        )


@pytest.mark.parametrize("channel", ["alpha_alpha", "alpha_beta", "beta_beta"])
def test_energy_source_weights_match_full_moving_frame_force(channel: str) -> None:
    rng = np.random.default_rng(502)
    dims = {"alpha": (2, 3), "beta": (3, 4)}
    l, r = channel.split("_")
    center, tangent = {}, {}
    for spin in sorted({l, r}):
        o, v = dims[spin]
        center["B_" + spin] = rng.normal(size=(5, o, v))
        center["eo_" + spin] = -np.arange(1, o + 1, dtype=float)
        center["ev_" + spin] = np.arange(1, v + 1, dtype=float)
        tangent["dB_" + spin] = rng.normal(size=(2, 5, o, v))
        for name, size in (("dFo_", o), ("dFv_", v)):
            x = rng.normal(size=(2, size, size))
            tangent[name + spin] = x + x.transpose(0, 2, 1)
    args = (channel, 5, *dims[l], *dims[r])
    expected = execute(df_ump2_pair_response(*args, 2), {**center, **tangent}).outputs
    source = execute(
        df_ump2_pair_response(*args, 1, source_weights=True),
        {**center, "bar_force": np.array([-1.0])},
    ).outputs
    force = np.zeros(2)
    for name, t in tangent.items():
        force -= t.reshape(2, -1) @ source["bar_" + name[1:]].ravel()
    np.testing.assert_allclose(force, expected["force"], atol=2e-12, rtol=1e-12)
    assert source["energy"] == pytest.approx(expected["energy"])


@pytest.mark.parametrize("root", ["rhf", "uhf"])
def test_molecular_ordinary_mp2_against_two_independent_fits(
    hydrogen_bases: tuple,
    root: str,
) -> None:
    from generativeqc.experimental.f12_focal import FocalSeries, plan_singlepoint
    from pyscf import df, gto, scf
    from pyscf.mp import dfmp2, dfump2

    from benchmarks.f12_focal_ump2 import molecular_df_ump2

    shell = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not shell:
        pytest.skip(
            "requires the full native molecular shell forward/derivative library"
        )
    basis, obs, jk = hydrogen_bases
    xyz = np.array([(0.10, -0.23, 0.20), (1.38, 0.27, -0.10), (0.67, 1.45, 0.31)])
    if root == "rhf":
        xyz = xyz[:2]
    symbols = ["H"] * len(xyz)
    mult = 1 if root == "rhf" else 2
    atoms = list(zip(symbols, xyz, strict=True))
    plan = plan_singlepoint(
        atoms,
        FocalSeries(basis),
        FocalSeries(basis),
        basis,
        root_id=root,
        correlation_space_id="all-active",
        multiplicity=mult,
        backend="cpu_reference",
    )
    req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    result = molecular_df_ump2(
        req, shell_library=Path(shell), one_electron_library=Path(shell)
    )

    def oracle(positions: np.ndarray) -> float:
        mol = gto.M(
            atom=list(zip(symbols, positions, strict=True)),
            basis=obs,
            unit="Bohr",
            spin=mult - 1,
            verbose=0,
        )
        mf = (scf.RHF(mol) if root == "rhf" else scf.UHF(mol)).density_fit(auxbasis=jk)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-9, 150
        mf.kernel()
        assert mf.converged
        mp = dfmp2.DFMP2(mf) if root == "rhf" else dfump2.DFUMP2(mf)
        mp.with_df = df.DF(mol, auxbasis="aug-cc-pvtz-ri")
        energy, _ = mp.kernel(with_t2=False)
        return float(energy)

    assert result.mp2_fit_id != result.jk_fit_id
    assert result.spin_energies[-1] == 0.0
    assert sum(result.spin_energies) == pytest.approx(oracle(xyz), abs=2e-9)
    delta = np.array([[0.1, -0.2, 0.3], [-0.2, 0.4, -0.1], [0.1, -0.2, -0.2]])
    delta = delta[: len(xyz)]
    for step in (4e-4, 2e-4):
        force = (oracle(xyz - step * delta) - oracle(xyz + step * delta)) / (2 * step)
        assert np.vdot(result.correlation_forces, delta) == pytest.approx(
            force, abs=2e-7
        )


def test_restricted_water_orbital_response_matches_pyscf_cphf() -> None:
    """Equal-spin response to each Cartesian electric field on a bent RHF root."""
    from generativeqc_compiler.mp2.f12_unrestricted_reference import (
        unrestricted_df_orbital_action,
    )
    from pyscf import gto, lib, scf

    mol = gto.M(
        atom="O .11 -.16 .21; H 1.46 .91 -.31; H -1.09 1.28 .37",
        unit="Bohr",
        basis="6-31g",
        verbose=0,
    )
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-13
    mf.kernel()
    assert mf.converged
    coeff = mf.mo_coeff
    occ = mol.nelectron // 2
    co, cv = coeff[:, :occ], coeff[:, occ:]
    b = np.concatenate([lib.unpack_tril(chunk) for chunk in mf.with_df.loop()])
    b = np.einsum("Ppq,pi,qj->Pij", b, coeff, coeff, optimize=True)
    gap = mf.mo_energy[occ:, None] - mf.mo_energy[:occ]
    raw = {
        "vo": b[:, occ:, :occ],
        "oo": b[:, :occ, :occ],
        "vv": b[:, occ:, occ:],
        "gap": gap,
    }
    factors = {
        f"{k}_{s}": np.ascontiguousarray(v)
        for k, v in raw.items()
        for s in ("alpha", "beta")
    }
    action = unrestricted_df_orbital_action(occ, occ, mol.nao, len(b))
    field = cv.T @ mol.intor("int1e_r", comp=3) @ co
    response = mf.gen_response(hermi=1)

    def fock_response(vector: np.ndarray) -> np.ndarray:
        density = 2 * cv @ vector.reshape(-1, *gap.shape) @ co.T
        v = response(density + density.transpose(0, 2, 1))
        return (cv.T @ v @ co).reshape(vector.shape)

    # Build the independent restricted CPHF matrix by perturbing each
    # spatial rotation in PySCF; solve all three field directions together.
    hessian = np.diag(gap.ravel()) + fock_response(np.eye(gap.size)).T
    solution = np.linalg.solve(hessian, -field.reshape(3, -1).T).T.reshape(field.shape)
    for trial, rhs in zip(solution, field, strict=True):
        x = np.ascontiguousarray(trial)
        value = execute(action, {**factors, "trial_alpha": x, "trial_beta": x}).outputs
        for spin in ("alpha", "beta"):
            np.testing.assert_allclose(value[spin], -rhs, rtol=0, atol=1e-9)
