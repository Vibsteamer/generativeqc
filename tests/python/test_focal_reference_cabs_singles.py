"""Physical H3 DF-HF + CABS singles, extended Fock and geometry response."""

import os
from pathlib import Path

import numpy as np
import pytest
import test_focal_df_reference as reference_tests
import test_focal_one_electron_sources as one_electron_tests
from generativeqc.experimental.f12_focal import (
    FocalSeries,
    df_reference,
    focal_cabs_projector,
    one_electron_sources,
    plan_singlepoint,
    reference_cabs_singles,
)

hydrogen_bases = reference_tests.hydrogen_bases
molecular_library = one_electron_tests.molecular_library


@pytest.mark.parametrize("root,charge,spin", [("rhf", 1, 0), ("uhf", 0, 1)])
def test_molecular_singles_and_source_response(
    hydrogen_bases: tuple, molecular_library: Path, root: str, charge: int, spin: int
) -> None:
    from pyscf import df, gto, scf
    from scipy.linalg import solve_sylvester, solve_triangular

    basis, obs, jk = hydrogen_bases
    opt = gto.basis.load("cc-pvdz-f12-optri", "H")
    xyz = np.array([(0.10, -0.23, 0.2), (1.38, 0.27, -0.1), (0.67, 1.45, 0.31)])
    direction = np.array([[0.3, -0.2, 0.1], [-0.4, 0.25, -0.1], [0.1, -0.05, 0.0]])
    direction /= np.linalg.norm(direction)

    def sources(x: np.ndarray) -> tuple:
        atoms = [("H", p) for p in x]
        req = plan_singlepoint(
            atoms,
            FocalSeries(basis),
            FocalSeries(basis),
            basis,
            root_id=root,
            correlation_space_id="all-active",
            charge=charge,
            multiplicity=spin + 1,
            backend="cpu_reference",
        ).steps[0][0]
        args = {"unit": "Bohr", "spin": spin, "charge": charge, "verbose": 0}
        mol = gto.M(atom=atoms, basis=obs, **args)
        union = gto.M(
            atom=atoms + [("ghost-H", p) for p in x],
            basis={**obs, "ghost-H": opt},
            **args,
        )
        aux = gto.M(atom=atoms, basis=jk, **args)
        metric = aux.intor("int2c2e")
        three = np.ascontiguousarray(
            df.incore.aux_e2(union, aux, intor="int3c2e", aosym="s1").transpose(2, 0, 1)
        )
        white = solve_triangular(
            np.linalg.cholesky(metric), np.eye(len(metric)), lower=True
        )
        mf = (scf.RHF(mol) if root == "rhf" else scf.UHF(mol)).density_fit(auxbasis=jk)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-9, 150
        mf.kernel()
        assert mf.converged
        s = union.intor("int1e_ovlp")
        h = union.intor("int1e_kin") + union.intor("int1e_nuc")
        n = mol.nao
        # QR in orthonormal AO coordinates generates an independent, full CABS complement.
        L = np.linalg.cholesky(s).T
        Q = np.linalg.qr(L[:, :n], mode="complete")[0][:, n:]
        cabs = solve_triangular(L, Q)
        scaled = (white @ three.reshape(len(metric), -1)).reshape(three.shape)
        d = np.asarray(mf.make_rdm1()).reshape(-1, n, n)
        da, db = (d[0] * 0.5, d[0] * 0.5) if root == "rhf" else d
        j = np.einsum(
            "Qpq,Q->pq", scaled, np.einsum("Qij,ij->Q", scaled[:, :n, :n], da + db)
        )
        occ = tuple(mol.nelec)
        energies = []
        focks = []
        for idx, density in enumerate((da, db)):
            c = np.atleast_3d(mf.mo_coeff) if root == "uhf" else np.array([mf.mo_coeff])
            cs = c[min(idx, len(c) - 1)]
            ao_f = (
                h
                + j
                - np.einsum(
                    "Qpi,ij,Qqj->pq",
                    scaled[:, :, :n],
                    density,
                    scaled[:, :, :n],
                    optimize=True,
                )
            )
            frame = np.column_stack([np.vstack([cs, np.zeros((len(s) - n, n))]), cabs])
            f = frame.T @ ao_f @ frame
            o = occ[idx]
            coupling = f[o:, :o]
            amp = solve_sylvester(f[o:, o:], -f[:o, :o], coupling)
            energies.append(-np.vdot(amp, coupling))
            focks.append(ao_f)
        return req, mf, s, h, three, metric, white, np.array(energies), focks

    central = sources(xyz)
    req, mf, s, _, b, m, w, expected, focks = central
    one = one_electron_sources(req, library=molecular_library)
    ref = df_reference(req)
    projector = focal_cabs_projector(req, one)
    result = reference_cabs_singles(
        req, ref, one, complement=projector.frame, three_center=b, metric=m, whitener=w
    )
    if root == "rhf":
        with pytest.raises(
            ValueError, match="interpreter logical retained-byte budget"
        ):
            reference_cabs_singles(
                req,
                ref,
                one,
                complement=projector.frame,
                three_center=b,
                metric=m,
                whitener=w,
                budget_mib=1,
                pullback=False,
            )
    n = one.obs_ao_count
    for i, f in enumerate(focks):
        coeff = ref.coefficients[min(i, len(ref.occupied) - 1)]
        frame = np.column_stack(
            [np.vstack([coeff, np.zeros((len(s) - n, n))]), projector.frame]
        )
        np.testing.assert_allclose(
            result.extended_fock[i], frame.T @ f @ frame, atol=2e-8, rtol=0
        )
    np.testing.assert_allclose(result.spin_energies, expected, atol=3e-9, rtol=0)
    assert result.hf_cabs_energy == pytest.approx(
        mf.e_tot + sum(expected), abs=4e-9, rel=0
    )
    assert max(result.max_obs_fock_error) < 2e-8
    relaxed = reference_cabs_singles(
        req,
        ref,
        one,
        complement=projector.frame,
        three_center=b,
        metric=m,
        whitener=w,
        relax=True,
    ).weights
    if shell_library := os.environ.get("GENERATIVEQC_F12_HOST_SINK"):
        from generativeqc.experimental.f12_focal import MolecularFocalProvider

        endpoint = MolecularFocalProvider(
            shell_library=Path(shell_library),
            one_electron_library=molecular_library,
        )(req)
        assert endpoint.term.energy == pytest.approx(
            result.hf_cabs_energy, abs=4e-9, rel=0
        )
        for step in (4e-4, 2e-4):
            force = np.empty_like(xyz)
            for atom, axis in np.ndindex(xyz.shape):
                delta = np.zeros_like(xyz)
                delta[atom, axis] = step
                plus, minus = (sources(xyz + sign * delta) for sign in (1, -1))
                force[atom, axis] = -(
                    plus[1].e_tot + sum(plus[7]) - minus[1].e_tot - sum(minus[7])
                ) / (2 * step)
            np.testing.assert_allclose(
                endpoint.term.forces, force, atol=4e-7 if step == 4e-4 else 1e-7, rtol=0
            )
    weights = result.weights
    bar = focal_cabs_projector(
        req, one, expected_rank=projector.rank, cotangent=weights["cabs_kernel"]
    ).overlap_bar
    dh = np.einsum("ax,axij->ij", direction, one.d_hcore)
    ds = np.einsum("ax,axij->ij", direction, one.d_overlap)

    def aligned(c: np.ndarray) -> np.ndarray:
        cs = np.asarray(c).reshape(-1, n, n)
        out = []
        for x, y, occ in zip(cs, ref.coefficients, ref.occupied, strict=True):
            blocks = []
            for subset in (slice(None, occ), slice(occ, None)):
                z = x[:, subset]
                u, _, v = np.linalg.svd(z.T @ s[:n, :n] @ y[:, subset])
                blocks.append(z @ (u @ v))
            out.append(np.column_stack(blocks))
        return np.stack(out)

    for step in (4e-4, 2e-4):
        p, minus = (sources(xyz + k * step * direction) for k in (1, -1))
        actual = (
            -np.vdot(ref.forces, direction)
            + np.vdot(weights["hcore"], dh)
            + np.vdot(bar, ds)
            + (
                np.vdot(weights["three_center"], p[4] - minus[4])
                + np.vdot(weights["metric"], p[5] - minus[5])
                + np.vdot(
                    weights["obs_coefficients"],
                    aligned(p[1].mo_coeff) - aligned(minus[1].mo_coeff),
                )
            )
            / (2 * step)
        )
        finite = (p[1].e_tot + sum(p[7]) - minus[1].e_tot - sum(minus[7])) / (2 * step)
        assert actual == pytest.approx(finite, abs=3e-7, rel=0)
        analytic = (
            -np.vdot(ref.forces, direction)
            + np.vdot(relaxed["hcore"], dh)
            + np.vdot(relaxed["overlap"], ds)
            + (
                np.vdot(relaxed["three_center"], p[4] - minus[4])
                + np.vdot(relaxed["metric"], p[5] - minus[5])
            )
            / (2 * step)
        )
        assert analytic == pytest.approx(finite, abs=3e-7, rel=0)
