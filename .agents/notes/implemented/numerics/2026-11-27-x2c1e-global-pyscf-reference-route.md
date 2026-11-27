# Decision: Global X2C-1e energy parity uses PySCF's FW route

Status: implemented
Date: 2026-11-27

## Problem

`test_df_hf_mp2_and_delta_sr_parity_vs_pyscf` compares the native DF-HF, DF-MP2
and additive Delta-SR endpoint with PySCF 2.14.0 at 1e-10 Eh. Its global1e HCl
and SO2 cases failed by 1.05e-10 and 1.77e-10 Eh in `e_hf_sr` with PySCF's
default `1E` reference. The native value was repeatable (bitwise within a
process, about 2e-13 Eh across thread counts), which made the failure look
like a native accuracy defect.

The reference was the inaccurate side. PySCF's default `1E` hcore (Eq. 2 of
`pyscf.x2c.x2c`) assembles `h = S C_L G^{-1/2} e G^{-1/2} C_L^T S` from the
individual positive-energy Dirac eigenvalues `e`. A backward-stable FP64
eigensolver bounds their absolute errors by about `eps ||H||` of the 2n x 2n
Dirac matrix, which for the tight s functions of uncontracted x2c-SVPall on Cl
and S reaches 1e-10 Eh in the energy.

## Decision

The reference SCF takes PySCF's FW formula on PySCF's own X matrix: `sfx2c1e`
with a `get_xmat` override that returns the class's X, the route the hcore parity
test in the same file already used for S/SO2. The gate stays at 1e-10 Eh. The
X2C-1e definition, the libcint integrals, DF and MP2 are unchanged. `ATOM1E` and
`dlu1e` are unaffected.

## Rejected alternatives

- Relaxing the gate. The native endpoint is within about 1e-12 Eh of exact
  arithmetic, so the gate is sound once the reference is accurate.
- Attributing the residual to integral differences. Exact arithmetic on native
  and on libcint integrals differs by less than 6e-13 Eh; the integrals agree to
  1-3 ulps of their largest elements.
- An exact-arithmetic (mpmath) oracle inside the test. It is decisive, but adds a
  test dependency and about two minutes for SO2.
- A whitened symmetric Dirac problem with the Eq. 2 assembly in NumPy. It still
  uses the eigenvalues directly: 1-ulp spread about 4e-11 Eh, and 8.7e-11 Eh from
  exact for SO2.

## Invariants

- An FP64 X2C-1e reference must not assemble h from individual positive-energy
  Dirac eigenvalues. Routes that depend only on the positive-energy subspace (X,
  FW formula, renormalization) inherit the conditioning of the 2c^2 gap.
- An accuracy claim for a reference route needs an exact-arithmetic check on the
  same FP64 integrals and the spread under 1-ulp integral perturbations.

## Evidence

Exact arithmetic is mpmath at 40 digits; 60 digits changes it by less than
1e-36. Energies are DF-HF SCF energies minus the SCF energy with the exact hcore
on libcint integrals, in Eh. Spreads are first-order energy changes over 32
random symmetric {-1, 0, +1} ulp perturbations of S, T, V and W.

| Quantity | HCl (43 AOs) | SO2 (84 AOs) |
| --- | --- | --- |
| PySCF default 1E (Eq. 2) | +1.03e-10 | -1.78e-10 |
| Eq. 2 spread, sigma (max) | 7.9e-11 (3.1e-10) | 8.1e-11 (3.3e-10) |
| PySCF FW route | +1.4e-12 | +2.3e-13 |
| FW route spread, sigma (max) | 4.2e-13 (1.7e-12) | 7.8e-13 (2.2e-12) |
| Native endpoint | -1.1e-12 | -1.0e-12 |
| Exact on native integrals (first order) | +7.1e-14 | -5.6e-13 |

The native global hcore on its own integrals is -6.2e-13 and +2.1e-12 Eh from
exact (first order). With the FW reference, `tests/python/test_x2c1e_dlu_gate1.py`
passes 32/32, global1e 5/5.

## Consequences

The energy parity test no longer runs PySCF's Eq. 2 code path. PySCF serves as an
independent reference here, not as a test subject.

## Revisit when

- PySCF changes its 1E hcore assembly.
- Elements heavier than Ar or larger uncontracted bases enter this gate; measure
  the FW route against exact arithmetic again.
- The native global path starts to assemble h from Dirac eigenvalues.

## References

- `test_df_hf_mp2_and_delta_sr_parity_vs_pyscf` and
  `test_hcore_and_analytic_jacobian_parity_vs_pyscf` in
  `tests/python/test_x2c1e_dlu_gate1.py` exercise the FW reference for
  energy and hcore parity, respectively.
