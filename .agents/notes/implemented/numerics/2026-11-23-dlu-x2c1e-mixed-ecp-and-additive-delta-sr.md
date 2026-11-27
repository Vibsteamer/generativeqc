# Scalar-relativistic DLU-X2C response for mixed all-electron and ECP bases

Status: implemented
Date: 2026-11-23

## Problem

Adding scalar relativistic corrections (`ΔSR`) to GenerativeQC's focal F12 pipeline requires:
1. Exact 1-electron scalar-relativistic `hcore` and analytic nuclear gradients on all-electron molecules (`H–Cl`) matching PySCF `sfx2c1e` / `sfx2c1e_grad`.
2. Zero relativistic double-counting on heavy ECP atoms (`Br ECP10MDF`, `I ECP28MDF`), which already embed scalar relativity in their pseudopotentials.
3. Zero contraction error (`< 0.01 kcal/mol`) when taking the pure additive difference `ΔE_SR = E_DF-MP2(DLU-X2C-1e) - E_DF-MP2(NR)` in a shared basis.
4. Machine-precision rotational/translational invariance (`net force, net torque < 1e-12`) even on mixed AE+ECP systems with scalar Jacobi eigensolvers when OpenBLAS LAPACKE is unavailable.

## Decision

1. **Mixed AE+ECP `DLU-X2C-1e` Hamiltonian (`src/integrals/x2c1e.cpp`)**:
   - Decontract only all-electron atoms (`ecp_core == 0`) with exponent deduplication (`1e-9` relative tolerance); leave ECP atoms contracted with identity contraction blocks.
   - Define `X = block_diag(X_A on AE, I on ECP)`, `dX = X - I` (zero on ECP), and `X_W = block_diag(X_A on AE, 0 on ECP)`.
   - Assemble:
     - `S_nesc = S + (1/2c^2) X_W^T T X_W`
     - `H_nesc = V + T - dX^T T dX + (1/4c^2) X_W^T W X_W` (eliminating `10^5 Eh` kinetic cancellation of `T X + X^T T - X^T T X`)
     - Molecular metric renormalization `R = S^{-1/2} (I + delta_K)^{-1/2} S^{1/2}` where `delta_K = S^{-1/2} (S_nesc - S) S^{-1/2}` is diagonalized directly (eliminating `+1.0` shift dilution on ECP/valence modes).
   - This preserves exact potential gauge invariance (`V -> V + a S`, `W -> W + 2a T` shifts `h_X2C -> h_X2C + a S`), leaves pure ECP atoms strictly NR (`Delta h_SR == 0` identically), and matches PySCF `sfx2c1e(approx='ATOM1E')` to `< 2e-12 Eh` on all-electron systems.
2. **Exact `SO(3)` Atomic Block Solver & Adjoint Nuclear Gradients**:
   - Solve spherical isolated-atom Dirac blocks once per angular momentum `l` (`m = 0`) and scatter bitwise-identically across all `m = -l..+l`, guaranteeing exact `SO(3)` invariance.
   - Use O(1)-Sylvester/eigenvector reverse-mode pullback into uncontracted `(S, T, V, W)` weights contracted through scalar McMurchie-Davidson kernels (zero `Jet` allocations, zero `(3*natom, n, n)` materialization). For `Global1e`, pull back only through the negative-positive subspace (`E_+ - E_- >= 2c^2 ~ 37,557 Eh`), avoiding all degenerate positive-positive divisions.
   - Tighten scalar Jacobi fallback in `build_r_metric` to `1e-15` absolute off-diagonal tolerance when LAPACKE is absent.
3. **Pure Additive `ΔSR` Basis (`unc-x2c-TZVPall` + `cc-pVDZ-PP` + shared `def2-universal-JKFit`)**:
   - Fully decontract Karlsruhe `x2c-TZVPall` on AE atoms `H–Cl` (`0.000 kcal/mol` contraction error, full 12-element BSE coverage, `max_l <= 3`), keep `cc-pVDZ-PP` contracted on `Br`/`I`, and share `def2-universal-JKFit` across NR and DLU-X2C-1e (`<= 0.00005 kcal/mol` difference vs split `def2-TZVP-RIFIT` on `ΔE_SR`).

## Rejected alternatives

- **Contracted relativistic bases (`x2c-TZVPall`, `cc-pVTZ-DK`, `cc-pCVTZ-DK`) for additive `E(X2C) - E(NR)`**: Rejected because evaluating `E(NR)` in a relativistically contracted `s/p` core basis causes `-157` to `-245 kcal/mol` total contraction error on `Cl` and `-0.17` to `-0.22 kcal/mol` bias on `BDE(HCl)`. Conversely, contracted NR `cc-pCVDZ-F12` causes `+94 kcal/mol` total error and `+0.10 kcal/mol` bias on `BDE(HCl)`.
- **Contracted `unc-x2c-SVPall` for `ΔSR`**: Rejected because `gscdb/IL16_12` (`Cl-` ion pair) deviates by `-0.077 kcal/mol` vs `unc-x2c-TZVPall` / `unc-cc-pCVTZ-DK` (whereas `unc-x2c-TZVPall` agrees with gold-standard `unc-cc-pCVTZ-DK` to `<= 0.0018 kcal/mol` on `BDE99_65`, `VIB5/CH4/33156`, and `IL16_12`).
- **Diagonalizing `K = I + delta_K` with default `1e-14 * scale` scalar Jacobi**: Rejected after diagnosis on `CH3Br` showed `1.46e-12` spurious rotational torque due to `1e-14 * scale` Jacobi noise amplified by `S^{-1/2}` on the 27-fold degenerate Br ECP block; diagonalizing `delta_K` with `1e-15` tolerance reduced torque 13x to `1.13e-13`.
- **Setting `X_B = I` inside `S_nesc` and `W` on ECP atoms**: Rejected because it injects spurious `(1/2c^2) T_BB` and `(1/4c^2) W_BB` relativistic shifts onto ECP atoms, double-counting scalar relativity.

## Invariants

- Pure ECP systems (`Br ECP10MDF`, `I ECP28MDF`) must yield identically zero `Delta h_SR`, `d(Delta h_SR)/dR`, and `Delta E_SR` under `Dlu1e`.
- `Global1e` must explicitly reject systems with ECP atoms.
- Scale-invariant ECP gradient convergence checks in `contract_weighted_ecp_derivative` ensure `eval(W_sr - W_nr)` selects the identical converged quadrature grid as `eval(W_sr) - eval(W_nr)`.

## Evidence

- `tests/python/test_x2c1e_dlu_gate1.py` + `tests/python/test_focal_preopt_gate1.py`: 40/40 passed against PySCF 2.14.0 (`sfx2c1e`, `sfx2c1e_grad`) and two-step Richardson central finite differences (`< 1e-10 Eh`, `< 1e-10` Jacobian, `< 1e-8 Eh/bohr` FD forces on `H2O`, `OH`, `HBr`, `CH3Br`, net force & torque `< 1e-12`).

For the metric-factor derivative the factorization and cotangent must
describe the metric from the same Hamiltonian. Geometry-specific atomic
blocks share only reusable basis-dependent inputs. The molecular metric
and its reverse weights use the live geometry.

## Revisit when

- CUDA-accelerated DF-MP2 Z-vector / DF gradient kernels replace the host OpenBLAS path for large (`> 400 AO`) `ΔSR` E+F evaluations.
- Full-molecular DLU-X2C-1e + 2-electron AMFI/SNSO picture-change corrections are introduced beyond 1-electron scalar relativity.

## References

- Implementation files: `src/integrals/x2c1e.{hpp,cpp}`, `src/api/c_api_x2c1e.cpp`, `python/generativeqc/_x2c1e.py`, `tests/python/test_x2c1e_dlu_gate1.py`, `tools/eval_delta_sr_x2c1e.py`.
