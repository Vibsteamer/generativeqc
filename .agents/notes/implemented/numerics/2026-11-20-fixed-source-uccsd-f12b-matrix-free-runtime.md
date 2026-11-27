# Matrix-free fixed-source UCCSD-F12b iteration and response

Status: implemented
Date: 2026-11-20

## Decision

The fixed-source owner binds canonical spin-MO Fock and integral blocks,
fixed-geminal feeds, spin occupations and the equation identity. It keeps
detached values so repeated primal and Lambda work uses the same reference.
Conventional UCCSD and fixed-source UCCSD-F12b share packed iteration and
response policy while retaining the identity of each method.

DIIS proposes packed amplitude steps. The resulting state passes fresh
energy and residual checks before use. Lambda applies the compiler
transpose to the packed response and the physical Lagrangian is
rechecked. Same-spin dense residual seeds use one quarter of the signed
packed expansion. Orbital-gap preconditioning may reduce transpose
iterations, but the solve checks the unpreconditioned residual.

The explicit CUDA executor checks the combined retained host iteration
state and device arena before copying the inputs. The host manages
amplitude and Lambda iterations; the CUDA executor owns its planned
contractions. A checkpoint may replay detached vectors only against
their bound feeds, with fresh primal and Lagrangian checks and the
ordinary response budget.

## Evidence

`tests/python/test_uccsd_fixed_f12.py` checks the equation and its
relaxed response against small independent dense controls. A full dense
Jacobian would need quadratic memory in the packed excitation count.
Reduced-space replay can therefore serve as a reference without
becoming the production solver.

## References

- `tools/generativeqc_cc/uccsd_solver.py`
- `tools/generativeqc_cc/uccsd_lambda.py`
- [Stationary response of the fixed-cusp inputs](2026-11-20-unrestricted-fixed-source-cc-f12b-response.md)
