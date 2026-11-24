# Decision: qualify the separate conventional triples force with the missing CCSD response

Status: implemented adapter; real-device numerical qualification pending
Date: 2026-11-18

## Problem

The retained conventional CCSD(T)-minus-MP2 controls qualify a combined high-order
correction, not the independent perturbative triples force needed by candidate
F12 focal recipes. The scalar triples energy already exists, but subtracting an
unrelaxed CCSD gradient or changing the RI-JK reference would change its derivative.

## Decision

The benchmark-only df_focal_component.cpp calls the existing full native
run_df_ccsdt_native with with_triples=false and explicit demand.df_reference=true.
It solves the separate CCSD Lambda/orbital response and returns the complete
CCSD energy and nuclear force. Its input is only the original geometry and full
orbital/auxiliary coefficients. There is no imported electronic state or CPU
scientific fallback.

Subtract this force from a separately authenticated, already qualified CCSD(T)
force at the identical geometry, basis, fitting, electron space and reference.
df_focal_triples.py checks both the CCSD response and their triples difference
against retained independent two-step displaced energies.

## Invariants

- A CCSD(T)-MP2 gradient cannot be relabelled as a triples gradient.
- Both response equations must be solved; the energy decomposition alone does
  not supply the separate relaxed force.
- Reusing a prior result requires exact input identities and independent energy
  and force comparisons; a matching method name is insufficient.
- The internal C++ ABI is not a compatibility contract. Compile this adapter
  against the exact headers/source of the pinned qualified native library.
  Record the adapter commit and library commit/hash separately.
- Retain the physical RI-JK trace, response residuals, full returned vectors and
  failure diagnostics. Do not substitute Direct RHF.
- The current independent oracle has one direction at two step sizes. Passing
  it is directional qualification, not all-Cartesian or complete F12 qualification.
- Charge the newly executed central CCSD E/F cost separately. Reused CCSD(T)
  compute is not newly billed work, but still belongs in the cost of a future
  fresh end-to-end recipe when that recipe requires the conventional CC solve.

## Evidence and follow-up

The first panel retains three original methane geometries and ordinary/augmented
DZ variants. Its six prior CCSD(T) energies and force projections match the
independent DF-CCSD(T) two-step controls within the predeclared gates.
The adapter compiles and links against the existing L20 library. Six missing
CCSD central E/F computations are registered as one short native L20 validation
package; they do not rerun the existing CCSD(T) or CPU oracle calculations.

Numerical acceptance remains 1e-7 Eh for energies, 2e-6 Eh/bohr for both projected
force comparisons, 2e-7 Eh/bohr for triples translation and 1e-8 for the recorded
response residuals. Full Cartesian checks and complete molecular F12 force
composition remain separate gates.
