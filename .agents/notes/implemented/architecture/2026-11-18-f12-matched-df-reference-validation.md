# Decision: retain the forward DF owner through molecular source response

Status: implemented (validation bridge; full F12 endpoint remains incomplete)
Date: 2026-11-18

## Problem

The F12 molecular source tests reverse supplied AO integrals and SCF/CABS frames.
The next connection needs a physical orbital response. Copying an oracle frame
into an otherwise native response, or rebuilding a direct-HF operator, would not
validate the derivative of the original RI-JK stationary reference.

## Decision

`benchmarks/df_reference_session.cpp` retains the native `hf::DFReference` and
the exact `shared_ptr<const PhysicalReference>` returned by its forward solve.
The same owner is supplied to `rhf_frame_response_cuda`. The bridge accepts full
basis metadata, not electronic oracle arrays, for the forward calculation. It
exposes detached snapshots for source construction without replacing its owned
physical reference. It performs no scientific CPU fallback.

This is a validation seam, not a new public SDK ABI. Native response already owns
the generated matrix maps, DF J/K actions, orbital solve and AO/Pulay derivatives.
Do not reimplement those under F12 or store only the dimensions as source identity.

## Invariants

- Preserve the immutable forward DF plan, metric policy, reference and geometry.
- `bar_fock_mo` uses full Frobenius coordinates; `bar_frame` uses the original
  orbital AO frame. Consume CABS frame weights before this connection.
- The existing response includes **HF with energy seed one** and the bridge adds
  nuclear repulsion once. Explicit F12 integral/metric derivatives are still a
  separate addition. Combining this with an HF-inclusive MP2 force without
  removing the duplicate HF term is wrong.
- Preserve complete capacity admission and the independent physical orbital
  residual. Failed response requests cannot publish partial forces or replace
  the reference with an exact-integral model.
- Passing this bridge's controlled-source test does not certify complete F12
  energies/forces, stock-water scalar parity, open shells, ECP, or CCSD-F12b.

## Evidence and intended GPU qualification

The controlled scalar is a fixed-AO occupied-projector contraction plus traces
of the occupied and virtual Fock blocks. Independent PySCF uses the same complete
cc-pVDZ-F12 / aug-cc-pVTZ-RIFIT inputs on H2 and water. Its occupied projector is
invariant to orbital phases and internal degeneracies, but has nonzero OV torque.
The native test checks reference matrices, HF analytic forces, two directions
at two nuclear step sizes, response residuals, capacity/nonfinite rejection and
owner reuse. CPU source-derivative/gauge tests and compilation are separate from
the pending physical L20 receipt. Timing reports include forward SCF and response
for the controlled endpoint, not the entire F12 endpoint.

## Consequences

The complete molecular F12 driver must next construct its AO sources from native
integrals in this actual forward frame, add the canonical-source multiplier, and
contract all explicit nuclear sources. Existing fixed-source scalar and Cartesian
water discrepancies retain their original acceptance gates until resolved.
