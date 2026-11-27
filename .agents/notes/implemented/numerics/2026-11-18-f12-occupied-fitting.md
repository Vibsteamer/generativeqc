# Decision: contract the occupied density before extended-frame Coulomb fitting

Status: implemented; molecular L20 component qualification tracked separately
Date: 2026-11-18

## Problem

The initial F12 molecular source chain solved the Coulomb metric for every pair
in the extended OBS/CABS space. Most columns were used only to contract a single
occupied density trace. That adds work and storage before either the geminal
pair functional or the reverse program needs them.

## Decision

Let G[A,p,q] be the raw three-center Coulomb source and
D[A,i,p] = (J^-1 G[:,i,p])[A] with occupied i. Generate

    J[p,q] = sum_A G[A,p,q] sum_i D[A,i,i]
    K[p,q] = sum_Ai D[A,i,p] G[A,i,q]
    F = h + 2 J - K.

Self-adjointness of the same Coulomb metric solve makes this identical to
fitting every p,q first. The reverse program accumulates both Fock and F12-pair
weights on the one occupied-fit input before the existing implicit solve
response. Its raw weight is added to the occupied slice of the full Coulomb
source weight; the shared metric weight is accumulated once.

The compiler owns these contractions; the benchmark orchestration owns only
views, accumulation and qualification. No new recurrence or solver is added.
Callers and reverse consumers migrate to `fit`, `raw_coulomb_all`, and `hcore`.
Historical snapshots/fixtures remain tied to their own source versions; absent
raw sources must not be synthesized silently from historical fitted arrays.

## Invariants and limitations

- The metric, orbital frame and electron partition must be identical throughout.
- The fit domain changes from naux*nri*nri to naux*nocc*nri. This reduces
  metric-solve columns and fitted-array storage, not the remaining full raw
  Coulomb array or the entire F12 endpoint by the same factor.
- The full Coulomb source, occupied fits, one-electron matrix and fitting metric
  all remain differentiable. No occupied-density term is treated as constant.
- The scalar component test uses supplied molecular sources and records cold
  compilation, warm execution and transfers. It is not complete nuclear E/F.
- The H2/H2O full source-chain versus stock Psi4 discrepancy is retained.
  Conditioning diagnostics or a same-equation stable solve comparison do not
  authorize relaxing the full molecular energy/force acceptance gates.

## Evidence

The executable unit oracle constructs the complete four-index DF interaction
using an independent dense solve, then forms F and K and checks both displaced
metric/raw/one-electron sources. This tests the mathematical reordering and its
implicit response, not a second copy of the occupied-only schedule.

The physical component fixture keeps the existing H2/H2O AO sources and original
geometries. An independent SciPy Cholesky solve fits all extended-space columns
and evaluates two displaced objectives at steps 2e-4 and 1e-4. Values retain the
5e-10 absolute Fock/exchange gate and response uses the existing metric-component
2e-7 bound. The distinct full F12 energy gate remains 5e-11 Eh. The fixture also
retains its differences from stock Psi4 rather than replacing those records.

## Revisit when

The full AO provider is tiled and weights remain resident. Then contract the
occupied density and orbital transforms into source tiles without materializing
the full extended Coulomb array. Keep this exact response contract and measure
complete E/F work before claiming a molecular speedup.
