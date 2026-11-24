# Bounded native unrestricted triples response

Status implemented  
Date 2026-11-21

## Decision

The retained conventional UCCSD(T) force owner evaluates independent
32-lane canonical triples and reverse tiles from the existing TensorIR
antisymmetrizer. RHF retained determinants use the same spin-resolved
owner. The same-space Fock response contracts panels of 32 shared
complement coordinates and scatters each upper-triangular moment to its
two transposed destinations.

The selected lanes keep their physical amplitude and integral bindings.
Repeated same-spin coordinates are omitted. An unused lane gets zero
numerators and the denominator of the first physical lane. The reverse
seed is zero on that lane. The force response independently replays the
conventional triples energy of the qualified parent. Energy-only calls
retain the existing triangular canonical solver.

## Numeric bounds

Generated input, arena, output, and contraction-work queries include the
panel extents. Work admission still charges each generated contraction,
including physical padding. This batches the existing fixed-amplitude
response. Dense physical reference and relaxed-response storage still
contribute separately to the full force budget.

For the retained 44-AO OH/ha-D case with five alpha and four beta
occupied orbitals, the previous loops required 1,663,338,855 triples
graph invocations. A conservative bound for the new loops is 3,909,925
invocations, or 57,479,610,368 semantic scalar contraction terms
including padded lanes. Neither count includes the CC or Lambda solve.
The counts describe execution structure and do not estimate wall time.

## Qualification

The packed reverse test compares every energy, resolvent and amplitude or
integral cotangent against the existing singleton graph in all four spin
channels. Its final lane exercises finite padding. The resolvent test
compares each packed same-space block with both the original contraction
and the independent off-diagonal Fock oracle, including degenerate
eigenvalues and an incomplete final moment panel.
