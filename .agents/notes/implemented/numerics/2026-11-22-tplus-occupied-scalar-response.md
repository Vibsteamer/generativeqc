# Occupied-orbital scalar and response for Kállay (T+)

Status: implemented
Date: 2026-11-22

## Decision

Equation (12) uses the conventional MP2 contribution (M_i), the F12
increment (F_i), and the conventional triples contribution (T_i) in one
matched occupied frame. Its scalar is the sum of (q_i T_i) with
(q_i=(M_i+F_i)/M_i). The derivative seeds for the three sources are
(-F_i T_i/M_i^2), (T_i/M_i), and (q_i), respectively. A triples
implementation that visits unordered triplets can consume the average of
the three occupied seeds from Eq. (13). The paired permutation convention
belongs to the triples producer.

The packed TensorIR implementation uses the shared scaled bilinear
primitive to avoid forming a possibly unrepresentable squared denominator.
This retains the finite response of shrinking but nonzero regular
directions. The reference evaluator removes entries certified as
structurally inactive by their source and scatters zero response into
those original locations. A zero denominator in an active orbital raises
because independent variations of (M_i), (F_i), and (T_i) can give
different limits even if all three center values vanish. Arbitrary
regularization would change the model and its derivative.

## Invariants

An upstream consumer must match the occupied root and basis, physical
correlation space, parent amplitudes, and fitting records before pairing
the three source vectors. The incoming vectors for disconnected fragments
may be combined only when their occupied frame preserves the fragment
partition. Appendix B discusses degenerate orbitals and the residual
ambiguity of accidental heterofragment degeneracy. Composition tests here
cover fragment-preserving coordinates. The production orbital-response
owner must include the change of occupied orbitals and of all three source
vectors, with method-specific treatment of degeneracy.

With split fitting, carry the parent JK identity and the matching
correlation and F12 fit identities into the join. Each occupied component
must sum back to its parent scalar before use. The molecular adjoint adds
the conventional weighted triples response and the derivatives of both
MP2 partitions, including the orbital and fitting metric weights.

## Evidence

Independent Decimal differences at both 0.0004 and 0.0002 verify all three
energy response paths together. The tests also cover unequal fragment
scales, empty spin blocks, certified structural zeros, singular active
denominators, the CBS energy limit, and regular shrinking inputs down to
the FP64 subnormal range. The full scalar and cotangents lower together for
SM89. These results cover the orbital contraction and its reverse seeds.
The molecular source test in `tests/python/test_focal_tplus_mp2_sources.py`
checks the joined frame and the weighted source contraction. The
`tests/python/test_f12_tplus_ram_gate1.py` water control recomputes the
weighted MP2/F12 partitions at a signed displacement.

## Reference

[Kállay et al., J. Chem. Phys. 155, 034107 (2021)](https://doi.org/10.1063/5.0057426).
