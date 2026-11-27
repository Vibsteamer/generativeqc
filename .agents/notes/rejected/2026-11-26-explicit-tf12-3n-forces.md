# Keep explicit Köhn (T)-F12 as an opt-in energy model

Status: rejected
Date: 2026-11-26

## Problem

Adding full Cartesian forces to the explicit extended-SP triples energy
would require differentiating the 33 A/Z2 terms and C2 coupling in the
extended basis, including the correlated and orbital response. The
occupied-scaled `(T+)` model already has a distinct force path using
the conventional triples pullback and the occupied MP2/F12 partitions.

## Decision

The explicit `(T)-F12` model remains a closed-shell energy experiment.
The FfgCompleted Z2 construction adds large extended-basis contractions
and memory pressure. Independent energy qualification of the valence
OptRI space is insufficient for Br/I semicore. Future derivative
qualification requires a conditioned complementary basis, bounded
work and independent signed nuclear displacements for the full model.

`GENERATIVEQC_ENABLE_EXPLICIT_TF12` defaults to OFF. Default builds
retain the C ABI symbol stubs and return
`GENERATIVEQC_STATUS_NOT_IMPLEMENTED` for explicit model calls.
Production `(T)` and `(T+)` are independent of this build option.

## Evidence

`tests/python/test_explicit_tf12_sp_gate1.py` checks the default stubs
and the opt-in energy algebra, including BH term decomposition and the
PySCF zero-coupling control. Those controls cover the energy formulation.

## References

- `CMakeLists.txt`
- `src/api/c_api_explicit_tf12.cpp`
- [Triples model scope](../implemented/numerics/2026-11-26-focal-triples-model-scope.md)
