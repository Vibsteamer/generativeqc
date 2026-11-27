# Decision: Admit scalar-ECP orbital g at the C ABI only on fitted electronic states

Status: implemented
Date: 2026-11-26

## Problem

`generativeqc_system_create_ecp` admitted spherical orbital `g` shells whenever
the atoms matched a small-core Br/I halide topology, independent of charge,
multiplicity and electron count. The fitted domain that the qualification
evidence covers is state dependent: `SmallCoreHalideComposition::
fitted_orbital_angular_limit` and its Python mirror
`ecp._fitted_ecp_orbital_angular_limit` exclude, for example, CH3I 2+ singlets,
parity-inconsistent multiplicities, dianions, triplets and charges outside
`-1..1`. Direct C ABI consumers bypass the Python gate, so they could build `g`
systems outside the qualified domain. Sweeping 18 topologies (qualified halides
and near misses) over charges `-2..2` and multiplicities `1..4`, the previous
ABI admitted `g` for 192 of 356 spherical states that the fitted domain rejects.

## Decision

The ABI computes post-ECP electrons as the neutral valence count minus the
charge and asks `fitted_orbital_angular_limit(charge, multiplicity, electrons)`.
Orbital `g` requires a spherical basis, a limit of at least 4 and the generated
`ecp_max_orbital_angular`; every other ECP system keeps orbital `s..f`. The
rejection keeps `GENERATIVEQC_STATUS_INVALID_ARGUMENT` and its detail text,
which Python callers match.

## Rejected alternatives

- Keep the topology rule and rely on Python gating: direct ABI consumers and
  other language bindings never pass through Python.
- A separate C ABI table of admitted states: a second authority drifts from
  the one that the Python mirror and the native consumers already share.

## Invariants

- `SmallCoreHalideComposition::fitted_orbital_angular_limit` stays the single
  authority; Python mirrors it and the ABI calls it.
- Orbital `s..f` admission does not depend on the electronic state.

## Evidence

- `tests/native/test_ecp_fitted_g_admission.cpp`: public ABI allow/reject
  matrix (out-of-range charge, parity mismatch, NaI, multi-halide, CH3I 2+
  singlet, C4H6Br2O dianion, Cartesian, large-core and non-halide ECPs).
- `tests/python/test_ecp_fitted_g_admission.py`: the ABI decision equals the
  Python mirror for all 356 swept spherical states plus Cartesian anchors;
  0 disagreements after the change.

## Consequences

ABI users that created `g` systems outside the fitted states now receive
`INVALID_ARGUMENT` instead of an unqualified calculation.

## Revisit when

The fitted domain gains states or elements; extend the C++ authority and its
Python mirror together.
