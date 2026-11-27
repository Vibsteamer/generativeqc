# Molecular triples method selection and explicit energy scope

Status: implemented
Date: 2026-11-26

The focal dispatcher keeps the conventional `(T)`, Kállay occupied-scaled
`(T+)` and explicit Köhn extended-SP `(T)-F12` components distinct. Each
alternative includes its own triples increment exactly once. A bare
triples basis selects the conventional correction, subject to any
explicitly qualified recipe predicate for `(T+)`.

With `GENERATIVEQC_ENABLE_EXPLICIT_TF12=ON`, a closed-shell RHF energy
request can evaluate the stationary 33-term extended-SP model. It uses
the matched CCSD-F12b parent and physical triples coordinates with the
C2 coupling and FfgCompleted Z2. An independently accumulated
same-cardinal doubles contribution can reuse the parent. The basis,
CABS rank, frozen spaces, normalization and execution precision belong
to the component's identity or telemetry.

The experimental path admits singlet energies only and refuses Br/I
semicore correlation with the current complementary-space support. The
default-off CMake option and the compatibility behavior are recorded
in the [explicit-force decision](../../rejected/2026-11-26-explicit-tf12-3n-forces.md).

`tests/python/test_focal_triples_alternatives.py` checks contribution
selection and reuse. `tests/python/test_explicit_tf12_sp_gate1.py`
checks the default-disabled contract and opt-in energy controls,
including GeCCo BH term decomposition and the PySCF zero-coupling limit.
