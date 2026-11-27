# Decision: admit single-pass FP32 triples W forces on the NVIDIA L20 only

Status: implemented
Date: 2026-11-27

## Problem

Signed differences of single-pass FP32 W energies amplify rounding by
`1/(2h)`. The error can increase as the step shrinks even when the
analytic result agrees with an independent FP64 derivative. Qualify
single-pass energies to 1e-9 Eh against FP64 and analytic forces to
1e-8 Eh/bohr against FP64 central differences. Record self-differences
as a diagnostic.

## Decision

- The three force entries (conventional triples and RHF and UHF occupied-scaled
  (T+)) admit a single-pass W request for production only when the native work
  runs on an NVIDIA L20 (`sm_89`). The request is refused before native work
  with `TriplesWPrecisionError` (a `ValueError`) on any other GPU and on the
  host backend.
- Admission matches the driver product name and the compute capability. The
  L40S, L4 and RTX 4090 share `sm_89` without the L20 evidence.
- The device is CUDA ordinal 0 after `CUDA_VISIBLE_DEVICES` filtering. The
  provider's native library and references use that ordinal. The driver query
  creates no context. A provider-private callable replaces the query in tests.
- A single-pass request whose native telemetry reports another executed W
  schedule raises `TriplesWPrecisionError` and returns no forces. This includes
  `fp64`, `fp32_split3` and the retained v1 path's `unreported`.
- Force telemetry records `device_name`, `device_architecture` and
  `preconvergence` alongside the executed `w_precision_schedule`. The host
  backend records `cpu` and `n/a`.
- `diagnostic_fp32_single_w_forces=True` still runs single-pass W on any
  device and tags the result `production_eligible=False`.
- Production validation and assembly admit single-pass force telemetry only
  with an L20 device record. They refuse tagged diagnostics, other devices and
  telemetry without a device record.
- Defaults are unchanged: FP64 W and `preconvergence="auto"`. The recipe requests any alternate W schedule explicitly.
- Energy-only single-pass W remains admitted on every device.

The bounded spatial RHF CUDA response can keep FP32 forward W while switching
its reverse W to FP64. Its execution telemetry reports this resource fallback
separately from the forward-derived schedule. Production single-pass forces
reject the fallback. UHF CUDA single-pass W already uses an FP64 pullback and
reports no fallback. The assembled force component's type determines whether
to apply this check even when a provider omits reverse W telemetry.

## Rejected alternatives

- Admitting every `sm_89` device would extend the L20 evidence to other
  products by architecture alone.
- Choosing single-pass W automatically on the L20 would change the upstream
  default and hide the precision choice from the recipe.
- Substituting FP64 when single-pass W is unavailable would silently change
  the requested precision and its cost.
- An environment variable would hide the choice from the call site and from
  result provenance.
- Keeping the self-differenced single-pass FD as a gate would keep refusing
  forces whose FP64-reference error is two orders below the threshold.
- Accepting results without a device record would let a direct low-level call
  or an older result enter production unqualified.

## Invariants

- A production single-pass W force result carries an L20 device record and the
  executed `fp32_single` schedule.
- Non-L20 single-pass force requests fail before native work unless they are
  explicit diagnostics. Diagnostics always carry the tag.
- The FP64 default and `fp32_split3` force schedules run on every device and
  record the same provenance.
- Acceptance of single-pass W compares with FP64. The self-differenced FD is a
  diagnostic.

## Evidence

- Earlier L20 controls on H2O, IHB93 and HBr had analytic
  forces within 1.2e-9 to 1.6e-9 Eh/bohr of the FP64 reference.
- `auto` preconvergence on CNa3O3 differs from all-FP64 preconvergence by
  4e-12 Eh and 5.4e-11 Eh/bohr.
- `tests/python/test_rhf_triples_fp32w_gate1.py`, CPU emulation of single-pass W
  for H2O and HBr, (T) and (T+), two axes, h = 2e-4 and 1e-4 bohr:
  - displaced energies within 1.3e-10 Eh of FP64;
  - analytic forces within 4.2e-11 Eh/bohr of the FP64 central difference;
  - self-differenced error 2.8e-8 to 6.4e-7 Eh/bohr, roughly doubling when h
    halves, recorded as a diagnostic.
- `tests/python/test_triples_single_pass_fp32_force_policy.py` covers each
  admission path on CPU with an injected device. It covers L20 production, the
  A100, L40S and host refusals, defaults, the diagnostic, an executed-schedule
  mismatch, `singlepoint` and assembly. `tests/python/test_cuda_device_selection.py`
  covers the driver name query.

## Consequences

Production recipes must request single-pass W explicitly on the L20. A100 and
other devices keep FP64 or `fp32_split3`. Results produced before the device
record existed cannot enter production assembly with single-pass forces.

## Revisit when

Another product passes the FP64-reference gates and should join the admitted
devices. Revisit also if the native library reports its own device identity, or
if a single-pass schedule's FP64-reference force error approaches 1e-8 Eh/bohr.

## References

- `python/generativeqc/_f12_triples_w_policy.py`
- `python/generativeqc/_cuda_device.py`
- `docs/user/ccsdt_cbs.md`
