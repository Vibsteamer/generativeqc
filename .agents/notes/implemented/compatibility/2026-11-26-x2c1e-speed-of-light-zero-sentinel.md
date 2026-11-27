# Decision: Only 0 selects the CODATA speed of light at the X2C-1e C ABI

Status: implemented
Date: 2026-11-26

## Problem

`generativeqc_system_set_x2c1e_v1`, `generativeqc_system_x2c1e_hcore_v1`,
`generativeqc_system_x2c1e_hcore_derivatives_v1` and
`generativeqc_system_x2c1e_weighted_gradient_v1` documented
`speed_of_light <= 0` as "select CODATA 137.03599967994 a.u.". Non-finite
values were rejected, but any negative value, typically a sign error or an
uninitialized argument, silently ran a CODATA calculation that the caller had
not requested and could not distinguish from an intentional default.

## Decision

`resolve_speed_of_light` maps exactly 0 (either sign) to CODATA, passes finite
positive values through, and rejects negative or non-finite values with
`GENERATIVEQC_STATUS_INVALID_ARGUMENT` before the system or any output buffer is
written. `ucc_fitting.h` states the rule on all four entry points.

## Rejected alternatives

- Keep `<= 0` as the sentinel: preserves the silent substitution for sign
  errors.
- Remove the sentinel and require an explicit value: breaks callers that pass 0
  for the default, with no accuracy gain because 0 is never a physical speed of
  light.
- Accept negative values with a warning: the ABI policy is fail-closed.

## Invariants

- 0 and -0 produce results identical to an explicit CODATA value.
- A rejected value changes neither the system nor any output buffer.

## Evidence

- `tests/native/test_x2c1e_speed_of_light_abi.cpp`: on all four entry points and
  both `delta_only` settings, 0 and -0 equal an explicit CODATA value, 100 a.u.
  changes the result, and -1, -CODATA, the smallest negative subnormal, NaN and
  both infinities return `INVALID_ARGUMENT` with the system and outputs
  unchanged.
- `tests/python/test_x2c1e_speed_of_light_abi.py`: the same contract through the
  Python bindings.

## Consequences

ABI users that passed a negative value to request CODATA now receive
`INVALID_ARGUMENT` and must pass 0. Repository callers pass 0 or a positive
value.

## Revisit when

The ABI gains a separate way to request the default constant; the 0 sentinel
can then be removed in a new ABI version.
