# Decision: bound the exact rational fallback of the compensated CABS products

Status: implemented
Date: 2026-11-26

## Problem

`_ozaki_mm` evaluates host double-double products for the full-rank CABS
overlap cotangent and the compensated CABS frame checks on diffuse bases.
Ozaki slicing can fail to represent its inputs in three ways: scaling overflow,
a partial product overflow, or a tail still above 1e-15 |result| after 16
slices per operand. In each case it escalated to `_exact_mm_dd`, a Python
`fractions.Fraction` triple loop with no size limit. One rational multiply-add
costs several microseconds, so one escalated product at production CABS
dimensions stalls an endpoint for minutes to hours.

## Decision

- The exact fallback admits at most `EXACT_PRODUCT_MULTIPLY_ADD_LIMIT = 2**18`
  (262,144) dense multiply-adds m·k·n. The check precedes any rational work.
- Above the bound `_ozaki_mm` raises `CompensatedProductWorkLimit`, a
  `RuntimeError` exported from `generativeqc.experimental.f12_focal`. It
  records `shape`, `trigger` (`scaling_overflow`, `partial_product_overflow`
  or `slice_limit`), `multiply_adds` and `limit`.
- A refusal returns no result. Callers propagate it instead of using the
  unconverged Ozaki sum or a plain FP64 product.

## Rejected alternatives

- Return the Ozaki result with a warning or precision flag: it silently
  downgrades the verified precision that the CABS checks and cotangents rely on.
- Fall back to FP64 or long double: plain FP64 loses more than nine digits to
  cancellation on diffuse QZ frames, which is why the compensated path exists.
- A wall-time limit: not reproducible across hosts. A work count is.
- An environment variable to raise the bound: it would hide the condition the
  refusal reports.
- A faster exact kernel (aligned integer mantissas, or `math.fsum` over exact
  two-product splits): it raises the admissible size but still needs a bound,
  and no physical input observed so far reaches the fallback.

## Invariants

- The work bound is enforced before rational work starts.
- A refusal never returns a partial or lower-precision product.
- Products that stay within the Ozaki range are unchanged.

## Evidence

Measured 2026-11-26 with CPython 3.11.18 and NumPy 2.4.6, 4 BLAS threads.

| Measurement | Result |
|---|---|
| Rational cost, dense random products (8³ to 64×64×32) | 3.5–4.7 µs per multiply-add |
| At the bound: overflow-adjacent dense 64³ | 1.35 s |
| At the bound: deep-scale 102×20×128 | 0.20 s |
| Previous code, water cc-pVQZ-F12/OptRI overlap × frame shape (341×341×186, 2.16e7 multiply-adds), overflow trigger | 72.2 s for one product |
| Same input with the bound | refused in under 1 ms |

Escalation frequency in existing tests: 773 Ozaki products over 184 tests in
eight CABS projector, frame response, molecular provider and AOT endpoint
modules (largest 570,834 multiply-adds, 0.67 s total) escalated once, in the
deliberate 1×20×1 deep-scale counterexample (20 multiply-adds). Twenty-seven
of those tests stopped early for unrelated AOT environment and library reasons
in the CPU-only run.

Representative QZ frames: water and methanol with cc-pVQZ-F12/OptRI and
aug-cc-pVQZ/aug-cc-pVQZ-OptRI (341–774 AOs, full CABS rank 186–430, overlap
condition 7e7–1e9) did not escalate in compensated orthonormality, frame
cotangent products or the full-rank overlap cotangent (at most 1.7 s each).
The overlap cotangent agreed with the closed form for
K = S⁻¹ − embed(S_oo⁻¹) to 2.9e-10 relative or better; orthonormality errors
were about 1e-12. Had methanol aug-cc-pVQZ escalated, one 774×774×430 product
would cost about 15 min, and the full-rank overlap cotangent forms six products.

The bound admits every observed escalation with a 13,000× margin and costs at
most 1.4 s in the slowest measured form. It is 45× below the smallest QZ
production product (186×341×186). A production-sized escalation means
overflow-adjacent magnitudes or a within-row dynamic range beyond the 256–336
bits that 16 slices resolve, and such inputs are refused in milliseconds.

## Consequences

An input that the previous code would eventually have resolved exactly at
production size now fails the endpoint with a structured error. No such input
was observed in tests or representative QZ frames.

## Revisit when

- A physical input class reaches the exact fallback at production size. The
  representative-QZ regression test refuses rational work and would detect
  it. Prefer a faster exact or wider Ozaki path over raising the bound.
- The Ozaki slice budget or tail tolerance changes.

## References

- `python/generativeqc/_f12_focal_projector.py`
- `tests/python/test_compensated_product_exact_bound.py`
