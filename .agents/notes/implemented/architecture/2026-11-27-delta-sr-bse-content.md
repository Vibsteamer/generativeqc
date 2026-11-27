# Decision: pin ΔSR Basis Set Exchange records by element content

Status: implemented
Date: 2026-11-27

## Problem

The ΔSR basis builders read from the installed Basis Set Exchange package.
Recording the package version and export checksum in the basis identity could
identify a different input after construction, while the calculation could
already use the changed data.

## Decision

`python/generativeqc/_delta_sr_bse.py` verifies one spherical element record
before the ΔSR builder consumes it. The pins were generated from the Basis Set
Exchange 0.12 JSON exports for the production orbital and fitting bases and the
additional reference-test bases. Each export must contain exactly the requested
element. Canonical content includes shell exponents and contraction coefficients
as binary64 values, AO representation, and the scalar ECP for Br and I.

The package version and raw export checksum remain basis provenance. Content
verification accepts equivalent exports with different JSON formatting or
decimal spelling, and rejects changed values even if the package version is
unchanged. Each additional basis or element requires a verified content hash.
The builders retain their existing decontraction and assembly rules.

## Evidence

An independent implementation computed the same 54 hashes directly from the
Basis Set Exchange 0.12 exports. All four builder configurations reproduced their
previous element content and basis identities. They reproduced the numerical
element data in 140 sealed production records. The required tests cover the
production element set and reject changed orbital, ECP and fitting exponents,
exports with an extra element, and requests without a pin.

## Revisit when

When a calculation needs another basis or element, verify the requested JSON
record with `bse_element_records` and `element_content_sha256`, review its
scientific qualification, and record the verified package release with the
new pin.
