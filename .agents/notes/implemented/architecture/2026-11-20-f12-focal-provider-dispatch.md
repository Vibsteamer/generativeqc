# Geometry-first F12 focal dispatch and reference ownership

Status: implemented
Date: 2026-11-20

## Decision

The focal entry resolves immutable orbital, JKFit, MP2Fit, F12Fit and
CABS records before dispatch. The surface identity binds ordered geometry,
charge, multiplicity, scalar ECP, electronic root and correlation space.
The basis identity also binds fitting assignments and the geminal exponent.
The dispatcher checks each returned identity and the executed backend.
`FocalPlan.collect` applies the same checks to independent jobs and uses
the planned summation order. An explicit protocol ID binds a stored plan
to the intended method and implementation.

Each distinct complete basis identity at a molecular center owns one
canonical DF reference. Keep the geometry, Coulomb/exchange plan, fitting
metric and spin occupations together through the orbital adjoint. A
displaced energy transports the center root and CABS rank, then evaluates
the primal energy on that frame. A copied orbital-only snapshot cannot
substitute for the live DF reference in the response.

The one-electron and molecular shell providers share normalized
real-spherical order with orbital AOs before the complementary blocks.
One-electron sources enumerate orbital and CABS shells only. Fitting
dimensions belong to the two- and three-center providers. The source
transpose reverses that order and contracts the physical-center derivatives.

## Consequences

The public assembler requires each electronic term and force contribution
to describe the same surface. A combined conventional CCSD(T) minus MP2
correction can reuse the grouped same-reference native result. A split
recipe needs the matching individual energy and force contributions.
A frozen fit layout, a separately converged determinant or a repeated HF
term would change the resulting derivative.

## Evidence

`tests/python/test_f12_focal_entry.py` exercises stale and incomplete
components, independent collection and center-to-displacement transport.
`tests/python/test_focal_molecular_provider.py` checks the physical LiH
component forces and retained CABS rank with signed displacements.
`tests/python/test_f12_molecular_reference_response.py` covers the
matched DF reference derivative. Build and provider setup are in
`docs/user/ccsdt_cbs.md`.
