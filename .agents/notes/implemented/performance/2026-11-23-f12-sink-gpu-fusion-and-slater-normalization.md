# Bounded spherical F12 integral and fixed-cusp pair contractions

Status: implemented
Date: 2026-11-23

## Decision

The spherical shell C ABI packs and unpacks Cartesian panels on device.
Separable angular transforms, fitting-shell slabs and direct nuclear
accumulation avoid retaining full Cartesian AO values and gradients.
Primitive geometry and radial moments are shared within a shell batch.
The small two-center metric keeps the exact contraction order because
Coulomb whitening can amplify a changed rounding order.

The planner computes a workspace envelope from the largest simultaneously
allocated spherical and Cartesian panels. The widest panels can occur
in different slabs, so bounding each slab by itself underestimates the
allocation and can make admission nonmonotone. Seed the shared estimate
with the widest single fitting shell and check the running envelope
before selecting a slab boundary. Failure below the single-shell
minimum reports that minimum. At or above it, the workspace is bounded
by the caller's budget and the nominal 1.5 GiB slab limit, with the
single-shell minimum admitted when it exceeds the nominal limit.

The fixed-cusp W, C and scalar pullbacks share the orbital projection,
extended-Fock and Coulomb-metric response. The non-Coulomb projection
contracts the narrow occupied index first. Streams keep one fitting panel
live and reload it when a later contribution needs it. Exact RHF
alpha/beta symmetry shares work in the reverse contraction, while mixed
spin ordered products retain their distinct raw and fitted legs.
Host-side production buffers obey the same independent host-resource
admission as the rest of the provider. Device workspace limits are
transported explicitly by the endpoint.

Normalization is a part of the operator identity. The shared Python/C++
default is the cusp-normalized Slater factor and the Psi4 reference
convention remains explicitly selectable.

## Evidence

`tests/python/test_f12_sink_gpu_gate1.py` checks the spherical values
and weighted nuclear derivatives against the Cartesian route on the
same source, the normalization scale and signed forces, and the stream
and pair pullback parity. Its synthetic s/g test sweeps the slab budgets
and checks monotonic admission against the Cartesian route.
`tests/python/test_f12_contracted_integrals.py` checks contracted
sources against the Libint2 fixtures. The provider-budget tests in
`tests/python/test_cc_pair_pullback_provider_budget.py` distinguish
minimum-workspace refusal from a valid caller-supplied larger budget.

## References

- `src/posthf/f12/f12_molecular_integral_sink.cu`
- `src/posthf/f12/f12_pair_pullback_cuda.cu`
- `python/generativeqc/_f12_focal_cc_fixed.py`
- [Device budget transport](../architecture/2026-11-26-f12-endpoint-device-budget-transport.md)
- [Geminal normalization](../numerics/2026-11-23-f12-additive-cv-and-geminal-normalization.md)
