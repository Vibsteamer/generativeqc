# Generated geminal DF center derivatives

Implemented on 2026-11-18.

## Equations and ownership

Two- and three-center F12 integrals reuse the generated DF angular expressions
and their physical-center derivatives. The existing native fitted-Slater
provider supplies Hermite moments for the geminal, geminal over distance,
geminal product and gradient product. All these moments obey
`dM_n/dT = -M_(n+1)`. Nuclear differentiation also includes the Gaussian product
decay, the product-center shifts and the radial argument. Exponents and geminal
range parameters remain fixed under a nuclear displacement.

The compiler emits FP64 host/device functions for an explicitly requested
Cartesian component. Normalization, spherical transformations and contraction
coefficients remain the caller's responsibility. The current declared angular
domain follows the shared DF owner through orbital g and auxiliary h, including
the extra radial order needed for the nuclear derivative. Missing factors in
two-/three-center integrals are explicit unit slots in the radial provider.

Reusing this DAG keeps the center and translation conventions shared with the
ordinary DF derivatives. Duplicating an angular recurrence for F12 would create
another numerical owner that needed independent maintenance.

## Independent evidence

The checked-in fixture contains 108 primitive cases from Libint2 in the Psi4
1.11 environment. It covers nine Cartesian signatures, four operators and three
range parameters, 0.9, 1.4 and 2.2 inverse bohr. The largest signature is ggh.
Libint2 uses its four-center operator with explicit unit shells for all cases.
Its specialized three-center geminal dispatcher lacks that operator route in
this build. Primitive coefficients are unnormalized. The derivative reference
uses independently evaluated raised/lowered Gaussian integrals.

The host build agrees with all values and nuclear derivatives to a maximum
scaled error of 9.49e-15. The scale for one primitive is the larger of one and
the largest absolute value or gradient component. Two finite-difference steps
of 2e-4 and 1e-4 bohr give maximum scaled residuals of 1.10e-8 and 2.73e-9.
The generated sum over center derivatives satisfies translation invariance.

The mandatory test compiles a smaller two-/three-center subset. The benchmark
runs the complete fixture and reports compilation, cold and warm batch times
on the explicitly selected backend. The L20 follow-up is registered separately
with its source and input identities. Actual device results qualify only the
registered primitive panel. Complete molecular force qualification also needs
contracted spherical source integrals, the fitting solve, SCF response, CABS
response and assembly with the other energy components.

## Reproduction

```sh
PYTHONPATH=python:. python -m pytest -q tests/python/test_f12_df_integrals.py
GENERATIVEQC_TEST_CUDA=1 PYTHONPATH=python:. python benchmarks/f12_df_integrals.py \
  --arch sm_89 --fixture tests/reference_data/f12/libint_df_primitives.json \
  --cache /tmp/f12-df-primitives --output /tmp/f12-df-primitives.json
```

The external reference export and complete raw test artifacts are under the
local `validation/native-f12-angular-response-20261118` directory. The fixture
records the source digest of the Libint2 adapter and the independent archived
geminal parameters. It contains numerical data and no external implementation.

## Remaining production work

The current generated component route supplies the source derivatives needed
by the molecular pullback. A production shell scheduler must reuse geometry and
radial work across components, contract final weights early and retain a bounded
fallback. The ggh test emits about 2.2 MB of combined test source across the
whole panel. Measure compilation and the complete molecular endpoint before
selecting this component schedule as a production default.
