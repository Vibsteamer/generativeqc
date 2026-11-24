# Decision: Join focal (T+) partitions on one retained determinant

Status: implemented
Date: 2026-11-22

The focal provider couples a conventional triples partition to the MP2 and F12
partitions only when they share the retained canonical determinant, correlation
space and fit. It also checks that the lower occupied contributions recover
their unscaled energies. A duplicated reference with the same nominal geometry
can use a different canonical orbital frame and cannot supply this join.

RHF groups the two spin contributions into spatial orbitals before evaluating
the scalar and gives each spin contribution the same resulting weight. UHF
retains the independent spin vectors, including a zero-length block. The
existing TensorIR program evaluates the scalar and all three cotangents using
the focal FP64 CUDA runner when requested. Existing host consumers use the
reference interpreter.

Tests exercise frame and fit mismatch refusal, partition closure, RHF seed
distribution, the empty UHF block and refusal of a host executor for a CUDA
request. The weighted native triples and nuclear-response owner remains a
separate implementation gate for the public energy and force endpoint.
