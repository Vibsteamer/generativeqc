# Bounded nuclear response for unrestricted auxiliary h fitting

Status: implemented
Date: 2026-11-21

## Problem

The unrestricted DF correlation owner reused the raw Cartesian integral export for first nuclear derivatives. That path transfers coordinate-indexed metric and three-center tensors and converts all of them to the public AO representation. The h-shell auxiliary domain increases the number of these elements and needs the larger generated raised-moment policy.

## Decision

The CUDA unrestricted interaction source gives its fixed public-order three-center and metric cotangents to the existing synchronous weighted DF gradient adapter when an auxiliary shell has angular momentum 5. The generated device sink selects the six-term metadata domain and contracts weight tiles before transferring the (3N) gradient. The source admission charges the raw value tensors and the sink allowance. It no longer charges detached all-coordinate tensors for this path. Other angular domains retain their existing response owner.

## Invariants

The two cotangents use the same fitted interaction and metric cutoff as the forward reference. Both are full public AO layouts. The returned vector is the positive electronic derivative. ECP one-electron response, orbital relaxation and nuclear repulsion stay with their existing consumers. CUDA errors propagate through the adapter and the process never changes backend.

The public orbital-f plus ECP and auxiliary-h correlation gates remain independently controlled. Method registration requires physical qualification of those end-to-end combinations. A reported primitive or source response does not upgrade an unqualified molecular route.

## Evidence

Generated h derivatives have independent signed primitive/full-center checks in `tests/python/test_df_auxiliary_g_derivatives.py`. The bounded source admission and device metadata checks are in `tests/python/test_df_nuclear_sink_admission.py`. Compile verification covers both CUDA and CPU branches of the method adapter and interaction source.

The compact HBr/ECP10MDF independent reference uses 26 active electrons, 24 orbitals with f, and 235 fitting functions with one Br h contraction. Its retained g/h comparisons isolate the h contribution from the f-orbital change. The original private physical L20 diagnostic is job `2123869721592860672`. It belongs to its frozen numerical implementation. The new bounded response requires a separate physical comparison against retained HBr data before a public capability change.

## Revisit when

Compare the raw public-layout A and M weights against the retained two-step H-z controls, verify the completed four-role energy and forces on the same electronic state, and measure source staging and actual device peaks before widening production use.
