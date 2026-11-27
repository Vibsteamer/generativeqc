# Decision: Unrestricted CC CUDA memory admission

Status: implemented
Date: 2026-11-22

## Problem

Preparing a CUDA UCCSD(T) plan charged the CPU contraction shapes and the
reference's full nuclear derivative allowance. A pure energy call then created
its CUDA graph arena before building MO feeds, overlapping the graph with
temporary CUDA transform buffers.

## Decision

Use the CUDA primal shape and value-only DF reference capacity at preparation.
Admit the full derivative reference and force resources when the caller asks
for forces, before preparing the respective providers. Build MO feeds before
allocating the graph arena. The transforms return synchronized host buffers
and release their temporary device buffers before this allocation. The graph
checks its own live device allocation against free memory on the selected
device. Detached host result storage remains part of the combined numeric cap.

## Invariants

The descriptor's combined host and device numeric capacity bound remains
conservative and unchanged in meaning. The work-policy cap, prepared
Hamiltonian, spin occupations, stationary and replay checks, and forces
requested by the caller retain their existing requirements.

## Evidence

The original water–hydroxide IHB100x2_93 and _94 geometries each use 134
ordinary ha-T orbitals with [10, 10] occupations. The frozen shape probe
estimated about 37.3 GiB of graph allocation and about 65.1 GiB of combined
capacity before the reference's separate derivative-capacity allowance. The
shape probe is an estimate. It provides no measured memory or cost for the
changed path. The native admission test exercises a CUDA value plan and the
derivative fitting refusal without allocating on a device.
