# Native canonical unrestricted CC response

Status implemented

Date 2026-11-21

## Decision

The native conventional CC force owner builds its Hamiltonian in the exact
canonical spin frame. It slices the occupied and virtual blocks before gathering
chemist-to-physicist axes. Occupied Coulomb and exchange traces also use only
the occupied subdomains. The original general-frame program remains an
independent oracle.

The stationary CC parameter adjoint runs once. Linear canonical sources from
CC, conventional triples and the reference constraints then feed a separate
program for each raw integral or one-electron owner. Each raw source produces
its full cotangent and contributes matrix contractions to the general
alpha/beta rotation bars. The generated frame transport derives the symmetric
overlap weights and the antisymmetric orbital RHS. The ordinary and separate
JKFit sources retain distinct owners. The first orbital pass discards each
large source output after accumulating its rotation contribution. The nuclear
pass detaches the source weights and retires the executor before evaluating
the integral derivatives.

The synchronous CUDA arena selects the input offset for the currently
admitted action. The maximum joint input and graph footprint sets the device
arena, with the largest detached output charged separately. Device uploads
borrow input views over existing storage. The primal Pulay admission counts
the actual packed excitations and uses the concrete integral-provider request
shapes once available. A shape-only query without a bound provider retains
a conservative transform allowance.

The conventional energy-only CUDA route batches 32 independent canonical
triangle terms into each forward action. Its last panel uses the first
physical denominator and zero numerators for unused lanes. The number and
multiplicity of physical terms are unchanged. Work admission charges the
entire final panel including padding.

## Evidence

The grouped source program is compared with the general-frame CC and raw
cotangent oracles at identity for RHF, unequal UHF and empty beta occupancy.
The comparisons include separate JK and correlation fitting, arbitrary
canonical source weights, the pre-orbital and post-orbital constraints, every
raw cotangent and both rotation matrices.

For the standard Br ha-T 101-AO RHF shape, generated CUDA response storage
is 12.242 GB device and 14.722 GB with the largest detached output. The old
combined reservation was 56.823 GB. The 119-AO UHF shape uses 23.856 GB device
and 28.790 GB combined, down from 110.205 GB. The corresponding 104-AO and
122-AO iodine shapes are 16.601 GB and 32.014 GB combined. These figures
describe the executor. A complete endpoint also owns the converged reference,
DF providers, host amplitude and response state, and integral derivative
workspace. Physical L20 verification remains an independent gate.

The unchanged larger light-system cases can still have large dense Lambda
actions even when their nuclear pullback is partitioned. The existing
conventional Br/I f-orbital and h-auxiliary capability checks still apply.

## Host force admission

Force admission counts the unique raw and canonical maps that are live
during replay, Lambda and the source actions. Retired maps do not overlap
with the derivative scratch. The native provider query adds the maximum
initial raw block reservation across the separate physical fits. CPU
generated graph storage and CUDA executor storage are charged separately.

The already admitted native Lambda plan supplies the GMRES allocator's
workspace allowance. This keeps the existing 30-vector restart and physical
residual thresholds while avoiding the generic 64 MiB default for larger
packed amplitudes. The complete molecular byte limit and scalar work policy
continue to apply.
