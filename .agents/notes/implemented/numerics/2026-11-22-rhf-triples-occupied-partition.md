# Decision: Reuse restricted DF triples tiles for the occupied partition

Status: implemented
Date: 2026-11-22

## Decision

The (T+) energy uses the occupied-orbital partition of the conventional (T)
scalar. Equation (13) of Kállay et al. weights each physical triplet by the
average of its three orbital ratios. The weighted energy is consequently the
dot product of the ratios with a vector that credits one third of each
conventional tile energy to each occupied slot, including repeated slots.

The existing restricted CUDA owner already stores one scalar per occupied
tile in canonical index order. An opt-in device reduction gathers these
values without rebuilding W or V. One block owns each occupied orbital and
visits unordered pairs of remaining indices. Duplicate occurrences in the
same tile enter as an integer count. The output occupies one FP64 vector on
device and host and remains charged to the numerical budget. Ordinary (T)
uses its existing scalar path. The native DF-RCCSD(T) energy owner requests the
partition through its internal correlation demand and evaluates it from the
same retained SCF, fitted Hamiltonian, and converged CCSD amplitudes.

The companion TensorIR scalar accepts separate canonical weights for each
slot. Its reverse yields the weighted conventional cotangents and three
orbital-ratio seeds. Orbital-indexed ratios depend on the canonical occupied
frame. A full implementation must include responses of those orbital
projections in addition to the existing (T) same-space Fock inverse response.
The paper's Appendix B describes reference-frame dependence when
accidentally degenerate orbitals from unlike fragments can mix freely.

## Evidence

The independent ordered virtual sum, two-fragment direct sum, and analytic
scalar pullback are checked in `tests/python/test_df_occupied_triples.py`.
The opt-in native-device probe exercises the same case on a fresh CUDA build.
The two patched CUDA translation units compiled for SM89 with CUDA 12.9 and
ccache. The local host has no live CUDA device for the native probe.

## References

- [Kállay et al., Eq. (13) and Appendix B](https://doi.org/10.1063/5.0057426)
