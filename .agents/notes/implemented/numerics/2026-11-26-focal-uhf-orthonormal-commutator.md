# Decision: Isolate focal UHF orthonormal commutator stopping behind an explicit opt-in flag

Status: implemented
Date: 2026-11-26

## Problem

On diffuse and near-linearly-dependent basis sets (such as focal DF-UHF on aug-cc-pVQZ /
aug-cc-pV5Z), the AO-basis commutator maximum `max|FDS - SDF|` and the whitened RMS norm can
both fall below `density_tolerance` (e.g., `1e-11`) while the orthonormal MO orbital-gradient
maximum `max|X^T (FDS - SDF) X|` remains `O(1e-8)`. Because focal post-HF methods such as
MP2-F12 are non-variational with respect to orbital rotations, residual orbital gradients at
that level leak into correlation energies.

Earlier fixes gated this check heuristically on `export_physical_reference && density_tolerance <= 1e-10`
on CPU (`src/scf/solver/mean_field_driver.cpp`) and `diis_history >= 2 && energy_tolerance <= 1e-12 &&
density_tolerance <= 1e-10` on CUDA (`src/scf/cuda/df_uhf_scf.cpp`). Those implicit predicates:

1. Altered stopping criteria for unrelated upstream UHF consumers (`ump2_method.cpp`,
   `uccsd_method.cpp`) whenever tight tolerances were requested; and
2. Diverged between CPU and CUDA (different threshold predicates, no CUDA Graph invalidation
   when the mode toggled, and silent omission when `diis_history < 2`).

## Decision

Introduce an explicit opt-in field `ScfOptions::enforce_orthonormal_uhf_commutator` (default
`false`) and unify CPU and CUDA DF-UHF behavior around it:

- `ScfOptions::enforce_orthonormal_uhf_commutator` defaults to `false`, restoring 100% baseline
  UHF stopping for all non-focal consumers (`ump2_method.cpp`, `uccsd_method.cpp`, standard UHF).
- Only the focal DF reference driver (`src/api/c_api_f12_focal_reference.cpp`) opts in by
  setting `options.enforce_orthonormal_uhf_commutator = true` (leaving coarse recovery stages
  `stage2` / `stage2b` at `false` before final tight polish).
- CPU `run_uhf_host_plan` (`src/scf/solver/mean_field_driver.cpp`) gates the `max|X^T(FDS-SDF)X| <=
  density_tolerance` check solely on `options.enforce_orthonormal_uhf_commutator`.
- CUDA `run_cuda_density_fitting_uhf_device_scf` (`src/scf/cuda/df_uhf_scf.cpp`,
  `src/scf/cuda_density_fitting.hpp`, `src/scf/cuda_density_fitting_stub.cpp`, `src/scf/rhf.cpp`)
  threads the same explicit boolean parameter:
  - When `true`, requires `diis_history >= 2` (returning `GENERATIVEQC_STATUS_INVALID_ARGUMENT`
    otherwise so `rhf.cpp` falls back cleanly to its host DIIS loop, which enforces the exact
    same orthonormal max-abs commutator bound);
  - Whitens the idle post-DIIS residual copy in place to `X^T(FDS-SDF)X` for both spins before
    `update_device_uhf_convergence_kernel` evaluates the max-abs criterion; and
  - Tracks `enforce_orthonormal_uhf_commutator` in `PersistentScfState` and invalidates the
    captured CUDA Graph whenever the flag changes.

## Rejected alternatives

- **Keep implicit tolerance heuristics (`export_physical_reference && density_tolerance <= 1e-10`)**:
  Rejected because it silently changes convergence work and iteration counts for upstream UMP2
  and UCCSD methods that never opted into focal F12 stopping semantics, and causes CPU/CUDA
  threshold drift.
- **Unconditionally enforce orthonormal max-abs commutator on all UHF solves**:
  Rejected because ill-conditioned general UHF solves would incur extra iterations and risk
  changing existing upstream UHF convergence behavior without scientific necessity outside
  non-variational focal F12 consumers.
- **Silently skip the check on CUDA when `diis_history < 2`**:
  Rejected because failing open would let non-stationary focal references pass undetected when
  device DIIS is disabled; failing with `GENERATIVEQC_STATUS_INVALID_ARGUMENT` routes the solve
  through the host DIIS fallback that enforces the same orthonormal bound.

## Invariants

- `ScfOptions{}.enforce_orthonormal_uhf_commutator` is `false`.
- Non-focal UHF callers (`export_physical_reference = true` with tight tolerances and default
  flag) retain identical stopping behavior to baseline UHF.
- Focal DF-UHF solves (`enforce_orthonormal_uhf_commutator = true`) do not accept convergence
  until `max(|X^T(F_a D_a S - S D_a F_a)X|_max, |X^T(F_b D_b S - S D_b F_b)X|_max) <= density_tolerance`
  on both CPU and CUDA paths.
- Changing `enforce_orthonormal_uhf_commutator` on a reused CUDA DF plan invalidates and
  rebuilds the cached CUDA iteration graph.

## Evidence

- Native CTest `generativeqc_focal_uhf_orthonormal_commutator_tests`
  (`tests/native/test_focal_uhf_orthonormal_commutator.cpp`) constructs a diffuse doublet
  fixture where AO-basis `max|FDS-SDF| <= 1e-11` is reached while `max|X^T(FDS-SDF)X| > 1e-11`,
  and verifies both baseline invariance under default `false` (even with
  `export_physical_reference = true`) and strict orthonormal convergence under `true`.
- `tests/python/test_df_uhf_error_details.py` verifies CUDA DF-UHF bucket error propagation
  with the updated device-SCF signature.

## Consequences

- One boolean option added to `ScfOptions`, `PersistentScfState`, and the DIIS overload of
  `run_cuda_density_fitting_uhf_device_scf` (with default `false`).
- Zero extra device memory allocation on CUDA (reuses the idle post-DIIS residual buffer and
  temporary matrix already owned by `PersistentScfState`).

## Revisit when

- If all correlated UHF consumers adopt a shared orthonormal MO-gradient stopping contract
  across CPU and CUDA with full upstream benchmark qualification, the focal-only flag can be
  unified with that broader convergence policy.

## References

- `src/scf/types.hpp`
- `src/scf/solver/mean_field_driver.cpp`
- `src/scf/cuda/df_uhf_scf.cpp`
- `src/scf/cuda/df_scf_state.hpp`
- `src/scf/rhf.cpp`
- `src/api/c_api_f12_focal_reference.cpp`
- `tests/native/test_focal_uhf_orthonormal_commutator.cpp`

## Update: evaluation order (2026-11-27)

The decision above stands: same quantity, same tolerance, same opt-in. The host
solve (`run_uhf_host_plan`) now evaluates it as `max|M - M^T|` with
`M = (X^T F)(D (S X))`, because forming `F D S - S D F` first and then the
congruence put a rounding floor above `1e-11` on diffuse bases. See
`2026-11-27-focal-uhf-commutator-evaluation-order.md`. The CUDA DF-UHF solvers
still use the earlier order.
