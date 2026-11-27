# Decision: pass the endpoint device budget to CUDA F12 AOT kernels

Status: implemented
Date: 2026-11-26

## Problem

The conventional DF-MP2 Delta-CV CUDA ABI takes a `budget_bytes` argument in
which `0` selects fixed per-kernel caps: 1 GiB for chunked factor, J/K Fock and
CPHF stages, 2.5 GiB for the pair stage and 2.8 GiB for the J/K MO Fock,
pullback and nuclear source stages. Every production binding site bound the
kernels without a budget: the Delta-CV energy, the Delta-CV energy and forces,
the focal-source DF-HF fallback and the ordinary MP2 factor build. The
provider's `budget_mib` therefore neither limited nor enabled their device
workspace. A tight endpoint budget was exceeded silently, and a system needing
more than a fixed cap was refused even when the endpoint allowed it.

The CC-F12 pair pullbacks for the W, C and scalar V+B-X channels have the same
ABI convention, with a fixed 4 GiB cap for `0`. Their three accumulators had no
budget parameter, so the fixed-cusp CC force callers could not pass one.

## Decision

Bind each CUDA Delta-CV kernel set with `budget_mib << 20` from the provider
that owns the request. A CUDA binding rejects `budget_bytes=0`, both when it is
created and as a per-call override, instead of selecting the fixed native
caps. Host-reference kernels allocate no device memory and keep accepting `0`.

The pair-pullback accumulators take `budget_bytes`. When a CUDA kernel is bound
they require a positive value and forward it to every pair call. The spatial
RHF and UHF fixed-cusp CC force callers pass the provider's budget to all three
channels. The TensorIR and host-reference paths do not consume it.

The C ABI convention is unchanged: a direct ABI caller may still pass `0` for
the native defaults, and the direct admission cases keep covering them. An
admission failure keeps surfacing as the existing native status error. There
is no retry with a larger budget and no host or TensorIR substitution.

## Rejected alternatives

- Raising the fixed native caps still ignores the endpoint budget in both
  directions.
- Keeping `0` as "use the defaults" with a warning silently violates the
  endpoint memory contract.
- A required argument for host-reference and TensorIR callers adds a parameter
  with no device meaning. Only bound CUDA kernels require a positive value.

## Invariants

- Production code never forwards `0` to a CUDA Delta-CV kernel or CUDA pair
  pullback.
- Every CUDA Delta-CV kernel and pair pullback of one endpoint receives the same
  budget as the provider's other device stages, such as canonical-frame
  seeding.
- A positive budget is a cap. The kernel plans its tiles within it and refuses
  when its minimum workspace does not fit.

## Evidence

`tests/python/test_delta_cv_provider_budget_propagation.py` drives the
provider's Delta-CV energy, energy and forces, and focal-source DF-HF entries,
plus the ordinary MP2 factor build, against a fake CUDA C ABI that applies the
native admission rule. A 128 MiB budget with a 256 MiB requirement is refused.
A 4096 MiB budget admits a 3 GiB requirement that every old fixed cap refuses.
Each kernel observes the provider budget.

`tests/python/test_cc_pair_pullback_provider_budget.py` runs each accumulator
through a CUDA double that applies the native admission rule and delegates the
arithmetic to the host-reference kernel. A tight budget is refused, a 6 GiB
budget admits a 5 GiB requirement above the old 4 GiB cap and matches TensorIR,
and an omitted budget is rejected before any kernel call. The RHF and UHF
fixed-cusp CC force callers pass the provider budget to all three channels.

Both tests fail on their parent commits and pass with these changes.

On an earlier L20 comparison with the same native build, changing only
the Python budget transport changed the measured energies by at most
4.6e-15 Eh and forces by at most 1.1e-12 Eh/bohr across 88 pairs.
The executable admission tests above reproduce the cap and refusal
behavior using small inputs. Real-device capacity and timing depend on
the complete endpoint and its selected budgets.

## Consequences

With the default `budget_mib=1536`, the Delta-CV pair and J/K stages and the
pair pullbacks get less device workspace than their former fixed caps. They use
more tiles, or refuse a system whose minimum workspace exceeds the budget.
For a larger calculation, select a budget that admits the requested
workspace. Device chunk plans follow that budget.

## Revisit when

A shared device-memory lease or free-memory-aware native admission replaces
per-call budgets.

## References

- `src/posthf/f12/f12_dfmp2_cv_cuda.cu`: `kDefault*BudgetBytes` and
  `effective_budget`.
- `src/posthf/f12/f12_pair_pullback_cuda.cu`: `kDefaultDeviceBudgetBytes`.
- `python/generativeqc/_f12_cv_dfmp2_native.py`: `bind_delta_cv_native`.
- `python/generativeqc/_f12_cc_pair_pullback_native.py`:
  `checked_device_budget`.
