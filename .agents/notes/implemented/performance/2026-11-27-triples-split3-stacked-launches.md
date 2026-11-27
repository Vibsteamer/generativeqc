# Decision: Stack compensated split3 W limb products into fewer launches

Status: implemented
Date: 2026-11-27

## Problem

An L20 profile of the CNa3O3 ha-D frozen-core (T+) energy and force reported spatial (T) 12x and Lambda 2.4x slower on the integration build with per-pair split3 launches than on the earlier build, before explicit workspace configuration for the DF-UCC v2 source handle. The intervening audit build set that handle's cuBLAS workspace to zero. A microprobe found non-split-K kernels up to 84x slower on small-output, long-reduction GEMMs, making the zero-workspace change a suspected cause.

With single-pass FP32 W and auto preconvergence on the same L20, the earlier build, the zero-workspace audit build and the later integration build with a DIIS-only source workspace took 692, 688 and 690 s, respectively. With single-pass FP32 W and FP64 preconvergence, the earlier and later builds took 1129 and 1127 s. The old profile used single-pass FP32 W and auto preconvergence. The later profile used `fp32_split3` W and FP64 preconvergence, which explains the Lambda difference (650 s versus 276 s). The qualification for the later build exercised FP64 preconvergence and checked for zero FP32 CCSD or Lambda work.

The (T) difference was real, though, and the schedule caused it. For every 64-wide K segment, `fp32_split3` issued nine pedantic SGEMMs, nine pack audits and nine FP64 combines. Reverse W products have K = v^2, so each launched tens of thousands of kernels. CNa3O3 (T) took 436 s with split3 and 35 s with single-pass FP32 W, for the same 31.7 million FP32 products.

## Decision

`src/cc/df_triples_compensated_w.cuh` packs the three limbs of each operand vector into one stacked left block, `[limb * m + row, t]`, and one stacked right block, `[t, limb * n + column]`, per segment. It then chooses the fewest launches that fit the split-W scratch already reserved for the schedule:

- `batched`: one strided-batched SGEMM forms all nine limb pairs for as many segments as fit, and one combine adds them;
- `rows`: each left limb multiplies the stacked right block, so a segment takes three SGEMMs; and
- `pairs`: the original per-pair loop, which needs the least scratch.

Every schedule packs the same digits and adds the same FP64 terms in the original order: segment, then left limb, then right limb. FP32 represents the high and middle partial sums exactly, so only the residual-limb FP32 rounding can depend on which SGEMM kernel ran.

## Production precision

The November 27 precision decision specifies `auto` CCSD/Lambda preconvergence on A100 and L20. A100 uses FP64 W. L20 uses `fp32_single` W when the recipe explicitly requests it. The production analytic-force entry points reject single-pass W on A100, and the provider reports failures without silent fallback. The L20 acceptance gates compare energy to FP64 within 1e-9 Eh and force to FP64 finite differences within 1e-8 Eh/bohr. Same-mode finite differences are diagnostic. Results record the W precision, preconvergence mode and card type. The stacked launches apply to the optional `fp32_split3` mode.

## Rejected alternatives

- Adding cuBLAS workspace to the RHF (T) and Lambda handles. Same-card, same-settings timings did not change across the revisions that changed workspace policy, so it would not remove the launch bound.
- Changing the W precision to reduce launches. The schedule preserves the compensated W products used in the per-pair measurements. The same-mode displacement results are documented in [the split3 numerics note](../numerics/2026-11-26-triples-fp32-split3.md).
- Reserving more scratch so that more segments batch together. That would change the admitted peak and the budget at which strict FP64 fallbacks begin. The batched schedule already removes the launch bound within the existing reservation.

## Invariants

- Packing, digits, FP64 recombination order, the pack audit and nonfinite reporting are the same for all three schedules.
- Split-W scratch reservation, admission, the schedule identity `spatial-triples/three-limb-fp32-sgemm-k64-high8-fp64-combine-v1` and the nine-per-segment product counters are unchanged. The counters measure limb products, not launches.
- Schedule selection depends only on the operand shape and the reserved scratch. The per-pair loop remains the minimum-scratch fallback.

## Evidence

`tests/native/test_triples_compensated_w_cuda.cu`, which `tests/python/test_triples_compensated_w_cuda.py` drives, forces each schedule through scratch limits. It checks the result against a long-double oracle with an a priori bound, against the other schedules and against repeated execution. It also checks nonfinite and lost-digit reporting and the minimum per-pair scratch. Its first L20 run stopped at "TN 9x4x64 pairs: device error 26". The lost-low-limb audit correctly rejected generated operands that spanned 2^138 inside one segment. The test scales its oracle vectors and adds a 2^-200 injection that every schedule must report. The test then passed twice on an A100.

The complete CNa3O3 ha-D frozen-core (T+) energy and force ran in a fresh process per measurement on one L20 card (cabs rank 471, 257 basis functions, 31 occupied orbitals with 19 frozen). The comparison used the original per-pair split3 schedule and the stacked schedule in `src/cc/df_triples_compensated_w.cuh`.

| Preconvergence | Split3 launches | E+F, s | (T), s | Lambda, s | CCSD, s |
|---|---|---:|---:|---:|---:|
| FP64 | Per-pair | 1533.9 | 436.5 | 650.5 | 251.6 |
| FP64 | Stacked | 1245.3 | 148.4 | 650.6 | 251.6 |
| auto | Per-pair | 1091.8 | 434.9 | 276.7 | 185.9 |
| auto | Stacked | 804.5 | 148.3 | 276.6 | 185.8 |

The schedules gave bitwise-identical energies in every component and (T+) forces within 2.4e-13 Eh/bohr. Both executed 31,733,424 FP32 products (79,164 forward, 31,654,260 reverse) and 9,828 FP64 GEMMs, with a 4,932,556,032-byte triples workspace. GPU peak memory was 21,672.6 MiB with FP64 preconvergence and differed by 2 MiB with auto preconvergence. Changing CNa3O3 from FP64 to auto preconvergence changed the (T+) energy by 4.0e-12 Eh and forces by 5.4e-11 Eh/bohr. The stacked build's FP64-preconverged split3 qualification passed the ihb93, H2O and OH ha-D panels and the H2O and OH Richardson finite-difference gates. Across all components, their best errors differed from the per-pair baseline by at most 5.4e-13 Eh/bohr, and the largest was 1.62e-9 Eh/bohr. For H2O (T+), they were 8.9135e-11 and 8.9111e-11 Eh/bohr.

## Consequences

In the split3, FP64-preconverged CNa3O3 comparison, (T) took 2.9x less time and the full energy-and-force endpoint took 18.8% less time. With auto preconvergence the corresponding stacked split3 endpoint took 804.5 s. The batched schedule fits in the reserved split-W scratch.

## Revisit when

- Scratch-limited shapes run the `rows` or `pairs` schedule in production endpoints.
- The pedantic provider can no longer guarantee strided-batched SGEMM, or its kernels can no longer represent the high and middle partial sums exactly.
- A different compensated W scheme replaces `fp32_split3`.

## References

- `src/cc/df_triples_compensated_w.cuh` implements the stacked launches.
  `tests/native/test_triples_compensated_w_cuda.cu` covers the bounded oracle
  and lost-low-limb injection.
- [DF-UCC v2 CUDA working-set admission](2026-11-26-df-ucc-v2-cuda-capacity.md)
  describes the 4 MiB source cuBLAS workspace included in the stacked build.
- `../numerics/2026-11-26-triples-fp32-split3.md` and `../numerics/2026-11-26-triples-w-execution-precision.md`.
