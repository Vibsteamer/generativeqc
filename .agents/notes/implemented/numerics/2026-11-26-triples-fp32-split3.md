# Compensated FP32 W products in molecular triples

Status: implemented
Date: 2026-11-26

The separately requested `fp32_split3` schedule addresses the displacement noise of single-pass FP32 DF panels and W moments. The original CPU reference rounds inputs and outputs around a double GEMM. For the frozen-core all-electron HBr test, its fixed-weight (T+) force and strict FP64 displaced energies agreed within 4.2e-11 Eh/bohr at h = 2e-4 bohr. Its own displaced energies gave a force error of 2.11e-7 Eh/bohr. Both displacements and the independent strict controls are required at each declared step.

## Product and pullback

Each matrix product partitions its reduction dimension into chunks of up to 64. Within a chunk, each left row and right column has its own power-of-two scales. Two successive rounded eight-bit-magnitude slices encode the high and middle parts of the FP64 inputs. A normalized FP32 residual supplies the third part. Nine FP32 matrix products form all pairs. An FP64 sum recombines their rescaled results and publishes the panel or W moment. Keeping the panel in FP64 between products avoids rounding the recovered low bits away before the next contraction.

For a pair of high or middle slices, the integer coefficients have magnitude at most 256. Every possible partial sum has magnitude at most 64 × 256² = 2²² in a common grid unit. FP32 exactly represents these integer partial sums; products involving the residual still incur FP32 rounding. Plain high/low FP32 casts leave the leading product and its accumulation rounded even when the cross terms are included.

The host spin-blocked implementation uses OpenBLAS SGEMM. Forward DF-panel and W products use the expansion. The reverse applies the same expansion to both adjoints of each virtual and occupied W product and to both DF-factor pullbacks for the forward panel. Fock and disconnected V response, denominators and outer force accumulation use FP64. Total and reverse FP32 counters measure physical matrix products. Energy-only evaluation has zero reverse work.

For an `m × k` times `k × n` product with `s = min(k, 64)`, the host reserves FP32 storage for `3s(m+n) + mn` elements, integer scales for `3(m+n)` limbs, `mn` FP64 elements for publication after validation, and one 64-double stack buffer. Eighteen compiled energy/force checks compared the calculated peak against the reserved input buffers plus actual C++ heap allocations; the nonempty cases differed by the explicitly accounted 512-byte stack buffer. BLAS-internal allocations are outside this numerical-buffer measurement. The stage adds these bytes to the retained input, output, permutation and occupied-response storage at the peak of each sequential spin channel. Scalar-only host builds reject this opt-in. Exponent normalization supports ordinary FP64 scales outside the FP32 range. Unrepresentable normalized residuals and nonfinite recombination are reported as errors.

The spatial CUDA schedule uses the shared pedantic cuBLAS provider and reserves space for the accepted three-panel policy, FP64 panels, split operands, scales, SGEMM partials and FP64 W assembly. Its identity is `spatial-triples/three-limb-fp32-sgemm-k64-high8-fp64-combine-v1`. Its reverse expansion covers the full panel/W graph, including the occupied W and DF-factor cotangents. It returns a resource or provider error if it cannot execute that schedule. The spin-blocked CUDA endpoint currently supports the original strict and raw forward policies.

## Numerical and timing evidence

The scratch comparison tested conventional (T) and the fixed-weight part of (T+) on one Cartesian coordinate each in H2O and all-electron HBr, with independent displacements at 1e-4, 2e-4, 4e-4 and 8e-4 bohr. The native host forward changed for these prototypes and their reverse used the original strict FP64 implementation.

| Host forward schedule | Max own FD error, Eh/bohr | Max center difference from strict FP64, Eh | Max center force difference, Eh/bohr | Single-run campaign, s |
|---|---:|---:|---:|---:|
| Original rounded DGEMM | 6.37e-7 | 4.26e-11 | 2.47e-11 | 22.38 |
| Two slices, four SGEMMs per chunk | 5.89e-10 | 2.42e-13 | 2.76e-13 | 27.75 |
| Three slices, nine SGEMMs per chunk | 2.03e-10 | 5.95e-16 | 1.01e-13 | 28.31 |

Runs used one OpenBLAS thread and shared the machine with other compilation work. Times include the reference calculation, four center force evaluations and sixteen displaced energy calculations per molecule. They describe the host prototypes only.

A separate 662-coordinate two-stage panel/W polynomial tested compensated forward and adjoint products for all five inputs. At an energy of −0.006825 Eh, two slices gave a maximum same-mode finite-difference error of 1.37e-8 at the declared steps. Three slices gave 3.97e-11 and a maximum adjoint difference from FP64 of 2.11e-14. The absolute finite-difference tolerance depends on the energy, derivative and step scales. Geometry-dependent (T+) weights also require the separate scaling response of the molecular endpoint.

A compiled, open-shell host check of the full three-slice implementation activated all four spin channels with four alpha and three beta occupied orbitals, three alpha and four beta virtuals, and nine auxiliary functions. It counted 3,222 forward and 6,444 reverse FP32 matrix products. At the two declared steps, eleven independent factor, amplitude and occupied-integral perturbations gave a maximum same-mode directional error of 1.04e-9. The largest difference from the strict analytic directional gradient was 1.90e-13.

The molecular acceptance tests live in `tests/python/test_rhf_triples_fp32w_gate1.py`. The companion note `2026-11-26-triples-w-execution-precision.md` describes directional telemetry and the legacy precision schedules. `../performance/2026-11-27-triples-split3-stacked-launches.md` describes how the CUDA schedule batches the nine limb products into fewer launches.
