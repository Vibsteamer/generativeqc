# Run bounded CCSD-F12b through the molecular RHF owner

Implemented an explicit provider option, 2026-11-21.

Use `MolecularFocalProvider.cuda(..., rhf_df_ladder=RHFDFLadderOptions(...))` to choose the bounded route on a canonical all-active RHF determinant. The native route builds ordinary Hamiltonian terms without four-virtual blocks and retains one spatial pair of raw and metric-solved MP2Fit factors. The same fixed-source DIIS owner iterates the remainder and packed virtual operator. Its Λ owner charges one fixed-amplitude preparation and combines each repeated replay with the packed bare transpose. Existing reference, physical residual, Λ, spin, and fitting identities still qualify the solution. The default dense path and UHF dispatch retain their existing behavior.

The current stationary source pass discards the packed spatial operators and the Λ frontier after independent replay checks. It stages the retained three-virtual source group and the remaining ordinary and fixed-geminal source group sequentially. The independent auxiliary-sliced factor reverse supplies each spin's cotangents. The ordinary source owner includes them in the MP2Fit metric, AO and canonical orbital response and uses the same analytic nuclear sink. The fixed-geminal path retains its own CABS, F12Fit, and scalar ECP pullback. The pure-energy route runs T only.

## Evidence

The compiled native CPU comparison covers a complete cold T, Λ, grouped ordinary and fixed source evaluation, and pure energy. Archived dense T and Λ are independently qualified on the same bounded equations with zero new solve iterations. Existing tests cover factor, raw AO and metric cotangent parity. A same-root physical GPU molecular replay remains necessary.

Static logical numeric bounds for the original CH₃Br RHF geometries on SM89 follow. Numbers include retained and graph workspace for the named electronic stage, without claiming device allocations for the whole molecular calculation.

| Cardinal | T owner, bytes | Λ owner with GMRES, bytes | Source VJP owner, bytes | Work per Λ transpose |
| --- | ---: | ---: | ---: | ---: |
| ha-D, restart 20 | 8,030,892,488 | 12,533,707,672 | 6,879,141,576 | 662,377,695,156 |
| ha-T, restart 8 | 39,034,164,520 | 47,677,736,480 | 38,439,086,096 | 4,536,075,126,366 |

With the previous $2.10\times10^{14}$ Λ work budget, the ha-T model admits 44 actions. The staged electronic domains fit an explicit 46 GiB combined numeric allowance with the restart-8 Krylov layout. A combined allowance does not indicate device occupancy. The largest named source graph needs 20,958,016,512 device bytes independently of its host staging and retained inputs. The reported solver budgets do not bound native integral and nuclear scratch or determine convergence.

## Physical acceptance

First run the original ha-D replay on the retained RHF determinant, its archived CABS rank and the same fitting identities and geometry. Reuse both adaptive scalar ECP libraries and the retained accepted `stationary.npz`. The worker can use `replay_focal_uccsd_f12b_with_cotangents` on the new sources and create `FocalUCCStationary(sources, primal, dual, weights)` for the same molecular provider's ordinary and fixed nuclear sinks. Require the existing energy, residual and stationarity thresholds, compare all 15 forces against the accepted ha-D replay, and retain the existing Br-z two-step evidence. The separate four-virtual fit factors are immutable snapshots from the same MP2Fit. Do not rebuild the original canonical SCF.

For the exact ha-T candidate, plan the cold T limit using `max_primal_work=940000000000000` with the default 100 T iterations. Use 44 or fewer Λ actions under the previous total-work budget and an explicit restart and Krylov budget. Verify actual CUDA free memory and the host cgroup at each guarded stage including AO and ECP, and record numerical failures in their own scope. Measure both the iteration compute and any compilation separately. The loader and worker can reuse the exact accepted CUDA chemistry libraries only after confirming compatible source, APIs and hashes. No ha-T endpoint is qualified by static admission alone.
