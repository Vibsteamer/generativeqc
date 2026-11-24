# Split the dressed CCSD-F12b virtual ladder and its response

Implemented compiler graphs and ordinary force source pullback, 2026-11-21.

The full F12b equations can evaluate the bare DF virtual ladder in auxiliary slices. The remainder contracts tau with the ordinary three- and two-virtual integral blocks before dressing the residual. It reuses the resulting occupied contraction in the dressed occupied ladder. No graph in the split produces a four-virtual integral tensor. The existing dense equations and their derivative graph identities were verified unchanged.

Each slice uses the same T1/T2-to-tau definition as the remainder. Its amplitude JVP and VJP include the disconnected singles term. Add the slice residuals, tangents, or cotangents to the matching remainder action before packing. The energy action is wholly in the remainder. Generate separate source VJPs for the three-virtual integrals and all other retained sources. The four factor cotangents for each slice fill its auxiliary range exactly once. The ordinary MP2Fit pullback adds these fit/raw weights before differentiating the metric solve and AO projection. The resulting ordinary AO and coefficient weights follow the existing orbital and nuclear response.

## Evidence

The FP64 gates compare all five complete residuals, correlation energy, amplitude directions, physical transpose and stationarity, remaining source cotangents, and factor directional contractions against the independently assembled dense virtual integrals. They cover unequal spin spaces, an empty occupied spin channel, and a ragged auxiliary slice. The full ordinary AO, fitting metric, and two orbital cotangents also agree with the dense virtual-integral path. The test tolerance for these comparisons is $2\times10^{-13}$.

At CH₃Br/ha-T with 17 occupied and 185 virtual orbitals per spin and 572 auxiliaries, static SM89 planning gives 14,121,788,416 device bytes for the remainder primal and 20,958,016,512 device bytes plus 5,641,611,008 host-staging bytes for the largest separate source VJP. These sizes exclude the rest of the stationary solver, stored feeds and the nuclear workspace.

With 71 eight-auxiliary slices and one four-auxiliary slice, the generated scalar-loop work model counts 15,166,628,580,070 operations for one full primal action, 17,720,086,843,612 for one full transpose action, and 17,721,443,200,346 for one stationarity replay. After one primal and one stationarity replay and the energy RHS, the existing $2.10\times10^{14}$ Lambda budget admits at most nine transpose actions. The retained ha-D replay has 41 historical actions, so the planner cannot infer convergence for the exact ha-T solve. These are static work counts, without a GPU timing claim.

## Revisit gate

The molecular solver still uses the dense entry point. Before enabling the split path, implement a repeat-action schedule that budgets and reduces the auxiliary-slice work and data movement. A reusable RHF spatial Coulomb representation may amortize repeated factor contractions if its exact spin mappings, measured resident storage and source cotangents pass validation. Qualify complete T and Lambda, ordinary and F12 forces on the retained CH₃Br/ha-D center with the same protocol before sending the exact ha-T case. Preserve work and memory admission at the complete endpoint.

## References

- [Bare DF virtual ladder](2026-11-21-uccsd-df-virtual-ladder-slices.md)
- `tests/python/test_uccsd_df_virtual_ladder.py` and `tests/python/test_uccsd_eri_nuclear.py`
