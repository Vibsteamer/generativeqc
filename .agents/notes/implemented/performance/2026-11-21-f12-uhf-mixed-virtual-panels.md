# Independent-spin virtual Coulomb panels for UCCSD-F12b

The bounded RHF path uses symmetric and antisymmetric virtual pair operators because both spins share canonical orbitals and DF factors. General UHF needs a rectangular mixed-spin operator on the independent alpha and beta virtual domains. The same-spin pair reduction cannot supply its opposite-spin action.

The mixed-spin panel uses the original metric-solved alpha factor and raw beta factor to stage each disjoint range of target alpha virtual orbitals. Each native fitting graph writes one panel. The prepared operator holds the panels on the host, and the generated repeated residual action takes only one panel at a time. Its transpose takes the matching target seed, returns full mixed-spin amplitude and two singles adjoints, and sums them over panels. This preserves the exact disconnected-singles T derivative. The analytical nuclear derivative of the factor is still supplied by the existing auxiliary-sliced stationary source transpose. The implementation accepts distinct, nonsymmetric factors; it performs no projection of either canonical spin space.

An explicit host admission counts retained panels, the current native output, factor copies, and finite checks. The native executor separately plans and admits each GPU graph. For a heavy open-shell layout with 17/16 alpha/beta occupied, 185/186 virtual, and 572 fitting functions, width 16 gives the following static result.

| Domain | Bytes or scalar operations |
|---|---:|
| Retained host panels | 9,472,384,800 bytes |
| Host preparation cap including retained panels | 10,565,878,496 bytes |
| Maximum additional native host staging among the tested graphs | 991,224,704 bytes |
| Largest native `sm_89` device graph | 1,223,778,048 bytes |
| One-time mixed-panel fitting | 1,354,808,070,044 operations |
| All panels for one bare mixed residual | 645,980,173,392 operations |
| All panels for one bare mixed adjoint | 645,989,532,912 operations |

The same-spin action uses a separate folded antisymmetric matrix in each spin frame. Preparation confirms the within-pair and Coulomb pair-exchange symmetries with the existing FP64 thresholds. A spin with an empty physical doubles space needs no matrix. It retains 2,317,443,200 alpha and 2,368,096,200 beta bytes for the modeled layout. The largest same-spin device graph is 2,977,460,224 bytes. In combination with the mixed term, the three bare operators retain 14,157,924,200 host bytes. Preparing both same-spin terms before the mixed term bounds incremental host preparation at 15,251,417,896 bytes before the separately admitted native staging and inputs. The three-channel bare residual and adjoint cost 796,797,377,873 and 797,056,980,850 scalar operations respectively.

`prepare_uhf_virtual_ladder` plans all three sectors before fitting. It gates incremental host residency, peak device graph, total fitting work and per-action work separately. Native output panels stack into the physical mixed residual. The stationarity action splits mixed seeds at the same boundaries and sums the physical bars before the packed dual map. Empty beta occupation instantiates interaction-free mixed and beta channels as zero residuals without planning nonexistent native work.

These counts exclude the remainder equations. The opt-in `MolecularFocalProvider.cuda(..., uhf_df_ladder=UHFDFLadderOptions(...))` integrates all three bare channels with that remainder and the complete native T and Λ solvers. It borrows the four independent immutable MP2Fit factors, charges both stages' coincident host and device allocations and guards the iteration and action counts. Source reverse first requalifies the same T/Λ and releases the prepared ladder. It then accumulates the four separate virtual fit cotangents and sends them to the same orbital, metric, AO and nuclear/ECP force owner. Both energy-only calls and replay of archived T/Λ use the same molecular equations. The previous public dense UHF route remains the default.

Three panel widths, a host-budget rejection, and both same-spin operators pass offline with unequal spin counts. The generated native CPU actions at width two match the complete auxiliary-space primal, JVP, and VJP for all three mixed-amplitude adjoints at absolute tolerance $4\times10^{-14}$. A compiled native CPU stationary regression compares bounded and dense UHF T, Λ, energy, source weights and archive replay with distinct alpha/beta energy arrays and fitted factors. The AO/nuclear regression also compares the independent-spin fit VJP and validates the production builder's omitted four-virtual blocks. The separate L20 build frozen at `72c3b45c2` contains the RHF owner only. The new UHF path still needs a compatible L20 build and unchanged real molecular controls.
