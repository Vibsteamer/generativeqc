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

These counts exclude the same-spin and remainder equations. The coordinator will also have to admit repeated transfer of the mixed operators, sampled native peak, fully retained sources and the state/Λ workspaces. The fitting sequence and complete native equations need to run together before claiming molecular qualification.

Three panel widths and a host-budget rejection pass offline with unequal spin counts and independent, deliberately nonsymmetric factors. The generated native CPU actions at width two match the complete auxiliary-space primal, JVP, and VJP for all three mixed-amplitude adjoints at absolute tolerance $4\times10^{-14}$. The public UHF molecular owner remains dense pending same-spin preparation and complete stationary integration. The RHF molecular owner and its L20 candidate remain unchanged by this work.
