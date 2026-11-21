# Reuse spatial Coulomb pair matrices for RHF CC response

Implemented the bounded preparation and repeated-action graphs, 2026-11-21.

Reapplying the auxiliary-factor virtual ladder in every Λ transpose spends most of the CH₃Br/ha-T budget on repeated contractions. A restricted reference supplies one spatial Coulomb operator. Prepare its unique chemists' pair matrix in row tiles from the matched MP2Fit factor and raw tensor. Fold this operator into symmetric and antisymmetric virtual-pair matrices. Mixed-spin residuals use both matrices and keep every ordered occupied pair. Same-spin residuals contract only distinct occupied and virtual pairs and expand with antisymmetric signs. The paired projections and their generated transposes operate on the physical amplitude and dual spaces. Keep the factor reverse from the bounded auxiliary path for the single fresh source pullback after T and Λ qualification.

This representation requires the same spatial orbitals for both spin lines. The provider must verify that identity. Equal dimensions alone are insufficient. Preparation checks within-pair and pair-exchange symmetry and admits its added host memory before running the fitting graphs. The caller still admits the fitting executor, retained inputs, complete iterative endpoint, and source and nuclear workspace.

## Evidence

For CH₃Br/ha-T with 17 occupied, 185 virtual orbitals, and 572 auxiliaries, the two persistent matrices take 4,685,539,400 bytes. With a 256-row staging tile, the host preparation upper bound is 7,055,561,080 bytes before the separately admitted executor workspace and input ownership. The repeated transpose SM89 graph plans 5,902,909,952 device bytes and 5,160,544,408 host-staging bytes. The generated scalar work for the isolated bare transpose is 497,690,320,750, down from 12,597,845,720,120 for the 72 auxiliary slices. The packed path changes FP64 reduction order.

Including the existing remainder graph, the static action estimate is 5,619,931,444,242 per Λ transpose. With the one-time pair build, two qualification actions, and the energy RHS, the existing $2.10\times10^{14}$ budget accommodates 35 transpose actions. The existing dependency analysis identifies 1,089,510,910,698 invariant remainder contraction operations and 1,195,133,992 bytes in its immutable frontier. Replaying only the dynamic portion would permit approximately 44 actions under the same model. That frontier still needs to be bound by the stationary runtime owner. These are static models. No device timing or convergence guarantee follows.

Offline numerical comparisons cover the bare residual, full amplitude tangent, and packed physical Λ transpose with unequal singles/doubles values across the two spin lines, ragged matrix construction, and an empty same-spin occupied-pair sector. The tests use explicit positive-definite metrics and independently solved factors. Existing F12 and factor-source checks preserve the reference derivative path.

## Revisit gate

The public molecular owner still uses the dense route. Integrate the prepared matrices and an admitted invariant frontier into the stationary epoch with shared provenance and one-time accounting. Reuse the accepted CH₃Br/ha-D state to compare energy, the physical T and Λ residuals, source responses, and all 15 nuclear force components before running the exact ha-T geometry.

## References

- [Split CCSD-F12b virtual ladder](2026-11-21-uccsd-f12-df-ladder-response.md)
- `tests/python/test_uccsd_rhf_virtual_pairs.py`
