# Molecular fixed-source contribution to the UCCSD-F12b gradient

For a physical coordinate the same perturbation changes W, C, and the scalar
V+B−X in both spin frames. A derivative of an independent CC input, such as
one mixed-spin W block, captures only that block's contribution.

The molecular source builder applies a TensorIR JVP to the same pair graphs
and the same spin permutations used for the forward inputs. Projected MO
sources and extended Fock/exchange matrices carry their total tangents in the
moving frame. Occupation selectors stay fixed. The full OBS Coulomb fit uses
the equation J dD = dG − dJ D. Its inverse comes from the central full-rank
whitener, giving a result independent of the choice of whitening gauge.

The response owner freshly checks the fixed-source primal and Lambda before
evaluating the joint source direction of E + Lambda·R. The full residual seeds
come from the packed-dual lift, with a factor of one quarter on each same-spin
doubles block. The ordinary CC Hamiltonian, reference and separate CABS singles
contributions must be assembled to obtain the full gradient.

The unit tests compare every molecular source against two-step finite
differences with a nondiagonal fitting metric. Independent dense spin-orbital
solutions at the two source perturbations check the native stationary
contraction. On the retained bent NH2 cc-pVDZ-F12 H1-z tangent, all 36 pairs
gave a source gradient of +0.0000543050714 Eh/bohr. Central differences of the
source-interpolated stationary Lagrangian at steps 0.0002 and 0.0001 bohr
differed by 1.1e-9 and 2.6e-10 Eh/bohr. The molecular geometry and stationary
amplitudes were fixed during this interpolation.

The molecular directional derivative also accepts all ordinary canonical
Hamiltonian directions on this frame and the full UHF reference derivative.
A DF adapter differentiates each chemists' tile through its fitted and raw
legs, applying the same permutations and exchange signs used by the ordinary
UCC solver. Its output can then enter the joint correlation Lagrangian JVP
alongside the geminal sources. The reference derivative enters once, outside
the correlation graph. Separately computed CABS singles stay external.

For the retained NH2 H1-z direction, the ordinary CC contribution was
−0.007019536472 Eh/bohr and the reference derivative was −0.018665872204
Eh/bohr. Adding the fixed-source term gave −0.025631103605 Eh/bohr. Two
source-interpolation checks of the correlation derivative differed by 1.21e-9
and 2.81e-10 Eh/bohr. Independent displacements of converged molecular UCC
roots remain necessary to qualify this as a finite-difference nuclear check.

## Cartesian energy and forces

`tools/generativeqc_cc/uccsd_f12_cartesian.py` evaluates the molecular
energy and all nuclear forces from converged T/Lambda, central extended-spin
DF data and one relaxed molecular source tangent for each Cartesian
coordinate. It rebuilds the physical CC primal and transposes its Lagrangian
once. The packed stationarity norm checks the supplied Lambda before the
driver contracts the ordinary DF and fixed-cusp geminal derivatives. The
moving extended Fock also supplies the two spin-resolved CABS singles
contributions when requested.

The all-electron doublet NH2 example uses cc-pVDZ-F12, cc-pVDZ-F12-OptRI
and common aug-cc-pVTZ-RI fitting. All 36 occupied pairs and 52,937
packed amplitudes enter the molecular calculation. The energy was
−55.84912880607271 Eh with CABS singles, including −0.262928016620654 Eh of
CC correlation and −0.002235910899654 Eh of CABS singles. The maximum
amplitude residual was 3.57e-12 and the packed Lambda stationarity norm was
4.71e-12. Nine independently supplied analytic source directions gave the
following forces in Eh/bohr.

| Nucleus | x | y | z |
| --- | ---: | ---: | ---: |
| N | −0.052883362478 | +5.36e-16 | −0.008935941019 |
| H1 | +0.026031110064 | +2.48e-16 | +0.024443099893 |
| H2 | +0.026852252413 | −9.20e-16 | −0.015507158874 |

Independent displacements rebuilt the UHF orbitals, all integrals, CABS
singles and the 52,937-amplitude CC root at each of six geometries. H1-z
errors against total-energy differences were 2.38e-8 and 4.77e-8 Eh/bohr
for steps of 0.0002 and 0.0001 bohr. N-x gave 3.20e-8 Eh/bohr at 0.0002
bohr. The largest correlation force discrepancy over the three differences
was 6.98e-10 Eh/bohr. Independent UHF and CABS differences cover all nine
coordinates. The complete-CC displaced-energy checks cover N-x and H1-z.
The local CPU receipts, full source hashes and displaced roots are in
`validation/focal-convergence-20261118/fixed_source_full_cuda/source_response/molecular_cartesian_nh2/{analytic,verified}.json`
in the surrounding workspace.
