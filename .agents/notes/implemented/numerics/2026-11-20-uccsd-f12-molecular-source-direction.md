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

### GPU execution boundary

The Cartesian driver can use a bounded `CudaCCTensorExecutor` for the fresh CC
residual and the full stationary source pullback, independently of the runner
used to build molecular integrals and their derivatives. The same executor's
`run` method supplies the TensorIR result expected by the tiled molecular force
contraction and source builders. The complete NH2/D pullback returns 38 ordinary
Hamiltonian cotangents and 18 active fixed-cusp source cotangents, together
with the five amplitude cotangents used to check Lambda. One redundant supplied
F12 source drops out of the energy and residual graph.

The archived L20 amplitude/Lambda solve used 27 amplitude iterations and 47
transposed actions. Its orbital-gap GMRES iteration ran on the host. The new
single-center comparison uses those retained T/Lambda and independently checks
all source weights and their contraction with the physical H1-z tangent.
Source construction, prior orbital response, reference force and CABS singles
are supplied by the recorded molecular calculation for this comparison. The
per-coordinate traversal, host staging, packed stationarity norm and final
scalar accumulation also remain on the host.

L20 jobs 2123729892435570688 and 2123734003381960704 completed the electronic
source contractions for all nine NH2/D coordinates. The first measured H1-z.
The second used its archived input dataset and stationary vectors to evaluate
one fresh full pullback and contract the other eight coordinate tangents. It
retained all 56 active device cotangents. Independent FP64 replay of the arrays
gave a maximum elementwise error of 1.05e-17 and reproduced all eight new
ordinary-CC and F12 source force components within 1e-10 Eh/bohr. Including
the recorded UHF and CABS terms gave −55.84912880607278 Eh. Finite differences
of independently reoptimized total energies agreed within 3.21e-8 Eh/bohr at
N-x and 4.78e-8 Eh/bohr across the two H1-z spacings. The signed archives,
source tensor hashes and numerical projections are in
`molecular_cartesian_nh2/l20_cartesian/passed.json` in the validation area
above. These runs consume the host-prepared moving Gaussian, spin-orbital and
fitting-metric directions, and add the host-verified UHF and CABS contributions.

The six ordinary Fock weights (oo, ov and vv in alpha and beta) can consume
the native geometry-derived Hcore and its Cartesian derivatives directly.
`tools/generativeqc_cc/uccsd_fock_nuclear.py` transforms the symmetric
Fock weights into the real spherical AO basis, applies both spin orbital
coefficient responses, and adds the independently supplied moving-MO
Coulomb-minus-exchange derivatives. It preserves the single copy of ov
and excludes frozen occupied rows and columns. The full CC Fock contribution
at NH2 H1-z is +0.009285805230 Eh/bohr, with +0.070221184863 from the
explicit AO core derivative, −0.030608596275 from the moving orbitals acting
on Hcore, and −0.030326783358 from Coulomb and exchange. The retained native
host execution reproduced the archived nuclear Fock force for all nine
coordinates within 3e-10 Eh/bohr. The other 50 CC source products, UHF and
CABS singles are separately assembled from their retained values.

Single-L20 job 2123741553728958464 subsequently generated the geometric Hcore
and all nine AO derivatives on device and contracted the six retained CC Fock
weights in FP64. Independent raw-array replay gave a maximum AO dH error of
1.01e-13, a maximum six-block nuclear force error of 1.81e-12 Eh/bohr, and a
maximum discrepancy of 2.25e-10 Eh/bohr against the three independent Fock
displacements. The signed `molecular_cartesian_nh2/l20_ordinary_fock_hcore/passed.json`
identifies the returned combined-basis H, dH, S and dS and the checked 54
spin-block derivatives.

The ordinary spin Fock cotangents can now propagate through the shared DF
Coulomb charge, spin-diagonal exchange and the implicit fit. The reverse map
returns energy weights for the Coulomb metric, AO three-center tensor and
the two complete ordinary MO coefficient matrices. Frozen occupied electrons
participate in the mean field. Nuclear force uses the negative action of
these four weights. Independent finite differences differentiate an AO
density-matrix J/K with one frozen alpha orbital and unequal occupations.

With the six NH2/D device weights, the geometry-derived CPU reference fit
and both native Gaussian derivative contractions reproduce the nine retained
moving-MO J−K force components within 1.50e-12 Eh/bohr. At H1-z the fitting
metric gives +0.001018189062, three-center integrals −0.055121304711 and
the two spin coefficient responses +0.023776332292 Eh/bohr. Independent
displaced Fock checks at H1-z and N-x agree within 2.27e-10 Eh/bohr.
The separate molecular qualification uses three shell centers, 158 combined
spherical AOs and the 198-function common fit. The coefficient derivatives,
the two previously verified Hcore terms and the other source, UHF and CABS
actions remain inputs. The validation directory is
`molecular_cartesian_nh2/l20_ordinary_fock_jk/` in the area described above.

For the six ordinary Fock inputs, Hcore contributes 2 H C W to each spin's
coefficient cotangent, where W is the symmetric CC Fock weight in that MO
frame. The measured DF Coulomb/exchange coefficient cotangent adds to it.
The resulting two weights include the norm-changing part of the AO overlap
and all canonical rotations. The same DF-UHF root supplies the metric
whitener, MO fitting factors and energy gaps to one transposed coupled-spin
orbital solve. Its output is a set of orbital-response weights on S, Hcore,
the Coulomb three-center tensor and the fitting metric. Their negative
nuclear contractions add to the explicit Fock force. The source no longer
takes Cartesian MO coefficient derivatives.

For the NH2/D six-weight observable the dense CPU CPHF control used 391
occupied/virtual rotations with residual 1.74e-17 and recovered the nine
original moving-orbital forces within 9.03e-13 Eh/bohr. The pure orbital
force on H1-z was −0.006832263983 Eh/bohr. The center reproduced both
retained canonical spin Fock matrices within 2.35e-10 Eh. Native CUDA
and independent centered controls for this route live in
`molecular_cartesian_nh2/l20_ordinary_fock_orbital_response/` in the
validation area above.
