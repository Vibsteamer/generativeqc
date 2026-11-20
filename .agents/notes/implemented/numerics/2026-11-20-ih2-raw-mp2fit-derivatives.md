# Raw Coulomb integrals for the IH2/D correlation fitting basis

The bent IH2 doublet uses 71 spherical orbital functions. Its correlation fit
contains 209 functions on iodine (through h) and 35 on each hydrogen
(through f). The 254-function diagnostic JKFit solves a different metric.
At fixed exponents and contractions the bare two-electron Coulomb integrals
depend on the shell geometry. ECP28MDF acts in the one-electron Hamiltonian.

`benchmarks/f12_raw_df_probe.cpp` streams the existing native full-range CUDA
DF integral source for one ordered AO-pair tile and one nuclear coordinate
at a time. It normalizes a supplied contracted spherical basis with the same
routine as the original native DF probe. All ordered pairs carry unit weight.
The output convention on disk is A[mu,nu,P], with the coordinate index first
on dA and dM. The qualification package transposes the fitting index of A
and dA to the first electronic axis for the spin-resolved molecular consumer.

The standalone comparison in `public_ih2_mp2fit_raw_cuda` in the
2026-11-18 focal validation area uses all three original atom centers and
the full nontruncated 279-function MP2Fit. Its libcint oracle differentiates
both orbital centers and the independent auxiliary center of the three-center
integral and both centers of the metric. Two centered shell displacements
independently reproduce those derivatives within 8.27e-9 at 1e-4 bohr. This
qualification computes raw integral data independently of the UHF solve and
of any subsequent whitening or MP2 spin contraction.
