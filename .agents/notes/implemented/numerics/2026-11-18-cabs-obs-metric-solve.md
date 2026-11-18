# Solving the orbital-basis overlap in CABS projection

For the combined orbital and complementary auxiliary basis (CABS), write the overlap blocks as
`A, D, B`, with `A` the orbital-basis overlap. The CABS lift needs `X` solving
`A X = -D`. Its projected metric is `B + D.T X`. A small eigenvalue in this last matrix
amplifies an inaccurate `X` when the complement columns are normalized.

Three original methane geometries with cc-pVDZ-F12 and its OptRI span exposed this effect on
an NVIDIA L20. The forward path computed a spectral inverse of `A` and multiplied by `D`.
The largest entries of `C.T S C - I` were 2.74e-9, 6.52e-9 and 3.52e-9, respectively.
All three exceeded the original 2e-9 gate. The skew gauge residual was below 1.4e-13.
The orbital-space condition numbers were between 5,858 and 7,591. All 154 resolved complement
directions were expected in a basis of 220 functions.

The native path factors the admitted `A` using FP64 Cholesky and applies two triangular
solves. The same factor also supplies the orbital-block inverse action on the reverse path.
The spectral admission test still protects the orbital space. The separate Schur eigensystem
selects the CABS rank and propagates its fixed-rank pseudoinverse derivative as before.
Factoring the entire combined overlap would impose positive definiteness even on disposable
auxiliary directions. Enlarging the residual threshold would hide the loss at the solve.

An independent CPU Cholesky calculation on the three Psi4 overlap matrices gave metric
residuals from 8.6e-12 or below to 3.1e-11. These host observations diagnose the algebra.
The CUDA regression in `tests/python/test_f12_projector_cuda.py` exercises a conditioned
orbital block, the complement metric, its orthogonality to the orbital basis, and the
analytic response to uniform overlap scaling at fixed seed. Matching a stock energy and
checking the molecular forces require the separate real-device endpoint.

The benchmark saves a measured central energy while the molecular adjoints are still being
assembled. The scalar checkpoint has a separate duration. A later failure of a derivative
therefore preserves the native energy and its computational provenance.
