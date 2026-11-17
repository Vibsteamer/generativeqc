# Native radial integrals for fitted Slater geminals

Status: implemented at the raw Gaussian integral boundary
Date: 2026-11-17

## Geometry and radial moments

Reuse the Coulomb product-center binder and its ordinary Hermite prefactor for the two-center auxiliary metric, three-center orbital-pair/auxiliary terms and four-center controls. For the auxiliary metric and fitting cases the unit slots have exponent and angular momentum zero. The integral operator supplies only the normalized radial seeds. This keeps coordinate shifts and Gaussian-product derivatives identical across operators.

Let ρ be the reduced exponent of the two electronic Gaussian products, T = ρ|P−Q|², z the exponent of exp(−z r₁₂²), g = z/(ρ+z) and h = ρ/(ρ+z). The nth Hermite seed is (−∂/∂T)ⁿ applied to the zeroth seed. For the Gaussian it is

    M_n = √π h exp(−gT) gⁿ / (2√(ρ+z)).

For exp(−z r₁₂²)/r₁₂, transform the Coulomb Gaussian integral with u = s/(ρ+z+s²)¹ᐟ². The result is

    M_n = h exp(−gT) ∫₀¹ (g+h u²)ⁿ exp(−h T u²) du.

Use a single ordinary Boys vector at hT and repeated positive binomial averaging. This eliminates per-moment numerical quadrature in the CUDA operator. The third primitive uses −∂/∂z of the Gaussian seed for r₁₂² exp(−z r₁₂²). Its product with 4 z₁ z₂ at z = z₁+z₂ produces the dot product of the two radial gradients on electron 1.

## Fit identity and cross terms

A six-Gaussian fit is prepared from β in bohr⁻¹. Scale each primitive exponent by β². The raw Psi4 and the 1/β-normalized Slater conventions are explicit choices. The fixed-amplitude contraction and its integral convention must be qualified together. Geometry changes keep the chosen β fixed. Products use 21 distinct terms when both references denote the same fit and 36 for two different fits. The mixed products require both exponents, including in the gradient product. Accumulate all primitive terms with compensated FP64 summation and apply the shared geometry prefactor once. [Psi4 1.11 numeric fit](https://github.com/psi4/psi4/blob/v1.11/psi4/src/psi4/libmints/mintshelper.cc) · [A normalized Slater factor in CC theory](https://doi.org/10.1063/1.4963748)

## Qualification

`tests/python/test_f12_radial_integrals.py` uses independent real-space radial integration for physical 2c/3c/4c s primitives and their nuclear-center derivatives, with displacements of 3×10⁻⁴ and 10⁻⁴ bohr. It tests radial orders 0 through 20 across 270 input sets. Forty-eight contracted 2c/3c cases include the two normalizations, three ordered β combinations (0.9/2.2, 2.2/0.9 and 1.4/1.4), four operators, translation or directional derivative checks, and independent Psi4 Gaussian coefficients. Increasing the real-space oracle quadrature from 320 to 480 nodes changes the sampled values by less than 10⁻¹³ relative to max(1, |integral|). Resolving the tightest primitive matters for the gradient product.

Run the same test on an allocated CUDA device with `GENERATIVEQC_TEST_CUDA=1`. The device build and host build use identical scientific functions. The tested radial order counts moments. Certification of a complete g/h/i shell or an ECP molecule also requires the corresponding contracted spherical angular blocks, transforms and response.

## Robust density fitting of an orbital-pair tile

Let D be the coefficients of orbital-pair densities fitted in the Coulomb metric, O the raw three-center integrals of the desired geminal operator and K its own auxiliary two-center metric. With possibly different left and right tiles the four-center approximation is

    D_Lᵀ (O_R − K D_R) + O_Lᵀ D_R.

Expanding both true densities about their fits shows that the omitted contribution is the matrix element of the two fitting residuals with each other. Using a single whitened product for every geminal operator loses that cancellation. If the operator is Coulomb, the defining metric equation reduces the expression to one two-factor contraction. Evaluate a tile through TensorIR in FP64 and apply its transpose with the actual downstream weights. The five general-operator cotangents are for D_L, D_R, O_L, O_R and K. D still depends on the three-center Coulomb source and its metric. On a nonsingular fixed space, J D = G gives J dD = dG − dJ D. Rank truncation and its geometric response belong to the upstream fitting construction. [Psi4 robust DF implementation](https://github.com/psi4/psi4/blob/v1.11/psi4/src/psi4/f12/integrals.cc) · [Explicitly correlated DF treatment](https://doi.org/10.1021/acs.jctc.2c01031)

The molecular regression reconstructs the independently computed Psi4 V and X diagonal occupied-pair entries and the entire virtual–virtual Coulomb block for that pair in DF-H₂ and DF-H₂O. It uses three operators with distinct auxiliary metrics and a rectangular occupied/CABS tile. The independent finite-model regression checks that the missing exact matrix element is quadratic in the fitting residuals, then compares the joint adjoint of all five inputs to two directional finite differences. Molecular tests use the input orbitals and fitting coefficients stored by Psi4. AO basis and metric nuclear derivatives, CABS construction and SCF response enter before this boundary.

On one NVIDIA L20 the largest absolute differences against those molecular references were 3.68×10⁻¹⁶ for V and 5.38×10⁻¹⁷ for X. The computed Coulomb blocks matched the stored FP64 entries in both cases. The absolute errors in the synthetic rectangular directional derivative were 8.21×10⁻⁹ and 8.87×10⁻¹⁰ at step sizes 3×10⁻⁴ and 10⁻⁴. Compiling and exercising the generated programs for these three cases took 60.1 s. The molecular cases used 92 and 198 auxiliary functions for H₂ and H₂O, respectively.
